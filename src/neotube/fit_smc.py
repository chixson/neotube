from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from astropy.time import Time, TimeDelta
from functools import lru_cache
import multiprocessing as mp
from scipy import stats
from scipy.stats import qmc

from .admissible_seed import (
    SeedConfig,
    seed_cloud_from_observations,
    _attrib_from_state_cached,
)
from .fit_cli import load_observations
from .horizons import fetch_horizons_state
from .constants import AU_KM, C_KM_S, GM_SUN
from .models import Attributable, Observation, ReplicaCloud
from .ranging import (
    attrib_from_state_with_observer_time,
    build_attributable_vector_fit,
    build_state_from_ranging,
    _observer_helio_state,
)
from .propagate import propagate_state_kepler, _body_posvel_km_single, _site_states
from .rng import make_rng
from .sbdb import fetch_sbdb_covariance
from .site_checks import filter_special_sites
from .three_pt_exact import (
    _default_nbody_accel,
    _compute_accels_for_rho,
    _fit_tangent_plane_quadratic_coeffs,
    _tangent_basis,
    solve_three_point_coupled,
    solve_three_point_exact,
    solve_three_point_gauss,
)


def _midpoint_time(observations: Sequence[Observation]) -> Time:
    if not observations:
        raise ValueError("Need observations to compute midpoint.")
    jd = np.array([ob.time.tdb.jd for ob in observations], dtype=float)
    return Time(jd.mean(), format="jd", scale="tdb")


def _ranging_reference_observation(
    observations: Sequence[Observation],
    epoch: Time,
) -> Observation:
    if not observations:
        raise ValueError("Need observations to choose reference observation.")
    return min(observations, key=lambda ob: abs((ob.time.tdb - epoch.tdb).to_value("s")))


def _attrib_vector_from_state(
    state: np.ndarray, obs_ref: Observation, t_obs: Time
) -> np.ndarray:
    attrib, _, _ = attrib_from_state_with_observer_time(state, obs_ref, t_obs)
    return np.array(
        [
            attrib.ra_deg,
            attrib.dec_deg,
            attrib.ra_dot_deg_per_day,
            attrib.dec_dot_deg_per_day,
        ],
        dtype=float,
    )


def _wrap_deg(delta_deg: float) -> float:
    return (delta_deg + 180.0) % 360.0 - 180.0


_MC_CTX: dict[str, object] = {}


def _mc_worker_init(ctx: dict[str, object], jd_round_digits: int = 9) -> None:
    # Install per-process caches and store context.
    global _MC_CTX
    _MC_CTX = ctx
    from . import ranging as _ranging_mod
    _orig_body = _ranging_mod._body_posvel_km_single
    _orig_site = _ranging_mod._site_states

    @lru_cache(maxsize=10000)
    def _body_cache(body: str, jd_rounded: float):
        t = Time(jd_rounded, format="jd", scale="tdb")
        return _orig_body(body, t)

    def _body_wrapper(body: str, t: Time):
        jd_key = round(float(t.tdb.jd), jd_round_digits)
        return _body_cache(body, jd_key)

    @lru_cache(maxsize=10000)
    def _site_cache(jd_rounded: float, site_code: str, obs_pos_tuple, allow_unknown_site: bool):
        t = Time(jd_rounded, format="jd", scale="tdb")
        obs_pos = None if obs_pos_tuple is None else np.asarray(obs_pos_tuple)
        return _orig_site(
            [t],
            [site_code],
            observer_positions_km=[obs_pos],
            observer_velocities_km_s=None,
            allow_unknown_site=allow_unknown_site,
        )

    def _site_wrapper(
        times,
        site_codes,
        observer_positions_km=None,
        observer_velocities_km_s=None,
        allow_unknown_site=True,
    ):
        if len(times) == 1 and len(site_codes) == 1:
            t = times[0]
            site = site_codes[0]
            obs_pos_tuple = None
            if observer_positions_km is not None:
                pos0 = observer_positions_km[0]
                obs_pos_tuple = None if pos0 is None else tuple(np.asarray(pos0).tolist())
            jd_key = round(float(t.tdb.jd), jd_round_digits)
            return _site_cache(jd_key, site, obs_pos_tuple, bool(allow_unknown_site))
        return _orig_site(
            times,
            site_codes,
            observer_positions_km=observer_positions_km,
            observer_velocities_km_s=observer_velocities_km_s,
            allow_unknown_site=allow_unknown_site,
        )

    # Monkey patch in worker to use cached ephemerides/site states.
    global _body_posvel_km_single, _site_states
    _body_posvel_km_single = _body_wrapper
    _site_states = _site_wrapper
    _ranging_mod._body_posvel_km_single = _body_wrapper
    _ranging_mod._site_states = _site_wrapper


def _mc_worker(draw_vec: np.ndarray) -> dict[str, object]:
    ctx = _MC_CTX
    rho_grid = ctx["rho_grid"]
    rhodot_grid = ctx["rhodot_grid"]
    obs_window = ctx["obs_window"]
    obs_ref = ctx["obs_ref"]
    epoch = ctx["epoch"]
    chi2_delta = ctx["chi2_delta"]

    attrib_draw = Attributable(
        ra_deg=float(draw_vec[0]),
        dec_deg=float(draw_vec[1]),
        ra_dot_deg_per_day=float(draw_vec[2]),
        dec_dot_deg_per_day=float(draw_vec[3]),
    )
    obs_cache: list[tuple[np.ndarray, np.ndarray]] = []
    for ob in obs_window:
        obs_pos, obs_vel = _observer_helio_state(ob, ob.time)
        obs_cache.append((obs_pos, obs_vel))

    chi2_grid = np.full((len(rho_grid), len(rhodot_grid)), np.inf, dtype=float)

    for i, rho_km in enumerate(rho_grid):
        states = []
        j_indices = []
        for j, rhodot_km_s in enumerate(rhodot_grid):
            try:
                st = build_state_from_ranging(
                    obs_ref, epoch, attrib_draw, float(rho_km), float(rhodot_km_s)
                )
            except Exception:
                continue
            if not _state_basic_ok(st):
                continue
            states.append(st)
            j_indices.append(j)

        if not states:
            continue

        states_arr = np.asarray(states, dtype=float)
        n_valid = states_arr.shape[0]
        chi2_vec = np.zeros(n_valid, dtype=float)

        for k, ob in enumerate(obs_window):
            obs_pos, obs_vel = obs_cache[k]
            st_em_list: list[np.ndarray | None] = []
            for sidx in range(n_valid):
                st = states_arr[sidx]
                try:
                    _, st_em = _emission_epoch_for_state(
                        st,
                        epoch,
                        ob,
                        ob.time,
                        max_iter=None,
                        tol_sec=1e-6,
                        obs_pos=obs_pos,
                        obs_vel=obs_vel,
                    )
                except Exception:
                    st_em = None
                st_em_list.append(st_em)

            ras = np.empty(n_valid, dtype=float)
            decs = np.empty(n_valid, dtype=float)
            valid_mask = np.ones(n_valid, dtype=bool)
            for sidx, st_em in enumerate(st_em_list):
                if st_em is None:
                    valid_mask[sidx] = False
                    continue
                try:
                    attrib_pred, _, _ = _attrib_from_state_cached(st_em, obs_pos, obs_vel)
                except Exception:
                    valid_mask[sidx] = False
                    continue
                ras[sidx] = attrib_pred.ra_deg
                decs[sidx] = attrib_pred.dec_deg

            if not np.any(valid_mask):
                chi2_vec[:] = np.inf
                break

            sigma = max(1e-6, float(ob.sigma_arcsec))
            dra_deg = ((ras - ob.ra_deg + 180.0) % 360.0) - 180.0
            dra_deg = dra_deg * np.cos(np.deg2rad(decs))
            ddec_deg = decs - ob.dec_deg
            dra_arcsec = dra_deg * 3600.0
            ddec_arcsec = ddec_deg * 3600.0
            dra_arcsec[~valid_mask] = 1e12
            ddec_arcsec[~valid_mask] = 1e12

            chi2_vec += (dra_arcsec / sigma) ** 2 + (ddec_arcsec / sigma) ** 2

        for local_idx, j in enumerate(j_indices):
            chi2_grid[i, j] = float(chi2_vec[local_idx])
    chi2_min = float(np.nanmin(chi2_grid))
    min_idx = int(np.nanargmin(chi2_grid))
    min_i, min_j = np.unravel_index(min_idx, chi2_grid.shape)
    threshold = chi2_min + chi2_delta
    mask = chi2_grid <= threshold
    flat = mask.ravel(order="C")
    accepted_count = int(np.sum(flat))
    packed = np.packbits(flat).tobytes()
    return {
        "packed_mask": packed,
        "nbits": flat.size,
        "accepted_count": accepted_count,
        "theta": {
            "alpha_deg": float(draw_vec[0]),
            "alpha_dot_deg_per_day": float(draw_vec[2]),
            "delta_deg": float(draw_vec[1]),
            "delta_dot_deg_per_day": float(draw_vec[3]),
            "rho_km": float(rho_grid[min_i]),
            "rhodot_km_s": float(rhodot_grid[min_j]),
            "chi2_min": chi2_min,
        },
    }

def _state_physicality(
    state: np.ndarray, mu: float = GM_SUN
) -> tuple[bool, float, float, float, float]:
    try:
        r = np.asarray(state[:3], dtype=float)
        v = np.asarray(state[3:6], dtype=float)
        rnorm = float(np.linalg.norm(r))
        vnorm = float(np.linalg.norm(v))
        eps = 0.5 * (vnorm**2) - mu / (rnorm + 1e-300)
        a = float("inf")
        if eps < 0.0:
            a = -mu / (2.0 * eps)
        h = np.cross(r, v)
        evec = (np.cross(v, h) / mu) - (r / (rnorm + 1e-300))
        e = float(np.linalg.norm(evec))
        ok = (
            np.isfinite(eps)
            and np.isfinite(e)
            and eps < 0.0
            and e < 0.9999
            and 1e5 < rnorm < 1e10
        )
        return bool(ok), float(eps), float(a), float(e), float(rnorm)
    except Exception:
        return False, float("nan"), float("nan"), float("nan"), float("nan")


def _state_basic_ok(state: np.ndarray) -> bool:
    try:
        r = np.asarray(state[:3], dtype=float)
        rnorm = float(np.linalg.norm(r))
        return bool(np.isfinite(rnorm) and 1e5 < rnorm < 1e11)
    except Exception:
        return False


def _chol_cov(cov: np.ndarray) -> np.ndarray:
    cov = np.asarray(cov, dtype=float)
    try:
        return np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh(cov)
        vals = np.clip(vals, 1e-12, None)
        return vecs @ np.diag(np.sqrt(vals))


def _sobol_samples(
    n: int,
    dim: int,
    *,
    seed: int | None = None,
    scramble: bool = True,
) -> np.ndarray:
    sob = qmc.Sobol(dim, scramble=scramble, seed=seed)
    return sob.random(n)


def _sample_attributables(
    mean: np.ndarray,
    cov: np.ndarray,
    u: np.ndarray,
) -> np.ndarray:
    u_clip = np.clip(u, 1e-12, 1.0 - 1e-12)
    z = stats.norm.ppf(u_clip)
    L = _chol_cov(cov)
    return mean[None, :] + z @ L.T


def _seed_states_from_sobol(
    obs_ref: Observation,
    t_obs: Time,
    attrib_mean: Attributable,
    attrib_cov: np.ndarray,
    n_samples: int,
    *,
    rho_min_km: float,
    rho_max_au: float,
    rhodot_df: float,
    rhodot_scale_kms: float,
    v_max_km_s: float | None,
    rate_max_deg_day: float | None,
    seed: int | None,
) -> np.ndarray:
    if n_samples <= 0:
        return np.empty((0, 6), dtype=float)
    u = _sobol_samples(n_samples, 6, seed=seed, scramble=True)
    gamma_samples = _sample_attributables(
        np.array(
            [
                attrib_mean.ra_deg,
                attrib_mean.dec_deg,
                attrib_mean.ra_dot_deg_per_day,
                attrib_mean.dec_dot_deg_per_day,
            ],
            dtype=float,
        ),
        attrib_cov,
        u[:, :4],
    )

    logrho_min = math.log(max(1e-12, rho_min_km / AU_KM))
    logrho_max = math.log(max(rho_min_km / AU_KM, rho_max_au))
    logrho = logrho_min + u[:, 4] * (logrho_max - logrho_min)
    rho_km = np.exp(logrho) * AU_KM

    rhodot = stats.t.ppf(np.clip(u[:, 5], 1e-12, 1.0 - 1e-12), df=rhodot_df)
    rhodot = rhodot * float(rhodot_scale_kms)
    rhodot = np.clip(rhodot, -200.0, 200.0)

    states = []
    for attrib_vec, rho_val, rhodot_val in zip(gamma_samples, rho_km, rhodot):
        attrib = Attributable(
            ra_deg=float(attrib_vec[0]),
            dec_deg=float(attrib_vec[1]),
            ra_dot_deg_per_day=float(attrib_vec[2]),
            dec_dot_deg_per_day=float(attrib_vec[3]),
        )
        try:
            state = build_state_from_ranging(obs_ref, t_obs, attrib, float(rho_val), float(rhodot_val))
        except Exception:
            continue
        if v_max_km_s is not None:
            vnorm = float(np.linalg.norm(state[3:6]))
            if not np.isfinite(vnorm) or vnorm > float(v_max_km_s):
                continue
        if not _state_basic_ok(state):
            continue
        if rate_max_deg_day is not None:
            attrib_state, _, _ = attrib_from_state_with_observer_time(state, obs_ref, t_obs)
            if (
                abs(attrib_state.ra_dot_deg_per_day) > float(rate_max_deg_day)
                or abs(attrib_state.dec_dot_deg_per_day) > float(rate_max_deg_day)
            ):
                continue
        states.append(state)
    if not states:
        return np.empty((0, 6), dtype=float)
    return np.vstack(states)


def _emission_epoch_for_state(
    state: np.ndarray,
    state_epoch: Time,
    obs_ref: Observation,
    t_obs: Time,
    *,
    max_iter: int | None = None,
    tol_sec: float = 1e-3,
    debug_label: str | None = None,
    obs_pos: np.ndarray | None = None,
    obs_vel: np.ndarray | None = None,
) -> tuple[Time, np.ndarray]:
    t_em = t_obs
    state_em = np.asarray(state, dtype=float)
    last_dt = None
    iter_count = 0
    while True:
        if obs_pos is None or obs_vel is None:
            _, rho, _ = attrib_from_state_with_observer_time(state_em, obs_ref, t_obs)
        else:
            r_topo = state_em[:3].astype(float) - obs_pos
            rho = float(np.linalg.norm(r_topo))
        dt = float(rho) / C_KM_S
        t_em = t_obs - TimeDelta(dt, format="sec")
        try:
            state_em = propagate_state_kepler(state, state_epoch, t_em)
        except Exception as exc:
            r0 = np.asarray(state[:3], dtype=float)
            v0 = np.asarray(state[3:6], dtype=float)
            r0_norm = float(np.linalg.norm(r0))
            v0_sq = float(np.dot(v0, v0))
            alpha = 2.0 / (r0_norm + 1e-30) - v0_sq / GM_SUN
            label = f" label={debug_label}" if debug_label else ""
            raise RuntimeError(
                f"Emission-time propagation failed{label} dt={dt:.6f}s "
                f"r0={r0.tolist()} v0={v0.tolist()} alpha={alpha:.6e}"
            ) from exc
        if last_dt is not None and abs(dt - last_dt) <= tol_sec:
            break
        if max_iter is not None:
            iter_count += 1
            if iter_count >= max_iter:
                break
        else:
            iter_count += 1
            if iter_count >= 10000:
                label = f" label={debug_label}" if debug_label else ""
                raise RuntimeError(
                    f"Emission-time iteration did not converge{label} dt={dt:.6f}s"
                )
        last_dt = dt
    return t_em, state_em


def seed_and_test_cloud(
    observations: Sequence[Observation],
    *,
    n_init: int = 3,
    n_beads: int = 2000,
    n_next: int = 3,
    rho_min_km: float = 6471.0,
    rho_max_au: float = 100.0,
    rhodot_df: float = 3.0,
    rhodot_scale_kms: float = 30.0,
    v_max_km_s: float | None = 120.0,
    rate_max_deg_day: float | None = 5.0,
    seed: int | None = None,
    jpl_state: np.ndarray | None = None,
    jpl_target: str | None = None,
    jpl_location: str = "@sun",
    jpl_refplane: str = "earth",
    jpl_cov: np.ndarray | None = None,
    jpl_cov_source: str | None = "sbdb",
    jpl_cov_scale: float = 1.0,
    jpl_n_jitter: int = 0,
    chi2_conf: float = 0.99,
    chi2_obs_df: int | None = None,
    accept_mode: str = "obs_chi2",
    skip_special_sites: bool = False,
    use_admissible_seed: bool = True,
) -> dict[str, object]:
    observations = filter_special_sites(
        observations,
        skip_special_sites=skip_special_sites,
        fail_unknown_site=True,
    )
    if len(observations) < max(n_init, n_next + n_init):
        raise ValueError("Not enough observations for requested windows.")

    obs_init = observations[:n_init]
    t0 = _midpoint_time(obs_init)
    attrib_init, cov_init = build_attributable_vector_fit(obs_init, t0, return_cov=True)
    obs_ref = _ranging_reference_observation(obs_init, t0)

    rng = make_rng(seed)
    if use_admissible_seed:
        base_jpl = 1 if (jpl_state is not None or jpl_target is not None) else 0
        planned = max(0, n_beads - base_jpl - int(jpl_n_jitter))
        n_jitter = min(SeedConfig().n_jitter, planned)
        n_sobol_local = max(0, planned - n_jitter)
        cfg = SeedConfig(
            n_jitter=n_jitter,
            n_sobol_local=n_sobol_local,
            rho_min_km=rho_min_km,
            rho_max_au=rho_max_au,
            rhodot_df=rhodot_df,
            rhodot_scale_kms=rhodot_scale_kms,
            v_max_km_s=v_max_km_s,
            rate_max_deg_day=rate_max_deg_day,
            seed=seed,
            seed_obs_max_keep=n_beads,
        )
        seed_result = seed_cloud_from_observations(
            observations,
            n_init=n_init,
            cfg=cfg,
            jpl_target=jpl_target,
            jpl_state=jpl_state,
            jpl_location=jpl_location,
            jpl_refplane=jpl_refplane,
            jpl_cov=jpl_cov,
            jpl_cov_source=jpl_cov_source,
            jpl_cov_scale=jpl_cov_scale,
            jpl_n_jitter=jpl_n_jitter,
            skip_special_sites=skip_special_sites,
        )
        states = seed_result.states
        state_epochs = seed_result.epochs
        attrib_init = seed_result.attributable
        cov_init = seed_result.cov
        obs_ref = seed_result.obs_ref
        t0 = seed_result.epoch
        if len(states) > n_beads:
            pick = rng.choice(len(states), size=int(n_beads), replace=False)
            states = states[pick]
            state_epochs = state_epochs[pick]
    else:
        jpl_states: list[tuple[np.ndarray, Time]] = []
        if jpl_state is None and jpl_target is not None:
            jpl_state = fetch_horizons_state(
                jpl_target, t0, location=jpl_location, refplane=jpl_refplane
            )
        if jpl_state is not None:
            jpl_state = np.asarray(jpl_state, dtype=float).reshape(1, 6)
            jpl_t_em, jpl_state_em = _emission_epoch_for_state(
                jpl_state[0], t0, obs_ref, t0, debug_label="jpl_seed"
            )
            jpl_states.append((jpl_state_em, jpl_t_em))
            if jpl_cov is None and jpl_n_jitter > 0:
                if not jpl_cov_source:
                    raise ValueError("Missing JPL covariance; provide jpl_cov or jpl_cov_source.")
                if jpl_cov_source.lower() == "sbdb":
                    if not jpl_target:
                        raise ValueError("SBDB covariance requires jpl_target.")
                    jpl_cov = fetch_sbdb_covariance(jpl_target)
                else:
                    raise ValueError("Unknown jpl_cov_source; provide jpl_cov explicitly.")
            if jpl_cov is not None and jpl_n_jitter > 0:
                cov = np.asarray(jpl_cov, dtype=float).reshape(6, 6) * float(jpl_cov_scale)
                jitter = rng.multivariate_normal(np.zeros(6), cov, size=int(jpl_n_jitter))
                for delta in jitter:
                    jpl_states.append((jpl_state_em + delta, jpl_t_em))

        if jpl_states:
            jpl_states = [item for item in jpl_states if _state_physicality(item[0])[0]]

        if len(jpl_states) > n_beads:
            raise ValueError("Requested n_beads smaller than JPL seed count.")

        seed_states = _seed_states_from_sobol(
            obs_ref,
            t0,
            attrib_init,
            cov_init,
            n_beads - len(jpl_states),
            rho_min_km=rho_min_km,
            rho_max_au=rho_max_au,
            rhodot_df=rhodot_df,
            rhodot_scale_kms=rhodot_scale_kms,
            v_max_km_s=v_max_km_s,
            rate_max_deg_day=rate_max_deg_day,
            seed=seed,
        )
        if jpl_states:
            jpl_states_arr = np.asarray([item[0] for item in jpl_states], dtype=float).reshape(-1, 6)
            states = np.vstack([jpl_states_arr, seed_states]) if len(seed_states) else jpl_states_arr
            state_epochs = np.array([item[1] for item in jpl_states], dtype=object)
            if len(seed_states):
                seed_epochs = np.full(len(seed_states), t0, dtype=object)
                state_epochs = np.concatenate([state_epochs, seed_epochs])
        else:
            states = seed_states
            state_epochs = np.full(len(states), t0, dtype=object)

    if len(states) == 0:
        raise RuntimeError("No valid seed states generated.")

    for i, st in enumerate(states):
        t_em, st_em = _emission_epoch_for_state(
            st, state_epochs[i], obs_ref, t0, debug_label=f"seed[{i}]"
        )
        state_epochs[i] = t_em
        states[i] = st_em

    weights = np.ones(len(states), dtype=float)
    weights /= float(weights.sum())

    cloud = ReplicaCloud(
        states=states,
        weights=weights,
        epoch=t0,
        metadata={
            "n_init": n_init,
            "n_beads": int(n_beads),
            "rho_min_km": float(rho_min_km),
            "rho_max_au": float(rho_max_au),
            "rhodot_df": float(rhodot_df),
            "rhodot_scale_kms": float(rhodot_scale_kms),
            "v_max_km_s": None if v_max_km_s is None else float(v_max_km_s),
            "rate_max_deg_day": None if rate_max_deg_day is None else float(rate_max_deg_day),
            "chi2_conf": float(chi2_conf),
            "accept_mode": str(accept_mode),
            "state_epoch_mode": "emission",
        },
    )

    obs_next = observations[n_init : n_init + n_next]
    t_next = obs_next[0].time
    attrib_next, cov_next = build_attributable_vector_fit(obs_next, t_next, return_cov=True)
    dec0_rad = math.radians(float(np.mean([ob.dec_deg for ob in obs_next])))
    cosd0 = math.cos(dec0_rad)
    inv_cov_next = np.linalg.inv(cov_next)

    obs_next_ref = _ranging_reference_observation(obs_next, t_next)
    propagated = np.empty_like(states)
    propagated_epochs = np.empty(len(states), dtype=object)
    for i, st in enumerate(states):
        t_em_next, st_next = _emission_epoch_for_state(
            st, state_epochs[i], obs_next_ref, t_next, debug_label=f"propagate[{i}]"
        )
        propagated[i] = st_next
        propagated_epochs[i] = t_em_next
    propagated_weights = weights.copy()

    accepts = np.zeros(len(propagated), dtype=bool)
    attribs = np.zeros((len(propagated), 4), dtype=float)
    if accept_mode == "obs_chi2":
        df = int(chi2_obs_df) if chi2_obs_df is not None else 2 * len(obs_next)
        chi2_star = stats.chi2.ppf(chi2_conf, df=df)
    else:
        chi2_star = stats.chi2.ppf(chi2_conf, df=4)

    for i, st in enumerate(propagated):
        attrib_vec = _attrib_vector_from_state(st, obs_next_ref, t_next)
        attribs[i] = attrib_vec
        if accept_mode == "obs_chi2":
            chi2_sum = 0.0
            for obs_i in obs_next:
                t_em_i, st_i = _emission_epoch_for_state(
                    st, propagated_epochs[i], obs_i, obs_i.time, debug_label=f"obs_chi2[{i}]"
                )
                attrib_i, _, _ = attrib_from_state_with_observer_time(st_i, obs_i, obs_i.time)
                dalpha = _wrap_deg(attrib_i.ra_deg - obs_i.ra_deg)
                dec_rad = math.radians(attrib_i.dec_deg)
                sigma_arc = float(obs_i.sigma_arcsec)
                sigma_arc = max(1e-6, sigma_arc)
                dra_arc = dalpha * math.cos(dec_rad) * 3600.0
                ddec_arc = (attrib_i.dec_deg - obs_i.dec_deg) * 3600.0
                chi2_sum += (dra_arc / sigma_arc) ** 2 + (ddec_arc / sigma_arc) ** 2
            accepts[i] = chi2_sum <= chi2_star
        else:
            d = np.array(
                [
                    _wrap_deg(attrib_vec[0] - attrib_next.ra_deg) * cosd0,
                    attrib_vec[1] - attrib_next.dec_deg,
                    (attrib_vec[2] - attrib_next.ra_dot_deg_per_day) * cosd0,
                    attrib_vec[3] - attrib_next.dec_dot_deg_per_day,
                ],
                dtype=float,
            )
            d2 = float(d @ (inv_cov_next @ d))
            accepts[i] = d2 <= chi2_star

    return {
        "cloud": cloud,
        "states": states,
        "weights": weights,
        "propagated": propagated,
        "propagated_weights": propagated_weights,
        "propagated_epochs": propagated_epochs,
        "accepts": accepts,
        "attribs": attribs,
        "t0": t0,
        "t_next": t_next,
        "state_epochs": state_epochs,
        "attrib_init": attrib_init,
        "cov_init": cov_init,
        "attrib_next": attrib_next,
        "cov_next": cov_next,
    }


if __name__ == "__main__":
    from pathlib import Path
    from astropy.utils import iers
    import json
    import os
    import numpy as np

    iers.conf.auto_download = False
    iers.conf.iers_degraded_accuracy = "warn"

    obs_path = Path(os.getenv("OBS_PATH", "runs/ceres/obs.csv"))
    observations = load_observations(obs_path, sigma=None, skip_special_sites=False)

    mode = os.getenv("MODE", "beads")
    n_draws = int(os.getenv("N_DRAWS", "50"))
    n_rho = int(os.getenv("N_RHO", "40"))
    n_rhodot = int(os.getenv("N_RHODOT", "40"))
    rho_min_au = float(os.getenv("RHO_MIN_AU", "1e-6"))
    rho_max_au = float(os.getenv("RHO_MAX_AU", "200.0"))
    rhodot_max_km_s = float(os.getenv("RHODOT_MAX", "100.0"))
    conf = float(os.getenv("CONF", "0.99"))
    workers = int(os.getenv("WORKERS", "1"))
    k_sigma = float(os.getenv("K_SIGMA", "3.0"))
    n_beads = int(os.getenv("N_BEADS", "10"))

    obs_window = observations[:3]
    epoch = _midpoint_time(obs_window)
    obs_ref = _ranging_reference_observation(obs_window, epoch)
    attrib_fit, cov = build_attributable_vector_fit(obs_window, epoch, return_cov=True)

    mean_vec = np.array(
        [
            attrib_fit.ra_deg,
            attrib_fit.dec_deg,
            attrib_fit.ra_dot_deg_per_day,
            attrib_fit.dec_dot_deg_per_day,
        ],
        dtype=float,
    )
    jitter = 1e-12 * np.eye(4)
    L = np.linalg.cholesky(cov + jitter)
    rng = np.random.default_rng(12345)

    rho_grid = np.logspace(math.log10(rho_min_au), math.log10(rho_max_au), n_rho) * AU_KM
    rhodot_grid = np.linspace(-rhodot_max_km_s, rhodot_max_km_s, n_rhodot)
    union_mask = np.zeros((n_rho, n_rhodot), dtype=bool)

    def _wrap_ra_deg(delta_deg: float) -> float:
        return (delta_deg + 180.0) % 360.0 - 180.0

    chi2_delta = float(stats.chi2.ppf(conf, df=2))

    draw_vecs = []
    for _ in range(n_draws):
        z = rng.normal(size=4)
        draw_vecs.append(mean_vec + L @ z)

    ctx = {
        "rho_grid": rho_grid,
        "rhodot_grid": rhodot_grid,
        "obs_window": obs_window,
        "obs_ref": obs_ref,
        "epoch": epoch,
        "chi2_delta": chi2_delta,
    }

    theta_draws = []
    if mode == "beads":
        obs_window = observations[:3]
        epoch = _midpoint_time(obs_window)
        obs_ref = _ranging_reference_observation(obs_window, epoch)
        print(f"Running bead check: n_beads={n_beads} k_sigma={k_sigma}")
        for bead_idx in range(1, n_beads + 1):
            z = rng.normal(size=4)
            draw_vec = mean_vec + L @ z
            attrib_draw = Attributable(
                ra_deg=float(draw_vec[0]),
                dec_deg=float(draw_vec[1]),
                ra_dot_deg_per_day=float(draw_vec[2]),
                dec_dot_deg_per_day=float(draw_vec[3]),
            )
            rho_km = float(np.exp(rng.uniform(math.log(rho_min_au * AU_KM), math.log(rho_max_au * AU_KM))))
            rhodot_km_s = float(rng.uniform(-rhodot_max_km_s, rhodot_max_km_s))
            try:
                state = build_state_from_ranging(obs_ref, epoch, attrib_draw, rho_km, rhodot_km_s)
            except Exception:
                print(
                    f"bead {bead_idx}/{n_beads} "
                    f"theta=({attrib_draw.ra_deg:.6f}, {attrib_draw.ra_dot_deg_per_day:.6f}, "
                    f"{attrib_draw.dec_deg:.6f}, {attrib_draw.dec_dot_deg_per_day:.6f}, "
                    f"{rho_km:.6f}, {rhodot_km_s:.6f}) accepted=False"
                )
                continue
            accepted = True
            for ob in obs_window:
                _, st_em = _emission_epoch_for_state(
                    state,
                    epoch,
                    ob,
                    ob.time,
                    max_iter=None,
                    tol_sec=1e-6,
                    debug_label="bead_check",
                )
                attrib_pred, _, _ = attrib_from_state_with_observer_time(st_em, ob, ob.time)
                dra = _wrap_deg(attrib_pred.ra_deg - ob.ra_deg)
                dra *= math.cos(math.radians(attrib_pred.dec_deg))
                ddec = attrib_pred.dec_deg - ob.dec_deg
                dra_arcsec = dra * 3600.0
                ddec_arcsec = ddec * 3600.0
                sigma = max(1e-6, float(ob.sigma_arcsec))
                # Diagonal covariance: chi2 = (dra/sigma)^2 + (ddec/sigma)^2
                chi2 = (dra_arcsec / sigma) ** 2 + (ddec_arcsec / sigma) ** 2
                if chi2 > (k_sigma ** 2):
                    accepted = False
                    break
            print(
                f"bead {bead_idx}/{n_beads} "
                f"theta=({attrib_draw.ra_deg:.6f}, {attrib_draw.ra_dot_deg_per_day:.6f}, "
                f"{attrib_draw.dec_deg:.6f}, {attrib_draw.dec_dot_deg_per_day:.6f}, "
                f"{rho_km:.6f}, {rhodot_km_s:.6f}) accepted={accepted}"
            )
    else:
        if workers > 1:
            with mp.Pool(processes=workers, initializer=_mc_worker_init, initargs=(ctx,)) as pool:
                for draw_idx, result in enumerate(pool.imap_unordered(_mc_worker, draw_vecs), start=1):
                    packed = np.frombuffer(result["packed_mask"], dtype=np.uint8)
                    bits = np.unpackbits(packed)[: result["nbits"]]
                    draw_mask = bits.reshape((n_rho, n_rhodot), order="C").astype(bool)
                    union_mask |= draw_mask
                    theta_draws.append(result["theta"])
                    theta = result["theta"]
                    accepted = result["accepted_count"] > 0
                    print(
                        f"draw {draw_idx}/{n_draws} "
                        f"theta=({theta['alpha_deg']:.6f}, {theta['alpha_dot_deg_per_day']:.6f}, "
                        f"{theta['delta_deg']:.6f}, {theta['delta_dot_deg_per_day']:.6f}, "
                        f"{theta['rho_km']:.6f}, {theta['rhodot_km_s']:.6f}) "
                        f"chi2_min={theta['chi2_min']:.3f} accepted={accepted}"
                    )
        else:
            _mc_worker_init(ctx)
            for draw_idx, draw_vec in enumerate(draw_vecs, start=1):
                result = _mc_worker(draw_vec)
                packed = np.frombuffer(result["packed_mask"], dtype=np.uint8)
                bits = np.unpackbits(packed)[: result["nbits"]]
                draw_mask = bits.reshape((n_rho, n_rhodot), order="C").astype(bool)
                union_mask |= draw_mask
                theta_draws.append(result["theta"])
                theta = result["theta"]
                accepted = result["accepted_count"] > 0
                print(
                    f"draw {draw_idx}/{n_draws} "
                    f"theta=({theta['alpha_deg']:.6f}, {theta['alpha_dot_deg_per_day']:.6f}, "
                    f"{theta['delta_deg']:.6f}, {theta['delta_dot_deg_per_day']:.6f}, "
                    f"{theta['rho_km']:.6f}, {theta['rhodot_km_s']:.6f}) "
                    f"chi2_min={theta['chi2_min']:.3f} accepted={accepted}"
                )

    if mode != "beads":
        accepted = int(np.sum(union_mask))
        total = int(union_mask.size)
        payload = {
            "obs_times": [str(ob.time.utc.iso) for ob in obs_window],
            "epoch": str(epoch.utc.iso),
            "rho_grid_km": rho_grid.tolist(),
            "rhodot_grid_km_s": rhodot_grid.tolist(),
            "union_mask": union_mask.astype(int).tolist(),
            "theta_draws": theta_draws,
            "accepted_cells": accepted,
            "total_cells": total,
            "accepted_fraction": accepted / max(1, total),
            "n_draws": n_draws,
            "conf": conf,
        }
        out_path = obs_path.parent / "admissible_mc.json"
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"Wrote {out_path}")
        print(f"Accepted fraction: {payload['accepted_fraction']:.3f}")
