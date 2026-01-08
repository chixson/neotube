from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from astropy.time import Time, TimeDelta
from scipy import stats
from scipy.stats import qmc

from .horizons import fetch_horizons_state
from .constants import AU_KM, C_KM_S, GM_SUN
from .models import Attributable, Observation, ReplicaCloud
from .ranging import (
    attrib_from_state_with_observer_time,
    build_attributable_vector_fit,
    build_state_from_ranging,
)
from .propagate import propagate_state_kepler
from .rng import make_rng
from .sbdb import fetch_sbdb_covariance
from .site_checks import filter_special_sites


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
    max_iter: int = 10,
    tol_sec: float = 1e-3,
    debug_label: str | None = None,
) -> tuple[Time, np.ndarray]:
    t_em = t_obs
    state_em = np.asarray(state, dtype=float)
    last_dt = None
    for _ in range(max_iter):
        _, rho, _ = attrib_from_state_with_observer_time(state_em, obs_ref, t_obs)
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

    jpl_states: list[tuple[np.ndarray, Time]] = []
    rng = make_rng(seed)
    if jpl_state is None and jpl_target is not None:
        jpl_state = fetch_horizons_state(jpl_target, t0, location=jpl_location, refplane=jpl_refplane)
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
