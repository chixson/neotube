from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from astropy.time import Time, TimeDelta
from scipy import stats
from scipy.stats import qmc

from .constants import AU_KM, C_KM_S, GM_SUN
from .horizons import fetch_horizons_state
from .models import Attributable, Observation
from .propagate import propagate_state_kepler
from .ranging import (
    attrib_from_state_with_observer_time,
    build_attributable_vector_fit,
    build_state_from_ranging,
    s_and_sdot,
    _observer_helio_state,
)
from .sbdb import fetch_sbdb_covariance
from .site_checks import filter_special_sites


@dataclass(frozen=True)
class SeedConfig:
    n_jitter: int = 500
    n_sobol_local: int = 2000
    rho_min_km: float = 6471.0
    rho_max_au: float = 100.0
    rhodot_df: float = 3.0
    rhodot_scale_kms: float = 30.0
    v_max_km_s: float | None = 120.0
    rate_max_deg_day: float | None = 5.0
    sobol_scramble: bool = True
    seed: int | None = None
    seed_obs_chi2_conf: float | None = 0.995
    seed_obs_df: int | None = None
    seed_obs_max_keep: int | None = None
    admissible_n_rho: int = 400
    admissible_n_per_gamma: int = 8
    admissible_bound_only: bool = True


@dataclass(frozen=True)
class SeedResult:
    states: np.ndarray
    epochs: np.ndarray
    attributable: Attributable
    cov: np.ndarray
    obs_ref: Observation
    epoch: Time


def _midpoint_time(observations: Sequence[Observation]) -> Time:
    if not observations:
        raise ValueError("Need observations to compute midpoint.")
    jd = np.array([ob.time.tdb.jd for ob in observations], dtype=float)
    return Time(jd.mean(), format="jd", scale="tdb")


def _ranging_reference_observation(
    observations: Sequence[Observation], epoch: Time
) -> Observation:
    if not observations:
        raise ValueError("Need observations to choose reference observation.")
    return min(observations, key=lambda ob: abs((ob.time.tdb - epoch.tdb).to_value("s")))


def _basic_state_ok(state: np.ndarray) -> bool:
    try:
        r = np.asarray(state[:3], dtype=float)
        rnorm = float(np.linalg.norm(r))
        return bool(np.isfinite(rnorm) and 1e5 < rnorm < 1e11)
    except Exception:
        return False


def _sobol_samples(
    n: int,
    dim: int,
    *,
    seed: int | None = None,
    scramble: bool = True,
) -> np.ndarray:
    sob = qmc.Sobol(dim, scramble=scramble, seed=seed)
    return sob.random(n)


def _chol_cov(cov: np.ndarray) -> np.ndarray:
    cov = np.asarray(cov, dtype=float)
    try:
        return np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh(cov)
        vals = np.clip(vals, 1e-12, None)
        return vecs @ np.diag(np.sqrt(vals))


def _sample_attributables(
    mean: np.ndarray,
    cov: np.ndarray,
    u: np.ndarray,
) -> np.ndarray:
    u_clip = np.clip(u, 1e-12, 1.0 - 1e-12)
    z = stats.norm.ppf(u_clip)
    L = _chol_cov(cov)
    return mean[None, :] + z @ L.T


def _emission_epoch_for_state(
    state: np.ndarray,
    state_epoch: Time,
    obs_ref: Observation,
    t_obs: Time,
    *,
    max_iter: int = 10,
    tol_sec: float = 1e-3,
    obs_pos: np.ndarray | None = None,
    obs_vel: np.ndarray | None = None,
) -> tuple[Time, np.ndarray]:
    t_em = t_obs
    state_em = np.asarray(state, dtype=float)
    last_dt = None
    for _ in range(max_iter):
        if obs_pos is None or obs_vel is None:
            _, rho, _ = attrib_from_state_with_observer_time(state_em, obs_ref, t_obs)
        else:
            r_topo = state_em[:3].astype(float) - obs_pos
            rho = float(np.linalg.norm(r_topo))
        dt = float(rho) / C_KM_S
        t_em = t_obs - TimeDelta(dt, format="sec")
        state_em = propagate_state_kepler(state, state_epoch, t_em)
        if last_dt is not None and abs(dt - last_dt) <= tol_sec:
            break
        last_dt = dt
    return t_em, state_em


def _wrap_deg(delta_deg: float) -> float:
    return (delta_deg + 180.0) % 360.0 - 180.0


def _attrib_from_state_cached(
    state: np.ndarray,
    obs_pos: np.ndarray,
    obs_vel: np.ndarray,
) -> tuple[Attributable, float, float]:
    r_helio = state[:3].astype(float)
    v_helio = state[3:].astype(float)
    r_topo = r_helio - obs_pos
    v_topo = v_helio - obs_vel
    rho = float(np.linalg.norm(r_topo))
    if rho <= 0:
        raise RuntimeError("Non-positive rho in attributable conversion.")
    s = r_topo / rho
    rhodot = float(np.dot(v_topo, s))
    sdot = (v_topo - rhodot * s) / max(rho, 1e-12)

    x, y, z = s
    xd, yd, zd = sdot
    rxy2 = max(x * x + y * y, 1e-12)
    ra = math.atan2(y, x)
    dec = math.asin(np.clip(z, -1.0, 1.0))
    ra_dot = (x * yd - y * xd) / rxy2
    cosdec = max(math.sqrt(rxy2), 1e-12)
    dec_dot = (zd - z * (x * xd + y * yd) / rxy2) / cosdec

    attrib = Attributable(
        ra_deg=float(math.degrees(ra) % 360.0),
        dec_deg=float(math.degrees(dec)),
        ra_dot_deg_per_day=float(math.degrees(ra_dot) * 86400.0),
        dec_dot_deg_per_day=float(math.degrees(dec_dot) * 86400.0),
    )
    return attrib, rho, rhodot


def _admissible_intervals(
    attrib: Attributable,
    obs_ref: Observation,
    epoch: Time,
    rho_grid_km: np.ndarray,
    *,
    mu: float = GM_SUN,
    rho_min_km: float = 6471.0,
    bound_only: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    s, sdot = s_and_sdot(attrib)
    obs_pos, obs_vel = _observer_helio_state(obs_ref, epoch)

    rvec = obs_pos[None, :] + rho_grid_km[:, None] * s[None, :]
    vlin = obs_vel[None, :] + rho_grid_km[:, None] * sdot[None, :]
    rnorm = np.linalg.norm(rvec, axis=1)
    b = np.einsum("ij,j->i", vlin, s)
    vlin2 = np.einsum("ij,ij->i", vlin, vlin)
    disc = b * b - (vlin2 - 2.0 * mu / (rnorm + 1e-300))

    dotmin = np.full_like(rho_grid_km, np.nan, dtype=float)
    dotmax = np.full_like(rho_grid_km, np.nan, dtype=float)

    valid = rnorm > float(rho_min_km)
    if bound_only:
        valid = valid & (disc >= 0.0)
    else:
        valid = valid & np.isfinite(disc)

    if not np.any(valid):
        return dotmin, dotmax

    sqrt_disc = np.sqrt(np.clip(disc[valid], 0.0, None))
    dot1 = -b[valid] - sqrt_disc
    dot2 = -b[valid] + sqrt_disc
    dotmin[valid] = np.minimum(dot1, dot2)
    dotmax[valid] = np.maximum(dot1, dot2)
    return dotmin, dotmax


def _obs_chi2_for_state(
    state: np.ndarray,
    state_epoch: Time,
    observations: Sequence[Observation],
    obs_cache: Sequence[tuple[np.ndarray, np.ndarray]] | None = None,
) -> float:
    chi2_sum = 0.0
    for idx, obs_i in enumerate(observations):
        if obs_cache is None:
            t_em, st_em = _emission_epoch_for_state(state, state_epoch, obs_i, obs_i.time)
            attrib_i, _, _ = attrib_from_state_with_observer_time(st_em, obs_i, obs_i.time)
        else:
            obs_pos, obs_vel = obs_cache[idx]
            t_em, st_em = _emission_epoch_for_state(
                state,
                state_epoch,
                obs_i,
                obs_i.time,
                obs_pos=obs_pos,
                obs_vel=obs_vel,
            )
            attrib_i, _, _ = _attrib_from_state_cached(st_em, obs_pos, obs_vel)
        dalpha = _wrap_deg(attrib_i.ra_deg - obs_i.ra_deg)
        dec_rad = math.radians(attrib_i.dec_deg)
        sigma_arc = max(1e-6, float(obs_i.sigma_arcsec))
        dra_arc = dalpha * math.cos(dec_rad) * 3600.0
        ddec_arc = (attrib_i.dec_deg - obs_i.dec_deg) * 3600.0
        chi2_sum += (dra_arc / sigma_arc) ** 2 + (ddec_arc / sigma_arc) ** 2
    return float(chi2_sum)


def _build_state_from_sample(
    obs_ref: Observation,
    epoch: Time,
    attrib_vec: np.ndarray,
    rho_km: float,
    rhodot_km_s: float,
    *,
    v_max_km_s: float | None,
    rate_max_deg_day: float | None,
) -> np.ndarray | None:
    attrib = Attributable(
        ra_deg=float(attrib_vec[0]),
        dec_deg=float(attrib_vec[1]),
        ra_dot_deg_per_day=float(attrib_vec[2]),
        dec_dot_deg_per_day=float(attrib_vec[3]),
    )
    try:
        state = build_state_from_ranging(obs_ref, epoch, attrib, rho_km, rhodot_km_s)
    except Exception:
        return None
    if v_max_km_s is not None:
        vnorm = float(np.linalg.norm(state[3:6]))
        if not np.isfinite(vnorm) or vnorm > float(v_max_km_s):
            return None
    if not _basic_state_ok(state):
        return None
    if rate_max_deg_day is not None:
        attrib_state, _, _ = attrib_from_state_with_observer_time(state, obs_ref, epoch)
        if (
            abs(attrib_state.ra_dot_deg_per_day) > float(rate_max_deg_day)
            or abs(attrib_state.dec_dot_deg_per_day) > float(rate_max_deg_day)
        ):
            return None
    return state


def seed_local_from_attrib(
    obs_ref: Observation,
    epoch: Time,
    attrib_mean: Attributable,
    attrib_cov: np.ndarray,
    *,
    cfg: SeedConfig,
) -> np.ndarray:
    mean_vec = np.array(
        [
            attrib_mean.ra_deg,
            attrib_mean.dec_deg,
            attrib_mean.ra_dot_deg_per_day,
            attrib_mean.dec_dot_deg_per_day,
        ],
        dtype=float,
    )
    rng = np.random.default_rng(cfg.seed)
    states: list[np.ndarray] = []

    gamma_samples: list[np.ndarray] = []
    if cfg.n_jitter > 0:
        jitter = rng.normal(size=(cfg.n_jitter, 4))
        L = _chol_cov(attrib_cov)
        gamma_samples.extend(list(mean_vec[None, :] + jitter @ L.T))
    if cfg.n_sobol_local > 0:
        u = _sobol_samples(
            cfg.n_sobol_local, 4, seed=cfg.seed, scramble=cfg.sobol_scramble
        )
        gamma_samples.extend(list(_sample_attributables(mean_vec, attrib_cov, u)))

    rho_min = max(1e-12, float(cfg.rho_min_km))
    rho_max = max(rho_min / AU_KM, float(cfg.rho_max_au)) * AU_KM
    rho_grid = np.logspace(
        math.log10(rho_min), math.log10(rho_max), int(cfg.admissible_n_rho)
    )

    for gamma_vec in gamma_samples:
        attrib = Attributable(
            ra_deg=float(gamma_vec[0]),
            dec_deg=float(gamma_vec[1]),
            ra_dot_deg_per_day=float(gamma_vec[2]),
            dec_dot_deg_per_day=float(gamma_vec[3]),
        )
        dotmin, dotmax = _admissible_intervals(
            attrib,
            obs_ref,
            epoch,
            rho_grid,
            rho_min_km=float(cfg.rho_min_km),
            bound_only=cfg.admissible_bound_only,
        )
        valid = np.isfinite(dotmin) & np.isfinite(dotmax) & (dotmax >= dotmin)
        if not np.any(valid):
            continue
        valid_idx = np.where(valid)[0]
        span = dotmax[valid] - dotmin[valid]
        weights = span.copy()
        if np.all(weights <= 0.0) or not np.all(np.isfinite(weights)):
            weights = None
        else:
            weights = weights / weights.sum()

        for _ in range(max(1, int(cfg.admissible_n_per_gamma))):
            pick = int(rng.choice(valid_idx, p=weights))
            rho_val = float(rho_grid[pick])
            dotrho_val = float(rng.uniform(dotmin[pick], dotmax[pick]))
            state = _build_state_from_sample(
                obs_ref,
                epoch,
                gamma_vec,
                rho_val,
                dotrho_val,
                v_max_km_s=cfg.v_max_km_s,
                rate_max_deg_day=cfg.rate_max_deg_day,
            )
            if state is not None:
                states.append(state)

    if not states:
        return np.empty((0, 6), dtype=float)
    return np.vstack(states)


def seed_cloud_from_observations(
    observations: Sequence[Observation],
    *,
    n_init: int = 3,
    cfg: SeedConfig | None = None,
    jpl_target: str | None = None,
    jpl_state: np.ndarray | None = None,
    jpl_location: str = "@sun",
    jpl_refplane: str = "earth",
    jpl_cov: np.ndarray | None = None,
    jpl_cov_source: str | None = "sbdb",
    jpl_cov_scale: float = 1.0,
    jpl_n_jitter: int = 0,
    skip_special_sites: bool = False,
) -> SeedResult:
    observations = filter_special_sites(
        observations,
        skip_special_sites=skip_special_sites,
        fail_unknown_site=True,
    )
    if len(observations) < n_init:
        raise ValueError("Not enough observations for seeding.")
    if cfg is None:
        cfg = SeedConfig()

    obs_init = observations[:n_init]
    epoch = _midpoint_time(obs_init)
    attrib, cov = build_attributable_vector_fit(obs_init, epoch, return_cov=True)
    obs_ref = _ranging_reference_observation(obs_init, epoch)
    obs_init_cache = [_observer_helio_state(ob, ob.time) for ob in obs_init]
    obs_ref_pos, obs_ref_vel = _observer_helio_state(obs_ref, epoch)

    states: list[np.ndarray] = []
    epochs: list[Time] = []

    if jpl_state is None and jpl_target is not None:
        jpl_state = fetch_horizons_state(jpl_target, epoch, location=jpl_location, refplane=jpl_refplane)
    if jpl_state is not None:
        jpl_state = np.asarray(jpl_state, dtype=float).reshape(1, 6)
        jpl_t_em, jpl_state_em = _emission_epoch_for_state(
            jpl_state[0],
            epoch,
            obs_ref,
            epoch,
            obs_pos=obs_ref_pos,
            obs_vel=obs_ref_vel,
        )
        states.append(jpl_state_em)
        epochs.append(jpl_t_em)
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
            cov_jpl = np.asarray(jpl_cov, dtype=float).reshape(6, 6) * float(jpl_cov_scale)
            rng = np.random.default_rng(cfg.seed)
            jitter = rng.multivariate_normal(np.zeros(6), cov_jpl, size=int(jpl_n_jitter))
            for delta in jitter:
                states.append(jpl_state_em + delta)
                epochs.append(jpl_t_em)

    seed_states = seed_local_from_attrib(obs_ref, epoch, attrib, cov, cfg=cfg)
    for state in seed_states:
        t_em, st_em = _emission_epoch_for_state(
            state,
            epoch,
            obs_ref,
            epoch,
            obs_pos=obs_ref_pos,
            obs_vel=obs_ref_vel,
        )
        states.append(st_em)
        epochs.append(t_em)

    if not states:
        raise RuntimeError("No valid seed states generated.")

    if cfg.seed_obs_chi2_conf is not None:
        df = int(cfg.seed_obs_df) if cfg.seed_obs_df is not None else 2 * len(obs_init)
        chi2_star = stats.chi2.ppf(float(cfg.seed_obs_chi2_conf), df=df)
        scored: list[tuple[float, np.ndarray, Time]] = []
        for st, st_epoch in zip(states, epochs):
            chi2_val = _obs_chi2_for_state(st, st_epoch, obs_init, obs_cache=obs_init_cache)
            if chi2_val <= chi2_star:
                scored.append((chi2_val, st, st_epoch))
        if not scored:
            scored = [
                (_obs_chi2_for_state(st, ep, obs_init, obs_cache=obs_init_cache), st, ep)
                for st, ep in zip(states, epochs)
            ]
        scored.sort(key=lambda item: item[0])
        keep = len(scored)
        if cfg.seed_obs_max_keep is not None:
            keep = min(keep, int(cfg.seed_obs_max_keep))
        states = [item[1] for item in scored[:keep]]
        epochs = [item[2] for item in scored[:keep]]

    return SeedResult(
        states=np.asarray(states, dtype=float),
        epochs=np.asarray(epochs, dtype=object),
        attributable=attrib,
        cov=cov,
        obs_ref=obs_ref,
        epoch=epoch,
    )
