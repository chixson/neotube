from __future__ import annotations

import math
from dataclasses import dataclass
import multiprocessing as mp
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
"""
Notes (this patch)
-------------------
What this patch does:
- Makes the robust rho-grid + stable admissible-interval primitives the default
  path by switching the rho grid builder default to the hybrid approach.
- Adds an explicit `rhodot_max_km_s` config option (None by default) so callers
  can optionally clip |rhodot| when building admissible envelopes.
- Raises the default admissible rho resolution to reduce coarse-grid failures.

What this patch DOES NOT do:
- It does NOT implement adaptive mesh refinement (AMR) over rho.
"""


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
    admissible_n_rho: int = 1000
    admissible_n_per_gamma: int = 8
    admissible_bound_only: bool = True
    rho_prior_mode: str = "log"
    # Decouple the rho grid spacing from the rho prior.
    # None -> use rho_prior_mode for the grid (backwards-compatible).
    rho_grid_mode: str | None = "hybrid"
    rho_grid_points_per_decade: int = 24
    rho_grid_max_points: int = 20000
    rho_grid_inner_frac: float = 0.6
    rho_grid_inner_max_km: float = 1.0e9
    # Optional clipping of admissible rhodot intervals.
    rhodot_max_km_s: float | None = None
    # Admissible interval definition: "bound" or "speedcap".
    admissible_cap_mode: str = "bound"
    admissible_speed_cap_km_s: float = C_KM_S


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


def _rho_grid_from_cfg(cfg: SeedConfig) -> np.ndarray:
    rho_min = max(1e-12, float(cfg.rho_min_km))
    rho_max = max(rho_min / AU_KM, float(cfg.rho_max_au)) * AU_KM
    n = int(cfg.admissible_n_rho)
    mode = str((cfg.rho_grid_mode or cfg.rho_prior_mode or "log")).lower()
    if mode == "log":
        return np.logspace(math.log10(rho_min), math.log10(rho_max), n)
    if mode == "volume":
        # Uniform in volume => CDF proportional to rho^3
        r3 = np.linspace(rho_min**3, rho_max**3, n)
        return np.cbrt(r3)
    if mode == "skeleton":
        ppd = max(2, int(cfg.rho_grid_points_per_decade))
        decades = max(1e-12, math.log10(rho_max) - math.log10(rho_min))
        n_skel = int(math.ceil(decades * ppd)) + 1
        n_skel = min(int(cfg.rho_grid_max_points), max(n_skel, 2))
        return np.logspace(math.log10(rho_min), math.log10(rho_max), n_skel)
    if mode == "hybrid":
        inner_frac = float(np.clip(cfg.rho_grid_inner_frac, 0.05, 0.95))
        n_inner = max(2, int(math.floor(n * inner_frac)))
        n_outer = max(2, (n - n_inner) + 1)
        rho_cut = min(rho_max, float(cfg.rho_grid_inner_max_km))
        rho_cut = max(rho_cut, rho_min * 1.0001)
        inner = np.logspace(math.log10(rho_min), math.log10(rho_cut), n_inner, endpoint=False)
        r3 = np.linspace(rho_cut**3, rho_max**3, n_outer)
        outer = np.cbrt(r3)
        return np.unique(np.concatenate([inner, outer]))
    raise ValueError(f"Unknown rho_prior_mode: {cfg.rho_prior_mode}")


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


def bead_from_state_at_epoch(
    state: np.ndarray,
    obs_ref: Observation,
    epoch: Time,
) -> tuple[np.ndarray, Attributable, float, float]:
    attrib, rho, rhodot = attrib_from_state_with_observer_time(state, obs_ref, epoch)
    bead_state = build_state_from_ranging(obs_ref, epoch, attrib, rho, rhodot)
    return bead_state, attrib, float(rho), float(rhodot)


def _admissible_intervals(
    attrib: Attributable,
    obs_ref: Observation,
    epoch: Time,
    rho_grid_km: np.ndarray,
    *,
    mu: float = GM_SUN,
    rho_min_km: float = 6471.0,
    bound_only: bool = True,
    cap_mode: str = "bound",
    speed_cap_km_s: float = C_KM_S,
    rhodot_clip_km_s: float | None = None,
    disc_tol: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    s, sdot = s_and_sdot(attrib)
    obs_pos, obs_vel = _observer_helio_state(obs_ref, epoch)

    rvec = obs_pos[None, :] + rho_grid_km[:, None] * s[None, :]
    vlin = obs_vel[None, :] + rho_grid_km[:, None] * sdot[None, :]
    rnorm = np.linalg.norm(rvec, axis=1)
    b = np.einsum("ij,j->i", vlin, s)
    vlin2 = np.einsum("ij,ij->i", vlin, vlin)
    if not bound_only:
        cap_mode = "speedcap"
    if cap_mode == "bound":
        vcap2 = 2.0 * mu / (rnorm + 1e-300)
    elif cap_mode == "speedcap":
        vcap = float(speed_cap_km_s)
        vcap2 = vcap * vcap
    else:
        raise ValueError(f"Unknown cap_mode: {cap_mode}")
    disc = b * b - (vlin2 - vcap2)

    dotmin = np.full_like(rho_grid_km, np.nan, dtype=float)
    dotmax = np.full_like(rho_grid_km, np.nan, dtype=float)

    valid = rnorm > float(rho_min_km)
    valid = valid & np.isfinite(disc) & (disc >= -float(disc_tol))

    if not np.any(valid):
        return dotmin, dotmax

    sqrt_disc = np.sqrt(np.clip(disc[valid], 0.0, None))
    dot1 = -b[valid] - sqrt_disc
    dot2 = -b[valid] + sqrt_disc
    dotmin[valid] = np.minimum(dot1, dot2)
    dotmax[valid] = np.maximum(dot1, dot2)
    if rhodot_clip_km_s is not None:
        clip = float(rhodot_clip_km_s)
        dotmin = np.maximum(dotmin, -clip)
        dotmax = np.minimum(dotmax, clip)
    return dotmin, dotmax


def admissible_rho_rhodot_envelope(
    obs_ref: Observation,
    epoch: Time,
    gamma_samples: list[np.ndarray],
    cfg: SeedConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rho_grid = _rho_grid_from_cfg(cfg)
    env_min = np.full(rho_grid.shape, np.inf, dtype=float)
    env_max = np.full(rho_grid.shape, -np.inf, dtype=float)
    n_ok = np.zeros(rho_grid.shape, dtype=int)

    for attrib in gamma_samples:
        dotmin, dotmax = _admissible_intervals(
            attrib,
            obs_ref,
            epoch,
            rho_grid,
            bound_only=cfg.admissible_bound_only,
            cap_mode=cfg.admissible_cap_mode,
            speed_cap_km_s=cfg.admissible_speed_cap_km_s,
            rhodot_clip_km_s=cfg.rhodot_max_km_s,
        )
        valid = np.isfinite(dotmin) & np.isfinite(dotmax)
        if not np.any(valid):
            continue
        env_min[valid] = np.minimum(env_min[valid], dotmin[valid])
        env_max[valid] = np.maximum(env_max[valid], dotmax[valid])
        n_ok[valid] += 1

    env_min[~np.isfinite(env_min)] = np.nan
    env_max[~np.isfinite(env_max)] = np.nan
    return rho_grid, env_min, env_max, n_ok


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
    obs_ref_pos: np.ndarray | None = None,
    obs_ref_vel: np.ndarray | None = None,
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
        if obs_ref_pos is None or obs_ref_vel is None:
            attrib_state, _, _ = attrib_from_state_with_observer_time(state, obs_ref, epoch)
        else:
            attrib_state, _, _ = _attrib_from_state_cached(state, obs_ref_pos, obs_ref_vel)
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

    rho_grid = _rho_grid_from_cfg(cfg)
    print(f"[admissible] gamma_samples={len(gamma_samples)} rho_grid={len(rho_grid)}")

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


def _state_passes_hard_tube(
    state: np.ndarray,
    state_epoch: Time,
    observations: Sequence[Observation],
    obs_cache: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    k_sigma: float = 1.0,
    emission_iter: int = 10,
    tol_sec: float = 1e-3,
) -> bool:
    for idx, ob in enumerate(observations):
        obs_pos, obs_vel = obs_cache[idx]
        try:
            _, st_em = _emission_epoch_for_state(
                state,
                state_epoch,
                ob,
                ob.time,
                max_iter=emission_iter,
                tol_sec=tol_sec,
                obs_pos=obs_pos,
                obs_vel=obs_vel,
            )
            attrib_pred, _, _ = _attrib_from_state_cached(st_em, obs_pos, obs_vel)
        except Exception:
            return False
        dra_deg = _wrap_deg(attrib_pred.ra_deg - ob.ra_deg)
        dra_deg *= math.cos(math.radians(attrib_pred.dec_deg))
        ddec_deg = attrib_pred.dec_deg - ob.dec_deg
        dra_arcsec = dra_deg * 3600.0
        ddec_arcsec = ddec_deg * 3600.0
        sigma = max(1e-6, float(ob.sigma_arcsec))
        chi2 = (dra_arcsec / sigma) ** 2 + (ddec_arcsec / sigma) ** 2
        if chi2 > (k_sigma ** 2):
            return False
    return True


def _chi2_per_obs_for_state(
    state: np.ndarray,
    state_epoch: Time,
    observations: Sequence[Observation],
    obs_cache: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    emission_iter: int = 10,
    tol_sec: float = 1e-3,
) -> np.ndarray:
    chi2s = np.empty(len(observations), dtype=float)
    for idx, ob in enumerate(observations):
        obs_pos, obs_vel = obs_cache[idx]
        try:
            _, st_em = _emission_epoch_for_state(
                state,
                state_epoch,
                ob,
                ob.time,
                max_iter=emission_iter,
                tol_sec=tol_sec,
                obs_pos=obs_pos,
                obs_vel=obs_vel,
            )
            attrib_pred, _, _ = _attrib_from_state_cached(st_em, obs_pos, obs_vel)
        except Exception:
            chi2s[idx] = float("inf")
            continue
        dra_deg = _wrap_deg(attrib_pred.ra_deg - ob.ra_deg)
        dra_deg *= math.cos(math.radians(attrib_pred.dec_deg))
        ddec_deg = attrib_pred.dec_deg - ob.dec_deg
        dra_arc = dra_deg * 3600.0
        ddec_arc = ddec_deg * 3600.0
        sigma = max(1e-6, float(ob.sigma_arcsec))
        chi2s[idx] = (dra_arc / sigma) ** 2 + (ddec_arc / sigma) ** 2
    return chi2s


def _linearized_chi2(
    state: np.ndarray,
    observations: Sequence[Observation],
    obs_cache: Sequence[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    chi2s = np.empty(len(observations), dtype=float)
    r_helio = state[:3].astype(float)
    v_helio = state[3:].astype(float)
    for idx, ob in enumerate(observations):
        obs_pos, obs_vel = obs_cache[idx]
        r_topo = r_helio - obs_pos
        v_topo = v_helio - obs_vel
        rho = float(np.linalg.norm(r_topo))
        if rho <= 0:
            chi2s[idx] = float("inf")
            continue
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
        dra_deg = ((math.degrees(ra) % 360.0) - ob.ra_deg + 180.0) % 360.0 - 180.0
        dra_deg *= math.cos(math.radians(math.degrees(dec)))
        ddec_deg = math.degrees(dec) - ob.dec_deg
        dra_arc = dra_deg * 3600.0
        ddec_arc = ddec_deg * 3600.0
        sigma = max(1e-6, float(ob.sigma_arcsec))
        chi2s[idx] = (dra_arc / sigma) ** 2 + (ddec_arc / sigma) ** 2
    return chi2s


def _batch_emission_chi2(
    states: Sequence[np.ndarray],
    state_epoch: Time,
    observations: Sequence[Observation],
    obs_cache: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    emission_iter: int = 10,
    tol_sec: float = 1e-3,
) -> np.ndarray:
    n = len(states)
    m = len(observations)
    out = np.full((n, m), np.inf, dtype=float)
    for i, st in enumerate(states):
        per = _chi2_per_obs_for_state(
            st,
            state_epoch,
            observations,
            obs_cache,
            emission_iter=emission_iter,
            tol_sec=tol_sec,
        )
        out[i, :] = per
    return out


def _find_true_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    segs: list[tuple[int, int]] = []
    if mask.size == 0:
        return segs
    in_seg = False
    start = 0
    for i, v in enumerate(mask):
        if v and not in_seg:
            in_seg = True
            start = i
        elif not v and in_seg:
            in_seg = False
            segs.append((start, i - 1))
    if in_seg:
        segs.append((start, len(mask) - 1))
    return segs


def rhodot_intervals_for_rho(
    attrib: Attributable,
    obs_ref: Observation,
    epoch: Time,
    rho_km: float,
    observations: Sequence[Observation],
    obs_cache: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    k_sigma: float = 3.0,
    N_coarse: int = 41,
    refine_tol: float = 1e-6,
    max_bisect: int = 60,
    coarse_prefilter: bool = True,
) -> list[tuple[float, float]]:
    rho_grid = np.array([rho_km], dtype=float)
    dotmin, dotmax = _admissible_intervals(
        attrib,
        obs_ref,
        epoch,
        rho_grid,
        mu=GM_SUN,
        rho_min_km=float(6471.0),
        bound_only=True,
    )
    lo_phys = dotmin[0]
    hi_phys = dotmax[0]
    if not np.isfinite(lo_phys) or not np.isfinite(hi_phys) or hi_phys <= lo_phys:
        return []

    zs = np.linspace(lo_phys, hi_phys, N_coarse)
    pass_mask = np.zeros_like(zs, dtype=bool)
    gamma_vec = np.array(
        [
            attrib.ra_deg,
            attrib.dec_deg,
            attrib.ra_dot_deg_per_day,
            attrib.dec_dot_deg_per_day,
        ],
        dtype=float,
    )
    for i, z in enumerate(zs):
        st = _build_state_from_sample(
            obs_ref,
            epoch,
            gamma_vec,
            float(rho_km),
            float(z),
            v_max_km_s=None,
            rate_max_deg_day=None,
        )
        if st is None:
            pass_mask[i] = False
            continue
        if coarse_prefilter:
            lin_chi2 = _linearized_chi2(st, observations, obs_cache)
            if np.any(lin_chi2 > (4.0 * (k_sigma ** 2))):
                pass_mask[i] = False
                continue
        chi2s = _chi2_per_obs_for_state(st, epoch, observations, obs_cache)
        pass_mask[i] = np.all(chi2s <= (k_sigma ** 2))

    segments = _find_true_segments(pass_mask)
    if not segments:
        return []

    def G(z_val: float) -> float:
        st = _build_state_from_sample(
            obs_ref,
            epoch,
            gamma_vec,
            float(rho_km),
            float(z_val),
            v_max_km_s=None,
            rate_max_deg_day=None,
        )
        if st is None:
            return float("inf")
        chi2s = _chi2_per_obs_for_state(st, epoch, observations, obs_cache)
        return float(np.max(chi2s) - (k_sigma ** 2))

    def bisect_root(a: float, b: float) -> float | None:
        fa = G(a)
        fb = G(b)
        if not np.isfinite(fa) or not np.isfinite(fb):
            return None
        if fa <= 0 and fb <= 0:
            return a
        if fa > 0 and fb > 0:
            return None
        lo = a
        hi = b
        for _ in range(max_bisect):
            mid = 0.5 * (lo + hi)
            fm = G(mid)
            if not np.isfinite(fm):
                lo = mid
                continue
            if abs(fm) <= refine_tol:
                return float(mid)
            if fm > 0:
                lo = mid
            else:
                hi = mid
        return float(0.5 * (lo + hi))

    intervals: list[tuple[float, float]] = []
    for (a_idx, b_idx) in segments:
        if a_idx == 0:
            left_lo = zs[a_idx]
            left_hi = zs[a_idx]
        else:
            left_lo = zs[a_idx - 1]
            left_hi = zs[a_idx]
        if b_idx == len(zs) - 1:
            right_lo = zs[b_idx]
            right_hi = zs[b_idx]
        else:
            right_lo = zs[b_idx]
            right_hi = zs[b_idx + 1]
        left_edge = bisect_root(left_lo, left_hi)
        right_edge = bisect_root(right_lo, right_hi)
        if left_edge is None or right_edge is None:
            left_edge = float(zs[a_idx])
            right_edge = float(zs[b_idx])
        intervals.append((left_edge, right_edge))

    intervals.sort()
    merged: list[tuple[float, float]] = []
    for lo, hi in intervals:
        if not merged:
            merged.append((lo, hi))
            continue
        plo, phi = merged[-1]
        if lo <= phi + 1e-12:
            merged[-1] = (plo, max(phi, hi))
        else:
            merged.append((lo, hi))
    return merged


def _coarse_chunk_worker_intervals(
    gamma_chunk: list[np.ndarray],
    observations: Sequence[Observation],
    epoch: Time,
    obs_ref: Observation,
    rho_grid: np.ndarray,
    cfg: SeedConfig,
    coarse_k_sigma: float,
    N_coarse: int,
    seed_offset: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray, float, float, Attributable]], int]:
    try:
        local_rng = np.random.default_rng(int(cfg.seed or 0) + seed_offset)
        obs_cache = [_observer_helio_state(ob, ob.time) for ob in observations]
        accepted: list[tuple[np.ndarray, np.ndarray, float, float, Attributable]] = []
        attempted = 0
    except Exception as exc:
        return ([], 0, f"worker_init_error: {exc!r}")
    for gamma_vec in gamma_chunk:
        attrib = Attributable(
            ra_deg=float(gamma_vec[0]),
            dec_deg=float(gamma_vec[1]),
            ra_dot_deg_per_day=float(gamma_vec[2]),
            dec_dot_deg_per_day=float(gamma_vec[3]),
        )
        for rho_val in rho_grid:
            intervals = rhodot_intervals_for_rho(
                attrib,
                obs_ref,
                epoch,
                float(rho_val),
                observations,
                obs_cache,
                k_sigma=coarse_k_sigma,
                N_coarse=N_coarse,
            )
            if not intervals:
                continue
            for lo, hi in intervals:
                if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                    continue
                nper = int(cfg.admissible_n_per_gamma)
                for _ in range(nper):
                    attempted += 1
                    rhodot_val = lo + local_rng.random() * (hi - lo)
                    st = _build_state_from_sample(
                        obs_ref,
                        epoch,
                        gamma_vec,
                        float(rho_val),
                        float(rhodot_val),
                        v_max_km_s=cfg.v_max_km_s,
                        rate_max_deg_day=cfg.rate_max_deg_day,
                    )
                    if st is None:
                        continue
                    lin_chi2 = _linearized_chi2(st, observations, obs_cache)
                    if np.any(lin_chi2 > (16.0 * (coarse_k_sigma**2))):
                        continue
                    if _state_passes_hard_tube(
                        st, epoch, observations, obs_cache, k_sigma=coarse_k_sigma
                    ):
                        r0 = st[:3].astype(float)
                        v0 = st[3:].astype(float)
                        accepted.append(
                            (r0, v0, float(rho_val), float(rhodot_val), attrib)
                        )
        if len(accepted) >= max(200, cfg.admissible_n_per_gamma * 50):
            break
    return (accepted, attempted, None)


def _coarse_chunk_worker(
    gamma_chunk: list[np.ndarray],
    observations: Sequence[Observation],
    epoch: Time,
    obs_ref: Observation,
    rho_grid: np.ndarray,
    cfg: SeedConfig,
    coarse_k_sigma: float,
    obs_ref_pos: np.ndarray,
    obs_ref_vel: np.ndarray,
    seed_offset: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray, float, float, Attributable]], int]:
    local_rng = np.random.default_rng(int(cfg.seed or 0) + seed_offset)
    local_obs_cache = [_observer_helio_state(ob, ob.time) for ob in observations]
    local_accepted: list[tuple[np.ndarray, np.ndarray, float, float, Attributable]] = []
    local_attempted = 0
    for gamma_vec in gamma_chunk:
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
            bound_only=bool(cfg.admissible_bound_only),
        )
        valid_idx = np.where(np.isfinite(dotmin) & np.isfinite(dotmax))[0]
        if valid_idx.size == 0:
            continue
        for idx in valid_idx:
            rho_val = float(rho_grid[idx])
            lo = float(dotmin[idx])
            hi = float(dotmax[idx])
            if hi <= lo:
                continue
            nper = int(cfg.admissible_n_per_gamma)
            rhodot_samples = lo + local_rng.random(nper) * (hi - lo)
            for rhodot_val in rhodot_samples:
                local_attempted += 1
                st = _build_state_from_sample(
                    obs_ref,
                    epoch,
                    gamma_vec,
                    rho_val,
                    rhodot_val,
                    v_max_km_s=cfg.v_max_km_s,
                    rate_max_deg_day=cfg.rate_max_deg_day,
                    obs_ref_pos=obs_ref_pos,
                    obs_ref_vel=obs_ref_vel,
                )
                if st is None:
                    continue
                ok = _state_passes_hard_tube(
                    st, epoch, observations, local_obs_cache, k_sigma=coarse_k_sigma
                )
                if ok:
                    r0 = st[:3].astype(float)
                    v0 = st[3:].astype(float)
                    local_accepted.append((r0, v0, rho_val, rhodot_val, attrib))
    return local_accepted, local_attempted


def sample_admissible_beads(
    observations: Sequence[Observation],
    epoch: Time,
    attrib_mean: Attributable,
    attrib_cov: np.ndarray,
    *,
    cfg: SeedConfig,
    k_sigma: float = 1.0,
    target: int | None = None,
    batch_size: int = 128,
) -> dict[str, object]:
    rng = np.random.default_rng(cfg.seed)
    if target is None:
        target = int(cfg.admissible_n_rho * cfg.admissible_n_per_gamma)

    mean_vec = np.array(
        [
            attrib_mean.ra_deg,
            attrib_mean.dec_deg,
            attrib_mean.ra_dot_deg_per_day,
            attrib_mean.dec_dot_deg_per_day,
        ],
        dtype=float,
    )

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

    rho_grid = _rho_grid_from_cfg(cfg)

    obs_ref = _ranging_reference_observation(observations, epoch)
    obs_ref_pos, obs_ref_vel = _observer_helio_state(obs_ref, epoch)
    obs_cache = [_observer_helio_state(ob, ob.time) for ob in observations]

    accepted: list[tuple[np.ndarray, np.ndarray, float, float, Attributable]] = []
    attempted = 0

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

        for idx in valid_idx:
            rho_val = float(rho_grid[idx])
            lo = float(dotmin[idx])
            hi = float(dotmax[idx])
            if hi <= lo:
                continue
            nper = int(cfg.admissible_n_per_gamma)
            u_draws = rng.random(nper)
            rhodot_samples = lo + u_draws * (hi - lo)

            candidates: list[tuple[np.ndarray, float, float, Attributable]] = []
            for rhodot_val in rhodot_samples:
                attempted += 1
                state = _build_state_from_sample(
                    obs_ref,
                    epoch,
                    gamma_vec,
                    rho_val,
                    float(rhodot_val),
                    v_max_km_s=cfg.v_max_km_s,
                    rate_max_deg_day=cfg.rate_max_deg_day,
                    obs_ref_pos=obs_ref_pos,
                    obs_ref_vel=obs_ref_vel,
                )
                if state is None:
                    continue
                candidates.append((state, rho_val, float(rhodot_val), attrib))

            idx0 = 0
            while idx0 < len(candidates):
                batch = candidates[idx0 : idx0 + batch_size]
                idx0 += batch_size
                for st, rho_v, rhod_v, attrib_v in batch:
                    ok = _state_passes_hard_tube(
                        st, epoch, observations, obs_cache, k_sigma=k_sigma
                    )
                    if ok:
                        r0 = st[:3].astype(float)
                        v0 = st[3:].astype(float)
                        accepted.append((r0, v0, rho_v, rhod_v, attrib_v))
                        if len(accepted) >= target:
                            acc = np.array(accepted, dtype=object)
                            return {
                                "r0": np.vstack(acc[:, 0]),
                                "v0": np.vstack(acc[:, 1]),
                                "rho_km": np.array(list(acc[:, 2]), dtype=float),
                                "rhodot_km_s": np.array(list(acc[:, 3]), dtype=float),
                                "attrib": list(acc[:, 4]),
                                "accepted_count": len(accepted),
                                "attempted": attempted,
                            }

    if not accepted:
        return {"accepted_count": 0, "attempted": attempted}

    acc = np.array(accepted, dtype=object)
    return {
        "r0": np.vstack(acc[:, 0]),
        "v0": np.vstack(acc[:, 1]),
        "rho_km": np.array(list(acc[:, 2]), dtype=float),
        "rhodot_km_s": np.array(list(acc[:, 3]), dtype=float),
        "attrib": list(acc[:, 4]),
        "accepted_count": len(accepted),
        "attempted": attempted,
    }


def adaptive_sample_admissible_beads(
    observations: Sequence[Observation],
    epoch: Time,
    attrib_mean: Attributable,
    attrib_cov: np.ndarray,
    *,
    cfg: SeedConfig,
    target: int = 2000,
    coarse_k_sigma: float = 6.0,
    final_k_sigma: float = 3.0,
    bins_logrho: int = 40,
    bins_rhodot: int = 40,
    adaptive_rounds: int = 3,
    adaptive_samples_per_cell: int = 200,
    N_coarse: int = 41,
    n_workers: int | None = None,
) -> dict[str, object]:
    rng = np.random.default_rng(cfg.seed)
    obs_ref = _ranging_reference_observation(observations, epoch)
    obs_cache = [_observer_helio_state(ob, ob.time) for ob in observations]

    mean_vec = np.array(
        [
            attrib_mean.ra_deg,
            attrib_mean.dec_deg,
            attrib_mean.ra_dot_deg_per_day,
            attrib_mean.dec_dot_deg_per_day,
        ],
        dtype=float,
    )
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

    rho_grid = _rho_grid_from_cfg(cfg)
    print(f"[admissible] gamma_samples={len(gamma_samples)} rho_grid={len(rho_grid)}")

    accepted: list[tuple[np.ndarray, np.ndarray, float, float, Attributable]] = []
    attempted = 0

    if n_workers is None:
        n_workers = 1
    if n_workers > 1 and len(gamma_samples) > 1:
        chunks = np.array_split(np.array(gamma_samples, dtype=object), n_workers)
        with mp.Pool(processes=n_workers) as pool:
            results = pool.starmap(
                _coarse_chunk_worker_intervals,
                [
                    (
                        list(chunk),
                        observations,
                        epoch,
                        obs_ref,
                        rho_grid,
                        cfg,
                        coarse_k_sigma,
                        N_coarse,
                        i * 1000,
                    )
                    for i, chunk in enumerate(chunks)
                ],
            )
        for acc_chunk, att_chunk, err in results:
            if err:
                print(f"[admissible] worker error: {err}")
                continue
            accepted.extend(acc_chunk)
            attempted += att_chunk
    else:
        acc_chunk, att_chunk, err = _coarse_chunk_worker_intervals(
            gamma_samples,
            observations,
            epoch,
            obs_ref,
            rho_grid,
            cfg,
            coarse_k_sigma,
            N_coarse,
            0,
        )
        if err:
            print(f"[admissible] worker error: {err}")
        else:
            accepted.extend(acc_chunk)
            attempted += att_chunk

    if len(accepted) >= max(200, target // 4):
        accepted = accepted[: max(200, target // 4)]

    if len(accepted) == 0:
        return {"accepted_count": 0, "attempted": attempted}

    arr = np.array([[math.log10(a[2]), a[3]] for a in accepted], dtype=float)
    for _ in range(adaptive_rounds):
        hist, xedges, yedges = np.histogram2d(
            arr[:, 0], arr[:, 1], bins=[bins_logrho, bins_rhodot]
        )
        empty = np.where(hist == 0)
        empty_cells = list(zip(empty[0], empty[1]))
        if not empty_cells:
            break
        n_cells = min(len(empty_cells), max(50, len(empty_cells) // 4))
        chosen = rng.choice(len(empty_cells), size=n_cells, replace=False)
        for ci in chosen:
            ix, iy = empty_cells[ci]
            logrho_lo, logrho_hi = xedges[ix], xedges[ix + 1]
            rhodot_lo, rhodot_hi = yedges[iy], yedges[iy + 1]
            for _ in range(adaptive_samples_per_cell):
                gamma_vec = gamma_samples[int(rng.integers(len(gamma_samples)))]
                rho_val = 10 ** (logrho_lo + rng.random() * (logrho_hi - logrho_lo))
                rhodot_val = rhodot_lo + rng.random() * (rhodot_hi - rhodot_lo)
                attempted += 1
                st = _build_state_from_sample(
                    obs_ref,
                    epoch,
                    gamma_vec,
                    float(rho_val),
                    float(rhodot_val),
                    v_max_km_s=cfg.v_max_km_s,
                    rate_max_deg_day=cfg.rate_max_deg_day,
                )
                if st is None:
                    continue
                if not _state_passes_hard_tube(
                    st, epoch, observations, obs_cache, k_sigma=final_k_sigma
                ):
                    continue
                r0 = st[:3].astype(float)
                v0 = st[3:].astype(float)
                accepted.append(
                    (
                        r0,
                        v0,
                        float(rho_val),
                        float(rhodot_val),
                        Attributable(
                            ra_deg=float(gamma_vec[0]),
                            dec_deg=float(gamma_vec[1]),
                            ra_dot_deg_per_day=float(gamma_vec[2]),
                            dec_dot_deg_per_day=float(gamma_vec[3]),
                        ),
                    )
                )
                arr = np.vstack([arr, [math.log10(rho_val), rhodot_val]])
                if len(accepted) >= target:
                    break
            if len(accepted) >= target:
                break
        if len(accepted) >= target:
            break

    acc = np.array(accepted, dtype=object)
    return {
        "r0": np.vstack(acc[:, 0]),
        "v0": np.vstack(acc[:, 1]),
        "rho_km": np.array(list(acc[:, 2]), dtype=float),
        "rhodot_km_s": np.array(list(acc[:, 3]), dtype=float),
        "attrib": list(acc[:, 4]),
        "accepted_count": len(accepted),
        "attempted": attempted,
    }


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
