from __future__ import annotations

import math
from pathlib import Path
import time
from concurrent.futures import ProcessPoolExecutor
import os
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Any, Iterable, Sequence

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord, get_body_barycentric_posvel
from astropy.time import Time

from .fit import _predict_batch, _site_offset, _site_offset_cached
from .propagate import _site_states, GM_SUN, _prepare_obs_cache
from .sites import get_site_ephemeris, get_site_kind
from .models import Observation

AU_KM = 149597870.7
DAY_S = 86400.0


def tangent_basis_from_unit(s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Given unit LOS s (3,), return orthonormal basis e_alpha, e_delta."""
    z = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(s, z)) > 0.9:
        z = np.array([0.0, 1.0, 0.0])
    e_alpha = np.cross(z, s)
    e_alpha /= np.linalg.norm(e_alpha)
    e_delta = np.cross(s, e_alpha)
    e_delta /= np.linalg.norm(e_delta)
    return e_alpha, e_delta


def add_tangent_jitter(
    states: np.ndarray,
    obs: Sequence[Observation] | str,
    posterior: object,
    n_per_state: int = 10,
    sigma_arcsec: float = 0.5,
    fit_scale: float | None = None,
    site_kappas: dict[str, float] | None = None,
    vel_timescale_sec: float | None = None,
    vel_scale_factor: float = 1.0,
    seed: int | None = None,
) -> np.ndarray:
    """
    Improved tangent-plane jitter: produces local clouds around each state with
    consistent position and velocity perturbations.

    Parameters
    ----------
    states : (N,6) array
      Input candidate states (x,y,z,vx,vy,vz) in km and km/s.
    obs : observation list or obs CSV path
      Used to compute site offsets and a representative time baseline.
    posterior : posterior object (used for fit_scale, site_kappas lookup)
    n_per_state : int
      Number of jittered samples to generate per input state.
    sigma_arcsec : float
      Angular sigma for tangent jitter (arcsec); multiplied by fit_scale and site kappa.
    fit_scale : float or None
      Use posterior.fit_scale if None.
    site_kappas : dict or None
      Per-site kappa multipliers; if None read from posterior.
    vel_timescale_sec : float or None
      Timescale to map spatial jitter to velocity jitter: v_noise ~ dr / vel_timescale_sec.
      If None, use median(obs span) or 86400s default.
    vel_scale_factor : float
      Additional multiplier applied to computed velocity jitter (>=1.0 increases spread).

    Returns
    -------
    (N*n_per_state,6) np.ndarray of jittered states.
    """
    if site_kappas is None:
        site_kappas = getattr(posterior, "site_kappas", {}) or {}
    if fit_scale is None:
        fit_scale = float(getattr(posterior, "fit_scale", 1.0))

    if isinstance(obs, (str, Path)):
        from neotube.fit_cli import load_observations

        obs = load_observations(Path(obs), None)

    dt_seconds = 86400.0
    try:
        times = np.array([o.time.jd for o in obs])
        if len(times) >= 2:
            span_days = np.max(times) - np.min(times)
            dt_seconds = max(1.0, span_days * 86400.0)
    except Exception:
        dt_seconds = 86400.0
    if vel_timescale_sec is None:
        vel_timescale_sec = dt_seconds

    obs_count = len(obs)
    if obs_count == 0:
        sigma_rad_states = np.full(len(states), np.deg2rad(sigma_arcsec * fit_scale / 3600.0))
    else:
        site_kappas_arr = np.array([site_kappas.get(o.site, 1.0) for o in obs], dtype=float)
        if obs_count == 1:
            obs_idx = np.zeros(len(states), dtype=int)
        else:
            obs_idx = rng.integers(0, obs_count, size=len(states))
        sigma_eff_arcsec = sigma_arcsec * fit_scale * site_kappas_arr[obs_idx]
        sigma_rad_states = np.deg2rad(sigma_eff_arcsec / 3600.0)

    rng = np.random.default_rng(seed)
    out = []
    for i, st in enumerate(states):
        r = st[:3].astype(float)
        v = st[3:].astype(float)
        rho = np.linalg.norm(r)
        if rho <= 0:
            continue
        sigma_rad = float(sigma_rad_states[i])
        s = r / rho
        e_a, e_d = tangent_basis_from_unit(s)
        d_alpha = rng.normal(scale=sigma_rad, size=n_per_state)
        d_delta = rng.normal(scale=sigma_rad, size=n_per_state)
        for d_a, d_d in zip(d_alpha, d_delta):
            cosdec = np.sqrt(max(0.0, 1.0 - s[2] ** 2))
            dra_km = rho * cosdec * d_a
            ddec_km = rho * d_d
            dr = e_a * dra_km + e_d * ddec_km
            new_r = r + dr
            v_noise = (dr / max(1.0, vel_timescale_sec)) * vel_scale_factor
            v_noise += rng.normal(scale=0.1, size=3) * np.linalg.norm(v_noise + 1e-12)
            new_v = v + v_noise
            out.append(np.hstack([new_r, new_v]))
    if len(out) == 0:
        return np.empty((0, 6), dtype=float)
    return np.array(out, dtype=float)


def add_local_multit_jitter(
    states: np.ndarray,
    obs: Sequence[Observation] | str,
    posterior: object,
    n_per_state: int = 10,
    sigma_arcsec: float = 0.5,
    fit_scale: float | None = None,
    site_kappas: dict | None = None,
    vel_scale_factor: float = 1.0,
    df: float = 4.0,
    seed: int | None = None,
) -> np.ndarray:
    """For each 6D state, sample n_per_state correlated multivariate-t draws.

    - Position std (km) = rho * sigma_rad
    - Velocity std (km/s) = position_std / dt_seconds * vel_scale_factor
    - Covariance is diagonal ([pos_std^2]*3 + [vel_std^2]*3)
    Returns array (N*n_per_state,6).
    """
    from .fit_cli import load_observations

    if site_kappas is None:
        site_kappas = getattr(posterior, "site_kappas", {}) or {}
    if fit_scale is None:
        fit_scale = float(getattr(posterior, "fit_scale", 1.0))

    if isinstance(obs, (str, Path)):
        obs = load_observations(Path(obs), None)

    if len(obs) >= 2:
        times = np.array([o.time.jd for o in obs])
        dt_days = float(np.max(times) - np.min(times))
        dt_seconds = max(1.0, dt_days * 86400.0)
    else:
        dt_seconds = 86400.0

    obs_count = len(obs)
    if obs_count == 0:
        sigma_rad_states = np.full(len(states), np.deg2rad(sigma_arcsec * fit_scale / 3600.0))
    else:
        site_kappas_arr = np.array([site_kappas.get(o.site, 1.0) for o in obs], dtype=float)
        if obs_count == 1:
            obs_idx = np.zeros(len(states), dtype=int)
        else:
            obs_idx = rng.integers(0, obs_count, size=len(states))
        sigma_eff_arcsec = sigma_arcsec * fit_scale * site_kappas_arr[obs_idx]
        sigma_rad_states = np.deg2rad(sigma_eff_arcsec / 3600.0)

    out = []
    for i, st in enumerate(states):
        r = st[:3].astype(float)
        v = st[3:].astype(float)
        rho = np.linalg.norm(r)
        if rho <= 0:
            continue
        sigma_rad = float(sigma_rad_states[i])
        pos_std_km = float(rho * sigma_rad)
        vel_std_km_s = float((pos_std_km / max(1.0, dt_seconds)) * vel_scale_factor)

        mean = np.hstack([r, v])
        cov = np.diag([pos_std_km**2] * 3 + [vel_std_km_s**2] * 3)
        d = 6
        g = rng.gamma(df / 2.0, 2.0 / df, size=n_per_state)
        z = rng.multivariate_normal(np.zeros(d), cov, size=n_per_state)
        samples = mean[None, :] + z / np.sqrt(g)[:, None]
        out.extend(list(samples))
    if len(out) == 0:
        return np.empty((0, 6), dtype=float)
    return np.vstack(out)


def add_attributable_jitter(
    states: np.ndarray,
    obs: Sequence[Observation] | str,
    posterior: object,
    n_per_state: int = 10,
    sigma_arcsec: float = 0.5,
    fit_scale: float | None = None,
    site_kappas: dict | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Vectorized attributable jitter (ra, dec, ra_dot, dec_dot)."""
    from .fit_cli import load_observations

    if site_kappas is None:
        site_kappas = getattr(posterior, "site_kappas", {}) or {}
    if fit_scale is None:
        fit_scale = float(getattr(posterior, "fit_scale", 1.0))

    if isinstance(obs, str):
        obs = load_observations(obs, None)

    if len(obs) >= 2:
        times = np.array([o.time.jd for o in obs])
        dt_days = float(np.max(times) - np.min(times))
        dt_seconds = max(1.0, dt_days * DAY_S)
    else:
        dt_seconds = DAY_S

    rng = np.random.default_rng(seed)
    obs_count = len(obs)
    if obs_count == 0:
        site_offsets = np.zeros((states.shape[0], 3), dtype=float)
        site_vels = np.zeros((states.shape[0], 3), dtype=float)
        sigma_rad_states = np.full(states.shape[0], np.deg2rad(sigma_arcsec * fit_scale / 3600.0))
    else:
        site_kappas_arr = np.array([site_kappas.get(o.site, 1.0) for o in obs], dtype=float)
        if obs_count == 1:
            obs_idx = np.zeros(states.shape[0], dtype=int)
        else:
            # Select an observation per-state to respect varying sites (spacecraft).
            obs_idx = rng.integers(0, obs_count, size=states.shape[0])
        obs_positions = [o.observer_pos_km for o in obs]
        site_pos_all, site_vel_all = _site_states(
            [o.time for o in obs],
            [o.site for o in obs],
            observer_positions_km=obs_positions,
            observer_velocities_km_s=None,
            allow_unknown_site=True,
        )
        site_offsets = site_pos_all[obs_idx]
        site_vels = site_vel_all[obs_idx]
        sigma_eff_arcsec = sigma_arcsec * fit_scale * site_kappas_arr[obs_idx]
        sigma_rad_states = np.deg2rad(sigma_eff_arcsec / 3600.0)

    epoch = getattr(posterior, "epoch", None)
    if epoch is None:
        raise RuntimeError("posterior.epoch is required for attributable jitter")

    earth_pos, earth_vel = get_body_barycentric_posvel("earth", epoch)
    sun_pos, sun_vel = get_body_barycentric_posvel("sun", epoch)
    earth_helio = (earth_pos.xyz - sun_pos.xyz).to(u.km).value.flatten()
    earth_vel_helio = (earth_vel.xyz - sun_vel.xyz).to(u.km / u.s).value.flatten()
    site_offset = site_offsets

    states = np.asarray(states, dtype=float)
    if states.size == 0:
        return np.empty((0, 6), dtype=float)

    r_helio = states[:, :3]
    v_helio = states[:, 3:]
    r_geo = r_helio - earth_helio[None, :]
    v_geo = v_helio - earth_vel_helio[None, :]
    r_topo = r_geo - site_offset
    rho = np.linalg.norm(r_topo, axis=1)
    ok = rho > 0
    if not np.any(ok):
        return np.empty((0, 6), dtype=float)

    r_topo = r_topo[ok]
    v_geo = v_geo[ok]
    rho_valid = rho[ok]
    site_offset = site_offset[ok]
    site_vels = site_vels[ok]
    sigma_rad_states = sigma_rad_states[ok]

    s = r_topo / rho_valid[:, None]
    # Use observer-relative velocity for attributable conversion.
    v_geo_rel = v_geo - site_vels
    rhodot = np.einsum("ij,ij->i", v_geo_rel, s)
    sdot = (v_geo_rel - rhodot[:, None] * s) / np.maximum(rho_valid[:, None], 1e-12)

    x = s[:, 0]
    y = s[:, 1]
    z = s[:, 2]
    rxy2 = np.maximum(x * x + y * y, 1e-12)
    ra = np.arctan2(y, x)
    dec = np.arcsin(np.clip(z, -1.0, 1.0))
    xd = sdot[:, 0]
    yd = sdot[:, 1]
    zd = sdot[:, 2]
    ra_dot = (x * yd - y * xd) / rxy2
    cosdec = np.maximum(np.sqrt(rxy2), 1e-12)
    dec_dot = (zd - z * (x * xd + y * yd) / rxy2) / cosdec

    n_states = s.shape[0]
    n_draws = int(n_per_state)

    ra_j = ra[:, None] + rng.normal(scale=sigma_rad_states[:, None], size=(n_states, n_draws))
    dec_j = dec[:, None] + rng.normal(scale=sigma_rad_states[:, None], size=(n_states, n_draws))
    dec_j = np.clip(dec_j, -0.5 * math.pi, 0.5 * math.pi)
    ra_dot_j = ra_dot[:, None] + rng.normal(
        scale=sigma_rad_states[:, None] / dt_seconds, size=(n_states, n_draws)
    )
    dec_dot_j = dec_dot[:, None] + rng.normal(
        scale=sigma_rad_states[:, None] / dt_seconds, size=(n_states, n_draws)
    )

    cd = np.cos(dec_j)
    sd = np.sin(dec_j)
    cr = np.cos(ra_j)
    sr = np.sin(ra_j)
    s_prime = np.stack([cd * cr, cd * sr, sd], axis=2)

    sdot_x = -cd * sr * ra_dot_j - sd * cr * dec_dot_j
    sdot_y = cd * cr * ra_dot_j - sd * sr * dec_dot_j
    sdot_z = cd * dec_dot_j
    sdot_prime = np.stack([sdot_x, sdot_y, sdot_z], axis=2)

    rho_nm = rho_valid[:, None, None]
    rhodot_nm = rhodot[:, None, None]
    r_topo_new = rho_nm * s_prime
    r_geo_new = site_offset[:, None, :] + r_topo_new
    v_geo_new = rhodot_nm * s_prime + rho_nm * sdot_prime

    r_helio_new = earth_helio[None, None, :] + r_geo_new
    v_helio_new = earth_vel_helio[None, None, :] + v_geo_new

    r_flat = r_helio_new.reshape(-1, 3)
    v_flat = v_helio_new.reshape(-1, 3)
    return np.hstack([r_flat, v_flat])


def _attrib_rho_from_state(
    state: np.ndarray,
    obs: Observation,
    epoch: Time,
) -> tuple[Attributable, float, float]:
    """Compute attributable + (rho, rhodot) from a heliocentric state at epoch."""
    earth_pos, earth_vel = get_body_barycentric_posvel("earth", epoch)
    sun_pos, sun_vel = get_body_barycentric_posvel("sun", epoch)
    earth_helio = (earth_pos.xyz - sun_pos.xyz).to(u.km).value.flatten()
    earth_vel_helio = (earth_vel.xyz - sun_vel.xyz).to(u.km / u.s).value.flatten()
    site_pos, site_vel = _site_states(
        [epoch],
        [obs.site],
        observer_positions_km=[obs.observer_pos_km],
        observer_velocities_km_s=None,
        allow_unknown_site=True,
    )
    site_offset = site_pos[0]
    site_vel = site_vel[0]

    r_helio = state[:3].astype(float)
    v_helio = state[3:].astype(float)
    r_geo = r_helio - earth_helio
    v_geo = v_helio - earth_vel_helio - site_vel
    r_topo = r_geo - site_offset
    rho = float(np.linalg.norm(r_topo))
    if rho <= 0:
        raise RuntimeError("Non-positive rho in attributable conversion.")
    s = r_topo / rho
    rhodot = float(np.dot(v_geo, s))
    sdot = (v_geo - rhodot * s) / max(rho, 1e-12)

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
        ra_dot_deg_per_day=float(math.degrees(ra_dot) * DAY_S),
        dec_dot_deg_per_day=float(math.degrees(dec_dot) * DAY_S),
    )
    return attrib, rho, rhodot


def add_range_jitter(
    states: np.ndarray,
    obs: Sequence[Observation] | Path | str,
    epoch: Time,
    n_per_state: int = 10,
    rho_min_au: float = 1.8,
    rho_max_au: float = 4.5,
    rhodot_max_kms: float = 20.0,
    seed: int | None = None,
) -> np.ndarray:
    """Generate range-jittered states by resampling rho/rhodot per attributable."""
    if isinstance(obs, (str, Path)):
        from neotube.fit_cli import load_observations

        obs = load_observations(Path(obs), None)
    if not obs:
        return np.empty((0, 6), dtype=float)
    obs0 = obs[0]

    rng = np.random.default_rng(None if seed is None else int(seed))
    log_rho_min = math.log(max(1e-12, rho_min_au))
    log_rho_max = math.log(max(rho_min_au, rho_max_au))

    out = []
    for st in states:
        try:
            attrib, _, _ = _attrib_rho_from_state(st, obs0, epoch)
        except Exception:
            continue
        rhos = np.exp(rng.uniform(log_rho_min, log_rho_max, size=n_per_state)) * AU_KM
        rhodots = rng.uniform(-rhodot_max_kms, rhodot_max_kms, size=n_per_state)
        for rho_km, rhodot_km_s in zip(rhos, rhodots):
            out.append(build_state_from_ranging(obs0, epoch, attrib, float(rho_km), float(rhodot_km_s)))
    if not out:
        return np.empty((0, 6), dtype=float)
    return np.array(out, dtype=float)


_SPREAD_WORKER_CTX: dict[str, object] = {}


def _init_spread_worker(obs: Sequence[Observation], posterior_small: dict[str, object]) -> None:
    """Initializer for local-spread workers to avoid repeated pickling."""
    global _SPREAD_WORKER_CTX
    _SPREAD_WORKER_CTX["obs"] = obs

    class PosteriorLite:
        def __init__(self, epoch_jd: float, fit_scale: float, site_kappas: dict[str, float], nu: float | None):
            self.epoch = Time(epoch_jd, format="jd")
            self.fit_scale = fit_scale
            self.site_kappas = site_kappas
            self.nu = nu

    _SPREAD_WORKER_CTX["posterior"] = PosteriorLite(
        float(posterior_small.get("epoch_jd", Time.now().jd)),
        float(posterior_small.get("fit_scale", 1.0)),
        dict(posterior_small.get("site_kappas", {})),
        posterior_small.get("nu", None),
    )


def _spread_chunk_worker(job: dict[str, object]) -> np.ndarray:
    """Worker function for local-spread chunk jobs."""
    mode = job["mode"]
    states = job["states"]
    n_per_state = int(job.get("n_per_state", 10))
    sigma_arcsec = float(job.get("sigma_arcsec", 0.5))
    vel_scale_factor = float(job.get("vel_scale_factor", 1.0))
    df = float(job.get("df", 4.0))
    seed = job.get("seed", None)
    posterior = _SPREAD_WORKER_CTX["posterior"]
    obs = _SPREAD_WORKER_CTX["obs"]

    if mode == "tangent":
        return add_tangent_jitter(
            states,
            obs,
            posterior,
            n_per_state=n_per_state,
            sigma_arcsec=sigma_arcsec,
            fit_scale=posterior.fit_scale,
            site_kappas=posterior.site_kappas,
            vel_timescale_sec=None,
            vel_scale_factor=vel_scale_factor,
            seed=seed,
        )
    if mode == "multit":
        return add_local_multit_jitter(
            states,
            obs,
            posterior,
            n_per_state=n_per_state,
            sigma_arcsec=sigma_arcsec,
            fit_scale=posterior.fit_scale,
            site_kappas=posterior.site_kappas,
            vel_scale_factor=vel_scale_factor,
            df=df,
            seed=seed,
        )
    if mode == "attributable":
        return add_attributable_jitter(
            states,
            obs,
            posterior,
            n_per_state=n_per_state,
            sigma_arcsec=sigma_arcsec,
            fit_scale=posterior.fit_scale,
            site_kappas=posterior.site_kappas,
            seed=seed,
        )
    raise RuntimeError(f"Unknown local-spread mode: {mode}")


def add_local_spread_parallel(
    states: np.ndarray,
    obs: Sequence[Observation] | str,
    posterior: object,
    mode: str = "tangent",
    n_per_state: int = 10,
    sigma_arcsec: float = 0.5,
    fit_scale: float | None = None,
    site_kappas: dict[str, float] | None = None,
    vel_scale_factor: float = 1.0,
    df: float = 4.0,
    n_workers: int | None = None,
    chunk_size: int | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Parallel driver for local-spread jitter using a process pool."""
    if isinstance(obs, (str, Path)):
        from neotube.fit_cli import load_observations

        obs = load_observations(Path(obs), None)

    if site_kappas is None:
        site_kappas = getattr(posterior, "site_kappas", {}) or {}
    if fit_scale is None:
        fit_scale = float(getattr(posterior, "fit_scale", 1.0))

    n_states = len(states)
    if n_states == 0:
        return np.empty((0, 6), dtype=float)

    if n_workers is None:
        n_workers = max(1, min(32, os.cpu_count() or 1))
    if chunk_size is None:
        chunk_size = min(max(128, n_states // (max(1, n_workers) * 4 + 1)), 4096)

    posterior_small = {
        "epoch_jd": float(getattr(posterior, "epoch", Time.now()).jd),
        "fit_scale": fit_scale,
        "site_kappas": site_kappas,
        "nu": getattr(posterior, "nu", None),
    }

    chunks = []
    for i in range(0, n_states, chunk_size):
        chunks.append(states[i : i + chunk_size])

    jobs = []
    for idx, chunk in enumerate(chunks):
        job_seed = None if seed is None else int(seed) + idx
        jobs.append(
            {
                "mode": mode,
                "states": chunk,
                "n_per_state": n_per_state,
                "sigma_arcsec": sigma_arcsec,
                "vel_scale_factor": float(vel_scale_factor),
                "df": float(df),
                "seed": job_seed,
            }
        )

    results = []
    with ProcessPoolExecutor(
        max_workers=max(1, int(n_workers)),
        initializer=_init_spread_worker,
        initargs=(obs, posterior_small),
    ) as executor:
        for res in executor.map(_spread_chunk_worker, jobs):
            if isinstance(res, np.ndarray) and res.size:
                results.append(res)

    if not results:
        return np.empty((0, 6), dtype=float)
    return np.vstack(results)


def _rho_log_prior(
    rhos_km: np.ndarray,
    rho_prior_mode: str | None,
    rho_prior_power: float | None,
) -> np.ndarray:
    if rho_prior_mode is None:
        power = 2.0 if rho_prior_power is None else float(rho_prior_power)
        return power * np.log(np.maximum(rhos_km, 1e-12))
    mode = rho_prior_mode.lower()
    if mode == "volume":
        return 2.0 * np.log(np.maximum(rhos_km, 1e-12))
    if mode == "uniform":
        return np.zeros_like(rhos_km, dtype=float)
    if mode == "log":
        return -1.0 * np.log(np.maximum(rhos_km, 1e-12))
    raise ValueError(f"Unknown rho_prior_mode={rho_prior_mode!r}")


def _stratified_indices(weights: np.ndarray, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    if n_samples <= 0:
        return np.empty(0, dtype=int)
    cum = np.cumsum(weights)
    if cum[-1] <= 0.0:
        raise ValueError("Weights must sum to > 0 for resampling.")
    positions = (rng.random(n_samples) + np.arange(n_samples)) / n_samples
    return np.searchsorted(cum, positions, side="right")


def stratified_resample(
    states: np.ndarray,
    weights: np.ndarray,
    nrep: int,
    n_clusters: int = 1,
    jitter_scale: float = 0.0,
    nu: float | None = None,
    seed: int | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    weights = np.asarray(weights, dtype=float)
    weights = weights / np.sum(weights)
    nrep = int(nrep)
    if nrep <= 0:
        return np.empty((0, states.shape[1]), dtype=float)

    labels = np.zeros(len(states), dtype=int)
    if n_clusters > 1:
        try:
            from sklearn.cluster import KMeans

            kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=seed)
            labels = kmeans.fit_predict(states[:, :3])
        except Exception:
            labels = np.zeros(len(states), dtype=int)

    unique = np.unique(labels)
    if len(unique) == 1:
        idx = _stratified_indices(weights, nrep, rng)
        reps = states[idx]
    else:
        reps = []
        cluster_weights = np.array([weights[labels == lab].sum() for lab in unique], dtype=float)
        cluster_weights = cluster_weights / np.sum(cluster_weights)
        n_per = np.floor(cluster_weights * nrep).astype(int)
        remainder = nrep - n_per.sum()
        if remainder > 0:
            extra = rng.choice(len(unique), size=remainder, replace=True, p=cluster_weights)
            for lab in extra:
                n_per[lab] += 1
        for lab, count in zip(unique, n_per):
            if count <= 0:
                continue
            mask = labels == lab
            w = weights[mask]
            w = w / np.sum(w)
            idx = _stratified_indices(w, count, rng)
            reps.append(states[mask][idx])
        reps = np.vstack(reps) if reps else np.empty((0, states.shape[1]), dtype=float)

    if jitter_scale > 0.0 and len(reps) > 0:
        std = np.std(states, axis=0)
        jitter = rng.standard_normal(reps.shape) * (jitter_scale * std)
        reps = reps + jitter
    return reps


@dataclass(frozen=True)
class Attributable:
    ra_deg: float
    dec_deg: float
    ra_dot_deg_per_day: float
    dec_dot_deg_per_day: float


def build_attributable(observations: Sequence[Observation], epoch: Time) -> Attributable:
    times = np.array([(ob.time - epoch).to(u.day).value for ob in observations], dtype=float)
    ra_deg = np.array([ob.ra_deg for ob in observations], dtype=float)
    dec_deg = np.array([ob.dec_deg for ob in observations], dtype=float)

    ra_rad = np.unwrap(np.deg2rad(ra_deg))
    dec_rad = np.deg2rad(dec_deg)

    A = np.vstack([np.ones_like(times), times]).T
    ra_coef, *_ = np.linalg.lstsq(A, ra_rad, rcond=None)
    dec_coef, *_ = np.linalg.lstsq(A, dec_rad, rcond=None)

    ra0 = float(np.rad2deg(ra_coef[0]))
    dec0 = float(np.rad2deg(dec_coef[0]))
    ra_dot = float(np.rad2deg(ra_coef[1]))
    dec_dot = float(np.rad2deg(dec_coef[1]))
    return Attributable(ra_deg=ra0, dec_deg=dec0, ra_dot_deg_per_day=ra_dot, dec_dot_deg_per_day=dec_dot)


def _attrib_from_s_sdot(s: np.ndarray, sdot: np.ndarray) -> Attributable:
    x, y, z = s
    xd, yd, zd = sdot
    rxy2 = max(x * x + y * y, 1e-12)
    ra = math.atan2(y, x)
    dec = math.asin(np.clip(z, -1.0, 1.0))
    ra_dot = (x * yd - y * xd) / rxy2
    cosdec = max(math.sqrt(rxy2), 1e-12)
    dec_dot = (zd - z * (x * xd + y * yd) / rxy2) / cosdec
    return Attributable(
        ra_deg=float(np.degrees(ra) % 360.0),
        dec_deg=float(np.degrees(dec)),
        ra_dot_deg_per_day=float(np.degrees(ra_dot) * DAY_S),
        dec_dot_deg_per_day=float(np.degrees(dec_dot) * DAY_S),
    )


def _build_attributable_vector_fit_lstsq(
    observations: Sequence[Observation],
    epoch: Time,
) -> Attributable:
    """Fit a linear model to topocentric direction unit vectors in ICRS."""
    times = np.array([(ob.time - epoch).to(u.s).value for ob in observations], dtype=float)
    ra_deg = np.array([ob.ra_deg for ob in observations], dtype=float)
    dec_deg = np.array([ob.dec_deg for ob in observations], dtype=float)

    ra_rad = np.deg2rad(ra_deg)
    dec_rad = np.deg2rad(dec_deg)
    s = np.stack(
        [np.cos(dec_rad) * np.cos(ra_rad), np.cos(dec_rad) * np.sin(ra_rad), np.sin(dec_rad)],
        axis=1,
    )

    A = np.vstack([np.ones_like(times), times]).T
    coef_x, *_ = np.linalg.lstsq(A, s[:, 0], rcond=None)
    coef_y, *_ = np.linalg.lstsq(A, s[:, 1], rcond=None)
    coef_z, *_ = np.linalg.lstsq(A, s[:, 2], rcond=None)

    s0 = np.array([coef_x[0], coef_y[0], coef_z[0]], dtype=float)
    sdot = np.array([coef_x[1], coef_y[1], coef_z[1]], dtype=float)

    s0_norm = np.linalg.norm(s0)
    if s0_norm <= 0:
        return build_attributable(observations, epoch)
    s0 = s0 / s0_norm
    # Remove radial component from sdot for a tangent-plane rate.
    sdot = sdot - np.dot(s0, sdot) * s0
    return _attrib_from_s_sdot(s0, sdot)


def build_attributable_studentt(
    observations: Sequence[Observation],
    epoch: Time | None = None,
    *,
    nu: float = 4.0,
    max_iter: int = 10,
    tol: float = 1e-8,
    site_kappas: dict[str, float] | None = None,
) -> tuple[Attributable, np.ndarray]:
    """Robust attributable fit using Student-t IRLS (deg/day output)."""
    if site_kappas is None:
        site_kappas = {}
    if epoch is None:
        epoch = observations[len(observations) // 2].time

    dec0_rad = math.radians(float(np.mean([o.dec_deg for o in observations])))
    cosd0 = math.cos(dec0_rad)

    n = len(observations)
    y = np.zeros(2 * n, dtype=float)
    sigma = np.zeros(2 * n, dtype=float)
    dt_days = np.zeros(n, dtype=float)
    for i, ob in enumerate(observations):
        y[2 * i] = ob.ra_deg * cosd0 * 3600.0
        y[2 * i + 1] = ob.dec_deg * 3600.0
        kappa = site_kappas.get(ob.site or "UNK", 1.0)
        sigma_arc = max(1e-6, float(ob.sigma_arcsec) * float(kappa))
        sigma[2 * i] = sigma_arc * cosd0
        sigma[2 * i + 1] = sigma_arc
        dt_days[i] = float((ob.time.tdb - epoch.tdb).to_value("day"))

    G = np.zeros((2 * n, 4), dtype=float)
    for i in range(n):
        dt = float(dt_days[i])
        G[2 * i, 0] = cosd0 * 3600.0
        G[2 * i, 2] = cosd0 * 3600.0 * dt
        G[2 * i + 1, 1] = 3600.0
        G[2 * i + 1, 3] = 3600.0 * dt

    W = np.diag(1.0 / (sigma**2 + 1e-12))
    GTWG = G.T @ W @ G
    try:
        a = np.linalg.solve(GTWG, G.T @ W @ y)
    except np.linalg.LinAlgError:
        a = np.linalg.pinv(GTWG) @ (G.T @ W @ y)

    nu = float(nu)
    for _ in range(max_iter):
        r = G @ a - y
        scaled = (r / (sigma + 1e-12)) ** 2
        w = (nu + 1.0) / (nu + scaled)
        W_eff = np.diag(w / (sigma**2 + 1e-12))
        GTWG = G.T @ W_eff @ G
        rhs = G.T @ W_eff @ y
        try:
            a_new = np.linalg.solve(GTWG, rhs)
        except np.linalg.LinAlgError:
            a_new = np.linalg.pinv(GTWG) @ rhs
        if np.linalg.norm(a_new - a) < tol:
            a = a_new
            W = W_eff
            break
        a = a_new
        W = W_eff

    try:
        cov = np.linalg.inv(GTWG)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(GTWG)

    attrib = Attributable(
        ra_deg=float(a[0]),
        dec_deg=float(a[1]),
        ra_dot_deg_per_day=float(a[2]),
        dec_dot_deg_per_day=float(a[3]),
    )
    return attrib, cov


def build_attributable_vector_fit(
    observations: Sequence[Observation],
    epoch: Time,
    *,
    robust: bool = True,
    return_cov: bool = False,
    nu: float = 4.0,
    max_iter: int = 10,
    tol: float = 1e-8,
    site_kappas: dict[str, float] | None = None,
) -> Attributable | tuple[Attributable, np.ndarray]:
    """Fit a linear model to topocentric direction unit vectors in ICRS."""
    if robust:
        attrib, cov = build_attributable_studentt(
            observations,
            epoch=epoch,
            nu=nu,
            max_iter=max_iter,
            tol=tol,
            site_kappas=site_kappas,
        )
        return (attrib, cov) if return_cov else attrib
    attrib = _build_attributable_vector_fit_lstsq(observations, epoch)
    return (attrib, np.zeros((4, 4), dtype=float)) if return_cov else attrib

def s_and_sdot(attrib: Attributable) -> tuple[np.ndarray, np.ndarray]:
    ra = math.radians(attrib.ra_deg)
    dec = math.radians(attrib.dec_deg)
    ra_dot = math.radians(attrib.ra_dot_deg_per_day) / DAY_S
    dec_dot = math.radians(attrib.dec_dot_deg_per_day) / DAY_S

    s = np.array(
        [math.cos(dec) * math.cos(ra), math.cos(dec) * math.sin(ra), math.sin(dec)],
        dtype=float,
    )
    sdot = np.array(
        [
            -math.sin(ra) * math.cos(dec) * ra_dot - math.cos(ra) * math.sin(dec) * dec_dot,
            math.cos(ra) * math.cos(dec) * ra_dot - math.sin(ra) * math.sin(dec) * dec_dot,
            math.cos(dec) * dec_dot,
        ],
        dtype=float,
    )
    return s, sdot


def build_state_from_ranging(
    obs: Observation,
    epoch: Time,
    attrib: Attributable,
    rho_km: float,
    rhodot_km_s: float,
) -> np.ndarray:
    s, sdot = s_and_sdot(attrib)
    return build_state_from_ranging_s_sdot(obs, epoch, s, sdot, rho_km, rhodot_km_s)


def build_state_from_ranging_s_sdot(
    obs: Observation,
    epoch: Time,
    s: np.ndarray,
    sdot: np.ndarray,
    rho_km: float,
    rhodot_km_s: float,
) -> np.ndarray:
    """Build heliocentric state directly from (s, sdot) to preserve velocity precision."""
    earth_pos, earth_vel = get_body_barycentric_posvel("earth", epoch)
    sun_pos, sun_vel = get_body_barycentric_posvel("sun", epoch)
    earth_helio = (earth_pos.xyz - sun_pos.xyz).to(u.km).value.flatten()
    earth_vel_helio = (earth_vel.xyz - sun_vel.xyz).to(u.km / u.s).value.flatten()

    site_pos, site_vel = _site_states(
        [epoch],
        [obs.site],
        observer_positions_km=[obs.observer_pos_km],
        observer_velocities_km_s=None,
        allow_unknown_site=True,
    )
    site_offset = site_pos[0]
    site_vel = site_vel[0]
    return _build_state_from_ranging_cached(
        s,
        sdot,
        earth_helio,
        earth_vel_helio,
        site_offset,
        site_vel,
        rho_km,
        rhodot_km_s,
    )


def _build_state_from_ranging_cached(
    s: np.ndarray,
    sdot: np.ndarray,
    earth_helio: np.ndarray,
    earth_vel_helio: np.ndarray,
    site_offset: np.ndarray,
    site_vel: np.ndarray,
    rho_km: float,
    rhodot_km_s: float,
) -> np.ndarray:
    r_geo = site_offset + rho_km * s
    v_geo = site_vel + rhodot_km_s * s + rho_km * sdot
    r_helio = earth_helio + r_geo
    v_helio = earth_vel_helio + v_geo
    return np.hstack([r_helio, v_helio]).astype(float)


def _state_residuals(
    state: np.ndarray,
    epoch: Time,
    observations: Sequence[Observation],
    perturbers: Sequence[str],
    max_step: float,
    use_kepler: bool,
    obs_cache=None,
) -> np.ndarray:
    pred_ra, pred_dec = _predict_batch(
        state,
        epoch,
        list(observations),
        perturbers,
        max_step,
        use_kepler=use_kepler,
        obs_cache=obs_cache,
    )
    residuals = []
    for idx, (ra, dec, ob) in enumerate(zip(pred_ra, pred_dec, observations)):
        d_ra = ((ob.ra_deg - ra + 180.0) % 360.0) - 180.0
        ra_arcsec = d_ra * math.cos(math.radians(dec)) * 3600.0
        dec_arcsec = (ob.dec_deg - dec) * 3600.0
        residuals.extend([ra_arcsec, dec_arcsec])
    return np.array(residuals, dtype=float)


def build_state_from_ranging_multiobs(
    observations: Sequence[Observation],
    obs_ref: Observation,
    epoch: Time,
    attrib: Attributable,
    rho_km: float,
    rhodot_km_s: float,
    perturbers: Sequence[str],
    max_step: float,
    use_kepler: bool,
    max_iter: int = 2,
) -> np.ndarray:
    rho = float(rho_km)
    rhodot = float(rhodot_km_s)
    obs_list = list(observations)
    if not obs_list:
        return build_state_from_ranging(obs_ref, epoch, attrib, rho, rhodot)
    s, sdot = s_and_sdot(attrib)
    earth_pos, earth_vel = get_body_barycentric_posvel("earth", epoch)
    sun_pos, sun_vel = get_body_barycentric_posvel("sun", epoch)
    earth_helio = (earth_pos.xyz - sun_pos.xyz).to(u.km).value.flatten()
    earth_vel_helio = (earth_vel.xyz - sun_vel.xyz).to(u.km / u.s).value.flatten()
    site_pos, site_vel = _site_states(
        [epoch],
        [obs_ref.site],
        observer_positions_km=[obs_ref.observer_pos_km],
        observer_velocities_km_s=None,
        allow_unknown_site=True,
    )
    site_offset = site_pos[0]
    site_vel = site_vel[0]
    obs_cache = _prepare_obs_cache(obs_list, allow_unknown_site=True)
    for _ in range(max_iter):
        state = _build_state_from_ranging_cached(
            s,
            sdot,
            earth_helio,
            earth_vel_helio,
            site_offset,
            site_vel,
            rho,
            rhodot,
        )
        res = _state_residuals(
            state, epoch, obs_list, perturbers, max_step, use_kepler, obs_cache=obs_cache
        )
        if not np.all(np.isfinite(res)):
            break
        eps_rho = max(1e-6 * abs(rho), 1e3)
        eps_rhodot = max(1e-6 * abs(rhodot), 1e-3)
        st_rho = _build_state_from_ranging_cached(
            s,
            sdot,
            earth_helio,
            earth_vel_helio,
            site_offset,
            site_vel,
            rho + eps_rho,
            rhodot,
        )
        st_rhodot = _build_state_from_ranging_cached(
            s,
            sdot,
            earth_helio,
            earth_vel_helio,
            site_offset,
            site_vel,
            rho,
            rhodot + eps_rhodot,
        )
        res_rho = _state_residuals(
            st_rho, epoch, obs_list, perturbers, max_step, use_kepler, obs_cache=obs_cache
        )
        res_rhodot = _state_residuals(
            st_rhodot, epoch, obs_list, perturbers, max_step, use_kepler, obs_cache=obs_cache
        )
        if not (np.all(np.isfinite(res_rho)) and np.all(np.isfinite(res_rhodot))):
            break
        j_rho = (res_rho - res) / eps_rho
        j_rhodot = (res_rhodot - res) / eps_rhodot
        J = np.vstack([j_rho, j_rhodot]).T
        try:
            JTJ = J.T @ J + np.eye(2) * 1e-6
            delta = -np.linalg.solve(JTJ, J.T @ res)
        except np.linalg.LinAlgError:
            delta = -np.linalg.lstsq(J, res, rcond=None)[0]
        rho += float(delta[0])
        rhodot += float(delta[1])
        if rho <= 1e-6:
            rho = 1e-6
        if np.linalg.norm(delta) < 1e-6:
            break
    return _build_state_from_ranging_cached(
        s,
        sdot,
        earth_helio,
        earth_vel_helio,
        site_offset,
        site_vel,
        rho,
        rhodot,
    )


def _ranging_reference_observation(
    observations: Sequence[Observation],
    epoch: Time,
) -> Observation:
    if not observations:
        raise RuntimeError("No observations provided for ranging reference.")
    sites = [o.site for o in observations if o.site]
    site = None
    if sites:
        counts: dict[str, int] = {}
        for s in sites:
            counts[s] = counts.get(s, 0) + 1
        site = max(counts.items(), key=lambda kv: kv[1])[0]
        if len(counts) > 1:
            print(
                f"[ranging] multiple sites found; using site '{site}' as reference for proposals",
                flush=True,
            )

    obs_ref = Observation(
        time=epoch,
        ra_deg=0.0,
        dec_deg=0.0,
        sigma_arcsec=1.0,
        site=site,
        observer_pos_km=None,
    )
    if site is not None:
        site_kind = get_site_kind(site)
        if site_kind in {"spacecraft", "space", "satellite"} or get_site_ephemeris(site):
            return obs_ref
    obs_pos = [o.observer_pos_km for o in observations if o.observer_pos_km is not None]
    if obs_pos:
        obs_ref.observer_pos_km = np.mean(np.array(obs_pos, dtype=float), axis=0)
    return obs_ref


def studentt_loglike(residuals: np.ndarray, sigma_vec: np.ndarray, nu: float) -> float:
    t = (residuals / sigma_vec) ** 2
    return float(-0.5 * np.sum((nu + 1.0) * np.log1p(t / nu)))


def _phase_func_hg(alpha_rad: float, g: float) -> float:
    # Bowell HG phase function (approx).
    tan_half = math.tan(0.5 * alpha_rad)
    phi1 = math.exp(-3.33 * tan_half**0.63)
    phi2 = math.exp(-1.87 * tan_half**1.22)
    return (1.0 - g) * phi1 + g * phi2


def photometric_loglike(
    m_obs: float,
    r_au: float,
    delta_au: float,
    phase_rad: float,
    h0: float,
    h_sigma: float,
    g: float,
    sigma_mag: float,
) -> float:
    # Apparent magnitude model: m = H + 5 log10(r*delta) - 2.5 log10(phi)
    phi = max(_phase_func_hg(phase_rad, g), 1e-12)
    A = 5.0 * math.log10(max(r_au * delta_au, 1e-12)) - 2.5 * math.log10(phi)
    h_hat = m_obs - A
    sigma = max(1e-6, math.hypot(h_sigma, sigma_mag))
    return float(-0.5 * ((h_hat - h0) / sigma) ** 2 - math.log(sigma))


def _admissible_ok(
    state: np.ndarray,
    *,
    q_min_au: float | None,
    q_max_au: float | None,
    bound_only: bool,
    mu_km3_s2: float = GM_SUN,
) -> bool:
    r = state[:3].astype(float)
    v = state[3:].astype(float)
    r_norm = float(np.linalg.norm(r))
    if not np.isfinite(r_norm) or r_norm <= 0:
        return False
    v2 = float(np.dot(v, v))
    eps = 0.5 * v2 - mu_km3_s2 / r_norm
    if bound_only and not (eps < 0.0):
        return False
    if q_min_au is None and q_max_au is None:
        return True
    h = np.cross(r, v)
    h_norm = float(np.linalg.norm(h))
    if not np.isfinite(h_norm) or h_norm <= 0:
        return False
    e_vec = (np.cross(v, h) / mu_km3_s2) - (r / r_norm)
    e = float(np.linalg.norm(e_vec))
    if not np.isfinite(e):
        return False
    if eps < 0.0:
        a = -mu_km3_s2 / (2.0 * eps)
    else:
        a = None
    if a is None or not np.isfinite(a):
        return not bound_only
    q = a * (1.0 - e)
    q_au = q / AU_KM
    if q_min_au is not None and q_au < q_min_au:
        return False
    if q_max_au is not None and q_au > q_max_au:
        return False
    return True


def score_candidate(
    state: np.ndarray,
    epoch: Time,
    observations: Sequence[Observation],
    perturbers: Sequence[str],
    max_step: float,
    nu: float,
    site_kappas: dict[str, float],
    use_kepler: bool,
    photometry: dict[str, float] | None = None,
) -> tuple[float, np.ndarray]:
    if photometry:
        from .propagate import predict_radec_with_geometry

        pred_ra, pred_dec, r_au, delta_au, phase_rad = predict_radec_with_geometry(
            state,
            epoch,
            list(observations),
            perturbers,
            max_step,
            use_kepler=use_kepler,
        )
    else:
        pred_ra, pred_dec = _predict_batch(
            state, epoch, list(observations), perturbers, max_step, use_kepler=use_kepler
        )
    residuals = []
    sigma_vec = []
    photo_ll = 0.0
    have_photo = False
    for idx, (ra, dec, ob) in enumerate(zip(pred_ra, pred_dec, observations)):
        d_ra = ((ob.ra_deg - ra + 180.0) % 360.0) - 180.0
        ra_arcsec = d_ra * math.cos(math.radians(dec)) * 3600.0
        dec_arcsec = (ob.dec_deg - dec) * 3600.0
        residuals.extend([ra_arcsec, dec_arcsec])
        kappa = site_kappas.get(ob.site or "UNK", 1.0)
        sigma = max(1e-6, float(ob.sigma_arcsec) * float(kappa))
        sigma_vec.extend([sigma, sigma])
        if photometry and ob.mag is not None:
            sigma_mag = ob.sigma_mag if ob.sigma_mag is not None else photometry["sigma_mag_default"]
            photo_ll += photometric_loglike(
                float(ob.mag),
                float(r_au[idx]),
                float(delta_au[idx]),
                float(phase_rad[idx]),
                float(photometry["h0"]),
                float(photometry["h_sigma"]),
                float(photometry["g"]),
                float(sigma_mag),
            )
            have_photo = True
    res = np.array(residuals, dtype=float)
    sig = np.array(sigma_vec, dtype=float)
    ll = studentt_loglike(res, sig, nu)
    if have_photo:
        ll += photo_ll
    return ll, res


_RANGE_CTX: dict[str, object] = {}


def _init_worker(
    obs_ref: Observation,
    epoch: Time,
    attrib: Attributable,
    observations: Sequence[Observation],
    perturbers: Sequence[str],
    max_step: float,
    nu: float,
    site_kappas: dict[str, float],
    use_kepler: bool,
    multiobs: bool,
    multiobs_max_iter: int,
    obs_eval: Sequence[Observation] | None,
    photometry: dict[str, float] | None,
    admissible_bound: bool,
    admissible_q_min_au: float | None,
    admissible_q_max_au: float | None,
) -> None:
    global _RANGE_CTX
    _RANGE_CTX = {
        "obs_ref": obs_ref,
        "epoch": epoch,
        "attrib": attrib,
        "observations": observations,
        "perturbers": perturbers,
        "max_step": max_step,
        "nu": nu,
        "site_kappas": site_kappas,
        "use_kepler": use_kepler,
        "multiobs": multiobs,
        "multiobs_max_iter": multiobs_max_iter,
        "obs_eval": obs_eval,
        "photometry": photometry,
        "admissible_bound": admissible_bound,
        "admissible_q_min_au": admissible_q_min_au,
        "admissible_q_max_au": admissible_q_max_au,
    }


def _score_chunk(
    chunk: Sequence[tuple[float, float, float]]
) -> list[tuple[float, float, float, np.ndarray, float]]:
    ctx = _RANGE_CTX
    out: list[tuple[float, float, float, np.ndarray, float]] = []
    for rho_km, rhodot_km_s, log_q in chunk:
        if ctx.get("multiobs"):
            state = build_state_from_ranging_multiobs(
                ctx["obs_eval"] or ctx["observations"],
                ctx["obs_ref"],
                ctx["epoch"],
                ctx["attrib"],
                rho_km,
                rhodot_km_s,
                ctx["perturbers"],
                ctx["max_step"],
                ctx["use_kepler"],
                max_iter=int(ctx.get("multiobs_max_iter", 2)),
            )
        else:
            state = build_state_from_ranging(
                ctx["obs_ref"],
                ctx["epoch"],
                ctx["attrib"],
                rho_km,
                rhodot_km_s,
            )
        try:
            if not _admissible_ok(
                state,
                q_min_au=ctx.get("admissible_q_min_au"),
                q_max_au=ctx.get("admissible_q_max_au"),
                bound_only=bool(ctx.get("admissible_bound")),
            ):
                continue
            ll, _ = score_candidate(
                state,
                ctx["epoch"],
                ctx["observations"],
                ctx["perturbers"],
                ctx["max_step"],
                ctx["nu"],
                ctx["site_kappas"],
                ctx["use_kepler"],
                ctx.get("photometry"),
            )
        except Exception:
            continue
        out.append((rho_km, rhodot_km_s, ll, state, log_q))
    return out


def _init_worker_state(
    epoch: Time,
    observations: Sequence[Observation],
    perturbers: Sequence[str],
    max_step: float,
    nu: float,
    site_kappas: dict[str, float],
    use_kepler: bool,
    photometry: dict[str, float] | None,
) -> None:
    global _RANGE_CTX
    _RANGE_CTX = {
        "epoch": epoch,
        "observations": observations,
        "perturbers": perturbers,
        "max_step": max_step,
        "nu": nu,
        "site_kappas": site_kappas,
        "use_kepler": use_kepler,
        "photometry": photometry,
    }


def _score_state_chunk(states: np.ndarray) -> list[tuple[float, np.ndarray]]:
    ctx = _RANGE_CTX
    out: list[tuple[float, np.ndarray]] = []
    for state in states:
        try:
            ll, _ = score_candidate(
                state,
                ctx["epoch"],
                ctx["observations"],
                ctx["perturbers"],
                ctx["max_step"],
                ctx["nu"],
                ctx["site_kappas"],
                ctx["use_kepler"],
                ctx.get("photometry"),
            )
        except Exception:
            ll = -np.inf
        out.append((ll, state))
    return out


def _normal_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _truncnorm_sample(
    rng: np.random.Generator, *, a: float, b: float, sigma: float, max_tries: int = 64
) -> float | None:
    if sigma <= 0.0 or a >= b:
        return None
    for _ in range(max_tries):
        val = rng.normal(loc=0.0, scale=sigma)
        if a <= val <= b:
            return float(val)
    return None


def _truncnorm_pdf(x: float, *, a: float, b: float, sigma: float) -> float:
    if sigma <= 0.0 or a >= b or x < a or x > b:
        return 0.0
    z = (_normal_cdf(b / sigma) - _normal_cdf(a / sigma))
    if z <= 0.0:
        return 0.0
    return _normal_pdf(x / sigma) / (sigma * z)


def _maxwell_pdf(speed: float, sigma: float) -> float:
    if speed <= 0.0 or sigma <= 0.0:
        return 0.0
    coef = math.sqrt(2.0 / math.pi) / (sigma**3)
    return coef * (speed**2) * math.exp(-0.5 * (speed / sigma) ** 2)


def _sample_maxwell_speed(
    rng: np.random.Generator, *, sigma: float, vmax: float, max_tries: int = 64
) -> float | None:
    if sigma <= 0.0 or vmax <= 0.0:
        return None
    for _ in range(max_tries):
        vec = rng.normal(loc=0.0, scale=sigma, size=3)
        speed = float(np.linalg.norm(vec))
        if speed <= vmax:
            return speed
    return None


def _sample_conditioned_ranging_proposals(
    *,
    rng: np.random.Generator,
    n_proposals: int,
    rho_min_au: float,
    rho_max_au: float,
    rhodot_max_kms: float,
    s: np.ndarray,
    sdot: np.ndarray,
    earth_helio: np.ndarray,
    earth_vel_helio: np.ndarray,
    site_offset: np.ndarray,
    site_vel: np.ndarray,
    v_inf_max_kms: float | None,
    sigma_rad_scale: float,
    iso_fraction: float,
    sigma_iso_kms: float,
    oversample_factor: float = 3.0,
    max_batches: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    log_rho_min = math.log(max(1e-12, rho_min_au))
    log_rho_max = math.log(max(rho_min_au, rho_max_au))
    v_inf_cap = float(rhodot_max_kms if v_inf_max_kms is None else v_inf_max_kms)
    v_inf_cap = max(v_inf_cap, 1e-6)
    sigma_rad_scale = max(1e-6, float(sigma_rad_scale))
    iso_fraction = float(np.clip(iso_fraction, 0.0, 1.0))

    rhos: list[float] = []
    rhodots: list[float] = []
    log_qs: list[float] = []

    for _ in range(max_batches):
        remaining = n_proposals - len(rhos)
        if remaining <= 0:
            break
        batch_size = max(128, int(math.ceil(remaining * max(1.0, oversample_factor))))
        rhos_batch = np.exp(rng.uniform(log_rho_min, log_rho_max, size=batch_size)) * AU_KM
        for rho_km in rhos_batch:
            v0 = earth_vel_helio + site_vel + rho_km * sdot
            s_dot_v0 = float(np.dot(s, v0))
            v_t = v0 - s_dot_v0 * s
            v_t2 = float(np.dot(v_t, v_t))

            r_helio = earth_helio + site_offset + rho_km * s
            r_norm = float(np.linalg.norm(r_helio))
            if r_norm <= 0.0:
                continue
            v_esc = math.sqrt(2.0 * GM_SUN / r_norm)
            v_max = math.sqrt(v_inf_cap * v_inf_cap + v_esc * v_esc)
            if v_t2 >= v_max * v_max:
                continue
            v_rad_max = math.sqrt(max(0.0, v_max * v_max - v_t2))
            if v_rad_max <= 0.0:
                continue

            if iso_fraction > 0.0 and rng.random() < iso_fraction:
                v_inf = _sample_maxwell_speed(rng, sigma=sigma_iso_kms, vmax=v_inf_cap)
                if v_inf is None:
                    continue
                v = math.sqrt(v_inf * v_inf + v_esc * v_esc)
                if v * v <= v_t2:
                    continue
                v_rad_mag = math.sqrt(max(0.0, v * v - v_t2))
                v_rad = v_rad_mag if rng.random() < 0.5 else -v_rad_mag
                v_inf_from_vrad = math.sqrt(max(0.0, v_t2 + v_rad * v_rad - v_esc * v_esc))
                if v_inf_from_vrad <= 0.0:
                    continue
                p_vinf = _maxwell_pdf(v_inf_from_vrad, sigma_iso_kms)
                q_vrad = 0.5 * p_vinf * (abs(v_rad) / v_inf_from_vrad)
                q = iso_fraction * q_vrad
            else:
                v_circ = math.sqrt(max(GM_SUN / r_norm, 1e-12))
                sigma_rad = max(1e-6, sigma_rad_scale * v_circ)
                v_rad = _truncnorm_sample(rng, a=-v_rad_max, b=v_rad_max, sigma=sigma_rad)
                if v_rad is None:
                    continue
                q_bound = _truncnorm_pdf(v_rad, a=-v_rad_max, b=v_rad_max, sigma=sigma_rad)
                q = (1.0 - iso_fraction) * q_bound

            if not np.isfinite(q) or q <= 0.0:
                continue
            rhodot = float(v_rad - s_dot_v0)
            if abs(rhodot) > rhodot_max_kms:
                continue
            rhos.append(float(rho_km))
            rhodots.append(rhodot)
            log_qs.append(float(math.log(q)))
            if len(rhos) >= n_proposals:
                break
        if len(rhos) >= n_proposals:
            break

    if len(rhos) < n_proposals:
        n_missing = n_proposals - len(rhos)
        rhos_fallback = np.exp(rng.uniform(log_rho_min, log_rho_max, size=n_missing)) * AU_KM
        rhodots_fallback = rng.uniform(-rhodot_max_kms, rhodot_max_kms, size=n_missing)
        q_uniform = math.log(0.5 / max(rhodot_max_kms, 1e-12))
        rhos.extend(rhos_fallback.tolist())
        rhodots.extend(rhodots_fallback.tolist())
        log_qs.extend([q_uniform] * n_missing)

    return (
        np.asarray(rhos[:n_proposals], dtype=float),
        np.asarray(rhodots[:n_proposals], dtype=float),
        np.asarray(log_qs[:n_proposals], dtype=float),
    )


def sample_ranged_replicas(
    observations: Sequence[Observation],
    epoch: Time,
    n_replicas: int,
    n_proposals: int,
    rho_min_au: float,
    rho_max_au: float,
    rhodot_max_kms: float,
    perturbers: Sequence[str],
    max_step: float,
    nu: float,
    site_kappas: dict[str, float],
    seed: int,
    log_every: int = 0,
    scoring_mode: str = "kepler",
    n_workers: int = 1,
    chunk_size: int | None = None,
    top_k_nbody: int = 2000,
    rho_prior_power: float = 2.0,
    rho_prior_mode: str | None = "log",
    multiobs: bool = False,
    multiobs_indices: Sequence[int] | None = None,
    multiobs_max_iter: int = 2,
    attrib_mode: str = "vector",
    photometry: dict[str, float] | None = None,
    admissible_bound: bool = False,
    admissible_q_min_au: float | None = None,
    admissible_q_max_au: float | None = None,
    rhodot_sampler: str = "conditioned",
    rhodot_sigma_scale: float = 0.25,
    rhodot_iso_fraction: float = 1e-3,
    rhodot_iso_sigma_kms: float = 40.0,
    rhodot_vinf_max_kms: float | None = None,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    obs_ref = _ranging_reference_observation(observations, epoch)
    if multiobs_indices:
        obs_eval = [observations[i] for i in multiobs_indices]
    else:
        obs_eval = list(observations)
    if attrib_mode == "linear":
        attrib = build_attributable(observations, epoch)
    else:
        attrib = build_attributable_vector_fit(observations, epoch)
    log_rho_min = math.log(max(1e-12, rho_min_au))
    log_rho_max = math.log(max(rho_min_au, rho_max_au))

    print(
        "[ranging] init n_proposals={} rho=[{:.2e},{:.2e}] AU rhodot=[-{:.1f},{:.1f}] km/s".format(
            n_proposals, rho_min_au, rho_max_au, rhodot_max_kms, rhodot_max_kms
        ),
        flush=True,
    )
    sampler = rhodot_sampler.lower().strip()
    if sampler == "conditioned":
        s, sdot = s_and_sdot(attrib)
        earth_pos, earth_vel = get_body_barycentric_posvel("earth", epoch)
        sun_pos, sun_vel = get_body_barycentric_posvel("sun", epoch)
        earth_helio = (earth_pos.xyz - sun_pos.xyz).to(u.km).value.flatten()
        earth_vel_helio = (earth_vel.xyz - sun_vel.xyz).to(u.km / u.s).value.flatten()
        site_pos, site_vel = _site_states(
            [epoch],
            [obs_ref.site],
            observer_positions_km=[obs_ref.observer_pos_km],
            observer_velocities_km_s=None,
            allow_unknown_site=True,
        )
        rhos, rhodots, log_q = _sample_conditioned_ranging_proposals(
            rng=rng,
            n_proposals=n_proposals,
            rho_min_au=rho_min_au,
            rho_max_au=rho_max_au,
            rhodot_max_kms=rhodot_max_kms,
            s=s,
            sdot=sdot,
            earth_helio=earth_helio,
            earth_vel_helio=earth_vel_helio,
            site_offset=site_pos[0],
            site_vel=site_vel[0],
            v_inf_max_kms=rhodot_vinf_max_kms,
            sigma_rad_scale=rhodot_sigma_scale,
            iso_fraction=rhodot_iso_fraction,
            sigma_iso_kms=rhodot_iso_sigma_kms,
        )
    elif sampler == "uniform":
        rhos = np.exp(rng.uniform(log_rho_min, log_rho_max, size=n_proposals)) * AU_KM
        rhodots = rng.uniform(-rhodot_max_kms, rhodot_max_kms, size=n_proposals)
        log_q = np.full(n_proposals, math.log(0.5 / max(rhodot_max_kms, 1e-12)))
    else:
        raise ValueError(f"Unknown rhodot_sampler={rhodot_sampler!r}")
    print(f"[ranging] {sampler} proposals sampled; scoring...", flush=True)

    proposals = list(zip(rhos, rhodots, log_q))
    if n_workers < 1:
        n_workers = 1
    if chunk_size is None:
        chunk_size = max(128, n_proposals // max(1, n_workers))
    chunks = [proposals[i : i + chunk_size] for i in range(0, len(proposals), chunk_size)]

    use_kepler = scoring_mode == "kepler"
    t0 = time.perf_counter()
    if n_workers == 1:
        results = []
        n_fail = 0
        best_ll = -np.inf
        for i, (rho_km, rhodot_km_s, log_qi) in enumerate(proposals):
            if multiobs:
                state = build_state_from_ranging_multiobs(
                    obs_eval,
                    obs_ref,
                    epoch,
                    attrib,
                    rho_km,
                    rhodot_km_s,
                    perturbers,
                    max_step,
                    use_kepler,
                    max_iter=multiobs_max_iter,
                )
            else:
                state = build_state_from_ranging(obs_ref, epoch, attrib, rho_km, rhodot_km_s)
            try:
                if not _admissible_ok(
                    state,
                    q_min_au=admissible_q_min_au,
                    q_max_au=admissible_q_max_au,
                    bound_only=admissible_bound,
                ):
                    ll = -np.inf
                    n_fail += 1
                    results.append((rho_km, rhodot_km_s, ll, state, log_qi))
                    continue
                ll, _ = score_candidate(
                    state,
                    epoch,
                    observations,
                    perturbers,
                    max_step,
                    nu,
                    site_kappas,
                    use_kepler,
                    photometry,
                )
            except Exception:
                ll = -np.inf
                n_fail += 1
            if ll > best_ll:
                best_ll = ll
            results.append((rho_km, rhodot_km_s, ll, state, log_qi))
            if log_every and ((i + 1) % log_every == 0 or (i + 1) == n_proposals):
                elapsed = time.perf_counter() - t0
                rate = (i + 1) / max(elapsed, 1e-6)
                print(
                    "[ranging] scored {}/{} proposals; best_ll={:.2f}; failures={}; rate={:.1f}/s".format(
                        i + 1, n_proposals, best_ll, n_fail, rate
                    ),
                    flush=True,
                )
        t1 = time.perf_counter()
        results = [r for r in results if np.isfinite(r[2])]
    else:
        with Pool(
            processes=n_workers,
            initializer=_init_worker,
            initargs=(
                obs_ref,
                epoch,
                attrib,
                list(observations),
                tuple(perturbers),
                max_step,
                nu,
                site_kappas,
                use_kepler,
                multiobs,
                multiobs_max_iter,
                obs_eval,
                photometry,
                admissible_bound,
                admissible_q_min_au,
                admissible_q_max_au,
            ),
        ) as pool:
            chunk_results = pool.map(_score_chunk, chunks)
        t1 = time.perf_counter()
        results = [r for sub in chunk_results for r in sub]

    if len(results) == 0:
        raise RuntimeError("All ranged proposals failed scoring.")
    rhos_out = np.array([r[0] for r in results], dtype=float)
    rhodots_out = np.array([r[1] for r in results], dtype=float)
    loglikes = np.array([r[2] for r in results], dtype=float)
    states = np.array([r[3] for r in results], dtype=float)
    log_q = np.array([r[4] for r in results], dtype=float)

    log_prior = _rho_log_prior(rhos_out, rho_prior_mode, rho_prior_power)
    log_w = loglikes + log_prior - log_q
    max_lw = np.max(log_w)
    weights = np.exp(log_w - max_lw)
    weights /= np.sum(weights)

    prefilter_debug: dict[str, Any] | None = None
    if scoring_mode == "kepler" and top_k_nbody > 0:
        top_k = min(top_k_nbody, len(states))
        selected_idx, prefilter_debug = kepler_prefilter_and_select(
            raw_rhos_km=rhos_out,
            kepler_ll=loglikes,
            states=states,
            top_k=top_k,
            n_bins=None,
            au_km=AU_KM,
            emit_debug=False,
            path_to_debug_npz=None,
        )
        top_states = states[selected_idx]
        top_rhos = rhos_out[selected_idx]
        top_rhodots = rhodots_out[selected_idx]
        top_log_q = log_q[selected_idx]
        if n_workers > 1:
            state_chunks = [
                top_states[i : i + chunk_size] for i in range(0, len(top_states), chunk_size)
            ]
            with Pool(
                processes=n_workers,
                initializer=_init_worker_state,
                initargs=(
                    epoch,
                    list(observations),
                    tuple(perturbers),
                    max_step,
                    nu,
                    site_kappas,
                    False,
                    photometry,
                ),
            ) as pool:
                chunk_results = pool.map(_score_state_chunk, state_chunks)
            scored = [r for sub in chunk_results for r in sub]
        else:
            scored = []
            for st in top_states:
                ll, _ = score_candidate(
                    st,
                    epoch,
                    observations,
                    perturbers,
                    max_step,
                    nu,
                    site_kappas,
                    False,
                    photometry,
                )
                scored.append((ll, st))
        lls = np.array([r[0] for r in scored], dtype=float)
        log_prior_top = _rho_log_prior(top_rhos, rho_prior_mode, rho_prior_power)
        log_w_top = lls + log_prior_top - top_log_q
        max_lw_top = np.max(log_w_top)
        weights = np.exp(log_w_top - max_lw_top)
        weights /= np.sum(weights)
        states = top_states
        rhos_out = top_rhos
        rhodots_out = top_rhodots
        log_q = top_log_q

    return {
        "states": states,
        "weights": weights,
        "rhos": rhos_out,
        "rhodots": rhodots_out,
        "log_q_rhodot": log_q,
        "attrib": attrib,
        "diagnostics": prefilter_debug or {},
    }


def kepler_prefilter_and_select(
    raw_rhos_km: np.ndarray,
    kepler_ll: np.ndarray,
    states: np.ndarray,
    top_k: int,
    n_bins: int | None = None,
    au_km: float = AU_KM,
    emit_debug: bool = False,
    path_to_debug_npz: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Robust Kepler prefilter and stratified selection."""
    M = len(raw_rhos_km)
    if M == 0:
        return np.empty(0, dtype=int), {}

    kepler_ll_sane = np.asarray(kepler_ll, dtype=float).copy()
    is_bad = ~np.isfinite(kepler_ll_sane)
    kepler_invalid_count = int(np.sum(is_bad))
    if kepler_invalid_count > 0:
        finite_mask = np.isfinite(kepler_ll_sane)
        if np.any(finite_mask):
            finite_min = float(np.nanmin(kepler_ll_sane[finite_mask]))
            kepler_ll_sane[is_bad] = finite_min - 1e3
        else:
            kepler_ll_sane[:] = -1e300

    if n_bins is None:
        n_bins = min(20, max(4, int(M // 50)))
    n_bins = max(2, int(n_bins))

    rho_au = (raw_rhos_km / au_km).astype(float)
    bins = np.linspace(float(rho_au.min()), float(rho_au.max()), n_bins + 1)
    bin_idx = np.digitize(rho_au, bins) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    top_k = min(int(top_k), M)
    k_per_bin = max(1, top_k // n_bins)

    selected_indices: list[int] = []
    for b in range(n_bins):
        idxs = np.where(bin_idx == b)[0]
        if idxs.size == 0:
            continue
        ordering = np.argsort(-kepler_ll_sane[idxs])
        pick = idxs[ordering[:k_per_bin]]
        selected_indices.extend(pick.tolist())

    selected_indices = list(dict.fromkeys(selected_indices))
    if len(selected_indices) < top_k:
        needed = top_k - len(selected_indices)
        remaining = np.setdiff1d(np.arange(M), np.array(selected_indices), assume_unique=True)
        global_order = np.argsort(-kepler_ll_sane[remaining])
        extra = remaining[global_order[:needed]]
        selected_indices.extend(extra.tolist())

    selected_indices = np.unique(np.array(selected_indices, dtype=int))[:top_k]

    debug = {
        "raw_rhos": raw_rhos_km,
        "kepler_ll_raw": kepler_ll_sane,
        "prefilter_indices": selected_indices,
        "prefilter_rhos": raw_rhos_km[selected_indices],
        "prefilter_kepler_ll": kepler_ll_sane[selected_indices],
        "n_bins": n_bins,
        "k_per_bin": k_per_bin,
        "kepler_invalid_count": kepler_invalid_count,
    }

    if emit_debug and path_to_debug_npz:
        try:
            np.savez(path_to_debug_npz, **debug, allow_pickle=True)
        except Exception:
            pass

    return np.array(selected_indices, dtype=int), debug
