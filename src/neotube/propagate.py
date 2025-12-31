from __future__ import annotations

from dataclasses import dataclass
from math import ceil
import math
from typing import Iterable, Sequence, Optional
from functools import lru_cache

import numpy as np
from astropy import units as u
from astropy.coordinates import (
    BarycentricTrueEcliptic,
    CartesianDifferential,
    CartesianRepresentation,
    FK5,
    GCRS,
    ICRS,
    SkyCoord,
    SphericalRepresentation,
    TETE,
    get_body_barycentric_posvel,
    solar_system_ephemeris,
)
import astropy.constants as const
from astropy.time import Time, TimeDelta
from concurrent.futures import ProcessPoolExecutor
from scipy.integrate import solve_ivp
import os
from astroquery.jplhorizons import Horizons

from .sites import get_site_location, get_site_ephemeris

try:
    import numba as nb

    _HAS_NUMBA = True
except Exception:  # pragma: no cover - optional dependency
    nb = None
    _HAS_NUMBA = False

try:
    from astropy.numba import solar_system as nb_solar_system
    from astropy.numba import time as nb_time

    _HAS_ASTROPY_NUMBA = True
except Exception:  # pragma: no cover - optional dependency
    nb_solar_system = None
    nb_time = None
    _HAS_ASTROPY_NUMBA = False

if _HAS_NUMBA:
    _NUMBA_THREADS = int(os.environ.get("NEOTUBE_NUMBA_THREADS", "0") or "0")
    if _NUMBA_THREADS <= 0:
        _NUMBA_THREADS = max(1, min(32, os.cpu_count() or 1))
    try:
        nb.set_num_threads(_NUMBA_THREADS)
    except Exception:
        pass

_USE_FAST_EPHEM = os.environ.get("NEOTUBE_FAST_EPHEM", "1").lower() not in {
    "0",
    "false",
    "no",
}
_USE_NUMBA_ACCEL = _HAS_NUMBA and os.environ.get("NEOTUBE_NUMBA_ACCEL", "1").lower() not in {
    "0",
    "false",
    "no",
}

def _trace_astropy(message: str) -> None:
    if os.environ.get("NEOTUBE_TRACE_ASTROPY", "0") == "1":
        print(f"[astropy-trace] {message}", flush=True)

if _HAS_NUMBA:
    _NUMBA_NJIT = nb.njit(cache=True, nogil=True, fastmath=True)
    _NUMBA_NJIT_PAR = nb.njit(cache=True, nogil=True, fastmath=True, parallel=True)

    @_NUMBA_NJIT_PAR
    def _light_time_linear_numba(obj_bary, obj_bary_vel, obs_bary, iters, c_km_s):
        n = obj_bary.shape[0]
        out = obj_bary.copy()
        for _ in range(iters):
            for i in nb.prange(n):
                dx = out[i, 0] - obs_bary[i, 0]
                dy = out[i, 1] - obs_bary[i, 1]
                dz = out[i, 2] - obs_bary[i, 2]
                rho = math.sqrt(dx * dx + dy * dy + dz * dz)
                dt = rho / c_km_s
                out[i, 0] = obj_bary[i, 0] - obj_bary_vel[i, 0] * dt
                out[i, 1] = obj_bary[i, 1] - obj_bary_vel[i, 1] * dt
                out[i, 2] = obj_bary[i, 2] - obj_bary_vel[i, 2] * dt
        return out

    @_NUMBA_NJIT_PAR
    def _radec_from_topovec_numba(topovec):
        n = topovec.shape[0]
        ra_out = np.empty(n, dtype=np.float64)
        dec_out = np.empty(n, dtype=np.float64)
        for i in nb.prange(n):
            x = topovec[i, 0]
            y = topovec[i, 1]
            z = topovec[i, 2]
            xy = math.hypot(x, y)
            ra = math.degrees(math.atan2(y, x))
            if ra < 0.0:
                ra += 360.0
            ra_out[i] = ra
            dec_out[i] = math.degrees(math.atan2(z, xy))
        return ra_out, dec_out

__all__ = [
    "propagate_state",
    "propagate_state_kepler",
    "propagate_state_sun",
    "PropagationConfig",
    "default_propagation_ladder",
    "predict_radec",
    "predict_radec_batch",
    "predict_radec_from_epoch",
    "predict_radec_with_contract",
    "predict_radec_with_geometry",
    "ReplicaCloud",
    "propagate_replicas",
]

GM_SUN = 1.32712440018e11  # km^3 / s^2
AU_KM = 149597870.7
DAY_S = 86400.0

PLANET_GMS = {
    "mercury": 2.203233e4,
    "venus": 3.248585e5,
    "earth": 3.986004418e5,
    "mars": 4.282837e4,
    "jupiter": 1.26686534e8,
    "saturn": 3.7931187e7,
    "uranus": 5.794e6,
    "neptune": 6.835e6,
    "pluto": 8.71e3,
    "moon": 4.9048695e3,
}


@dataclass(frozen=True)
class ObsCache:
    times_obs: Time
    times_tdb: Time
    site_pos_km: np.ndarray
    site_vel_km_s: np.ndarray
    earth_bary_km: np.ndarray
    earth_bary_vel_km_s: np.ndarray


@dataclass(frozen=True)
class PropagationConfig:
    model: str = "kepler"
    perturbers: Sequence[str] = ("earth", "mars", "jupiter")
    max_step: float = 3600.0
    light_time_iters: int = 2
    full_physics: bool = False
    include_refraction: bool = False
    adaptive_step: bool = True
    step_scale: float = 0.05


def _prepare_obs_cache(
    obs: Sequence["Observation"], *, allow_unknown_site: bool = True
) -> ObsCache:
    from .models import Observation  # local import to avoid cycles

    if not obs:
        raise ValueError("obs must be non-empty")
    if not isinstance(obs[0], Observation):
        raise ValueError("obs must be a sequence of Observation objects")
    time_obs = Time([o.time for o in obs])
    time_tdb = time_obs.tdb
    site_codes = [o.site for o in obs]
    observer_positions = [o.observer_pos_km for o in obs]
    site_pos_km, site_vel_km_s = _site_states(
        time_obs,
        site_codes,
        observer_positions_km=observer_positions,
        observer_velocities_km_s=None,
        allow_unknown_site=allow_unknown_site,
    )
    earth_bary_km, earth_bary_vel_km_s = _body_posvel_km("earth", time_tdb)
    if earth_bary_km.shape[0] != len(obs):
        earth_bary_km = earth_bary_km[: len(obs)]
        earth_bary_vel_km_s = earth_bary_vel_km_s[: len(obs)]
    return ObsCache(
        times_obs=time_obs,
        times_tdb=time_tdb,
        site_pos_km=site_pos_km,
        site_vel_km_s=site_vel_km_s,
        earth_bary_km=earth_bary_km,
        earth_bary_vel_km_s=earth_bary_vel_km_s,
    )


def shapiro_delay_sun(
    obj_pos_bary_km: np.ndarray,
    obs_pos_bary_km: np.ndarray,
    sun_pos_bary_km: np.ndarray,
) -> float:
    """Compute the Shapiro delay for the Sun (seconds) for a single emitter/observer pair."""
    c_km_s = 299792.458
    r_e = float(np.linalg.norm(obj_pos_bary_km - sun_pos_bary_km))
    r_o = float(np.linalg.norm(obs_pos_bary_km - sun_pos_bary_km))
    R = float(np.linalg.norm(obj_pos_bary_km - obs_pos_bary_km))
    denom = max(r_e + r_o - R, 1e-12)
    arg = (r_e + r_o + R) / denom
    return float(2.0 * GM_SUN / (c_km_s**3) * np.log(arg))


def aberrate_direction_first_order(topovec: np.ndarray, obs_vel_km_s: np.ndarray) -> np.ndarray:
    """Apply first-order special-relativistic aberration to a topocentric direction."""
    c_km_s = 299792.458
    n = topovec / (np.linalg.norm(topovec) + 1e-30)
    beta = obs_vel_km_s / c_km_s
    nb_dot = float(np.dot(n, beta))
    s = n + beta - nb_dot * n
    return s / (np.linalg.norm(s) + 1e-30)


def bennett_refraction(zenith_rad: float) -> float:
    """Bennett refraction approximation (radians)."""
    z_deg = np.degrees(zenith_rad)
    if z_deg >= 89.999:
        return 0.0
    r_arcmin = 1.02 / np.tan(np.radians(z_deg + 10.3 / (z_deg + 5.11)))
    return float(np.radians(r_arcmin / 60.0))


def _apply_bennett_refraction(topounit: np.ndarray, site_pos_km: np.ndarray) -> np.ndarray:
    site_norm = float(np.linalg.norm(site_pos_km))
    if site_norm <= 0.0:
        return topounit
    if site_norm < 5000.0 or site_norm > 7000.0:
        return topounit
    z_hat = site_pos_km / site_norm
    cos_z = float(np.clip(np.dot(topounit, z_hat), -1.0, 1.0))
    z = float(np.arccos(cos_z))
    if not np.isfinite(z) or z <= 0.0:
        return topounit
    delta = bennett_refraction(z)
    if delta <= 0.0 or delta >= z:
        return topounit
    perp = topounit - cos_z * z_hat
    perp_norm = float(np.linalg.norm(perp))
    if perp_norm <= 0.0:
        return topounit
    e_perp = perp / perp_norm
    z_prime = z - delta
    return e_perp * np.sin(z_prime) + z_hat * np.cos(z_prime)


def default_propagation_ladder(max_step: float = 3600.0) -> list[PropagationConfig]:
    """Default escalation ladder for contract-based propagation."""
    return [
        PropagationConfig(
            model="kepler",
            perturbers=(),
            max_step=max_step,
            full_physics=False,
            adaptive_step=False,
        ),
        PropagationConfig(
            model="kepler",
            perturbers=(),
            max_step=max_step,
            full_physics=True,
            adaptive_step=False,
        ),
        PropagationConfig(
            model="nbody",
            perturbers=("earth", "mars", "jupiter"),
            max_step=max_step,
            full_physics=False,
            adaptive_step=True,
        ),
        PropagationConfig(
            model="nbody",
            perturbers=("earth", "mars", "jupiter"),
            max_step=max_step,
            full_physics=True,
            adaptive_step=True,
        ),
        PropagationConfig(
            model="nbody",
            perturbers=("mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune"),
            max_step=max_step / 6.0,
            full_physics=True,
            adaptive_step=True,
            step_scale=0.02,
        ),
    ]


def _epsilon_ast_from_observations(
    obs: Sequence["Observation"],
    *,
    scale: float = 0.1,
    floor_arcsec: float = 0.01,
    ceiling_arcsec: float | None = None,
) -> float:
    sigmas = np.array([float(o.sigma_arcsec) for o in obs], dtype=float)
    if sigmas.size == 0:
        return floor_arcsec
    eps = float(np.median(sigmas) * scale)
    eps = max(floor_arcsec, eps)
    if ceiling_arcsec is not None:
        eps = min(eps, ceiling_arcsec)
    return eps


def _angular_sep_arcsec(
    ra_a: np.ndarray,
    dec_a: np.ndarray,
    ra_b: np.ndarray,
    dec_b: np.ndarray,
) -> np.ndarray:
    ra1 = np.deg2rad(ra_a)
    dec1 = np.deg2rad(dec_a)
    ra2 = np.deg2rad(ra_b)
    dec2 = np.deg2rad(dec_b)
    cos_ang = np.sin(dec1) * np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2)
    cos_ang = np.clip(cos_ang, -1.0, 1.0)
    ang = np.arccos(cos_ang)
    return np.rad2deg(ang) * 3600.0


def _estimate_step_limit(
    state: np.ndarray,
    epoch: Time,
    perturbers: Sequence[str],
    step_scale: float,
    *,
    min_step: float = 1.0,
) -> float:
    r = state[:3].astype(float)
    v = state[3:].astype(float)
    r_norm = float(np.linalg.norm(r))
    v_norm = float(np.linalg.norm(v))
    if r_norm <= 0.0:
        return min_step
    a_sun = GM_SUN / max(r_norm * r_norm, 1e-30)
    t_curve = math.sqrt(r_norm / max(a_sun, 1e-30))
    t_cross = r_norm / max(v_norm, 1e-30)
    t_close = np.inf
    if perturbers:
        ephem = os.environ.get("NEOTUBE_EPHEMERIS", "de432s")
        sun_pos, sun_vel = _body_posvel_ephem("sun", epoch, ephem)
        sun_bary = sun_pos.xyz.to(u.km).value.flatten()
        sun_vel_kms = sun_vel.xyz.to(u.km / u.s).value.flatten()
        for body in perturbers:
            gm = PLANET_GMS.get(body.lower())
            if gm is None:
                continue
            body_pos, body_vel = _body_posvel_ephem(body, epoch, ephem)
            body_bary = body_pos.xyz.to(u.km).value.flatten()
            body_vel_kms = body_vel.xyz.to(u.km / u.s).value.flatten()
            body_helio = body_bary - sun_bary
            body_vel_helio = body_vel_kms - sun_vel_kms
            rel = r - body_helio
            d = float(np.linalg.norm(rel))
            v_rel = float(np.linalg.norm(v - body_vel_helio))
            if d > 0.0 and v_rel > 0.0:
                t_close = min(t_close, d / v_rel)
    t_local = min(t_curve, t_cross, t_close)
    if not np.isfinite(t_local):
        t_local = t_cross
    return max(min_step, step_scale * t_local)


def _predict_with_config(
    state: np.ndarray,
    epoch: Time,
    obs: Sequence["Observation"],
    config: PropagationConfig,
    *,
    allow_unknown_site: bool,
    obs_cache: ObsCache | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    use_kepler = config.model.lower() == "kepler"
    max_step = float(config.max_step)
    if config.adaptive_step:
        max_step = min(
            max_step,
            _estimate_step_limit(
                state,
                epoch,
                config.perturbers,
                config.step_scale,
            ),
        )
    return predict_radec_from_epoch(
        state,
        epoch,
        obs,
        config.perturbers,
        max_step,
        use_kepler=use_kepler,
        allow_unknown_site=allow_unknown_site,
        light_time_iters=config.light_time_iters,
        full_physics=config.full_physics,
        include_refraction=config.include_refraction,
        obs_cache=obs_cache,
    )


def predict_radec_with_contract(
    states: np.ndarray | Sequence[np.ndarray],
    epoch: Time,
    obs: Sequence["Observation"],
    *,
    configs: Sequence[PropagationConfig] | None = None,
    epsilon_ast_arcsec: float | None = 0.1,
    epsilon_ast_scale: float = 0.1,
    epsilon_ast_floor_arcsec: float = 0.01,
    epsilon_ast_ceiling_arcsec: float | None = None,
    allow_unknown_site: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Predict RA/Dec with a physics escalation ladder until angular error is below contract."""
    if not obs:
        return np.empty((0, 0), dtype=float), np.empty((0, 0), dtype=float), np.empty(0), np.empty(0)

    states_arr = np.asarray(states, dtype=float)
    if states_arr.ndim == 1:
        states_arr = states_arr[np.newaxis, :]
    if states_arr.shape[1] != 6:
        raise ValueError("states must have shape (N,6)")

    if configs is None:
        configs = default_propagation_ladder()
    if not configs:
        raise ValueError("configs must be non-empty")

    if epsilon_ast_arcsec is None:
        epsilon_ast_arcsec = _epsilon_ast_from_observations(
            obs,
            scale=epsilon_ast_scale,
            floor_arcsec=epsilon_ast_floor_arcsec,
            ceiling_arcsec=epsilon_ast_ceiling_arcsec,
        )

    n_states = states_arr.shape[0]
    n_obs = len(obs)
    ra_out = np.empty((n_states, n_obs), dtype=float)
    dec_out = np.empty((n_states, n_obs), dtype=float)
    used_level = np.zeros(n_states, dtype=int)
    max_delta = np.zeros(n_states, dtype=float)

    obs_cache = _prepare_obs_cache(obs, allow_unknown_site=allow_unknown_site)

    current_ra = np.empty((n_states, n_obs), dtype=float)
    current_dec = np.empty((n_states, n_obs), dtype=float)
    for i, state in enumerate(states_arr):
        ra_pred, dec_pred = _predict_with_config(
            state,
            epoch,
            obs,
            configs[0],
            allow_unknown_site=allow_unknown_site,
            obs_cache=obs_cache,
        )
        current_ra[i] = ra_pred
        current_dec[i] = dec_pred
    ra_out[:] = current_ra
    dec_out[:] = current_dec

    active = np.ones(n_states, dtype=bool)
    for level in range(1, len(configs)):
        if not np.any(active):
            break
        for i in np.where(active)[0]:
            ra_new, dec_new = _predict_with_config(
                states_arr[i],
                epoch,
                obs,
                configs[level],
                allow_unknown_site=allow_unknown_site,
                obs_cache=obs_cache,
            )
            delta = _angular_sep_arcsec(current_ra[i], current_dec[i], ra_new, dec_new)
            delta_max = float(np.max(delta))
            if delta_max <= epsilon_ast_arcsec:
                max_delta[i] = max(max_delta[i], delta_max)
                active[i] = False
            else:
                current_ra[i] = ra_new
                current_dec[i] = dec_new
                ra_out[i] = ra_new
                dec_out[i] = dec_new
                used_level[i] = level
                max_delta[i] = delta_max
        # accept all remaining if this is the last level
        if level == len(configs) - 1:
            active[:] = False

    return ra_out, dec_out, used_level, max_delta

# Cache for repeated ephemeris lookups (per process).
_BULK_EPHEM_CACHE: dict[tuple, np.ndarray] = {}


def _time_to_jd12_tdb(times: Time) -> tuple[np.ndarray, np.ndarray]:
    if times.scale == "tdb":
        tdb = times
    else:
        tdb = times.tdb
    if nb_time is not None:
        jd1, jd2 = nb_time.time_to_jd1_jd2(tdb, "tdb")
    else:
        jd1, jd2 = tdb.jd1, tdb.jd2
    return np.asarray(jd1), np.asarray(jd2)


def _body_posvel_km(body: str, times: Time, ephemeris: str = "de432s") -> tuple[np.ndarray, np.ndarray]:
    if _USE_FAST_EPHEM and _HAS_ASTROPY_NUMBA:
        jd1, jd2 = _time_to_jd12_tdb(times)
        scalar = jd1.shape == ()
        if scalar:
            jd1 = jd1.reshape(1)
            jd2 = jd2.reshape(1)
        with solar_system_ephemeris.set(ephemeris):
            pos_km, vel_km_day = nb_solar_system.get_body_barycentric_posvel_jd(
                body, jd1, jd2, ephemeris=None, unit="km"
            )
        pos_km = np.asarray(pos_km)
        vel_km_s = np.asarray(vel_km_day) / 86400.0
    else:
        with solar_system_ephemeris.set(ephemeris):
            pos, vel = get_body_barycentric_posvel(body, times)
        pos_val = pos.xyz.to(u.km).value
        vel_val = vel.xyz.to(u.km / u.s).value
        if pos_val.ndim == 1:
            pos_km = pos_val.reshape(1, 3)
            vel_km_s = vel_val.reshape(1, 3)
        else:
            pos_km = pos_val.T
            vel_km_s = vel_val.T
    return pos_km, vel_km_s


def _body_posvel_km_single(body: str, time: Time, ephemeris: str = "de432s") -> tuple[np.ndarray, np.ndarray]:
    pos_km, vel_km_s = _body_posvel_km(body, time, ephemeris=ephemeris)
    return pos_km[0], vel_km_s[0]


def _body_posvel(body: str, times: Time):
    _trace_astropy(f"get_body_barycentric_posvel start body={body} n={len(np.atleast_1d(times))}")
    with solar_system_ephemeris.set("de432s"):
        posvel = get_body_barycentric_posvel(body, times)
    _trace_astropy(f"get_body_barycentric_posvel end body={body}")
    return posvel


def _body_posvel_ephem(body: str, times: Time, ephemeris: str):
    with solar_system_ephemeris.set(ephemeris):
        return get_body_barycentric_posvel(body, times)


def _times_cache_key(times: Time) -> tuple:
    jd1 = np.atleast_1d(times.jd1)
    jd2 = np.atleast_1d(times.jd2)
    return (times.scale, jd1.tobytes(), jd2.tobytes())

def _stumpff_C2(z: float) -> float:
    try:
        if z > 1e-8:
            s = np.sqrt(z)
            if s > 1e6:
                raise OverflowError("stumpff C2 input s too large")
            return float((1.0 - np.cos(s)) / z)
        if z < -1e-8:
            s = np.sqrt(-z)
            if s > 1e6:
                raise OverflowError("stumpff C2 input s too large")
            return float((np.cosh(s) - 1.0) / (-z))
        # series expansion near 0
        return 0.5
    except FloatingPointError as exc:
        raise ValueError(f"stumpff_C2 overflow for z={z}") from exc


def _stumpff_C3(z: float) -> float:
    try:
        if z > 1e-8:
            s = np.sqrt(z)
            if s > 1e6:
                raise OverflowError("stumpff C3 input s too large")
            return float((s - np.sin(s)) / (s**3))
        if z < -1e-8:
            s = np.sqrt(-z)
            if s > 1e6:
                raise OverflowError("stumpff C3 input s too large")
            return float((np.sinh(s) - s) / (s**3))
        # series expansion near 0
        return 1.0 / 6.0
    except FloatingPointError as exc:
        raise ValueError(f"stumpff_C3 overflow for z={z}") from exc


def _propagate_state_kepler_single(
    r0: np.ndarray,
    v0: np.ndarray,
    dt: float,
    mu_km3_s2: float,
    max_iter: int,
    tol: float,
) -> np.ndarray:
    """Kepler propagation for a single dt using the universal variable formulation."""
    r0_norm = float(np.linalg.norm(r0))
    v0_sq = float(np.dot(v0, v0))
    vr0 = float(np.dot(r0, v0)) / (r0_norm + 1e-30)
    alpha = 2.0 / (r0_norm + 1e-30) - v0_sq / mu_km3_s2
    sqrt_mu = np.sqrt(mu_km3_s2)

    # initial guess for chi
    if alpha > 1e-12:
        chi = sqrt_mu * dt * alpha
    else:
        chi = np.sign(dt) * sqrt_mu * abs(dt) * 1e-3

    success = False
    for _ in range(max_iter):
        z = alpha * chi * chi
        if not np.isfinite(z) or abs(z) > 1.0e6:
            raise ValueError(f"Kepler solver produced pathological z={z:.3e} for dt={dt}")
        C2 = _stumpff_C2(z)
        C3 = _stumpff_C3(z)
        F = (
            (r0_norm * vr0 / sqrt_mu) * chi * chi * C2
            + (1.0 - alpha * r0_norm) * chi**3 * C3
            + r0_norm * chi
            - sqrt_mu * dt
        )
        dF = (
            (r0_norm * vr0 / sqrt_mu) * chi * (1.0 - z * C3)
            + (1.0 - alpha * r0_norm) * chi * chi * C2
            + r0_norm
        )
        step = F / (dF + 1e-30)
        if not np.isfinite(step):
            raise ValueError(f"Kepler update became non-finite for dt={dt}")
        chi -= step
        if not np.isfinite(chi) or abs(chi) > 1.0e8:
            raise ValueError(f"Kepler universal-anomaly chi unstable (chi={chi:.3e}) for dt={dt}")
        if abs(step) < tol:
            success = True
            break

    if not success:
        raise ValueError(f"Kepler solver did not converge for dt={dt}")

    z = alpha * chi * chi
    C2 = _stumpff_C2(z)
    C3 = _stumpff_C3(z)
    if not np.isfinite(C2) or not np.isfinite(C3):
        raise ValueError(f"Kepler coefficients became non-finite for dt={dt}")
    f = 1.0 - (chi * chi / (r0_norm + 1e-30)) * C2
    g = dt - (chi**3 / sqrt_mu) * C3
    r = f * r0 + g * v0
    r_norm = float(np.linalg.norm(r))
    if not np.isfinite(r_norm):
        raise ValueError(f"Kepler propagation produced non-finite radius for dt={dt}")
    gdot = 1.0 - (chi * chi / (r_norm + 1e-30)) * C2
    fdot = (sqrt_mu / ((r_norm + 1e-30) * (r0_norm + 1e-30))) * chi * (z * C3 - 1.0)
    v = fdot * r0 + gdot * v0
    return np.hstack([r, v])


if _HAS_NUMBA:
    @nb.njit(cache=True)
    def _stumpff_C2_numba(z):
        if z > 1e-8:
            s = np.sqrt(z)
            if s > 1e6:
                return np.nan
            return (1.0 - np.cos(s)) / z
        if z < -1e-8:
            s = np.sqrt(-z)
            if s > 1e6:
                return np.nan
            return (np.cosh(s) - 1.0) / (-z)
        return 0.5

    @nb.njit(cache=True)
    def _stumpff_C3_numba(z):
        if z > 1e-8:
            s = np.sqrt(z)
            if s > 1e6:
                return np.nan
            return (s - np.sin(s)) / (s * s * s)
        if z < -1e-8:
            s = np.sqrt(-z)
            if s > 1e6:
                return np.nan
            return (np.sinh(s) - s) / (s * s * s)
        return 1.0 / 6.0

    @nb.njit(cache=True, parallel=True)
    def _propagate_state_kepler_batch_numba(r0, v0, dt_arr, mu_km3_s2, max_iter, tol):
        n = dt_arr.shape[0]
        out = np.empty((n, 6), dtype=np.float64)
        ok = np.ones(n, dtype=np.bool_)

        r0_norm = np.sqrt(np.dot(r0, r0))
        v0_sq = np.dot(v0, v0)
        vr0 = np.dot(r0, v0) / (r0_norm + 1e-30)
        alpha = 2.0 / (r0_norm + 1e-30) - v0_sq / mu_km3_s2
        sqrt_mu = np.sqrt(mu_km3_s2)

        for i in nb.prange(n):
            dt = dt_arr[i]
            if abs(dt) < 1e-9:
                out[i, :3] = r0
                out[i, 3:] = v0
                continue
            if alpha > 1e-12:
                chi = sqrt_mu * dt * alpha
            else:
                chi = np.sign(dt) * sqrt_mu * abs(dt) * 1e-3
            success = False
            for _ in range(max_iter):
                z = alpha * chi * chi
                if not np.isfinite(z) or abs(z) > 1.0e6:
                    ok[i] = False
                    break
                C2 = _stumpff_C2_numba(z)
                C3 = _stumpff_C3_numba(z)
                if not np.isfinite(C2) or not np.isfinite(C3):
                    ok[i] = False
                    break
                F = (
                    (r0_norm * vr0 / sqrt_mu) * chi * chi * C2
                    + (1.0 - alpha * r0_norm) * chi**3 * C3
                    + r0_norm * chi
                    - sqrt_mu * dt
                )
                dF = (
                    (r0_norm * vr0 / sqrt_mu) * chi * (1.0 - z * C3)
                    + (1.0 - alpha * r0_norm) * chi * chi * C2
                    + r0_norm
                )
                step = F / (dF + 1e-30)
                if not np.isfinite(step):
                    ok[i] = False
                    break
                chi -= step
                if not np.isfinite(chi) or abs(chi) > 1.0e8:
                    ok[i] = False
                    break
                if abs(step) < tol:
                    success = True
                    break
            if not success:
                ok[i] = False
                continue
            z = alpha * chi * chi
            C2 = _stumpff_C2_numba(z)
            C3 = _stumpff_C3_numba(z)
            if not np.isfinite(C2) or not np.isfinite(C3):
                ok[i] = False
                continue
            f = 1.0 - (chi * chi / (r0_norm + 1e-30)) * C2
            g = dt - (chi**3 / sqrt_mu) * C3
            r = f * r0 + g * v0
            r_norm = np.sqrt(np.dot(r, r))
            if not np.isfinite(r_norm):
                ok[i] = False
                continue
            gdot = 1.0 - (chi * chi / (r_norm + 1e-30)) * C2
            fdot = (sqrt_mu / ((r_norm + 1e-30) * (r0_norm + 1e-30))) * chi * (z * C3 - 1.0)
            v = fdot * r0 + gdot * v0
            out[i, :3] = r
            out[i, 3:] = v
        return out, ok


def propagate_state_kepler(
    state: np.ndarray,
    epoch: Time,
    targets: Iterable[Time],
    *,
    mu_km3_s2: float = GM_SUN,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> np.ndarray:
    """Propagate a heliocentric state assuming a pure two-body (Sun-only) Keplerian orbit.

    Uses the universal variable formulation (Vallado-style f/g functions).
    Input state is [x,y,z,vx,vy,vz] in km and km/s. Targets are astropy Times.
    """
    targets = list(targets)
    if not targets:
        return np.empty((0, 6), dtype=float)

    r0 = np.array(state[:3], dtype=float)
    v0 = np.array(state[3:], dtype=float)

    dt_arr = np.array([(t - epoch).to(u.s).value for t in targets], dtype=float)

    if _HAS_NUMBA:
        out, ok = _propagate_state_kepler_batch_numba(r0, v0, dt_arr, mu_km3_s2, max_iter, tol)
        if np.all(ok):
            return out
        # fallback for failed steps
        for i in range(len(targets)):
            if ok[i]:
                continue
            dt = float(dt_arr[i])
            if abs(dt) < 1e-9:
                out[i] = state
                continue
            out[i] = _propagate_state_kepler_single(r0, v0, dt, mu_km3_s2, max_iter, tol)
        return out

    out = np.empty((len(targets), 6), dtype=float)
    for i, t in enumerate(targets):
        dt = float((t - epoch).to(u.s).value)
        if abs(dt) < 1e-9:
            out[i] = state
            continue
        out[i] = _propagate_state_kepler_single(r0, v0, dt, mu_km3_s2, max_iter, tol)
    return out


def propagate_state_sun(
    state: np.ndarray,
    epoch: Time,
    targets: Iterable[Time],
    *,
    mu_km3_s2: float = GM_SUN,
    max_step: float = 3600.0,
) -> np.ndarray:
    """Propagate a heliocentric state using a Sun-only (2-body) ODE integration.

    This is 'Keplerian' dynamics (no perturbers) but avoids universal-variable
    numerical issues for pathological states by relying on solve_ivp.
    """
    targets = list(targets)
    if not targets:
        return np.empty((0, 6), dtype=float)

    offsets = np.array([(target - epoch).to(u.s).value for target in targets], dtype=float)
    neg_mask = offsets < -1e-9
    pos_mask = offsets > 1e-9
    zero_mask = ~(neg_mask | pos_mask)

    results = np.empty((len(targets), 6), dtype=float)
    if np.any(zero_mask):
        results[zero_mask] = state

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        r = y[:3]
        norm = np.linalg.norm(r)
        acc = -mu_km3_s2 * r / (norm**3 + 1e-18)
        return np.concatenate((y[3:], acc))

    def _solve(offsets_subset: np.ndarray) -> np.ndarray:
        if offsets_subset.size == 0:
            return np.empty((0, 6), dtype=float)
        t_end = float(offsets_subset.min() if offsets_subset.min() < 0 else offsets_subset.max())
        t_eval = np.sort(offsets_subset)
        if t_end < 0:
            t_eval = t_eval[::-1]
        sol = solve_ivp(
            rhs,
            (0.0, t_end),
            state,
            t_eval=t_eval,
            max_step=abs(max_step),
            rtol=1e-9,
            atol=1e-12,
            method="DOP853",
        )
        if not sol.success:
            raise RuntimeError(f"Sun-only propagation failed: {sol.message}")
        return sol.y.T

    if np.any(pos_mask):
        pos_offsets = offsets[pos_mask]
        pos_states = _solve(pos_offsets)
        order_pos = np.argsort(pos_offsets)
        results[np.where(pos_mask)[0][order_pos]] = pos_states

    if np.any(neg_mask):
        neg_offsets = offsets[neg_mask]
        neg_states = _solve(neg_offsets)
        order_neg = np.argsort(neg_offsets)[::-1]
        results[np.where(neg_mask)[0][order_neg]] = neg_states

    return results


@dataclass
class PerturberTable:
    times: np.ndarray
    bodies: list[str]
    gms: np.ndarray
    positions: np.ndarray  # shape (len(bodies), len(times), 3)

    def position_vectors(self, t: float) -> np.ndarray:
        out = np.empty((len(self.bodies), 3), dtype=float)
        for i in range(len(self.bodies)):
            for dim in range(3):
                out[i, dim] = np.interp(t, self.times, self.positions[i, :, dim])
        return out


def _bulk_heliocentric_positions(body: str, times: Time, sun_pos_km: np.ndarray) -> np.ndarray:
    """Return heliocentric (km) positions for body over multiple epochs."""
    key = (body.lower(), _times_cache_key(times))
    cached = _BULK_EPHEM_CACHE.get(key)
    if cached is not None:
        return cached
    body_pos_km, _ = _body_posvel_km(body, times)
    out = body_pos_km - sun_pos_km
    _BULK_EPHEM_CACHE[key] = out
    return out


def _heliocentric_position(body: str, epoch: Time) -> np.ndarray:
    """Return heliocentric (km) position for a single epoch."""
    body_pos_km, _ = _body_posvel_km(body, epoch)
    sun_pos_km, _ = _body_posvel_km("sun", epoch)
    return body_pos_km[0] - sun_pos_km[0]


def _build_perturber_table(
    epoch: Time,
    perturbers: Sequence[str],
    max_offset: float,
    step: float,
) -> PerturberTable:
    """Precompute perturber positions on a regular grid of offsets (seconds)."""
    if step <= 0:
        step = 1.0

    n_steps = max(2, int(ceil(max_offset / step)) + 1)
    grid = np.linspace(0.0, max_offset, n_steps, dtype=float)
    times = epoch + TimeDelta(grid * u.s)
    sun_pos_km, _ = _body_posvel_km("sun", times)

    bodies = []
    position_list = []
    gms = []
    for body in perturbers:
        gm = PLANET_GMS.get(body.lower())
        if gm is None:
            continue
        body_pos = _bulk_heliocentric_positions(body, times, sun_pos_km)
        bodies.append(body.lower())
        position_list.append(body_pos)
        gms.append(gm)

    if not bodies:
        return PerturberTable(
            times=grid,
            bodies=[],
            gms=np.empty((0,), dtype=float),
            positions=np.empty((0, len(grid), 3), dtype=float),
        )

    stacked_positions = np.stack(position_list, axis=0)
    return PerturberTable(
        times=grid,
        bodies=bodies,
        gms=np.array(gms, dtype=float),
        positions=stacked_positions,
    )


def _acceleration(r: np.ndarray, perturber_table: PerturberTable, t: float) -> np.ndarray:
    """Compute heliocentric acceleration on a test particle."""
    norm = np.linalg.norm(r)
    acc = -GM_SUN * r / (norm**3 + 1e-18)
    if perturber_table.bodies:
        positions = perturber_table.position_vectors(t)
        diffs = r - positions
        norms = np.linalg.norm(diffs, axis=1)
        coeff = -perturber_table.gms / (norms**3 + 1e-18)
        acc += np.sum((coeff[:, np.newaxis] * diffs), axis=0)
    return acc


def _rhs(t: float, y: np.ndarray, perturber_table: PerturberTable) -> np.ndarray:
    """Ordinary differential equation for heliocentric motion."""
    r = y[:3]
    acc = _acceleration(r, perturber_table, t)
    return np.concatenate((y[3:], acc))


def propagate_state(
    state: np.ndarray,
    epoch: Time,
    targets: Iterable[Time],
    *,
    perturbers: Sequence[str] = ("earth", "mars", "jupiter"),
    max_step: float = 300.0,
) -> np.ndarray:
    """Propagate a single heliocentric state to multiple epochs."""
    targets = list(targets)
    if not targets:
        return np.empty((0, 6), dtype=float)

    offsets = np.array([(target - epoch).to(u.s).value for target in targets], dtype=float)
    neg_mask = offsets < -1e-9
    pos_mask = offsets > 1e-9
    zero_mask = ~(neg_mask | pos_mask)

    results = np.empty((len(targets), 6), dtype=float)
    if np.any(zero_mask):
        results[zero_mask] = state

    def _solve(offsets_subset: np.ndarray) -> np.ndarray:
        if offsets_subset.size == 0:
            return np.empty((0, 6), dtype=float)
        t_end = float(offsets_subset.min() if offsets_subset.min() < 0 else offsets_subset.max())
        t_eval = np.sort(offsets_subset)
        if t_end < 0:
            t_eval = t_eval[::-1]

        # Build a perturber table spanning the integration interval.
        if t_end < 0:
            table_times = np.linspace(
                t_end,
                0.0,
                max(2, int(ceil(abs(t_end) / max(abs(max_step), 1.0))) + 1),
                dtype=float,
            )
        else:
            table_times = np.linspace(
                0.0,
                t_end,
                max(2, int(ceil(abs(t_end) / max(abs(max_step), 1.0))) + 1),
                dtype=float,
            )
        times = epoch + TimeDelta(table_times * u.s)
        sun_pos_km, _ = _body_posvel_km("sun", times)
        bodies = []
        position_list = []
        gms = []
        for body in perturbers:
            gm = PLANET_GMS.get(body.lower())
            if gm is None:
                continue
            body_pos = _bulk_heliocentric_positions(body, times, sun_pos_km)
            bodies.append(body.lower())
            position_list.append(body_pos)
            gms.append(gm)
        if bodies:
            stacked_positions = np.stack(position_list, axis=0)
            pert_table = PerturberTable(
                times=table_times,
                bodies=bodies,
                gms=np.array(gms, dtype=float),
                positions=stacked_positions,
            )
        else:
            pert_table = PerturberTable(
                times=table_times,
                bodies=[],
                gms=np.empty((0,), dtype=float),
                positions=np.empty((0, len(table_times), 3), dtype=float),
            )

        sol = solve_ivp(
            lambda t, y: _rhs(t, y, pert_table),
            (0.0, t_end),
            state,
            t_eval=t_eval,
            max_step=abs(max_step),
            rtol=1e-9,
            atol=1e-12,
            method="DOP853",
        )
        if not sol.success:
            raise RuntimeError(f"Propagation to targets failed: {sol.message}")
        return sol.y.T

    if np.any(pos_mask):
        pos_offsets = offsets[pos_mask]
        pos_states = _solve(pos_offsets)
        order_pos = np.argsort(pos_offsets)
        results[np.where(pos_mask)[0][order_pos]] = pos_states

    if np.any(neg_mask):
        neg_offsets = offsets[neg_mask]
        neg_states = _solve(neg_offsets)
        order_neg = np.argsort(neg_offsets)[::-1]
        results[np.where(neg_mask)[0][order_neg]] = neg_states

    return results


def predict_radec(
    state: np.ndarray,
    epoch: Time,
    site_code: str | None = None,
    *,
    allow_unknown_site: bool = True,
) -> tuple[float, float]:
    """Compute topocentric RA/Dec (degrees) from a heliocentric state."""
    ra, dec = predict_radec_batch(
        state[np.newaxis, :],
        (epoch,),
        site_codes=(None if site_code is None else (site_code,)),
        allow_unknown_site=allow_unknown_site,
    )
    return float(ra[0]), float(dec[0])


def predict_radec_from_epoch(
    state: np.ndarray,
    epoch: Time,
    obs: Sequence["Observation"],
    perturbers: Sequence[str],
    max_step: float,
    use_kepler: bool = True,
    *,
    allow_unknown_site: bool = True,
    light_time_iters: int = 2,
    full_physics: bool = False,
    include_refraction: bool = False,
    obs_cache: ObsCache | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute astrometric topocentric RA/Dec (ICRS) from a heliocentric state with light-time."""
    if not obs:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)

    from .models import Observation  # local import to avoid cycles

    if not isinstance(obs[0], Observation):
        raise ValueError("obs must be a sequence of Observation objects")

    ra_vals = np.empty(len(obs), dtype=float)
    dec_vals = np.empty(len(obs), dtype=float)
    c_km_s = 299792.458

    if obs_cache is None:
        obs_cache = _prepare_obs_cache(obs, allow_unknown_site=allow_unknown_site)

    for idx, ob in enumerate(obs):
        t_obs_tdb = obs_cache.times_tdb[idx]
        site_pos = obs_cache.site_pos_km[idx]
        site_vel = obs_cache.site_vel_km_s[idx]
        earth_bary = obs_cache.earth_bary_km[idx]
        earth_bary_vel = obs_cache.earth_bary_vel_km_s[idx]
        obs_bary = earth_bary + site_pos
        obs_bary_vel = earth_bary_vel + site_vel

        t_emit = t_obs_tdb
        obj_bary = None
        sun_bary = None
        for _ in range(max(1, light_time_iters)):
            if use_kepler:
                try:
                    emit_state = propagate_state_kepler(state, epoch, (t_emit,))[0]
                except Exception:
                    emit_state = propagate_state(
                        state, epoch, (t_emit,), perturbers=perturbers, max_step=max_step
                    )[0]
            else:
                emit_state = propagate_state(
                    state, epoch, (t_emit,), perturbers=perturbers, max_step=max_step
                )[0]
            sun_bary, _ = _body_posvel_km_single("sun", t_emit)
            obj_bary = emit_state[:3] + sun_bary
            rho = float(np.linalg.norm(obj_bary - obs_bary))
            dt_geom = rho / c_km_s
            if full_physics:
                dt_shapiro = shapiro_delay_sun(obj_bary, obs_bary, sun_bary)
            else:
                dt_shapiro = 0.0
            t_emit = t_obs_tdb - TimeDelta(dt_geom + dt_shapiro, format="sec")

        if obj_bary is None:
            obj_bary = state[:3] + earth_bary * 0.0

        topovec = obj_bary - obs_bary
        if full_physics:
            topounit = aberrate_direction_first_order(topovec, obs_bary_vel)
            if include_refraction:
                topounit = _apply_bennett_refraction(topounit, site_pos)
            xy = np.hypot(topounit[0], topounit[1])
            ra_vals[idx] = (
                np.degrees(np.arctan2(topounit[1], topounit[0])) + 360.0
            ) % 360.0
            dec_vals[idx] = np.degrees(np.arctan2(topounit[2], xy))
        else:
            xy = np.hypot(topovec[0], topovec[1])
            ra_vals[idx] = (
                np.degrees(np.arctan2(topovec[1], topovec[0])) + 360.0
            ) % 360.0
            dec_vals[idx] = np.degrees(np.arctan2(topovec[2], xy))

    return ra_vals, dec_vals


def predict_radec_with_geometry(
    state: np.ndarray,
    epoch: Time,
    obs: Sequence["Observation"],
    perturbers: Sequence[str],
    max_step: float,
    use_kepler: bool = True,
    *,
    allow_unknown_site: bool = True,
    light_time_iters: int = 2,
    full_physics: bool = False,
    include_refraction: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute RA/Dec plus heliocentric r, topocentric delta, and phase angle (rad)."""
    if not obs:
        empty = np.empty(0, dtype=float)
        return empty, empty, empty, empty, empty

    from .models import Observation  # local import to avoid cycles

    if not isinstance(obs[0], Observation):
        raise ValueError("obs must be a sequence of Observation objects")

    ra_vals = np.empty(len(obs), dtype=float)
    dec_vals = np.empty(len(obs), dtype=float)
    r_au = np.empty(len(obs), dtype=float)
    delta_au = np.empty(len(obs), dtype=float)
    phase_rad = np.empty(len(obs), dtype=float)
    c_km_s = 299792.458

    for idx, ob in enumerate(obs):
        t_obs = ob.time
        t_obs_tdb = ob.time.tdb
        site_codes = [ob.site]
        observer_positions = [ob.observer_pos_km]
        obs_pos_km, obs_vel_km_s = _site_states(
            [t_obs],
            site_codes,
            observer_positions_km=observer_positions,
            observer_velocities_km_s=None,
            allow_unknown_site=allow_unknown_site,
        )
        site_pos = obs_pos_km[0]
        site_vel = obs_vel_km_s[0]

        earth_bary, earth_bary_vel = _body_posvel_km_single("earth", t_obs_tdb)

        t_emit = t_obs_tdb
        obj_bary = None
        sun_bary = None
        for _ in range(max(1, light_time_iters)):
            if use_kepler:
                try:
                    emit_state = propagate_state_kepler(state, epoch, (t_emit,))[0]
                except Exception:
                    emit_state = propagate_state(
                        state, epoch, (t_emit,), perturbers=perturbers, max_step=max_step
                    )[0]
            else:
                emit_state = propagate_state(
                    state, epoch, (t_emit,), perturbers=perturbers, max_step=max_step
                )[0]
            sun_bary, _ = _body_posvel_km_single("sun", t_emit)
            obj_bary = emit_state[:3] + sun_bary
            obs_bary = earth_bary + site_pos
            rho = float(np.linalg.norm(obj_bary - obs_bary))
            dt_geom = rho / c_km_s
            if full_physics:
                dt_shapiro = shapiro_delay_sun(obj_bary, obs_bary, sun_bary)
            else:
                dt_shapiro = 0.0
            t_emit = t_obs_tdb - TimeDelta(dt_geom + dt_shapiro, format="sec")

        if obj_bary is None or sun_bary is None:
            obj_bary = state[:3] + earth_bary * 0.0
            sun_bary, _ = _body_posvel_km_single("sun", t_obs_tdb)

        obs_bary = earth_bary + site_pos
        obs_bary_vel = earth_bary_vel + site_vel
        topovec = obj_bary - obs_bary
        if full_physics:
            topounit = aberrate_direction_first_order(topovec, obs_bary_vel)
            if include_refraction:
                topounit = _apply_bennett_refraction(topounit, site_pos)
            xy = np.hypot(topounit[0], topounit[1])
            ra_vals[idx] = (
                np.degrees(np.arctan2(topounit[1], topounit[0])) + 360.0
            ) % 360.0
            dec_vals[idx] = np.degrees(np.arctan2(topounit[2], xy))
        else:
            xy = np.hypot(topovec[0], topovec[1])
            ra_vals[idx] = (
                np.degrees(np.arctan2(topovec[1], topovec[0])) + 360.0
            ) % 360.0
            dec_vals[idx] = np.degrees(np.arctan2(topovec[2], xy))

        r_vec = obj_bary - sun_bary
        r_norm = float(np.linalg.norm(r_vec))
        delta_norm = float(np.linalg.norm(topovec))
        r_au[idx] = r_norm / float(u.au.to(u.km))
        delta_au[idx] = delta_norm / float(u.au.to(u.km))
        # phase angle between r_vec and -topovec
        denom = max(r_norm * delta_norm, 1e-30)
        cos_phase = np.dot(r_vec, -topovec) / denom
        cos_phase = float(np.clip(cos_phase, -1.0, 1.0))
        phase_rad[idx] = float(np.arccos(cos_phase))

    return ra_vals, dec_vals, r_au, delta_au, phase_rad


def _site_error_message(site: str | None, time: Time, message: str) -> str:
    site_label = (site or "UNKNOWN").strip() if site else "UNKNOWN"
    time_label = time.isot if hasattr(time, "isot") else str(time)
    return f"{message} (site='{site_label}', time='{time_label}')"


def _icrs_from_horizons_au(
    x_au: float,
    y_au: float,
    z_au: float,
    vx_au_day: float,
    vy_au_day: float,
    vz_au_day: float,
) -> tuple[np.ndarray, np.ndarray]:
    pos = np.array([x_au, y_au, z_au], dtype=float) * AU_KM
    vel = (
        np.array([vx_au_day, vy_au_day, vz_au_day], dtype=float) * (AU_KM / DAY_S)
    )
    return pos, vel


@lru_cache(maxsize=2048)
def _spacecraft_barycentric_state(
    ephemeris_id: str,
    ephemeris_id_type: str | None,
    ephemeris_location: str,
    jd_seconds: int,
    ephemeris_frame: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    t = Time(jd_seconds / 86400.0, format="jd")
    id_type = ephemeris_id_type
    if id_type is not None and id_type.lower() in {"id", "majorbody", "spacecraft"}:
        id_type = None
    obj = Horizons(
        id=ephemeris_id,
        id_type=id_type,
        location=ephemeris_location,
        epochs=t.jd,
    )
    # Use equatorial refplane to align with Horizons ephemerides.
    vec = obj.vectors(refplane="earth")
    x = float(vec["x"][0])
    y = float(vec["y"][0])
    z = float(vec["z"][0])
    vx = float(vec["vx"][0])
    vy = float(vec["vy"][0])
    vz = float(vec["vz"][0])
    frame_key = (ephemeris_frame or "icrs").strip().lower()
    if frame_key in {"icrs", "bcrs"}:
        return _icrs_from_horizons_au(x, y, z, vx, vy, vz)

    rep = CartesianRepresentation(x * u.au, y * u.au, z * u.au)
    diff = CartesianDifferential(vx * u.au / u.day, vy * u.au / u.day, vz * u.au / u.day)
    if frame_key in {"tete", "tod", "earth"}:
        coord = SkyCoord(rep.with_differentials(diff), frame=TETE(obstime=t))
        coord = coord.transform_to(ICRS())
    elif frame_key in {"fk5", "mean"}:
        coord = SkyCoord(rep.with_differentials(diff), frame=FK5(equinox=t))
        coord = coord.transform_to(ICRS())
    elif frame_key == "gcrs":
        coord = SkyCoord(rep.with_differentials(diff), frame=GCRS(obstime=t))
        coord = coord.transform_to(ICRS())
    else:
        coord = SkyCoord(rep.with_differentials(diff), frame=ICRS())
    pos = coord.cartesian.xyz.to(u.km).value
    vel = coord.cartesian.differentials["s"].d_xyz.to(u.km / u.s).value
    return pos, vel


def _site_states(
    epochs: Sequence[Time],
    site_codes: Sequence[str | None] | None,
    observer_positions_km: Sequence[np.ndarray | None] | None = None,
    observer_velocities_km_s: Sequence[np.ndarray | None] | None = None,
    *,
    allow_unknown_site: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    if observer_positions_km is not None and len(observer_positions_km) != len(epochs):
        raise ValueError("observer_positions_km must match epochs length")
    if observer_velocities_km_s is not None and len(observer_velocities_km_s) != len(epochs):
        raise ValueError("observer_velocities_km_s must match epochs length")
    if site_codes is not None and len(site_codes) != len(epochs):
        raise ValueError("site_codes must match epochs length")
    positions = np.zeros((len(epochs), 3), dtype=float)
    velocities = np.zeros((len(epochs), 3), dtype=float)
    site_iter = site_codes if site_codes is not None else [None] * len(epochs)
    observer_pos_iter = (
        observer_positions_km if observer_positions_km is not None else [None] * len(epochs)
    )
    observer_vel_iter = (
        observer_velocities_km_s
        if observer_velocities_km_s is not None
        else [None] * len(epochs)
    )
    for idx, (code, time, observer_pos, observer_vel) in enumerate(
        zip(site_iter, epochs, observer_pos_iter, observer_vel_iter)
    ):
        if observer_pos is not None:
            positions[idx] = np.asarray(observer_pos, dtype=float)
            if observer_vel is not None:
                velocities[idx] = np.asarray(observer_vel, dtype=float)
            continue
        if code is None:
            if allow_unknown_site:
                continue
            raise ValueError(_site_error_message(code, time, "Missing site code"))
        loc = get_site_location(code)
        if loc is None:
            config = get_site_ephemeris(code)
            if config is not None:
                ephemeris_id = str(config.get("ephemeris_id"))
                ephemeris_id_type = config.get("ephemeris_id_type")
                ephemeris_location = str(config.get("ephemeris_location", "500@399"))
                ephemeris_frame = config.get("ephemeris_frame") or "icrs"
                jd_seconds = int(round(float(time.jd) * 86400.0))
                pos_bary, vel_bary = _spacecraft_barycentric_state(
                    ephemeris_id,
                    ephemeris_id_type,
                    ephemeris_location,
                    jd_seconds,
                    ephemeris_frame,
                )
                if ephemeris_location.lower() in {"@ssb", "500@0", "@sun"}:
                    if ephemeris_location.lower() in {"@sun", "500@0"}:
                        sun_km, sun_kms = _body_posvel_km_single("sun", time)
                        pos_bary = pos_bary + sun_km
                        vel_bary = vel_bary + sun_kms
                    earth_km, earth_kms = _body_posvel_km_single("earth", time)
                    positions[idx] = pos_bary - earth_km
                    velocities[idx] = vel_bary - earth_kms
                else:
                    positions[idx] = pos_bary
                    velocities[idx] = vel_bary
                continue
            if allow_unknown_site:
                continue
            raise ValueError(
                _site_error_message(code, time, "Site code not found in observatory catalog")
            )
        gcrs = loc.get_gcrs(obstime=time)
        positions[idx] = gcrs.cartesian.xyz.to(u.km).value
        if gcrs.cartesian.differentials:
            velocities[idx] = (
                gcrs.cartesian.differentials["s"].d_xyz.to(u.km / u.s).value
            )
    return positions, velocities


def _site_offsets(
    epochs: Sequence[Time],
    site_codes: Sequence[str | None] | None,
    observer_positions_km: Sequence[np.ndarray | None] | None = None,
    *,
    allow_unknown_site: bool = True,
) -> np.ndarray:
    positions, _ = _site_states(
        epochs,
        site_codes,
        observer_positions_km,
        observer_velocities_km_s=None,
        allow_unknown_site=allow_unknown_site,
    )
    return positions


def predict_radec_batch(
    states: np.ndarray | Sequence[np.ndarray],
    epochs: Sequence[Time],
    site_codes: Sequence[str | None] | None = None,
    observer_positions_km: Sequence[np.ndarray | None] | None = None,
    observer_velocities_km_s: Sequence[np.ndarray | None] | None = None,
    *,
    allow_unknown_site: bool = True,
    light_time_mode: str = "auto",
    light_time_iters: int = 2,
    full_physics: bool = False,
    include_refraction: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized astrometric topocentric RA/Dec (ICRS) for many heliocentric states/times."""
    if len(epochs) == 0:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)

    states_arr = np.asarray(states, dtype=float)
    if states_arr.ndim == 1:
        states_arr = states_arr[np.newaxis, :]
    if states_arr.shape[1] != 6:
        raise ValueError("Expected states with 6 components (r/v).")

    time_obs = Time(epochs)
    time_tdb = time_obs.tdb
    sun_bary, sun_bary_vel = _body_posvel_km("sun", time_tdb)
    if sun_bary.shape[0] != len(epochs):
        sun_bary = sun_bary[: len(epochs)]
        sun_bary_vel = sun_bary_vel[: len(epochs)]

    obj_pos = states_arr[:, :3]
    obj_vel = states_arr[:, 3:]
    if len(epochs) != obj_pos.shape[0]:
        raise ValueError("Number of states and epochs must match.")
    obs_pos_km, obs_vel_km_s = _site_states(
        time_obs,
        site_codes,
        observer_positions_km,
        observer_velocities_km_s,
        allow_unknown_site=allow_unknown_site,
    )
    obj_bary = obj_pos + sun_bary
    obj_bary_vel = obj_vel + sun_bary_vel
    earth_bary, earth_bary_vel = _body_posvel_km("earth", time_tdb)
    if earth_bary.shape[0] != len(epochs):
        earth_bary = earth_bary[: len(epochs)]
        earth_bary_vel = earth_bary_vel[: len(epochs)]
    obs_bary = earth_bary + obs_pos_km
    obs_bary_vel = earth_bary_vel + obs_vel_km_s

    # Light-time correction: iterate emission time using barycentric positions.
    c_km_s = 299792.458
    mode = light_time_mode.lower()
    if mode not in {"auto", "linear", "kepler", "none"}:
        raise ValueError(f"Unsupported light_time_mode: {light_time_mode}")
    if mode == "auto":
        mode = "kepler" if len(epochs) <= 128 else "linear"

    obj_bary_lt = obj_bary.copy()
    if mode == "none":
        pass
    elif mode == "kepler":
        # Use Kepler propagation per-state to emission time; robust for small batches.
        for _ in range(max(1, light_time_iters)):
            vec = obj_bary_lt - obs_bary
            rho = np.linalg.norm(vec, axis=1)
            dt = rho / c_km_s
            obj_helio_emit = np.empty_like(obj_pos)
            for idx in range(len(epochs)):
                if not np.isfinite(dt[idx]) or dt[idx] <= 0.0:
                    obj_helio_emit[idx] = obj_pos[idx]
                    continue
                t_emit = time_tdb[idx] - TimeDelta(float(dt[idx]), format="sec")
                try:
                    emit_state = propagate_state_kepler(
                        states_arr[idx], time_tdb[idx], (t_emit,)
                    )[0]
                except Exception:
                    emit_state = states_arr[idx]
                obj_helio_emit[idx] = emit_state[:3]
            obj_bary_lt = obj_helio_emit + sun_bary.T
    else:
        iters = max(1, light_time_iters)
        if _USE_NUMBA_ACCEL:
            obj_bary_lt = _light_time_linear_numba(
                obj_bary, obj_bary_vel, obs_bary, iters, c_km_s
            )
        else:
            for _ in range(iters):
                vec = obj_bary_lt - obs_bary
                rho = np.linalg.norm(vec, axis=1)
                dt = rho / c_km_s
                obj_bary_lt = obj_bary - obj_bary_vel * dt[:, None]

    if full_physics:
        ra_vals = np.empty(len(epochs), dtype=float)
        dec_vals = np.empty(len(epochs), dtype=float)
        for idx in range(len(epochs)):
            t_obs_tdb = time_tdb[idx]
            obs_b = obs_bary[idx]
            obs_b_vel = obs_bary_vel[idx]
            t_emit = t_obs_tdb
            obj_bary_exact = None
            for _ in range(max(1, light_time_iters)):
                try:
                    emit_state = propagate_state_kepler(states_arr[idx], time_tdb[idx], (t_emit,))[
                        0
                    ]
                except Exception:
                    emit_state = states_arr[idx]
                sun_bary_emit, _ = _body_posvel_km_single("sun", t_emit)
                obj_bary_exact = emit_state[:3] + sun_bary_emit
                rho = float(np.linalg.norm(obj_bary_exact - obs_b))
                dt_geom = rho / c_km_s
                dt_sh = shapiro_delay_sun(obj_bary_exact, obs_b, sun_bary_emit)
                t_emit = t_obs_tdb - TimeDelta(dt_geom + dt_sh, format="sec")
            if obj_bary_exact is None:
                obj_bary_exact = obj_bary_lt[idx]
            topovec = obj_bary_exact - obs_b
            topounit = aberrate_direction_first_order(topovec, obs_b_vel)
            if include_refraction:
                topounit = _apply_bennett_refraction(topounit, obs_pos_km[idx])
            xy = np.hypot(topounit[0], topounit[1])
            ra_vals[idx] = (np.degrees(np.arctan2(topounit[1], topounit[0])) + 360.0) % 360.0
            dec_vals[idx] = np.degrees(np.arctan2(topounit[2], xy))
        return ra_vals, dec_vals

    topovec = obj_bary_lt - obs_bary
    if _USE_NUMBA_ACCEL:
        return _radec_from_topovec_numba(topovec)
    xy = np.hypot(topovec[:, 0], topovec[:, 1])
    ra_out = (np.degrees(np.arctan2(topovec[:, 1], topovec[:, 0])) + 360.0) % 360.0
    dec_out = np.degrees(np.arctan2(topovec[:, 2], xy))
    return ra_out, dec_out


@dataclass
class ReplicaCloud:
    epoch: Time
    states: np.ndarray  # shape (6, N)


def _chunk_propagate(args):
    chunk, epoch, targets, perturbers, max_step = args
    results = []
    for idx in range(chunk.shape[1]):
        results.append(
            propagate_state(
                chunk[:, idx],
                epoch,
                targets,
                perturbers=perturbers,
                max_step=max_step,
            )
        )
    return np.stack(results, axis=2)


def propagate_replicas(
    cloud: ReplicaCloud,
    targets: Iterable[Time],
    perturbers: Sequence[str],
    *,
    max_step: float = 300.0,
    workers: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> list[np.ndarray]:
    """Return propagated states for each target epoch with parallel replica batches."""
    targets_list = list(targets)
    if not targets_list:
        return []

    total = cloud.states.shape[1]
    if total == 0:
        return []

    cpu_count = os.cpu_count() or 1
    requested_workers = workers if workers is not None else cpu_count
    requested_workers = max(1, requested_workers)

    if batch_size is None or batch_size <= 0:
        batch_size = max(1, ceil(total / requested_workers))

    schedule = []
    for start in range(0, total, batch_size):
        chunk = cloud.states[:, start : min(start + batch_size, total)]
        schedule.append((chunk, cloud.epoch, targets_list, perturbers, max_step))

    actual_workers = min(requested_workers, len(schedule))

    outputs = np.empty((len(targets_list), 6, total), dtype=float)
    cursor = 0
    if actual_workers <= 1 or len(schedule) == 1:
        for unit in schedule:
            result = _chunk_propagate(unit)
            size = result.shape[2]
            outputs[:, :, cursor : cursor + size] = result
            cursor += size
    else:
        with ProcessPoolExecutor(max_workers=actual_workers) as executor:
            for result in executor.map(_chunk_propagate, schedule):
                size = result.shape[2]
                outputs[:, :, cursor : cursor + size] = result
                cursor += size

    return [outputs[idx] for idx in range(len(targets_list))]
