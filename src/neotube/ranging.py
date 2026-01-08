from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from astropy import units as u
from astropy.time import Time

from .constants import DAY_S
from .models import Attributable, Observation
from .propagate import _body_posvel_km_single, _site_states


def _observer_helio_state(obs: Observation, t_obs: Time) -> tuple[np.ndarray, np.ndarray]:
    earth_bary, earth_bary_vel = _body_posvel_km_single("earth", t_obs)
    sun_bary, sun_bary_vel = _body_posvel_km_single("sun", t_obs)
    earth_helio = earth_bary - sun_bary
    earth_vel_helio = earth_bary_vel - sun_bary_vel
    site_pos, site_vel = _site_states(
        [t_obs],
        [obs.site],
        observer_positions_km=[obs.observer_pos_km],
        observer_velocities_km_s=None,
        allow_unknown_site=True,
    )
    return earth_helio + site_pos[0], earth_vel_helio + site_vel[0]


def attrib_from_state_with_observer_time(
    state: np.ndarray,
    obs: Observation,
    t_obs: Time,
) -> tuple[Attributable, float, float]:
    obs_pos, obs_vel = _observer_helio_state(obs, t_obs)
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
        ra_dot_deg_per_day=float(math.degrees(ra_dot) * DAY_S),
        dec_dot_deg_per_day=float(math.degrees(dec_dot) * DAY_S),
    )
    return attrib, rho, rhodot


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
        ra_deg=float(math.degrees(ra) % 360.0),
        dec_deg=float(math.degrees(dec)),
        ra_dot_deg_per_day=float(math.degrees(ra_dot) * DAY_S),
        dec_dot_deg_per_day=float(math.degrees(dec_dot) * DAY_S),
    )


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


def fit_unit_vector_rate(
    observations: Sequence[Observation],
    epoch: Time,
) -> tuple[np.ndarray, np.ndarray]:
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
        fallback = build_attributable(observations, epoch)
        return s_and_sdot(fallback)
    s0 = s0 / s0_norm
    sdot = sdot - np.dot(s0, sdot) * s0
    return s0, sdot


def build_attributable_studentt(
    observations: Sequence[Observation],
    epoch: Time | None = None,
    *,
    nu: float = 4.0,
    max_iter: int = 10,
    tol: float = 1e-8,
    site_kappas: dict[str, float] | None = None,
) -> tuple[Attributable, np.ndarray]:
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
    return_s_sdot: bool = False,
    nu: float = 4.0,
    max_iter: int = 10,
    tol: float = 1e-8,
    site_kappas: dict[str, float] | None = None,
) -> Attributable | tuple[Attributable, np.ndarray]:
    if robust:
        attrib, cov = build_attributable_studentt(
            observations,
            epoch=epoch,
            nu=nu,
            max_iter=max_iter,
            tol=tol,
            site_kappas=site_kappas,
        )
        if return_s_sdot:
            s0, sdot = fit_unit_vector_rate(observations, epoch)
            return (attrib, cov, s0, sdot) if return_cov else (attrib, s0, sdot)
        return (attrib, cov) if return_cov else attrib
    s0, sdot = fit_unit_vector_rate(observations, epoch)
    attrib = _attrib_from_s_sdot(s0, sdot)
    if return_s_sdot:
        return (attrib, np.zeros((4, 4), dtype=float), s0, sdot) if return_cov else (attrib, s0, sdot)
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
    earth_bary, earth_bary_vel = _body_posvel_km_single("earth", epoch)
    sun_bary, sun_bary_vel = _body_posvel_km_single("sun", epoch)
    earth_helio = earth_bary - sun_bary
    earth_vel_helio = earth_bary_vel - sun_bary_vel

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


def _attrib_rho_from_state(
    state: np.ndarray,
    obs: Observation,
    epoch: Time,
) -> tuple[Attributable, float, float]:
    earth_bary, earth_bary_vel = _body_posvel_km_single("earth", epoch)
    sun_bary, sun_bary_vel = _body_posvel_km_single("sun", epoch)
    earth_helio = earth_bary - sun_bary
    earth_vel_helio = earth_bary_vel - sun_bary_vel
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
