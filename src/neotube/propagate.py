from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from astropy import units as u
from astropy.coordinates import ICRS, CartesianDifferential, CartesianRepresentation, SkyCoord, get_body_barycentric_posvel, solar_system_ephemeris
from astropy.time import Time

from .constants import GM_SUN
from .sites import get_site_location


def _body_posvel_km(body: str, times: Time, ephemeris: str = "de432s") -> tuple[np.ndarray, np.ndarray]:
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


def _stumpff_C2(z: float) -> float:
    if z > 1e-8:
        s = math.sqrt(z)
        if s > 1e6:
            raise OverflowError("stumpff C2 input s too large")
        return float((1.0 - math.cos(s)) / z)
    if z < -1e-8:
        s = math.sqrt(-z)
        if s > 50.0:
            exp_s = math.exp(min(s, 700.0))
            cosh_s = 0.5 * exp_s
            return float((cosh_s - 1.0) / (s * s))
        return float((math.cosh(s) - 1.0) / (s * s))
    return 0.5


def _stumpff_C3(z: float) -> float:
    if z > 1e-8:
        s = math.sqrt(z)
        if s > 1e6:
            raise OverflowError("stumpff C3 input s too large")
        return float((s - math.sin(s)) / (s**3))
    if z < -1e-8:
        s = math.sqrt(-z)
        if s > 50.0:
            exp_s = math.exp(min(s, 700.0))
            sinh_s = 0.5 * exp_s
            return float((sinh_s - s) / (s**3))
        return float((math.sinh(s) - s) / (s**3))
    return 1.0 / 6.0


def _propagate_state_kepler_single(
    r0: np.ndarray,
    v0: np.ndarray,
    dt: float,
    mu_km3_s2: float,
    max_iter: int,
    tol: float,
) -> np.ndarray:
    r0_norm = float(np.linalg.norm(r0))
    v0_sq = float(np.dot(v0, v0))
    vr0 = float(np.dot(r0, v0)) / (r0_norm + 1e-30)
    alpha = 2.0 / (r0_norm + 1e-30) - v0_sq / mu_km3_s2
    sqrt_mu = math.sqrt(mu_km3_s2)

    if alpha > 1e-12:
        chi = sqrt_mu * dt * alpha
    else:
        if alpha < -1e-12:
            sqrt_neg_alpha = math.sqrt(-1.0 / alpha)
            term = r0_norm * vr0 + math.copysign(1.0, dt) * sqrt_mu * sqrt_neg_alpha * (
                1.0 - r0_norm * alpha
            )
            if term != 0.0:
                arg = (-2.0 * mu_km3_s2 * alpha * dt) / term
                if arg > 0.0:
                    chi = math.copysign(1.0, dt) * sqrt_neg_alpha * math.log(arg)
                else:
                    chi = math.copysign(1.0, dt) * sqrt_mu * abs(dt) / max(r0_norm, 1e-6)
            else:
                chi = math.copysign(1.0, dt) * sqrt_mu * abs(dt) / max(r0_norm, 1e-6)
        else:
            chi = math.copysign(1.0, dt) * sqrt_mu * abs(dt) / max(r0_norm, 1e-6)

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
            (r0_norm * vr0 / sqrt_mu) * chi * (1.0 - alpha * chi * chi * C3)
            + (1.0 - alpha * r0_norm) * chi * chi * C2
            + r0_norm
        )
        delta = F / max(1e-16, dF)
        if chi != 0.0 and abs(delta) > 0.5 * abs(chi):
            delta = math.copysign(0.5 * abs(chi), delta)
        chi -= delta
        if abs(delta) < tol:
            break
    else:
        raise RuntimeError("Kepler solver failed to converge")

    z = alpha * chi * chi
    C2 = _stumpff_C2(z)
    C3 = _stumpff_C3(z)

    f = 1.0 - (chi * chi / r0_norm) * C2
    g = dt - (chi**3 / sqrt_mu) * C3
    r = f * r0 + g * v0
    r_norm = float(np.linalg.norm(r))
    if r_norm <= 0.0:
        raise RuntimeError("Kepler propagation produced non-positive radius")

    fdot = (sqrt_mu / (r_norm * r0_norm)) * (alpha * chi**3 * C3 - chi)
    gdot = 1.0 - (chi * chi / r_norm) * C2
    v = fdot * r0 + gdot * v0
    return np.hstack([r, v])


def propagate_state_kepler(
    state: np.ndarray,
    epoch: Time,
    target: Time,
    *,
    mu_km3_s2: float = GM_SUN,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> np.ndarray:
    dt = float((target.tdb - epoch.tdb).to_value("s"))
    r0 = np.asarray(state[:3], dtype=float)
    v0 = np.asarray(state[3:6], dtype=float)
    return _propagate_state_kepler_single(r0, v0, dt, mu_km3_s2, max_iter, tol)


def propagate_state(
    state: np.ndarray,
    epoch: Time,
    target: Time,
    *,
    mu_km3_s2: float = GM_SUN,
) -> np.ndarray:
    return propagate_state_kepler(state, epoch, target, mu_km3_s2=mu_km3_s2)


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
            raise ValueError(f"Missing site code at {time.isot}")
        loc = get_site_location(code)
        if loc is None:
            if allow_unknown_site:
                continue
            raise ValueError(f"Site code {code} not found")
        gcrs = loc.get_gcrs(obstime=time)
        positions[idx] = gcrs.cartesian.xyz.to(u.km).value
        if gcrs.cartesian.differentials:
            velocities[idx] = gcrs.cartesian.differentials["s"].d_xyz.to(u.km / u.s).value
    return positions, velocities


def _icrs_from_horizons_au(
    x: float,
    y: float,
    z: float,
    vx: float,
    vy: float,
    vz: float,
) -> tuple[np.ndarray, np.ndarray]:
    rep = CartesianRepresentation(x * u.au, y * u.au, z * u.au)
    diff = CartesianDifferential(vx * u.au / u.day, vy * u.au / u.day, vz * u.au / u.day)
    coord = SkyCoord(rep.with_differentials(diff), frame=ICRS())
    pos = coord.cartesian.xyz.to(u.km).value
    vel = coord.cartesian.differentials["s"].d_xyz.to(u.km / u.s).value
    return pos, vel
