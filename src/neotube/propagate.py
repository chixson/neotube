from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from astropy import units as u
from astropy.coordinates import (
    ICRS,
    CartesianDifferential,
    CartesianRepresentation,
    SkyCoord,
    get_body_barycentric_posvel,
    solar_system_ephemeris,
)
from astropy.time import Time

from .constants import GM_SUN
from .horizons import fetch_horizons_states
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


def _body_posvel_helio_km(
    body: str,
    times: Time,
    ephemeris: str = "de432s",
) -> tuple[np.ndarray, np.ndarray]:
    pos_km, vel_km_s = _body_posvel_km(body, times, ephemeris=ephemeris)
    sun_pos, sun_vel = _body_posvel_km("sun", times, ephemeris=ephemeris)
    return pos_km - sun_pos, vel_km_s - sun_vel


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


@dataclass(frozen=True)
class Perturber:
    name: str
    gm_km3_s2: float
    position_km: np.ndarray
    times_sec: np.ndarray


# Approximate GM values in km^3/s^2.
PLANET_GM = {
    "mercury": 22032.080,
    "venus": 324858.592,
    "earth": 398600.435436,
    "mars": 42828.375214,
    "jupiter": 126686534.0,
    "saturn": 37940626.0,
    "uranus": 5794548.0,
    "neptune": 6836527.0,
}

# GM for Ceres (km^3/s^2). Reference: Dawn mission value ~62.628.
CERES_GM = 62.628


def build_perturbers(
    epoch: Time,
    target: Time,
    *,
    step_sec: float,
    ephemeris: str = "de432s",
    include_planets: Iterable[str] | None = None,
    include_asteroids: Iterable[str] | None = None,
) -> list[Perturber]:
    dt = float((target.tdb - epoch.tdb).to_value("s"))
    if dt == 0.0:
        raise ValueError("Zero propagation interval for perturber table.")
    step = abs(float(step_sec))
    n_steps = int(math.ceil(abs(dt) / step))
    times_sec = np.linspace(0.0, dt, n_steps + 1)
    times = epoch.tdb + times_sec * u.s

    perturbers: list[Perturber] = []
    planets = include_planets or PLANET_GM.keys()
    for name in planets:
        pos_helio, _ = _body_posvel_helio_km(name, times, ephemeris=ephemeris)
        perturbers.append(
            Perturber(
                name=name,
                gm_km3_s2=PLANET_GM[name],
                position_km=pos_helio,
                times_sec=times_sec,
            )
        )

    asteroids = include_asteroids or ["ceres"]
    for name in asteroids:
        if name.lower() == "ceres":
            pos, _ = fetch_horizons_states("1", times, location="@sun", refplane="earth")
            perturbers.append(
                Perturber(
                    name="ceres",
                    gm_km3_s2=CERES_GM,
                    position_km=pos,
                    times_sec=times_sec,
                )
            )
    return perturbers


def _interp_pos(times_sec: np.ndarray, pos: np.ndarray, t: float) -> np.ndarray:
    if t <= times_sec[0]:
        return pos[0]
    if t >= times_sec[-1]:
        return pos[-1]
    idx = int(np.searchsorted(times_sec, t))
    t0 = times_sec[idx - 1]
    t1 = times_sec[idx]
    w = (t - t0) / max(1e-12, t1 - t0)
    return (1.0 - w) * pos[idx - 1] + w * pos[idx]


def _accel_nbody(
    r: np.ndarray,
    t: float,
    perturbers: Sequence[Perturber],
) -> np.ndarray:
    rnorm = float(np.linalg.norm(r))
    if rnorm <= 0.0:
        raise RuntimeError("Non-positive radius in n-body acceleration.")
    acc = -GM_SUN * r / (rnorm**3)
    for p in perturbers:
        r_p = _interp_pos(p.times_sec, p.position_km, t)
        dr = r_p - r
        dr_norm = float(np.linalg.norm(dr))
        if dr_norm <= 0.0:
            continue
        acc += p.gm_km3_s2 * (dr / (dr_norm**3) - r_p / (np.linalg.norm(r_p) ** 3))
    return acc


def propagate_state_nbody(
    state: np.ndarray,
    epoch: Time,
    target: Time,
    *,
    step_sec: float = 3600.0,
    perturbers: Sequence[Perturber] | None = None,
) -> np.ndarray:
    dt = float((target.tdb - epoch.tdb).to_value("s"))
    if dt == 0.0:
        return np.asarray(state, dtype=float).copy()
    if perturbers is None:
        perturbers = build_perturbers(epoch, target, step_sec=step_sec)
    n_steps = int(math.ceil(abs(dt) / abs(step_sec)))
    h = dt / n_steps
    y = np.asarray(state, dtype=float).copy()
    t = 0.0
    for _ in range(n_steps):
        r = y[:3]
        v = y[3:6]
        a1 = _accel_nbody(r, t, perturbers)
        k1 = np.hstack([v, a1])

        r2 = r + 0.5 * h * k1[:3]
        v2 = v + 0.5 * h * k1[3:]
        a2 = _accel_nbody(r2, t + 0.5 * h, perturbers)
        k2 = np.hstack([v2, a2])

        r3 = r + 0.5 * h * k2[:3]
        v3 = v + 0.5 * h * k2[3:]
        a3 = _accel_nbody(r3, t + 0.5 * h, perturbers)
        k3 = np.hstack([v3, a3])

        r4 = r + h * k3[:3]
        v4 = v + h * k3[3:]
        a4 = _accel_nbody(r4, t + h, perturbers)
        k4 = np.hstack([v4, a4])

        y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += h
    return y


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
        icrs = gcrs.transform_to(ICRS())
        earth_pos, earth_vel = _body_posvel_km_single("earth", time)
        positions[idx] = icrs.cartesian.xyz.to(u.km).value - earth_pos
        if icrs.cartesian.differentials:
            velocities[idx] = icrs.cartesian.differentials["s"].d_xyz.to(u.km / u.s).value - earth_vel
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
