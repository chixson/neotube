from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Sequence

import numpy as np
from astropy import units as u
from astropy.coordinates import (
    SkyCoord,
    SphericalRepresentation,
    get_body_barycentric_posvel,
)
from astropy.time import Time, TimeDelta
from scipy.integrate import solve_ivp

__all__ = [
    "propagate_state",
    "predict_radec",
    "ReplicaCloud",
    "propagate_replicas",
]

GM_SUN = 1.32712440018e11  # km^3 / s^2

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


def _heliocentric_position(body: str, epoch: Time) -> np.ndarray:
    """Return heliocentric (km) position for the requested body."""
    body_pos, _ = get_body_barycentric_posvel(body, epoch)
    sun_pos, _ = get_body_barycentric_posvel("sun", epoch)
    vec = body_pos.xyz - sun_pos.xyz
    return vec.to(u.km).value


def _acceleration(r: np.ndarray, epoch: Time, perturbers: Sequence[str]) -> np.ndarray:
    """Compute heliocentric acceleration on a test particle."""
    norm = np.linalg.norm(r)
    acc = -GM_SUN * r / (norm**3 + 1e-18)
    for body in perturbers:
        gm = PLANET_GMS.get(body.lower())
        if gm is None:
            continue
        body_pos = _heliocentric_position(body, epoch)
        diff = r - body_pos
        acc += -gm * diff / (np.linalg.norm(diff)**3 + 1e-18)
    return acc


def _rhs(t: float, y: np.ndarray, epoch: Time, perturbers: Sequence[str]) -> np.ndarray:
    """Ordinary differential equation for heliocentric motion."""
    current_epoch = epoch + TimeDelta(t * u.s)
    r = y[:3]
    acc = _acceleration(r, current_epoch, perturbers)
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
    results = []
    for target in targets:
        delta = (target - epoch).to(u.s).value
        if abs(delta) < 1e-8:
            results.append(state.copy())
            continue
        sol = solve_ivp(
            lambda t, y: _rhs(t, y, epoch, perturbers),
            (0.0, delta),
            state,
            max_step=max_step,
            rtol=1e-9,
            atol=1e-12,
            method="RK45",
        )
        if not sol.success:
            raise RuntimeError(f"Propagation to {target.iso} failed: {sol.message}")
        results.append(sol.y[:, -1])
    return np.stack(results, axis=0)


def predict_radec(state: np.ndarray, epoch: Time) -> tuple[float, float]:
    """Compute topocentric RA/Dec (degrees) from a heliocentric state."""
    obj_pos = state[:3]
    earth_pos = _heliocentric_position("earth", epoch)
    vector = obj_pos - earth_pos
    coord = SkyCoord(
        x=vector[0] * u.km,
        y=vector[1] * u.km,
        z=vector[2] * u.km,
        representation_type="cartesian",
        frame="icrs",
    )
    sph = coord.represent_as(SphericalRepresentation)
    return float(sph.lon.deg), float(sph.lat.deg)


@dataclass
class ReplicaCloud:
    epoch: Time
    states: np.ndarray  # shape (6, N)


def propagate_replicas(
    cloud: ReplicaCloud,
    targets: Iterable[Time],
    perturbers: Sequence[str],
) -> list[np.ndarray]:
    """Return propagated states (shape (6,N)) for each target epoch."""
    propagated = []
    for target in targets:
        per_epoch: list[np.ndarray] = []
        for col in cloud.states.T:
            state_at = propagate_state(col, cloud.epoch, [target], perturbers=perturbers)[0]
            per_epoch.append(state_at)
        propagated.append(np.stack(per_epoch, axis=1))
    return propagated
