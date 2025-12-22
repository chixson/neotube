from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Iterable, Sequence, Optional

import numpy as np
from astropy import units as u
from astropy.coordinates import (
    SkyCoord,
    SphericalRepresentation,
    get_body_barycentric_posvel,
)
from astropy.time import Time, TimeDelta
from concurrent.futures import ProcessPoolExecutor
from scipy.integrate import solve_ivp
import os

__all__ = [
    "propagate_state",
    "propagate_state_kepler",
    "propagate_state_sun",
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

def _stumpff_C2(z: float) -> float:
    if z > 1e-8:
        s = np.sqrt(z)
        return float((1.0 - np.cos(s)) / z)
    if z < -1e-8:
        s = np.sqrt(-z)
        return float((np.cosh(s) - 1.0) / (-z))
    # series expansion near 0
    return 0.5


def _stumpff_C3(z: float) -> float:
    if z > 1e-8:
        s = np.sqrt(z)
        return float((s - np.sin(s)) / (s**3))
    if z < -1e-8:
        s = np.sqrt(-z)
        return float((np.sinh(s) - s) / (s**3))
    # series expansion near 0
    return 1.0 / 6.0


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
    r0_norm = float(np.linalg.norm(r0))
    v0_sq = float(np.dot(v0, v0))
    vr0 = float(np.dot(r0, v0)) / (r0_norm + 1e-30)
    alpha = 2.0 / (r0_norm + 1e-30) - v0_sq / mu_km3_s2
    sqrt_mu = np.sqrt(mu_km3_s2)

    out = np.empty((len(targets), 6), dtype=float)
    for i, t in enumerate(targets):
        dt = float((t - epoch).to(u.s).value)
        if abs(dt) < 1e-9:
            out[i] = state
            continue

        # initial guess for chi
        if alpha > 1e-12:
            chi = sqrt_mu * dt * alpha
        else:
            chi = np.sign(dt) * sqrt_mu * abs(dt) * 1e-3

        # Newton solve for universal anomaly chi
        for _ in range(max_iter):
            z = alpha * chi * chi
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
            chi -= step
            if abs(step) < tol:
                break

        z = alpha * chi * chi
        C2 = _stumpff_C2(z)
        C3 = _stumpff_C3(z)
        f = 1.0 - (chi * chi / (r0_norm + 1e-30)) * C2
        g = dt - (chi**3 / sqrt_mu) * C3
        r = f * r0 + g * v0
        r_norm = float(np.linalg.norm(r))
        gdot = 1.0 - (chi * chi / (r_norm + 1e-30)) * C2
        fdot = (sqrt_mu / ((r_norm + 1e-30) * (r0_norm + 1e-30))) * chi * (z * C3 - 1.0)
        v = fdot * r0 + gdot * v0
        out[i, :3] = r
        out[i, 3:] = v
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
    order = np.argsort(offsets)
    sorted_offsets = offsets[order]
    if sorted_offsets[0] < 0:
        raise ValueError("Backward propagation is not supported yet.")
    if sorted_offsets[-1] < 1e-9:
        return np.tile(state, (len(targets), 1))

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        r = y[:3]
        norm = np.linalg.norm(r)
        acc = -mu_km3_s2 * r / (norm**3 + 1e-18)
        return np.concatenate((y[3:], acc))

    sol = solve_ivp(
        rhs,
        (0.0, sorted_offsets[-1]),
        state,
        t_eval=sorted_offsets,
        max_step=max_step,
        rtol=1e-9,
        atol=1e-12,
        method="DOP853",
    )
    if not sol.success:
        raise RuntimeError(f"Sun-only propagation failed: {sol.message}")
    results_sorted = sol.y.T
    results = np.empty_like(results_sorted)
    results[order] = results_sorted
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


def _bulk_heliocentric_positions(body: str, times: Time, sun_coord: SkyCoord) -> np.ndarray:
    """Return heliocentric (km) positions for body over multiple epochs."""
    body_pos, _ = get_body_barycentric_posvel(body, times)
    vec = body_pos.xyz - sun_coord.xyz
    return vec.to(u.km).value.T


def _heliocentric_position(body: str, epoch: Time) -> np.ndarray:
    """Return heliocentric (km) position for a single epoch."""
    body_pos, _ = get_body_barycentric_posvel(body, epoch)
    sun_pos, _ = get_body_barycentric_posvel("sun", epoch)
    vec = body_pos.xyz - sun_pos.xyz
    return vec.to(u.km).value


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
    sun_pos, _ = get_body_barycentric_posvel("sun", times)

    bodies = []
    position_list = []
    gms = []
    for body in perturbers:
        gm = PLANET_GMS.get(body.lower())
        if gm is None:
            continue
        body_pos = _bulk_heliocentric_positions(body, times, sun_pos)
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
    order = np.argsort(offsets)
    sorted_offsets = offsets[order]
    if sorted_offsets[0] < 0:
        raise ValueError("Backward propagation is not supported yet.")

    if sorted_offsets[-1] < 1e-9:
        return np.tile(state, (len(targets), 1))

    perturber_table = _build_perturber_table(epoch, perturbers, sorted_offsets[-1], max_step)
    sol = solve_ivp(
        lambda t, y: _rhs(t, y, perturber_table),
        (0.0, sorted_offsets[-1]),
        state,
        t_eval=sorted_offsets,
        max_step=max_step,
        rtol=1e-9,
        atol=1e-12,
        method="DOP853",
    )
    if not sol.success:
        raise RuntimeError(f"Propagation to targets failed: {sol.message}")

    results_sorted = sol.y.T
    results = np.empty_like(results_sorted)
    results[order] = results_sorted
    return results


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
