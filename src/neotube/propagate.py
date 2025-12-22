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


@dataclass
class PerturberTable:
    times: np.ndarray
    positions: dict[str, np.ndarray]

    def position(self, body: str, t: float) -> np.ndarray | None:
        arr = self.positions.get(body.lower())
        if arr is None:
            return None
        # interpolate each dimension
        return np.array(
            [
                np.interp(t, self.times, arr[:, dim])
                for dim in range(3)
            ],
            dtype=float,
        )


def _bulk_heliocentric_positions(body: str, times: Time, sun_coord: SkyCoord) -> np.ndarray:
    """Return heliocentric (km) positions for body over multiple epochs."""
    body_pos, _ = get_body_barycentric_posvel(body, times)
    vec = body_pos.xyz - sun_coord.xyz
    return vec.to(u.km).value.T


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

    positions = {}
    for body in perturbers:
        body_pos = _bulk_heliocentric_positions(body, times, sun_pos)
        positions[body.lower()] = body_pos

    return PerturberTable(times=grid, positions=positions)


def _acceleration(r: np.ndarray, perturber_table: PerturberTable, t: float) -> np.ndarray:
    """Compute heliocentric acceleration on a test particle."""
    norm = np.linalg.norm(r)
    acc = -GM_SUN * r / (norm**3 + 1e-18)
    for body, arr in perturber_table.positions.items():
        gm = PLANET_GMS.get(body)
        if gm is None:
            continue
        body_pos = perturber_table.position(body, t)
        if body_pos is None:
            continue
        diff = r - body_pos
        acc += -gm * diff / (np.linalg.norm(diff)**3 + 1e-18)
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
