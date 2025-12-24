from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Iterable, Sequence, Optional

import numpy as np
from astropy import units as u
from astropy.coordinates import (
    EarthLocation,
    GCRS,
    SkyCoord,
    SphericalRepresentation,
    get_body_barycentric_posvel,
)
from astropy.time import Time, TimeDelta
from concurrent.futures import ProcessPoolExecutor
from scipy.integrate import solve_ivp
import os

from .sites import get_site_location

__all__ = [
    "propagate_state",
    "propagate_state_kepler",
    "propagate_state_sun",
    "predict_radec",
    "predict_radec_batch",
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

# Cache for repeated ephemeris lookups (per process).
_BULK_EPHEM_CACHE: dict[tuple, np.ndarray] = {}


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


def _bulk_heliocentric_positions(body: str, times: Time, sun_coord: SkyCoord) -> np.ndarray:
    """Return heliocentric (km) positions for body over multiple epochs."""
    key = (body.lower(), _times_cache_key(times))
    cached = _BULK_EPHEM_CACHE.get(key)
    if cached is not None:
        return cached
    body_pos, _ = get_body_barycentric_posvel(body, times)
    vec = body_pos.xyz - sun_coord.xyz
    out = vec.to(u.km).value.T
    _BULK_EPHEM_CACHE[key] = out
    return out


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
) -> tuple[float, float]:
    """Compute topocentric RA/Dec (degrees) from a heliocentric state."""
    ra, dec = predict_radec_batch(
        state[np.newaxis, :],
        (epoch,),
        site_codes=(None if site_code is None else (site_code,)),
    )
    return float(ra[0]), float(dec[0])


def _site_offsets(
    epochs: Sequence[Time], site_codes: Sequence[str | None] | None
) -> np.ndarray:
    if site_codes is None:
        return np.zeros((len(epochs), 3), dtype=float)
    if len(site_codes) != len(epochs):
        raise ValueError("site_codes must match epochs length")
    offsets = np.zeros((len(epochs), 3), dtype=float)
    for idx, (code, time) in enumerate(zip(site_codes, epochs)):
        if code is None:
            continue
        loc = get_site_location(code)
        if loc is None:
            continue
        gcrs = loc.get_gcrs(obstime=time)
        offsets[idx] = gcrs.cartesian.xyz.to(u.km).value
    return offsets


def predict_radec_batch(
    states: np.ndarray | Sequence[np.ndarray],
    epochs: Sequence[Time],
    site_codes: Sequence[str | None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized RA/Dec computation for many heliocentric states/times."""
    if len(epochs) == 0:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)

    states_arr = np.asarray(states, dtype=float)
    if states_arr.ndim == 1:
        states_arr = states_arr[np.newaxis, :]
    if states_arr.shape[1] != 6:
        raise ValueError("Expected states with 6 components (r/v).")

    time_array = Time(epochs)
    earth_pos, _ = get_body_barycentric_posvel("earth", time_array)
    sun_pos, _ = get_body_barycentric_posvel("sun", time_array)
    earth_helio = (earth_pos.xyz - sun_pos.xyz).to(u.km).value
    if earth_helio.shape[1] != len(epochs):
        earth_helio = earth_helio[:, : len(epochs)]

    obj_pos = states_arr[:, :3]
    if len(epochs) != obj_pos.shape[0]:
        raise ValueError("Number of states and epochs must match.")
    site_offsets = _site_offsets(epochs, site_codes)
    vectors = obj_pos - earth_helio.T - site_offsets

    coord = SkyCoord(
        x=vectors[:, 0] * u.km,
        y=vectors[:, 1] * u.km,
        z=vectors[:, 2] * u.km,
        representation_type="cartesian",
        frame="icrs",
    )
    sph = coord.represent_as(SphericalRepresentation)
    return np.array(sph.lon.deg), np.array(sph.lat.deg)


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
