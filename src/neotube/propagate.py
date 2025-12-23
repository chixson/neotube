from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np
from astropy import constants as const
from astropy import units as u
from astropy.coordinates import SkyCoord, solar_system_ephemeris, get_body_barycentric_posvel, CartesianRepresentation
from astropy.time import Time
from scipy.integrate import solve_ivp


GRAVITATIONAL_PARAMS = {
    "sun": 1.32712440018e11,
    "mercury": 2.2032e4,
    "venus": 3.24858599e5,
    "earth": 3.986004418e5,
    "moon": 4.9048695e3,
    "mars": 4.282837e4,
    "jupiter": 1.26686534e8,
    "saturn": 3.7931187e7,
    "uranus": 5.793939e6,
    "neptune": 6.835e6,
}


@dataclass(frozen=True)
class OrbitalState:
    epoch: Time
    rv: np.ndarray  # shape (6,) r km and v km/s


solar_system_ephemeris.set("de432s")


def _body_position(name: str, time: Time) -> np.ndarray:
    posvel = get_body_barycentric_posvel(name, time)
    coord = posvel[0] if isinstance(posvel, tuple) else posvel
    return coord.x.to("km").value, coord.y.to("km").value, coord.z.to("km").value


def propagate_kepler(state: OrbitalState, target_time: Time, perturbers: Sequence[str] | None = None) -> OrbitalState:
    dt_seconds = (target_time.tcb.jd - state.epoch.tcb.jd) * 86400.0
    if abs(dt_seconds) < 1e-9:
        return state
    if perturbers is None:
        perturbers = ["earth", "moon", "mars", "jupiter", "saturn", "uranus", "neptune"]

    km_to_au = u.km.to(u.AU)
    scaled_state = state.rv.copy()
    scaled_state[:3] *= km_to_au
    scaled_state[3:] *= km_to_au * 86400.0

    def rhs(t, y):
        r = y[:3]
        norm_r = np.linalg.norm(r)
        if not np.isfinite(norm_r) or norm_r == 0:
            raise RuntimeError("Invalid position during propagation")
        acc = -GRAVITATIONAL_PARAMS["sun"] * r / norm_r**3
        current_jd = state.epoch.tcb.jd + t / 86400.0
        current_time = Time(current_jd, format="jd", scale="tdb")
        for body in perturbers:
            mu = GRAVITATIONAL_PARAMS.get(body)
            if mu is None:
                continue
            bx, by, bz = _body_position(body, current_time)
            bvec = np.array([bx * km_to_au, by * km_to_au, bz * km_to_au])
            delta = bvec - r
            acc += mu * delta / np.linalg.norm(delta) ** 3
        return np.concatenate([y[3:], acc])

    t_eval = [dt_seconds]
    max_step = min(abs(dt_seconds), 600.0)
    sol = solve_ivp(
        rhs,
        (0.0, dt_seconds),
        scaled_state,
        t_eval=t_eval,
        rtol=1e-10,
        atol=np.array([1e-9, 1e-9, 1e-9, 1e-11, 1e-11, 1e-11]),
        max_step=max_step,
    )
    if not sol.success:
        sol = solve_ivp(
            rhs,
            (0.0, dt_seconds),
            scaled_state,
            t_eval=t_eval,
            method="DOP853",
            rtol=1e-9,
            atol=np.array([1e-9, 1e-9, 1e-9, 1e-11, 1e-11, 1e-11]),
            max_step=max_step / 2,
        )
    if not sol.success:
        raise RuntimeError(f"Propagation failed: {sol.message}")

    final = sol.y[:, -1].copy()
    final[:3] /= km_to_au
    final[3:] /= km_to_au * 86400.0
    return OrbitalState(epoch=target_time, rv=final)

def state_to_radec(state: OrbitalState) -> tuple[float, float]:
    r = state.rv[:3]
    coord = SkyCoord(CartesianRepresentation(r[0] * u.km, r[1] * u.km, r[2] * u.km), frame="icrs")
    return float(coord.ra.deg), float(coord.dec.deg)

@lru_cache(maxsize=10)
def get_horizons_state(target: str, epoch: Time) -> OrbitalState:
    from astroquery.jplhorizons import Horizons

    obj = Horizons(id=target, location="500", epochs=epoch.jd)
    vect = obj.vectors()
    r = np.array([float(vect["x"][0]), float(vect["y"][0]), float(vect["z"][0])])
    v = np.array([float(vect["vx"][0]), float(vect["vy"][0]), float(vect["vz"][0])])
    rv = np.concatenate([r, v])
    return OrbitalState(epoch=Time(vect["datetime_jd"][0], format="jd"), rv=rv)
