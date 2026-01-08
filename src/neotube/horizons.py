from __future__ import annotations

import numpy as np
from astropy.time import Time
from astroquery.jplhorizons import Horizons

from .constants import AU_KM, DAY_S


def normalize_horizons_id(raw: str) -> str:
    s = raw.strip()
    if s.isdigit():
        n = int(s)
        if 1 <= n < 2000000:
            return str(2000000 + n)
    return s


def fetch_horizons_state(
    target: str,
    epoch: Time,
    *,
    location: str = "@sun",
    refplane: str = "earth",
) -> np.ndarray:
    obj = Horizons(id=normalize_horizons_id(target), location=location, epochs=epoch.jd)
    vec = obj.vectors(refplane=refplane)
    row = vec[0]
    au_km = AU_KM
    au_per_day_to_km_s = AU_KM / DAY_S
    return np.array(
        [
            float(row["x"]) * au_km,
            float(row["y"]) * au_km,
            float(row["z"]) * au_km,
            float(row["vx"]) * au_per_day_to_km_s,
            float(row["vy"]) * au_per_day_to_km_s,
            float(row["vz"]) * au_per_day_to_km_s,
        ],
        dtype=float,
    )


def fetch_horizons_states(
    target: str,
    epochs: Time,
    *,
    location: str = "@sun",
    refplane: str = "earth",
) -> tuple[np.ndarray, np.ndarray]:
    obj = Horizons(id=normalize_horizons_id(target), location=location, epochs=epochs.jd)
    vec = obj.vectors(refplane=refplane)
    au_km = AU_KM
    au_per_day_to_km_s = AU_KM / DAY_S
    pos = np.column_stack([vec["x"], vec["y"], vec["z"]]).astype(float) * au_km
    vel = np.column_stack([vec["vx"], vec["vy"], vec["vz"]]).astype(float) * au_per_day_to_km_s
    return pos, vel
