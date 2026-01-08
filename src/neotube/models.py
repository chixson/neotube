from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from astropy.time import Time


@dataclass
class Observation:
    time: Time
    ra_deg: float
    dec_deg: float
    sigma_arcsec: float
    site: str | None = None
    observer_pos_km: np.ndarray | None = None
    mag: float | None = None
    sigma_mag: float | None = None


@dataclass(frozen=True)
class Attributable:
    ra_deg: float
    dec_deg: float
    ra_dot_deg_per_day: float
    dec_dot_deg_per_day: float


@dataclass
class ReplicaCloud:
    states: np.ndarray
    weights: np.ndarray
    epoch: Time
    metadata: dict[str, Any]


class StateVector(np.ndarray):
    """Numpy-backed state vector with .pos/.vel/.epoch attributes."""

    def __new__(cls, arr: np.ndarray, epoch: object):
        obj = np.asarray(arr, dtype=float).view(cls)
        obj.epoch = epoch
        return obj

    def __array_finalize__(self, obj: object) -> None:
        if obj is None:
            return
        self.epoch = getattr(obj, "epoch", None)

    @property
    def pos(self) -> np.ndarray:
        return np.asarray(self[:3], dtype=float)

    @property
    def vel(self) -> np.ndarray:
        return np.asarray(self[3:6], dtype=float)
