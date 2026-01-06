from __future__ import annotations

from dataclasses import dataclass, field

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


@dataclass
class OrbitPosterior:
    epoch: Time
    state: np.ndarray  # shape (6,)
    cov: np.ndarray  # shape (6,6)
    residuals: np.ndarray  # length 2n (arcsec)
    rms_arcsec: float
    converged: bool
    seed_rms_arcsec: float | None = None
    fit_scale: float = 1.0
    nu: float | None = None
    site_kappas: dict[str, float] = field(default_factory=dict)
