from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy.time import Time


@dataclass
class Observation:
    time: Time
    ra_deg: float
    dec_deg: float
    sigma_arcsec: float
    site: str | None = None


@dataclass
class OrbitPosterior:
    epoch: Time
    state: np.ndarray  # shape (6,)
    cov: np.ndarray  # shape (6,6)
    residuals: np.ndarray  # length 2n (arcsec)
    rms_arcsec: float
    converged: bool
    seed_rms_arcsec: float | None = None
