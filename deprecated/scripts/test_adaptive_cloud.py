#!/usr/bin/env python3
"""Smoke test for adaptive cloud sizing on synthetic observations."""
from __future__ import annotations

import numpy as np
from astropy.time import Time, TimeDelta

from neotube.fit_smc import sequential_fit_replicas
from neotube.models import Observation


def main() -> int:
    t0 = Time("2025-01-01T00:00:00", scale="utc")
    obs = [
        Observation(time=t0, ra_deg=10.0, dec_deg=10.0, sigma_arcsec=0.5, site=None),
        Observation(
            time=t0 + TimeDelta(10.0, format="sec"),
            ra_deg=10.0003,
            dec_deg=10.0001,
            sigma_arcsec=0.5,
            site=None,
        ),
        Observation(
            time=t0 + TimeDelta(20.0, format="sec"),
            ra_deg=10.0006,
            dec_deg=10.0002,
            sigma_arcsec=0.5,
            site=None,
        ),
    ]
    cloud = sequential_fit_replicas(
        obs,
        n_particles=200,
        rho_min_au=0.1,
        rho_max_au=2.0,
        rhodot_max_km_s=30.0,
        auto_grow=True,
        auto_n_max=400,
        auto_n_add=100,
        auto_ess_target=50.0,
    )
    print("states:", cloud.states.shape, "weights:", cloud.weights.shape)
    print("meta keys:", sorted(cloud.metadata.keys()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
