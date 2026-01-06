import math

import numpy as np
from astropy.time import Time

from neotube.models import Observation
from neotube.propagate import _site_states, predict_radec_from_epoch


def _ang_sep_arcsec(ra1, dec1, ra2, dec2):
    ra1r, ra2r = math.radians(ra1), math.radians(ra2)
    d1r, d2r = math.radians(dec1), math.radians(dec2)
    cosang = math.sin(d1r) * math.sin(d2r) + math.cos(d1r) * math.cos(d2r) * math.cos(ra1r - ra2r)
    cosang = max(-1.0, min(1.0, cosang))
    return math.degrees(math.acos(cosang)) * 3600.0


def test_q1_geometry_matches_fallback():
    epoch = Time("2024-01-01T00:00:00", scale="tdb")
    times = Time(
        [
            "2024-01-01T00:00:00",
            "2024-01-01T01:00:00",
            "2024-01-01T02:00:00",
        ],
        scale="tdb",
    )
    state = np.array([1.2e8, -0.4e8, 0.9e8, -2.5, 4.0, 1.1], dtype=float)

    obs_geo = [
        Observation(
            time=t,
            ra_deg=10.0,
            dec_deg=20.0,
            sigma_arcsec=0.5,
            site="500",
            observer_pos_km=None,
        )
        for t in times
    ]

    site_pos_km, _ = _site_states(times, ["500"] * len(times), allow_unknown_site=True)
    obs_fallback = [
        Observation(
            time=t,
            ra_deg=10.0,
            dec_deg=20.0,
            sigma_arcsec=0.5,
            site=None,
            observer_pos_km=site_pos_km[i],
        )
        for i, t in enumerate(times)
    ]

    ra_geo, dec_geo = predict_radec_from_epoch(
        state,
        epoch,
        obs_geo,
        perturbers=(),
        max_step=3600.0,
        use_kepler=True,
        full_physics=False,
        light_time_iters=1,
    )
    ra_fb, dec_fb = predict_radec_from_epoch(
        state,
        epoch,
        obs_fallback,
        perturbers=(),
        max_step=3600.0,
        use_kepler=True,
        full_physics=False,
        light_time_iters=1,
    )

    sep = [
        _ang_sep_arcsec(float(r1), float(d1), float(r2), float(d2))
        for r1, d1, r2, d2 in zip(ra_geo, dec_geo, ra_fb, dec_fb)
    ]
    assert float(np.median(sep)) < 0.01
