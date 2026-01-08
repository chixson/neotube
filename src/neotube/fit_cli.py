from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from astropy.time import Time

from .models import Observation
from .site_checks import filter_special_sites


def load_observations(
    path: Path,
    sigma: float | None,
    *,
    skip_special_sites: bool = False,
) -> list[Observation]:
    observations: list[Observation] = []
    with path.open() as fh:
        reader = csv.DictReader(fh)
        pos_keys = ("obs_x_km", "obs_y_km", "obs_z_km")
        for row in reader:
            if not row.get("t_utc") or not row.get("ra_deg") or not row.get("dec_deg"):
                continue
            obs_sigma = sigma
            if row.get("sigma_arcsec"):
                obs_sigma = float(row["sigma_arcsec"])
            if obs_sigma is None:
                raise ValueError("Observation row missing sigma_arcsec and no default provided.")
            obs_time = Time(row["t_utc"], scale="utc")
            observer_pos_km = None
            if any(key in row for key in pos_keys):
                raw_vals = [row.get(key) for key in pos_keys]
                if any(val not in (None, "") for val in raw_vals):
                    if any(val in (None, "") for val in raw_vals):
                        raise ValueError(
                            "Observation row has incomplete observer position; "
                            "expected obs_x_km, obs_y_km, obs_z_km."
                        )
                    observer_pos_km = np.array([float(val) for val in raw_vals], dtype=float)
            mag_val = None
            sigma_mag = None
            for key in ("mag", "mag_app", "v_mag", "V", "Vmag", "v"):
                if row.get(key) not in (None, ""):
                    mag_val = float(row[key])
                    break
            for key in ("sigma_mag", "mag_sigma", "mag_err"):
                if row.get(key) not in (None, ""):
                    sigma_mag = float(row[key])
                    break
            obs = Observation(
                time=obs_time,
                ra_deg=float(row["ra_deg"]),
                dec_deg=float(row["dec_deg"]),
                sigma_arcsec=obs_sigma,
                site=row.get("site"),
                observer_pos_km=observer_pos_km,
                mag=mag_val,
                sigma_mag=sigma_mag,
            )
            observations.append(obs)
    if not observations:
        raise ValueError("No valid observations loaded from CSV.")
    observations.sort(key=lambda ob: ob.time)
    return filter_special_sites(
        observations,
        skip_special_sites=skip_special_sites,
        fail_unknown_site=True,
    )
