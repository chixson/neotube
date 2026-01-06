#!/usr/bin/env python3
"""
Compare posterior (runs/ceres/fit_studentt_em/posterior.npz) to
JPL/Horizons heliocentric ICRS state for Ceres (id=1) at the
posterior epoch. Also compute RA/Dec for the JPL state using the
project's predict code and print separations / Mahalanobis.
"""

import sys
from pathlib import Path
import numpy as np
from numpy.linalg import inv
from astropy.time import Time
import astropy.units as u
from astropy.coordinates import (
    CartesianDifferential,
    CartesianRepresentation,
    HeliocentricTrueEcliptic,
    ICRS,
    SkyCoord,
)

from neotube.fit import load_posterior, _predict_batch
from neotube.fit_cli import load_observations


# ------------- Configuration --------------
POST_PATH = "runs/ceres/fit_studentt_em/posterior.npz"
REPLICA_CSV = "runs/ceres/replicas_10k.csv"
OBJECT_ID = "1"  # MPC number for Ceres
HORIZONS_ID_TYPE = "smallbody"
# -----------------------------------------


def main() -> int:
    post = load_posterior(POST_PATH)
    epoch = post.epoch
    print("Posterior epoch:", epoch.isot)
    s = np.array(post.state, float)
    cov = np.array(post.cov, float)
    print("Posterior mean (state):", s)
    print("Posterior position norm (km):", np.linalg.norm(s[:3]))
    print("Posterior position stddevs (km):", np.sqrt(np.abs(np.diag(cov)))[:3])

    # --- Option A: use an existing JPL vector you have --- #
    USE_EXISTING_JPL = False
    jpl_km = None
    if USE_EXISTING_JPL:
        jpl_km = np.array([
            -58495683.68161769, -20737419.190706648, -5013912.943995145,
            6.387721463365289, -38.47384914627045, -21.214559704628464
        ])
    else:
        try:
            from astroquery.jplhorizons import Horizons
        except Exception as exc:
            print("astroquery not available or import failed:", exc)
            print("Set USE_EXISTING_JPL=True and provide jpl_km manually.")
            return 1

        obj = Horizons(
            id=OBJECT_ID,
            location="@sun",
            epochs=Time(epoch.isot).jd,
            id_type=HORIZONS_ID_TYPE,
        )
        vec = obj.vectors(refplane="ecliptic")
        x = float(vec["x"][0])   # AU (heliocentric ecliptic)
        y = float(vec["y"][0])
        z = float(vec["z"][0])
        vx = float(vec["vx"][0])  # AU/day
        vy = float(vec["vy"][0])
        vz = float(vec["vz"][0])

        rep = CartesianRepresentation(x * u.AU, y * u.AU, z * u.AU)
        diff = CartesianDifferential(vx * u.AU / u.day, vy * u.AU / u.day, vz * u.AU / u.day)
        rep_w = rep.with_differentials(diff)
        hce = HeliocentricTrueEcliptic(rep_w, obstime=Time(epoch.isot))
        sc_icrs = hce.transform_to(ICRS())

        pos_icrs_km = sc_icrs.cartesian.xyz.to(u.km).value
        diff_icrs = sc_icrs.cartesian.differentials.get("s", None)
        if diff_icrs is None:
            raise RuntimeError("Transformed coordinate lacks velocity differential.")
        vel_icrs_km_s = diff_icrs.d_xyz.to(u.km / u.s).value
        jpl_km = np.hstack((pos_icrs_km, vel_icrs_km_s))
        print("Fetched JPL (Heliocentric ICRS) at epoch:", epoch.isot)
        print("JPL state (km, km/s):", jpl_km)
        print("JPL position norm (km):", np.linalg.norm(jpl_km[:3]))

    d = jpl_km - s
    print("\nRaw difference (jpl - post):", d)
    print("norm(d) [km]:", np.linalg.norm(d))

    try:
        maha2 = float(d.T @ inv(cov) @ d)
        print("Mahalanobis^2:", maha2, "Mahalanobis:", np.sqrt(maha2))
    except Exception as exc:
        print("Mahalanobis failed (cov may be singular):", exc)

    obs = load_observations(Path("runs/ceres/obs.csv"), None)
    pred_ra_post, pred_dec_post = _predict_batch(
        s, epoch, obs, ("earth", "mars", "jupiter"), max_step=3600.0, use_kepler=False
    )
    print("\nPosterior predicted RA/Dec (mean over obs times):",
          np.mean(pred_ra_post), np.mean(pred_dec_post))

    jpl_ra, jpl_dec = _predict_batch(
        jpl_km, epoch, obs, ("earth", "mars", "jupiter"), max_step=3600.0, use_kepler=False
    )
    print("JPL predicted RA/Dec (mean over obs times):", np.mean(jpl_ra), np.mean(jpl_dec))

    c_rep = SkyCoord(np.mean(pred_ra_post) * u.deg, np.mean(pred_dec_post) * u.deg, frame="icrs")
    c_jpl = SkyCoord(np.mean(jpl_ra) * u.deg, np.mean(jpl_dec) * u.deg, frame="icrs")
    print("Posterior mean <-> JPL mean separation (arcsec):", c_rep.separation(c_jpl).arcsecond)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
