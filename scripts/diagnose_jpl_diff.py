#!/usr/bin/env python3
"""
Diagnose per-observation differences between posterior predictions
and a JPL/Horizons heliocentric ICRS state. Produces tables and plots.

Saves:
  runs/ceres/diagnostics_jpl_vs_post/deltas_by_obs.csv
  runs/ceres/diagnostics_jpl_vs_post/deltas_vs_time.png
  runs/ceres/diagnostics_jpl_vs_post/deltas_vector.png
"""

from pathlib import Path
import csv
from collections import defaultdict

import numpy as np
from numpy.linalg import lstsq
import matplotlib.pyplot as plt
from astropy.time import Time
import astropy.units as u

from neotube.fit import load_posterior, _predict_batch
from neotube.fit_cli import load_observations

# Config
POST_PATH = Path("runs/ceres/fit_studentt_em/posterior.npz")
OBS_CSV = Path("runs/ceres/obs.csv")
OUT_DIR = Path("runs/ceres/diagnostics_jpl_vs_post")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def signed_deltas(ra1_deg, dec1_deg, ra2_deg, dec2_deg):
    ra1 = np.deg2rad(np.array(ra1_deg))
    ra2 = np.deg2rad(np.array(ra2_deg))
    dec1 = np.deg2rad(np.array(dec1_deg))
    dec2 = np.deg2rad(np.array(dec2_deg))
    dra = ra1 - ra2
    dra = (dra + np.pi) % (2 * np.pi) - np.pi
    dx_arcsec = dra * np.cos(dec1) * 206265.0
    dy_arcsec = (dec1 - dec2) * 206265.0
    return dx_arcsec, dy_arcsec


def linfit(x, y):
    A = np.vstack([np.ones_like(x), x]).T
    coef, *_ = lstsq(A, y, rcond=None)
    residuals = y - (A @ coef)
    dof = max(1, len(y) - 2)
    s2 = np.sum(residuals**2) / dof
    cov = s2 * np.linalg.pinv(A.T @ A)
    se = np.sqrt(np.diag(cov))
    return coef, se


def main():
    post = load_posterior(POST_PATH)
    epoch = post.epoch
    print("posterior epoch:", epoch.isot)
    s = np.array(post.state, float)

    # fetch JPL vector via astroquery (heliocentric ICRS, converted to km/km/s)
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
            from astropy.coordinates import (
                CartesianRepresentation,
                CartesianDifferential,
                HeliocentricTrueEcliptic,
                ICRS,
            )
            obj = Horizons(id="1", location="@sun", epochs=Time(epoch.isot).jd, id_type="smallbody")
            vec = obj.vectors(refplane="ecliptic")
            x = float(vec["x"][0]); y = float(vec["y"][0]); z = float(vec["z"][0])
            vx = float(vec["vx"][0]); vy = float(vec["vy"][0]); vz = float(vec["vz"][0])
            rep = CartesianRepresentation(x * u.AU, y * u.AU, z * u.AU)
            diff = CartesianDifferential(vx * u.AU / u.day, vy * u.AU / u.day, vz * u.AU / u.day)
            rep_w = rep.with_differentials(diff)
            hce = HeliocentricTrueEcliptic(rep_w, obstime=Time(epoch.isot))
            sc_icrs = hce.transform_to(ICRS())
            pos_icrs_km = sc_icrs.cartesian.xyz.to(u.km).value
            diff_icrs = sc_icrs.cartesian.differentials.get("s", None)
            if diff_icrs is None:
                raise RuntimeError("Transformed coordinate lacks velocity differential")
            vel_icrs_km_s = diff_icrs.d_xyz.to(u.km / u.s).value
            jpl_km = np.hstack((pos_icrs_km, vel_icrs_km_s))
            print("Fetched JPL (heliocentric ICRS) km-state norm:", np.linalg.norm(jpl_km[:3]))
        except Exception as exc:
            raise RuntimeError("Could not fetch JPL via astroquery: " + str(exc))

    obs = load_observations(OBS_CSV, None)
    pred_ra_post, pred_dec_post = _predict_batch(
        s, epoch, obs, ("earth", "mars", "jupiter"), max_step=3600.0, use_kepler=False
    )
    pred_ra_jpl, pred_dec_jpl = _predict_batch(
        jpl_km, epoch, obs, ("earth", "mars", "jupiter"), max_step=3600.0, use_kepler=False
    )

    dx, dy = signed_deltas(pred_ra_post, pred_dec_post, pred_ra_jpl, pred_dec_jpl)
    sep = np.sqrt(dx * dx + dy * dy)

    out_csv = OUT_DIR / "deltas_by_obs.csv"
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["idx", "time_utc", "site", "dx_arcsec", "dy_arcsec", "sep_arcsec",
                    "post_ra", "post_dec", "jpl_ra", "jpl_dec"])
        for i, ob in enumerate(obs):
            w.writerow([i, ob.time.isot, ob.site or "", dx[i], dy[i], sep[i],
                        pred_ra_post[i], pred_dec_post[i], pred_ra_jpl[i], pred_dec_jpl[i]])
    print("Wrote per-observation deltas to", out_csv)

    print("\nSummary statistics (arcsec):")
    print("median dx,dy,sep:", np.median(dx), np.median(dy), np.median(sep))
    print("mean dx,dy,sep:", np.mean(dx), np.mean(dy), np.mean(sep))
    print("std dx,dy,sep:", np.std(dx), np.std(dy), np.std(sep))
    print("max sep:", np.max(sep), "min sep:", np.min(sep))

    site_groups = defaultdict(list)
    for i, ob in enumerate(obs):
        site_groups[ob.site or "UNK"].append(i)
    print("\nPer-site summary (site, n, mean_dx, mean_dy, mean_sep):")
    for site, idxs in site_groups.items():
        arr_dx = dx[idxs]
        arr_dy = dy[idxs]
        arr_sep = sep[idxs]
        print(site, len(idxs), np.mean(arr_dx), np.mean(arr_dy), np.mean(arr_sep))

    times = np.array([(ob.time - epoch).to(u.day).value for ob in obs])
    coef_dx, se_dx = linfit(times, dx)
    coef_dy, se_dy = linfit(times, dy)
    print("\nLinear trend dx (arcsec/day): intercept, slope, stderr_slope:",
          coef_dx[0], coef_dx[1], se_dx[1])
    print("Linear trend dy (arcsec/day): intercept, slope, stderr_slope:",
          coef_dy[0], coef_dy[1], se_dy[1])

    plt.figure(figsize=(8, 5))
    plt.plot(times, dx, "o", label="dx (arcsec)")
    plt.plot(times, dy, "o", label="dy (arcsec)")
    t_line = np.linspace(times.min(), times.max(), 50)
    plt.plot(t_line, coef_dx[0] + coef_dx[1] * t_line, "-", color="C0", alpha=0.6)
    plt.plot(t_line, coef_dy[0] + coef_dy[1] * t_line, "-", color="C1", alpha=0.6)
    plt.xlabel("dt (days) relative to posterior epoch")
    plt.ylabel("Signed residual (arcsec)")
    plt.legend()
    plt.title("Per-observation signed residuals: posterior - JPL")
    plt.grid(True)
    plt.savefig(OUT_DIR / "deltas_vs_time.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.quiver(np.zeros_like(dx), np.zeros_like(dy), dx, dy,
               angles="xy", scale_units="xy", scale=1, alpha=0.6)
    plt.scatter(dx, dy, s=10, alpha=0.6)
    plt.axhline(0, color="k", lw=0.5)
    plt.axvline(0, color="k", lw=0.5)
    plt.xlabel("ΔRA cos(dec) (arcsec)")
    plt.ylabel("ΔDec (arcsec)")
    plt.title("Vector residuals (posterior - JPL) at obs times")
    plt.grid(True)
    plt.gca().set_aspect("equal", "box")
    plt.savefig(OUT_DIR / "deltas_vector.png", dpi=200)
    plt.close()

    print("\nSaved plots to:", OUT_DIR)
    print("Done.")


if __name__ == "__main__":
    raise SystemExit(main())
