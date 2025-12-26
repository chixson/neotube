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
from numpy.linalg import lstsq

import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time
import astropy.units as u

from neotube.fit import load_posterior, _predict_batch
from neotube.fit_cli import load_observations

# Config
POST_PATH = Path("runs/ceres/fit_studentt_em/posterior.npz")
REPLICA_CSV = Path("runs/ceres/replicas_10k.csv")
OBS_CSV = Path("runs/ceres/obs.csv")
OUT_DIR = Path("runs/ceres/diagnostics_jpl_vs_post")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# load posterior
post = load_posterior(POST_PATH)
epoch = post.epoch
print("posterior epoch:", epoch.isot)
s = np.array(post.state, float)
cov = np.array(post.cov, float)

# fetch JPL vector via astroquery (heliocentric ICRS, converted to km/km/s)
USE_EXISTING_JPL = False
# If you have an existing jpl vector (km/km/s), set USE_EXISTING_JPL=True
jpl_km = None
if USE_EXISTING_JPL:
    jpl_km = np.array([
        -58495683.68161769, -20737419.190706648, -5013912.943995145,
         6.387721463365289, -38.47384914627045, -21.214559704628464
    ])
else:
    try:
        from astroquery.jplhorizons import Horizons
        from astropy.coordinates import CartesianRepresentation, CartesianDifferential, ICRS, SkyCoord
        obj = Horizons(id="1", location="@sun", epochs=Time(epoch.isot).jd, id_type="smallbody")
        vec = obj.vectors(refplane="earth")
        x = float(vec["x"][0])
        y = float(vec["y"][0])
        z = float(vec["z"][0])
        vx = float(vec["vx"][0])
        vy = float(vec["vy"][0])
        vz = float(vec["vz"][0])
        AU_KM = 149597870.7
        rep = CartesianRepresentation(x * u.AU, y * u.AU, z * u.AU)
        diff = CartesianDifferential(vx * u.AU / u.day, vy * u.AU / u.day, vz * u.AU / u.day)
        rep_w = rep.with_differentials(diff)
        coord = SkyCoord(rep_w, frame=ICRS())
        pos_icrs_km = coord.cartesian.xyz.to(u.km).value
        vel_icrs_km_s = coord.cartesian.differentials["s"].d_xyz.to(u.km / u.s).value
        jpl_km = np.hstack((pos_icrs_km, vel_icrs_km_s))
        print("Fetched JPL (heliocentric ICRS) km-state norm:", np.linalg.norm(jpl_km[:3]))
    except Exception as exc:
        raise RuntimeError("Could not fetch JPL via astroquery: " + str(exc))

# load observations
obs = load_observations(OBS_CSV, None)

# compute predicted RA/Dec arrays (one value per obs)
pred_ra_post, pred_dec_post = _predict_batch(
    s, epoch, obs, ("earth", "mars", "jupiter"), max_step=3600.0, use_kepler=False
)
pred_ra_jpl, pred_dec_jpl = _predict_batch(
    jpl_km, epoch, obs, ("earth", "mars", "jupiter"), max_step=3600.0, use_kepler=False
)


def signed_deltas(ra1_deg, dec1_deg, ra2_deg, dec2_deg):
    # returns delta_ra_arcsec = ra1 - ra2 (signed, cos(dec) accounted), delta_dec_arcsec
    ra1 = np.deg2rad(np.array(ra1_deg))
    ra2 = np.deg2rad(np.array(ra2_deg))
    dec1 = np.deg2rad(np.array(dec1_deg))
    dec2 = np.deg2rad(np.array(dec2_deg))
    dra = ra1 - ra2
    dra = (dra + np.pi) % (2 * np.pi) - np.pi
    dx_arcsec = dra * np.cos(dec1) * 206265.0
    dy_arcsec = (dec1 - dec2) * 206265.0
    return dx_arcsec, dy_arcsec


dx, dy = signed_deltas(pred_ra_post, pred_dec_post, pred_ra_jpl, pred_dec_jpl)
sep = np.sqrt(dx * dx + dy * dy)

# offsets relative to observations (tangent plane at obs)
obs_ra = np.array([o.ra_deg for o in obs], dtype=float)
obs_dec = np.array([o.dec_deg for o in obs], dtype=float)

def residuals_against_obs(pred_ra, pred_dec):
    d_ra = ((pred_ra - obs_ra + 180.0) % 360.0) - 180.0
    dx_arcsec = d_ra * np.cos(np.deg2rad(obs_dec)) * 3600.0
    dy_arcsec = (pred_dec - obs_dec) * 3600.0
    return dx_arcsec, dy_arcsec

dx_post, dy_post = residuals_against_obs(np.array(pred_ra_post), np.array(pred_dec_post))
dx_jpl, dy_jpl = residuals_against_obs(np.array(pred_ra_jpl), np.array(pred_dec_jpl))

# Optional: compare against Horizons apparent RA/Dec for the observer (C51/WISE).
hz_ra = None
hz_dec = None
try:
    from astroquery.jplhorizons import Horizons

    epochs = Time([o.time.isot for o in obs]).jd
    hz = Horizons(id="1", location="500@-163", epochs=epochs, id_type="smallbody")
    eph = hz.ephemerides()
    hz_ra = np.array(eph["RA"], dtype=float)
    hz_dec = np.array(eph["DEC"], dtype=float)
    dx_hz, dy_hz = residuals_against_obs(hz_ra, hz_dec)
except Exception as exc:
    print("Horizons apparent ephemerides fetch failed:", exc)
    dx_hz, dy_hz = None, None

# write per-observation CSV
out_csv = OUT_DIR / "deltas_by_obs.csv"
with open(out_csv, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(
        [
            "idx",
            "time_utc",
            "site",
            "dx_arcsec",
            "dy_arcsec",
            "sep_arcsec",
            "post_ra",
            "post_dec",
            "jpl_ra",
            "jpl_dec",
        ]
    )
    for i, ob in enumerate(obs):
        w.writerow(
            [
                i,
                ob.time.isot,
                ob.site or "",
                dx[i],
                dy[i],
                sep[i],
                pred_ra_post[i],
                pred_dec_post[i],
                pred_ra_jpl[i],
                pred_dec_jpl[i],
            ]
        )
print("Wrote per-observation deltas to", out_csv)

print("\nSummary statistics (arcsec):")
print("median dx,dy,sep:", np.median(dx), np.median(dy), np.median(sep))
print("mean dx,dy,sep:", np.mean(dx), np.mean(dy), np.mean(sep))
print("std dx,dy,sep:", np.std(dx), np.std(dy), np.std(sep))
print("max sep:", np.max(sep), "min sep:", np.min(sep))

# Per-site statistics
site_groups = defaultdict(list)
for i, ob in enumerate(obs):
    site_groups[ob.site or "UNK"].append(i)
print("\nPer-site summary (site, n, mean_dx, mean_dy, mean_sep):")
for site, idxs in site_groups.items():
    arr_dx = dx[idxs]
    arr_dy = dy[idxs]
    arr_sep = sep[idxs]
    print(site, len(idxs), np.mean(arr_dx), np.mean(arr_dy), np.mean(arr_sep))

# trend vs time: fit linear slope dx ~ a + b * dt
times = np.array([(ob.time - epoch).to(u.day).value for ob in obs])


def linfit(x, y):
    A = np.vstack([np.ones_like(x), x]).T
    coef, *_ = lstsq(A, y, rcond=None)
    residuals = y - (A @ coef)
    dof = max(1, len(y) - 2)
    s2 = np.sum(residuals ** 2) / dof
    cov = s2 * np.linalg.pinv(A.T @ A)
    se = np.sqrt(np.diag(cov))
    return coef, se


coef_dx, se_dx = linfit(times, dx)
coef_dy, se_dy = linfit(times, dy)
print("\nLinear trend dx (arcsec/day): intercept, slope, stderr_slope:", coef_dx[0], coef_dx[1], se_dx[1])
print("Linear trend dy (arcsec/day): intercept, slope, stderr_slope:", coef_dy[0], coef_dy[1], se_dy[1])

# plot dx,dy vs time
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

# vector plot (arrows)
plt.figure(figsize=(6, 6))
plt.quiver(np.zeros_like(dx), np.zeros_like(dy), dx, dy, angles="xy", scale_units="xy", scale=1, alpha=0.6)
plt.scatter(dx, dy, s=10, alpha=0.6)
plt.axhline(0, color="k", lw=0.5)
plt.axvline(0, color="k", lw=0.5)
plt.xlabel("ΔRA * cos(dec) (arcsec)")
plt.ylabel("ΔDec (arcsec)")
plt.title("Vector residuals (posterior - JPL) at obs times")
plt.grid(True)
plt.gca().set_aspect("equal", "box")
plt.savefig(OUT_DIR / "deltas_vector.png", dpi=200)
plt.close()

# overlay plot: posterior vs JPL residuals relative to obs
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(dx_post, dy_post, s=30, alpha=0.8, label="posterior vs obs")
ax.scatter(dx_jpl, dy_jpl, s=30, alpha=0.8, label="JPL vs obs")
for x1, y1, x2, y2 in zip(dx_post, dy_post, dx_jpl, dy_jpl):
    ax.plot([x1, x2], [y1, y2], color="#999999", alpha=0.5, linewidth=0.7)
ax.axhline(0, color="#bbbbbb", lw=1)
ax.axvline(0, color="#bbbbbb", lw=1)
ax.set_xlabel("ΔRA cosδ (arcsec) [pred − obs]")
ax.set_ylabel("ΔDec (arcsec) [pred − obs]")
ax.set_title("Posterior vs JPL residuals per observation")
ax.grid(True, lw=0.5, color="#dddddd")
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "obs_overlay.png", dpi=150)
plt.close(fig)

if dx_hz is not None:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(dx_post, dy_post, s=30, alpha=0.8, label="posterior vs obs")
    ax.scatter(dx_hz, dy_hz, s=30, alpha=0.8, label="Horizons apparent vs obs")
    for x1, y1, x2, y2 in zip(dx_post, dy_post, dx_hz, dy_hz):
        ax.plot([x1, x2], [y1, y2], color="#999999", alpha=0.5, linewidth=0.7)
    ax.axhline(0, color="#bbbbbb", lw=1)
    ax.axvline(0, color="#bbbbbb", lw=1)
    ax.set_xlabel("ΔRA cosδ (arcsec) [pred − obs]")
    ax.set_ylabel("ΔDec (arcsec) [pred − obs]")
    ax.set_title("Posterior vs Horizons (apparent) residuals")
    ax.grid(True, lw=0.5, color="#dddddd")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "obs_overlay_horizons.png", dpi=150)
    plt.close(fig)

print("\nSaved plots to:", OUT_DIR)
print("Done.")
