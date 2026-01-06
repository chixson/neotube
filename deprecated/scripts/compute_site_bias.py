import pandas as pd
import numpy as np
import json

df = pd.read_csv("runs/ceres/geom_validator_out.csv")

# recompute dx/dy if needed
def signed_deltas(ra1, dec1, ra2, dec2):
    ra1 = np.deg2rad(np.array(ra1))
    ra2 = np.deg2rad(np.array(ra2))
    dec1 = np.deg2rad(np.array(dec1))
    dec2 = np.deg2rad(np.array(dec2))
    dra = ra1 - ra2
    dra = (dra + np.pi) % (2 * np.pi) - np.pi
    dx = dra * np.cos(dec1) * 206265.0
    dy = (dec1 - dec2) * 206265.0
    return dx, dy


df["dx_arcsec"], df["dy_arcsec"] = signed_deltas(
    df["pred_ra"], df["pred_dec"], df["obs_ra"], df["obs_dec"]
)

out = {}
for site, g in df.groupby("site"):
    dx = g["dx_arcsec"].values
    dy = g["dy_arcsec"].values
    med_dx = float(np.median(dx))
    med_dy = float(np.median(dy))
    mad_dx = float(np.median(np.abs(dx - med_dx)) * 1.4826)
    mad_dy = float(np.median(np.abs(dy - med_dy)) * 1.4826)
    cov = (
        np.cov(np.vstack([dx, dy]), bias=True)
        if len(dx) > 1
        else [[mad_dx**2, 0.0], [0.0, mad_dy**2]]
    )
    out[site] = {
        "n": int(len(dx)),
        "median_dx": med_dx,
        "median_dy": med_dy,
        "mad_dx": mad_dx,
        "mad_dy": mad_dy,
        "cov": cov.tolist(),
    }

with open("runs/ceres/site_biases.json", "w") as fh:
    json.dump(out, fh, indent=2)

print("Wrote runs/ceres/site_biases.json")
