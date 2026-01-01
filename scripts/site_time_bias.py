from pathlib import Path
import csv
import math
import numpy as np

res_fn = Path("runs/ceres-ground-test/fit_est_kappa/residuals.csv")
sites = {}
with res_fn.open() as fh:
    r = csv.DictReader(fh)
    for row in r:
        site = (row.get("site") or "UNK").strip().upper()
        ra_res = float(row["res_ra_arcsec"])
        dec_pred = float(row.get("dec_pred_deg") or row.get("dec_obs_deg") or 0.0)
        sites.setdefault(site, []).append((ra_res, dec_pred))

print("site  n_obs   med_ra_res(arcsec)  med_dec_deg   implied_dt_seconds")
for s, vals in sorted(sites.items()):
    ras = np.array([v[0] for v in vals], dtype=float)
    decs = np.array([v[1] for v in vals], dtype=float)
    med_ra = float(np.median(ras))
    med_dec = float(np.median(decs))
    cosd = math.cos(math.radians(med_dec))
    implied_dt = med_ra / (15.0 * cosd)
    print(f"{s:4s} {len(vals):5d} {med_ra:18.3f} {med_dec:13.6f} {implied_dt:18.3f}")
