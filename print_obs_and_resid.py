import csv

obs_fn = "runs/ceres-ground-test/obs.csv"
res_fn = "runs/ceres-ground-test/fit_est_kappa/residuals.csv"

print("OBS rows for suspect sites:")
with open(obs_fn) as f:
    r = csv.DictReader(f)
    for i, row in enumerate(r):
        site = (row.get("site") or "").strip().upper()
        if site in {"Z22", "B50", "D29"}:
            print(i, row)

print("\nRESIDUALS rows for suspect sites:")
with open(res_fn) as f:
    r = csv.DictReader(f)
    for i, row in enumerate(r):
        site = (row.get("site") or "").strip().upper()
        if site in {"Z22", "B50", "D29"}:
            print(i, row)
