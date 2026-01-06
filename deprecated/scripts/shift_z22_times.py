from pathlib import Path
from datetime import datetime, timedelta
import csv

obs_in = Path("runs/ceres-ground-test/obs.csv")
obs_out = Path("runs/ceres-ground-test/obs_z22_shifted.csv")
dt_seconds = -5.2  # set this to -implied_dt from site_time_bias.py (trial and error OK)


def shift_iso(iso_str, secs):
    # Assumes ISO string like '2021-10-21T03:38:07.008'
    fmt_ms = "%Y-%m-%dT%H:%M:%S.%f"
    fmt_s = "%Y-%m-%dT%H:%M:%S"
    try:
        t = datetime.strptime(iso_str, fmt_ms)
    except ValueError:
        t = datetime.strptime(iso_str, fmt_s)
    t2 = t + timedelta(seconds=secs)
    # keep milliseconds if present
    return t2.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]


with obs_in.open() as inf, obs_out.open("w", newline="") as outf:
    r = csv.DictReader(inf)
    w = csv.DictWriter(outf, fieldnames=r.fieldnames)
    w.writeheader()
    for row in r:
        site = (row.get("site") or "").strip().upper()
        if site == "Z22":
            row["t_utc"] = shift_iso(row["t_utc"], dt_seconds)
        w.writerow(row)

print("Wrote", obs_out)
