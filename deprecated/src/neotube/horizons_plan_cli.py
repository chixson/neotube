from __future__ import annotations

import argparse
import csv
from pathlib import Path

from astropy.time import Time
from astroquery.jplhorizons import Horizons


def _normalize_horizons_id(raw: str) -> str:
    s = raw.strip()
    if s.isdigit():
        n = int(s)
        if 1 <= n < 2000000:
            return str(2000000 + n)
    return s


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rewrite a plan CSV so planned_center_ra/dec are taken from JPL Horizons at each exposure time."
    )
    parser.add_argument("--plan", type=Path, required=True, help="Input plan CSV (from neotube.plan_cli or coarse planners).")
    parser.add_argument("--target", required=True, help="Horizons target (e.g., 00001 or 2000001 or Ceres).")
    parser.add_argument("--location", default="500", help="Horizons observer location (default: geocenter 500).")
    parser.add_argument("--out", type=Path, required=True, help="Output plan CSV with Horizons centers.")
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    obsjds: list[float] = []
    with args.plan.open() as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise SystemExit("Input plan missing header.")
        for row in reader:
            if not row.get("obsjd"):
                continue
            rows.append(row)
            obsjds.append(float(row["obsjd"]))

    if not rows:
        raise SystemExit("No rows loaded from plan.")

    unique_epochs = sorted(set(obsjds))
    obj = Horizons(id=_normalize_horizons_id(args.target), location=args.location, epochs=unique_epochs[0] if len(unique_epochs) == 1 else unique_epochs)
    eph = obj.ephemerides()

    # Build a mapping from epoch JD -> (RA, DEC)
    ra_dec_by_jd: dict[float, tuple[float, float]] = {}
    if len(unique_epochs) == 1:
        ra_dec_by_jd[unique_epochs[0]] = (float(eph[0]["RA"]), float(eph[0]["DEC"]))
    else:
        if len(eph) != len(unique_epochs):
            raise SystemExit(f"Horizons returned {len(eph)} rows for {len(unique_epochs)} epochs.")
        for jd, row in zip(unique_epochs, eph):
            ra_dec_by_jd[jd] = (float(row["RA"]), float(row["DEC"]))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    # ensure planned_* columns exist
    for col in ("planned_center_ra", "planned_center_dec"):
        if col not in fieldnames:
            fieldnames.append(col)

    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row, jd in zip(rows, obsjds):
            row = dict(row)
            ra, dec = ra_dec_by_jd[jd]
            row["planned_center_ra"] = f"{ra:.6f}"
            row["planned_center_dec"] = f"{dec:.6f}"
            writer.writerow(row)

    first_time = Time(obsjds[0], format="jd", scale="utc").isot
    print(f"Wrote {len(rows)} rows to {args.out} (first obs time {first_time})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
