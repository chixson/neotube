from __future__ import annotations

import argparse
import csv
import json
from math import cos, radians, sqrt
from pathlib import Path


def angular_offset_arcsec(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    dra = ((ra1 - ra2 + 180.0) % 360.0) - 180.0
    dec_rad = radians(dec2)
    return sqrt((dra * cos(dec_rad)) ** 2 + (dec1 - dec2) ** 2) * 3600.0


def load_plan(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if not row.get("exposure_id"):
                continue
            rows.append(
                {
                    "exposure_id": row["exposure_id"],
                    "ra": float(row.get("ra", 0.0)),
                    "dec": float(row.get("dec", 0.0)),
                    "planned_center_ra": float(row.get("planned_center_ra", 0.0)) if row.get("planned_center_ra") else None,
                    "planned_center_dec": float(row.get("planned_center_dec", 0.0)) if row.get("planned_center_dec") else None,
                    "planned_radius_arcsec": float(row.get("planned_radius_arcsec", 0.0)) if row.get("planned_radius_arcsec") else None,
                }
            )
    return rows


def load_clean_index(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if not row.get("exposure_id") or not row.get("ra") or not row.get("dec"):
                continue
            rows.append(
                {
                    "exposure_id": row["exposure_id"],
                    "ra": float(row["ra"]),
                    "dec": float(row["dec"]),
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnostics for planned exposures and tubes.")
    parser.add_argument("--plan", type=Path, required=True, help="Plan CSV to inspect.")
    parser.add_argument("--clean-index", type=Path, help="Optional cleaned cutouts index CSV.")
    parser.add_argument("--output", type=Path, default=Path("diag_summary.json"), help="Diagnostic JSON output.")
    args = parser.parse_args()

    plan_rows = load_plan(args.plan)
    if not plan_rows:
        raise SystemExit("Plan file contains no exposures.")

    covered = []
    missed = []
    for row in plan_rows:
        if row["planned_center_ra"] is None or row["planned_center_dec"] is None:
            continue
        radius = row["planned_radius_arcsec"] or 0.0
        offs = angular_offset_arcsec(row["ra"], row["dec"], row["planned_center_ra"], row["planned_center_dec"])
        entry = {
            "exposure_id": row["exposure_id"],
            "offset_arcsec": offs,
            "radius_arcsec": radius,
        }
        if offs <= radius + 1e-6:
            covered.append(entry)
        else:
            missed.append(entry)

    summary = {
        "total_plan_exposures": len(plan_rows),
        "covered": len(covered),
        "missed": len(missed),
        "missed_details": missed[:20],
    }

    if args.clean_index and args.clean_index.exists():
        clean_rows = load_clean_index(args.clean_index)
        plan_map = {row["exposure_id"]: row for row in plan_rows}
        uncovered = []
        for clean in clean_rows:
            plan_row = plan_map.get(clean["exposure_id"])
            if not plan_row or plan_row["planned_center_ra"] is None:
                continue
            radius = plan_row["planned_radius_arcsec"] or 0.0
            offs = angular_offset_arcsec(clean["ra"], clean["dec"], plan_row["planned_center_ra"], plan_row["planned_center_dec"])
            record = {
                "exposure_id": clean["exposure_id"],
                "offset_arcsec": offs,
                "radius_arcsec": radius,
            }
            if offs > radius + 1e-6:
                uncovered.append(record)
        summary["clean_index_uncovered"] = len(uncovered)
        summary["clean_index_details"] = uncovered[:20]
    else:
        summary["clean_index_uncovered"] = None

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"Diagnostics written to {args.output}; missed {summary['missed']} exposures.")
    if summary["clean_index_uncovered"]:
        print(f"{summary['clean_index_uncovered']} cleaned cutouts lie outside the tube.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
