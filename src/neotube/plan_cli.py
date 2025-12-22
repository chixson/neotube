from __future__ import annotations

import argparse
import csv
from pathlib import Path

import requests
from astropy.time import Time

from .cli import DEFAULT_COLUMNS, GlobalRateLimiter, exposure_unique_id, query_exposures


def load_nodes(path: Path) -> list[dict]:
    nodes: list[dict] = []
    with path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            nodes.append(
                {
                    "time": Time(row["time_utc"], scale="utc"),
                    "ra": float(row["center_ra_deg"]),
                    "dec": float(row["center_dec_deg"]),
                    "radius": float(row["radius_arcsec"]),
                }
            )
    return nodes


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a plan file from tube nodes.")
    parser.add_argument("--tube", type=Path, required=True, help="Tube nodes CSV.")
    parser.add_argument("--out", type=Path, required=True, help="Output plan CSV.")
    parser.add_argument("--slot-s", type=float, default=3600.0, help="Time window (seconds) for metadata query.")
    parser.add_argument("--min-size-arcsec", type=float, default=30.0, help="Minimum search box size.")
    parser.add_argument("--filter", type=str, default=None, help="Optional filter code to restrict exposures.")
    parser.add_argument("--max-rps", type=float, default=0.5, help="Max requests per second.")
    parser.add_argument("--user-agent", type=str, default="neotube-plan/0.1", help="User-Agent header.")
    args = parser.parse_args()

    nodes = load_nodes(args.tube)
    if not nodes:
        raise SystemExit("No tube nodes loaded.")

    seen: set[str] = set()
    rows: list[dict] = []
    session = requests.Session()
    limiter = GlobalRateLimiter(args.max_rps)
    headers = {"User-Agent": args.user_agent}
    for node in nodes:
        jd_center = node["time"].jd
        window = args.slot_s / 86400.0
        size_deg = max(args.min_size_arcsec / 3600.0, node["radius"] * 2.0 / 3600.0)
        exposures = query_exposures(
            session,
            ra=node["ra"],
            dec=node["dec"],
            jd_start=jd_center - window / 2.0,
            jd_end=jd_center + window / 2.0,
            columns=DEFAULT_COLUMNS,
            headers=headers,
            limiter=limiter,
            size_deg=size_deg,
            filtercode=args.filter,
        )
        for exposure in exposures:
            eid = exposure_unique_id(exposure)
            if eid in seen:
                continue
            seen.add(eid)
            rows.append(
                {
                    "exposure_id": eid,
                    "obsjd": exposure.obsjd,
                    "obsdate": exposure.obsdate,
                    "filefracday": exposure.filefracday,
                    "field": exposure.field,
                    "ccdid": exposure.ccdid,
                    "qid": exposure.qid,
                    "filtercode": exposure.filtercode,
                    "imgtypecode": exposure.imgtypecode,
                    "ra": exposure.ra,
                    "dec": exposure.dec,
                    "planned_center_ra": node["ra"],
                    "planned_center_dec": node["dec"],
                    "planned_radius_arcsec": node["radius"],
                    "node_time_utc": node["time"].isot,
                }
            )

    if not rows:
        raise SystemExit("No exposures found for tube nodes.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "exposure_id",
        "obsjd",
        "obsdate",
        "filefracday",
        "field",
        "ccdid",
        "qid",
        "filtercode",
        "imgtypecode",
        "ra",
        "dec",
        "planned_center_ra",
        "planned_center_dec",
        "planned_radius_arcsec",
        "node_time_utc",
    ]
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Wrote plan with {len(rows)} exposures to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
