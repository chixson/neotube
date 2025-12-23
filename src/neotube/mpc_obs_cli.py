from __future__ import annotations

import argparse
import csv
from pathlib import Path

from astropy.time import Time
from astroquery.mpc import MPC


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch MPC observations for an object and write a neotube-ready CSV."
    )
    parser.add_argument(
        "--target",
        required=True,
        help="MPC target number or designation (e.g. 1, 2023 AB).",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Optional inclusive start time (UTC, ISO).",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="Optional inclusive end time (UTC, ISO).",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=0,
        help="If >0, write at most N observations after filtering.",
    )
    parser.add_argument(
        "--sigma-arcsec",
        type=float,
        default=0.5,
        help="Per-observation astrometric uncertainty (arcsec) to record in output.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output CSV path (columns: t_utc, ra_deg, dec_deg, sigma_arcsec, site).",
    )
    args = parser.parse_args()

    t_start = Time(args.start, scale="utc") if args.start else None
    t_end = Time(args.end, scale="utc") if args.end else None

    tab = MPC.get_observations(args.target)
    if len(tab) == 0:
        raise SystemExit("MPC returned no observations.")

    rows = []
    for row in tab:
        jd = row["epoch"]
        try:
            jd_val = float(jd.to_value("d"))
        except Exception:
            jd_val = float(jd)
        t = Time(jd_val, format="jd", scale="utc")
        if t_start and t < t_start:
            continue
        if t_end and t > t_end:
            continue
        rows.append(
            {
                "t_utc": t.isot,
                "ra_deg": float(row["RA"]),
                "dec_deg": float(row["DEC"]),
                "sigma_arcsec": float(args.sigma_arcsec),
                "site": str(row["observatory"]),
            }
        )

    rows.sort(key=lambda r: r["t_utc"])
    if args.n and args.n > 0:
        rows = rows[: args.n]

    if not rows:
        raise SystemExit("No observations in requested time window.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["t_utc", "ra_deg", "dec_deg", "sigma_arcsec", "site"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} observations to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
