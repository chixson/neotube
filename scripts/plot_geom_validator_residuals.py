#!/usr/bin/env python3
"""Plot geom validator residuals vs time and per-site.

Reads the CSV from geom_validator_cached_polite.py and writes PNGs.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _parse_time(s: str) -> datetime:
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _signed_deltas(ra1_deg, dec1_deg, ra2_deg, dec2_deg):
    ra1 = np.deg2rad(np.asarray(ra1_deg, dtype=float))
    ra2 = np.deg2rad(np.asarray(ra2_deg, dtype=float))
    dec1 = np.deg2rad(np.asarray(dec1_deg, dtype=float))
    dec2 = np.deg2rad(np.asarray(dec2_deg, dtype=float))
    dra = ra1 - ra2
    dra = (dra + np.pi) % (2.0 * np.pi) - np.pi
    dx_arcsec = dra * np.cos(dec1) * 206265.0
    dy_arcsec = (dec1 - dec2) * 206265.0
    return dx_arcsec, dy_arcsec


def load_rows(path: Path):
    rows = []
    with path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot geom validator residuals vs time/site.")
    parser.add_argument("--input", type=Path, required=True, help="CSV from geom_validator_cached_polite.py")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/ceres"), help="Output directory")
    args = parser.parse_args()

    rows = load_rows(args.input)
    if not rows:
        raise SystemExit("No rows in input CSV.")

    times = np.array([_parse_time(r["time_utc"]) for r in rows])
    obs_ra = np.array([float(r["obs_ra"]) for r in rows], dtype=float)
    obs_dec = np.array([float(r["obs_dec"]) for r in rows], dtype=float)
    pred_ra = np.array([float(r["pred_ra"]) for r in rows], dtype=float)
    pred_dec = np.array([float(r["pred_dec"]) for r in rows], dtype=float)
    sites = [r["site"].strip().upper() if r["site"] else "UNK" for r in rows]

    dx, dy = _signed_deltas(pred_ra, pred_dec, obs_ra, obs_dec)
    sep = np.sqrt(dx * dx + dy * dy)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Time series plot
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(times, dx, "o", ms=4, alpha=0.75, label="dRA*cos(dec) (arcsec)")
    ax.plot(times, dy, "o", ms=4, alpha=0.75, label="dDec (arcsec)")
    ax.axhline(0.0, color="k", lw=0.8, alpha=0.5)
    ax.set_title("Geom validator residuals vs time")
    ax.set_xlabel("UTC time")
    ax.set_ylabel("Residual (arcsec)")
    ax.legend(loc="best")
    fig.autofmt_xdate()
    out_time = args.out_dir / "geom_validator_residuals_time.png"
    fig.tight_layout()
    fig.savefig(out_time, dpi=150)
    plt.close(fig)

    # Per-site scatter (dx, dy)
    fig, ax = plt.subplots(figsize=(6, 6))
    site_groups = defaultdict(list)
    for i, s in enumerate(sites):
        site_groups[s].append(i)
    for site, idxs in sorted(site_groups.items(), key=lambda kv: -len(kv[1])):
        ax.scatter(dx[idxs], dy[idxs], s=18, alpha=0.75, label=f"{site} (n={len(idxs)})")
    ax.axhline(0.0, color="k", lw=0.8, alpha=0.4)
    ax.axvline(0.0, color="k", lw=0.8, alpha=0.4)
    ax.set_title("Geom validator residuals by site")
    ax.set_xlabel("dRA*cos(dec) (arcsec)")
    ax.set_ylabel("dDec (arcsec)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_site = args.out_dir / "geom_validator_residuals_sites.png"
    fig.savefig(out_site, dpi=150)
    plt.close(fig)

    # Separation vs time
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(times, sep, "o", ms=4, alpha=0.75)
    ax.set_title("Geom validator separation vs time")
    ax.set_xlabel("UTC time")
    ax.set_ylabel("Separation (arcsec)")
    fig.autofmt_xdate()
    fig.tight_layout()
    out_sep = args.out_dir / "geom_validator_sep_time.png"
    fig.savefig(out_sep, dpi=150)
    plt.close(fig)

    print("Wrote plots:")
    print(out_time)
    print(out_site)
    print(out_sep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
