#!/usr/bin/env python3
"""
pick_ceres_epoch.py

Drop-in script to:
  1) Fetch a pool of MPC observations (using the neotube CLI: `python -m neotube.mpc_obs_cli`)
  2) Select a subset of observations that:
       - contain between MIN_OBS and MAX_OBS observations (default 5..10),
       - are taken by at least MIN_UNIQUE_SITES different MPC observatories (default 3),
       - span between MIN_SPAN_DAYS and MAX_SPAN_DAYS (default 30..90 days),
       - are from ground-based observatories (site kind FIXED or ROVING).
  3) Write the selected epoch CSV (default: runs/ceres/selected_epoch.csv).

Notes / assumptions:
 - This script expects the neotube package to be importable (e.g. run with
     PYTHONPATH=src python pick_ceres_epoch.py
   when run from the neotube repo root).
 - We call the CLI `python -m neotube.mpc_obs_cli` to fetch the pool. If you have
   an existing `all_obs.csv` pool file, you can skip fetching with --no-fetch.
 - The script is tolerant of different CSV column names for time/site if they
   match common patterns; however the CLI output normally includes:
     t_utc, ra_deg, dec_deg, sigma_arcsec, site
 - Comments explain the selection logic below.
"""

from __future__ import annotations

import argparse
import csv
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
from astropy.time import Time

# Try importing neotube helpers (site classifiers). If this fails, show a clear
# message about running with PYTHONPATH.
try:
    from neotube.sites import SiteKind, get_site_kind
except Exception as exc:
    print(
        "ERROR: could not import neotube.sites.get_site_kind. "
        "Run this script with the neotube package on your PYTHONPATH.\n\n"
        "Example (from repo root):\n"
        "  PYTHONPATH=src python pick_ceres_epoch.py\n\n"
        "Original import error: ",
        exc,
        file=sys.stderr,
    )
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch and pick a Ceres epoch satisfying constraints.")
    p.add_argument("--target", default="1", help="Horizons/MPC target identifier. Default '1' (Ceres).")
    p.add_argument(
        "--start",
        default="2018-01-01T00:00:00",
        help="Start time (ISO) for pool fetch (CLI --start). Default 2018-01-01T00:00:00.",
    )
    p.add_argument("--pool-n", type=int, default=200, help="Number of observations to fetch into pool.")
    p.add_argument("--sigma", type=float, default=0.5, help="Default sigma_arcsec passed to mpc_obs_cli.")
    p.add_argument("--out-dir", default="runs/ceres", help="Output directory for pool and selected CSV.")
    p.add_argument("--pool-file", default="all_obs.csv", help="Filename for fetched pool inside out-dir.")
    p.add_argument("--selected-file", default="selected_epoch.csv", help="Filename for selected epoch inside out-dir.")
    p.add_argument("--min-obs", type=int, default=5, help="Minimum number of observations to select.")
    p.add_argument("--max-obs", type=int, default=10, help="Maximum number of observations to select.")
    p.add_argument("--min-unique-sites", type=int, default=3, help="Minimum distinct MPC sites required.")
    p.add_argument("--min-span-days", type=float, default=30.0, help="Minimum timespan (days) across selected obs.")
    p.add_argument("--max-span-days", type=float, default=90.0, help="Maximum timespan (days) across selected obs.")
    p.add_argument(
        "--no-fetch",
        action="store_true",
        help="Skip fetching pool from neotube.mpc_obs_cli if pool-file already exists.",
    )
    p.add_argument("--seed", type=int, default=12345, help="Random seed for fallback random trials.")
    p.add_argument("--verbose", action="store_true", help="Verbose output.")
    return p.parse_args()


def shell_fetch_pool(out_path: Path, target: str, start: str, pool_n: int, sigma: float, verbose: bool) -> None:
    """
    Use the neotube CLI module to fetch a pool of MPC observations.
    This runs:
      python -m neotube.mpc_obs_cli --target <target> --start <start> --n <pool_n> --sigma-arcsec <sigma> --out <out_path>
    """
    cmd = [
        sys.executable,
        "-m",
        "neotube.mpc_obs_cli",
        "--target",
        str(target),
        "--start",
        str(start),
        "--n",
        str(pool_n),
        "--sigma-arcsec",
        str(sigma),
        "--out",
        str(out_path),
    ]
    if verbose:
        print("Running:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Failed to fetch pool via neotube.mpc_obs_cli: {exc}")


def find_key(row: Dict[str, str], candidates: List[str]) -> str | None:
    """Return the first key in row that appears in candidates (case-insensitive)."""
    keys = {k.lower(): k for k in row.keys()}
    for candidate in candidates:
        if candidate.lower() in keys:
            return keys[candidate.lower()]
    return None


def load_csv_rows(path: Path) -> List[Dict[str, str]]:
    """Load CSV rows into a list of dicts."""
    with path.open() as fh:
        rdr = csv.DictReader(fh)
        rows = [r for r in rdr]
    return rows


def time_from_row(row: Dict[str, str]) -> Time:
    """Try common time columns; expect t_utc normally."""
    for candidate in ("t_utc", "time_utc", "time", "datetime", "datetime_utc", "obs_time"):
        if candidate in row and row[candidate]:
            return Time(row[candidate], scale="utc")
    tkey = find_key(row, ["t_utc", "time_utc", "time", "datetime", "datetime_utc", "obs_time"])
    if tkey:
        return Time(row[tkey], scale="utc")
    raise KeyError("No time column found in row; expected t_utc/time_utc/datetime")


def site_from_row(row: Dict[str, str]) -> str:
    for candidate in ("site", "observatory", "obs_code", "station"):
        if candidate in row and row[candidate] and row[candidate].strip():
            return row[candidate].strip().upper()
    skey = find_key(row, ["site", "observatory", "obs_code", "station"])
    return row[skey].strip().upper() if skey and row[skey] else ""


def pick_window_by_sliding(
    good: List[Dict[str, str]],
    min_obs: int,
    max_obs: int,
    min_span_days: float,
    max_span_days: float,
    min_unique_sites: int,
    verbose: bool = False,
) -> List[Dict[str, str]] | None:
    """
    Sliding-window attempt:
    Walk a time-sorted list and test windows for our constraints.
    Downsample windows longer than max_obs uniformly in time.
    """
    n = len(good)
    times = [r["_time"].jd for r in good]
    _ = times  # keep for future debug without recomputing

    for i in range(n):
        for j in range(i + min_obs - 1, min(n, i + 200)):
            window = good[i : j + 1]
            span_days = (window[-1]["_time"] - window[0]["_time"]).to("day").value
            if not (min_span_days <= span_days <= max_span_days):
                continue
            sites = {site_from_row(w) for w in window if site_from_row(w)}
            if len(sites) < min_unique_sites:
                continue
            if len(window) > max_obs:
                idxs = np.linspace(0, len(window) - 1, max_obs, dtype=int)
                picked = [window[k] for k in idxs]
            else:
                picked = list(window)
            sites2 = {site_from_row(w) for w in picked if site_from_row(w)}
            if len(sites2) >= min_unique_sites:
                if verbose:
                    print(
                        f"Sliding found window {i}..{j} -> {len(picked)} obs, span {span_days:.1f} d, sites {len(sites2)}"
                    )
                return picked
    return None


def pick_by_random_trials(
    good: List[Dict[str, str]],
    min_obs: int,
    max_obs: int,
    min_span_days: float,
    max_span_days: float,
    min_unique_sites: int,
    seed: int = 12345,
    trials: int = 2000,
    verbose: bool = False,
) -> List[Dict[str, str]] | None:
    """
    Fallback: random sampling trials to find a combination satisfying constraints.
    We try many random subsets and return the first acceptable.
    """
    rng = random.Random(seed)
    n = len(good)
    if n < min_obs:
        return None
    for t in range(trials):
        k = rng.randint(min_obs, min(max_obs, n))
        idxs = sorted(rng.sample(range(n), k))
        subset = [good[i] for i in idxs]
        span_days = (subset[-1]["_time"] - subset[0]["_time"]).to("day").value
        if not (min_span_days <= span_days <= max_span_days):
            continue
        sites = {site_from_row(w) for w in subset if site_from_row(w)}
        if len(sites) >= min_unique_sites:
            if verbose:
                print(f"Random trial success at trial {t}: {k} obs, span {span_days:.1f} d, sites {len(sites)}")
            return subset
    return None


def write_selected_csv(selected: List[Dict[str, str]], out_path: Path) -> None:
    if not selected:
        raise ValueError("No selected rows to write")
    canonical = ["t_utc", "ra_deg", "dec_deg", "sigma_arcsec", "site"]
    first = selected[0]
    headers: List[str] = []
    for key in canonical:
        if key in first:
            headers.append(key)
    for key in first.keys():
        if key.startswith("_"):
            continue
        if key not in headers:
            headers.append(key)
    with out_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=headers)
        w.writeheader()
        for row in selected:
            outrow = {k: row.get(k, "") for k in headers}
            if "t_utc" not in outrow or not outrow.get("t_utc"):
                outrow["t_utc"] = row["_time"].isot
            w.writerow(outrow)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pool_path = out_dir / args.pool_file
    selected_path = out_dir / args.selected_file

    if not args.no_fetch or not pool_path.exists():
        if pool_path.exists() and args.no_fetch:
            print(f"Using existing pool: {pool_path}")
        else:
            print(f"[{datetime.utcnow().isoformat()}] Fetching pool to {pool_path} ...")
            shell_fetch_pool(pool_path, args.target, args.start, args.pool_n, args.sigma, args.verbose)
            print("Fetch complete.")
    else:
        print(f"Using existing pool file: {pool_path}")

    rows = load_csv_rows(pool_path)
    if args.verbose:
        print(f"Loaded {len(rows)} rows from pool")

    good: List[Dict[str, str]] = []
    for row in rows:
        try:
            t = time_from_row(row)
        except Exception:
            continue
        row["_time"] = t
        site = site_from_row(row)
        kind = get_site_kind(site)
        if kind in (SiteKind.FIXED, SiteKind.ROVING):
            good.append(row)

    if args.verbose:
        print(f"After filtering ground-based sites: {len(good)} candidate observations")

    if len(good) < args.min_obs:
        raise SystemExit(f"Not enough ground-based observations in pool ({len(good)} < {args.min_obs})")

    good.sort(key=lambda r: r["_time"].jd)

    selected = pick_window_by_sliding(
        good,
        args.min_obs,
        args.max_obs,
        args.min_span_days,
        args.max_span_days,
        args.min_unique_sites,
        verbose=args.verbose,
    )

    if selected is None:
        if args.verbose:
            print("Sliding window failed - trying random trials fallback...")
        selected = pick_by_random_trials(
            good,
            args.min_obs,
            args.max_obs,
            args.min_span_days,
            args.max_span_days,
            args.min_unique_sites,
            seed=args.seed,
            trials=3000,
            verbose=args.verbose,
        )

    if selected is None:
        raise SystemExit("Could not find a subset satisfying constraints. Try increasing pool size or relaxing constraints.")

    write_selected_csv(selected, selected_path)

    times = [s["_time"] for s in selected]
    span_days = (times[-1] - times[0]).to("day").value
    unique_sites = {site_from_row(s) for s in selected}
    print(f"Wrote selected epoch to {selected_path}")
    print(f"n_obs: {len(selected)}, span_days: {span_days:.2f}, unique_sites: {len(unique_sites)}")
    print("Sites:", sorted(unique_sites))
    if args.verbose:
        for row in selected:
            print(f"  {row['_time'].isot}  {site_from_row(row)}  ra={row.get('ra_deg')} dec={row.get('dec_deg')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
