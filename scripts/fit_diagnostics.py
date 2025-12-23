#!/usr/bin/env python3
"""
fit_diagnostics.py
------------------
Summarize fit residuals from a posterior + observation CSV.

Outputs RMS, chi2, chi2_red, and per-observation residuals
using the same normalization as neotube fit diagnostics.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from neotube.fit import load_posterior
from neotube.fit_cli import load_observations


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize fit residuals vs observation sigmas.")
    parser.add_argument("--obs", type=Path, required=True, help="Observation CSV used in the fit.")
    parser.add_argument("--posterior", type=Path, required=True, help="Posterior .npz from neotube-fit.")
    parser.add_argument("--sigma-arcsec", type=float, default=None, help="Override sigma if not present in CSV.")
    parser.add_argument("--out-csv", type=Path, default=None, help="Optional CSV to write per-observation diagnostics.")
    args = parser.parse_args()

    obs = load_observations(args.obs, args.sigma_arcsec)
    posterior = load_posterior(args.posterior)

    res = posterior.residuals
    sigma = np.array([ob.sigma_arcsec for ob in obs for _ in (0, 1)], dtype=float)
    normed = res / sigma

    rms = float(np.sqrt(np.mean(res ** 2)))
    chi2 = float((normed ** 2).sum())
    ndof = 2 * len(obs) - 6
    chi2_red = chi2 / ndof if ndof > 0 else float("nan")

    print("RMS (arcsec):", rms)
    print("chi2:", chi2)
    print("chi2_red:", chi2_red)

    per_obs = np.sqrt(res.reshape(-1, 2) ** 2).sum(axis=1)
    rows: list[dict[str, object]] = []
    for i, o in enumerate(obs):
        row = {
            "index": i,
            "time_utc": o.time.isot,
            "per_obs_abs_arcsec": float(per_obs[i]),
            "res_ra_arcsec": float(res[2 * i]),
            "res_dec_arcsec": float(res[2 * i + 1]),
            "normed_ra": float(normed[2 * i]),
            "normed_dec": float(normed[2 * i + 1]),
            "site": o.site,
        }
        rows.append(row)
        print(i, o.time.isot, per_obs[i], "normed RA:", normed[2 * i], "normed Dec:", normed[2 * i + 1])

    if args.out_csv:
        _write_csv(args.out_csv, rows)


if __name__ == "__main__":
    main()
