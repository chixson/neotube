#!/usr/bin/env python3
"""
Recompute full-physics residuals for replicas and write Gaussian weights.

This is a lightweight example helper. It expects a replicas CSV that includes
state columns (x_km,y_km,z_km,vx_km_s,vy_km_s,vz_km_s) and an observations CSV
that load_observations can parse.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from astropy.time import Time

from neotube.fit_cli import load_observations
from neotube.propagate import predict_radec_from_epoch


def load_replicas_csv(path: Path) -> np.ndarray:
    rows = []
    with path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)
    states = []
    for row in rows:
        try:
            states.append(
                [
                    float(row.get("x_km", row.get("x", 0.0))),
                    float(row.get("y_km", row.get("y", 0.0))),
                    float(row.get("z_km", row.get("z", 0.0))),
                    float(row.get("vx_km_s", row.get("vx", 0.0))),
                    float(row.get("vy_km_s", row.get("vy", 0.0))),
                    float(row.get("vz_km_s", row.get("vz", 0.0))),
                ]
            )
        except ValueError as exc:
            raise ValueError("Replica CSV has non-numeric state columns.") from exc
    if not states:
        raise ValueError("No replicas found in CSV.")
    return np.array(states, dtype=float)


def tangent_residuals(ra_pred: np.ndarray, dec_pred: np.ndarray, obs) -> np.ndarray:
    res = np.zeros(2 * len(obs), dtype=float)
    for idx, (ra_p, dec_p, ob) in enumerate(zip(ra_pred, dec_pred, obs)):
        delta_ra = ((ob.ra_deg - ra_p + 180.0) % 360.0) - 180.0
        ra_arcsec = delta_ra * np.cos(np.deg2rad(dec_p)) * 3600.0
        dec_arcsec = (ob.dec_deg - dec_p) * 3600.0
        res[2 * idx] = ra_arcsec
        res[2 * idx + 1] = dec_arcsec
    return res


def sigma_vector(obs) -> np.ndarray:
    sigma = np.zeros(2 * len(obs), dtype=float)
    for idx, ob in enumerate(obs):
        sigma[2 * idx] = ob.sigma_arcsec
        sigma[2 * idx + 1] = ob.sigma_arcsec
    return sigma


def main() -> int:
    parser = argparse.ArgumentParser(description="Finalize replica weights with full-physics.")
    parser.add_argument("--replicas", type=Path, required=True)
    parser.add_argument("--obs", type=Path, required=True)
    parser.add_argument("--epoch", type=str, required=True, help="UTC epoch of replica states.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--perturbers", nargs="+", default=["earth", "mars", "jupiter"])
    parser.add_argument("--max-step", type=float, default=3600.0)
    parser.add_argument("--no-kepler", action="store_true")
    args = parser.parse_args()

    states = load_replicas_csv(args.replicas)
    obs = load_observations(args.obs, None)
    epoch = Time(args.epoch, scale="utc")
    use_kepler = not args.no_kepler

    sigmas = sigma_vector(obs)
    weights = np.empty(len(states), dtype=float)
    for idx, state in enumerate(states):
        ra, dec = predict_radec_from_epoch(
            state,
            epoch,
            obs,
            args.perturbers,
            args.max_step,
            use_kepler=use_kepler,
            full_physics=True,
        )
        res = tangent_residuals(ra, dec, obs)
        chi2 = float(np.sum((res / sigmas) ** 2))
        weights[idx] = float(np.exp(-0.5 * chi2))

    with args.out.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["replica_index", "weight"])
        for idx, w in enumerate(weights):
            writer.writerow([idx, w])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
