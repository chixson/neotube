from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from astropy.time import Time

from .fit import fit_orbit, predict_orbit
from .models import Observation


def load_observations(path: Path, sigma: float | None) -> list[Observation]:
    observations: list[Observation] = []
    with path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if not row.get("t_utc") or not row.get("ra_deg") or not row.get("dec_deg"):
                continue
            obs_sigma = sigma
            if row.get("sigma_arcsec"):
                obs_sigma = float(row["sigma_arcsec"])
            if obs_sigma is None:
                raise ValueError("Observation row missing sigma_arcsec and no default provided.")
            obs_time = Time(row["t_utc"], scale="utc")
            observations.append(
                Observation(
                    time=obs_time,
                    ra_deg=float(row["ra_deg"]),
                    dec_deg=float(row["dec_deg"]),
                    sigma_arcsec=obs_sigma,
                    site=row.get("site"),
                )
            )
    if not observations:
        raise ValueError("No valid observations loaded from CSV.")
    observations.sort(key=lambda ob: ob.time)
    return observations


def _tangent_residual(ra_obs: float, dec_obs: float, ra_pred: float, dec_pred: float) -> tuple[float, float]:
    delta_ra = ((ra_obs - ra_pred + 180.0) % 360.0) - 180.0
    ra_arcsec = delta_ra * np.cos(np.deg2rad(dec_pred)) * 3600.0
    dec_arcsec = (dec_obs - dec_pred) * 3600.0
    return ra_arcsec, dec_arcsec


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit MPC astrometry into an orbit posterior.")
    parser.add_argument("--obs", type=Path, required=True, help="CSV file with obs: t_utc, ra_deg, dec_deg, sigma_arcsec")
    parser.add_argument("--target", required=True, help="Identifier used for Horizons seed (e.g. MPC number).")
    parser.add_argument("--sigma-arcsec", type=float, default=None, help="Default sigma if not in CSV.")
    parser.add_argument("--perturbers", nargs="+", default=["earth", "mars", "jupiter"], help="Perturbers for fit.")
    parser.add_argument("--max-step", type=float, default=3600.0, help="Max step (seconds) for integration.")
    parser.add_argument("--max-iter", type=int, default=6, help="Max Gauss-Newton iterations.")
    parser.add_argument(
        "--seed-method",
        choices=["horizons", "observations", "gauss", "attributable"],
        default="attributable",
        help="Seed initializer: horizons, observations, gauss, or attributable.",
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory to write artifacts.")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARN"], default="INFO", help="Logging level.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    observations = load_observations(args.obs, args.sigma_arcsec)
    params = {
        "target": args.target,
        "perturbers": args.perturbers,
        "sigma_arcsec": args.sigma_arcsec,
        "obs_file": str(args.obs),
        "seed_method": args.seed_method,
    }
    with open(args.out_dir / "fit_params.json", "w") as fh:
        json.dump(params, fh, indent=2)

    try:
        posterior = fit_orbit(
            args.target,
            observations,
            perturbers=tuple(args.perturbers),
            max_step=args.max_step,
            max_iter=args.max_iter,
            seed_method=args.seed_method,
        )
    except RuntimeError as exc:
        summary = {
            "n_obs": len(observations),
            "rms_arcsec": None,
            "chi2": None,
            "converged": False,
            "perturbers": args.perturbers,
            "error": str(exc),
        }
        with open(args.out_dir / "fit_summary.json", "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"Fit failed: {exc}", file=sys.stderr)
        return 1

    pred_ra, pred_dec = predict_orbit(
        posterior.state,
        posterior.epoch,
        observations,
        tuple(args.perturbers),
        args.max_step,
    )

    residuals_rows = []
    chi2 = 0.0
    for obs, ra_pred, dec_pred in zip(observations, pred_ra, pred_dec):
        res_ra, res_dec = _tangent_residual(obs.ra_deg, obs.dec_deg, ra_pred, dec_pred)
        residuals_rows.append(
            {
                "time_utc": obs.time.isot,
                "ra_obs_deg": obs.ra_deg,
                "dec_obs_deg": obs.dec_deg,
                "ra_pred_deg": ra_pred,
                "dec_pred_deg": dec_pred,
                "res_ra_arcsec": res_ra,
                "res_dec_arcsec": res_dec,
                "sigma_arcsec": obs.sigma_arcsec,
            }
        )
        chi2 += (res_ra / obs.sigma_arcsec) ** 2 + (res_dec / obs.sigma_arcsec) ** 2

    summary = {
        "n_obs": len(observations),
        "rms_arcsec": posterior.rms_arcsec,
        "seed_rms_arcsec": posterior.seed_rms_arcsec,
        "chi2": chi2,
        "converged": posterior.converged,
        "perturbers": args.perturbers,
        "error": None,
    }

    posterior_json = {
        "epoch_utc": posterior.epoch.isot,
        "state_km": posterior.state.tolist(),
        "cov_km2": posterior.cov.tolist(),
        "fit": summary,
    }

    npz_path = args.out_dir / "posterior.npz"
    np.savez(
        npz_path,
        state=posterior.state,
        cov=posterior.cov,
        residuals=posterior.residuals,
        epoch=posterior.epoch.isot,
        rms=posterior.rms_arcsec,
        converged=posterior.converged,
        seed_rms=posterior.seed_rms_arcsec if posterior.seed_rms_arcsec is not None else np.nan,
    )

    with open(args.out_dir / "posterior.json", "w") as fh:
        json.dump(posterior_json, fh, indent=2)

    with open(args.out_dir / "fit_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    with open(args.out_dir / "residuals.csv", "w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "time_utc",
                "ra_obs_deg",
                "dec_obs_deg",
                "ra_pred_deg",
                "dec_pred_deg",
                "res_ra_arcsec",
                "res_dec_arcsec",
                "sigma_arcsec",
            ],
        )
        writer.writeheader()
        writer.writerows(residuals_rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
