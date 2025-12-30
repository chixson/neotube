from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from astropy.time import Time

from .fit import _resolve_spacecraft_offset, fit_orbit, predict_orbit
from .models import Observation


def load_observations(path: Path, sigma: float | None) -> list[Observation]:
    observations: list[Observation] = []
    with path.open() as fh:
        reader = csv.DictReader(fh)
        pos_keys = ("obs_x_km", "obs_y_km", "obs_z_km")
        for row in reader:
            if not row.get("t_utc") or not row.get("ra_deg") or not row.get("dec_deg"):
                continue
            obs_sigma = sigma
            if row.get("sigma_arcsec"):
                obs_sigma = float(row["sigma_arcsec"])
            if obs_sigma is None:
                raise ValueError("Observation row missing sigma_arcsec and no default provided.")
            obs_time = Time(row["t_utc"], scale="utc")
            observer_pos_km = None
            if any(key in row for key in pos_keys):
                raw_vals = [row.get(key) for key in pos_keys]
                if any(val not in (None, "") for val in raw_vals):
                    if any(val in (None, "") for val in raw_vals):
                        raise ValueError(
                            "Observation row has incomplete observer position; "
                            "expected obs_x_km, obs_y_km, obs_z_km."
                        )
                    observer_pos_km = np.array([float(val) for val in raw_vals], dtype=float)
            mag_val = None
            sigma_mag = None
            for key in ("mag", "mag_app", "v_mag", "V", "Vmag", "v"):
                if row.get(key) not in (None, ""):
                    mag_val = float(row[key])
                    break
            for key in ("sigma_mag", "mag_sigma", "mag_err"):
                if row.get(key) not in (None, ""):
                    sigma_mag = float(row[key])
                    break
            obs = Observation(
                time=obs_time,
                ra_deg=float(row["ra_deg"]),
                dec_deg=float(row["dec_deg"]),
                sigma_arcsec=obs_sigma,
                site=row.get("site"),
                observer_pos_km=observer_pos_km,
                mag=mag_val,
                sigma_mag=sigma_mag,
            )
            if obs.observer_pos_km is None and obs.site:
                try:
                    sc_offset = _resolve_spacecraft_offset(obs)
                except Exception:
                    sc_offset = None
                if sc_offset is not None:
                    obs.observer_pos_km = sc_offset
            observations.append(obs)
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
    parser.add_argument(
        "--likelihood",
        choices=["gaussian", "studentt"],
        default="gaussian",
        help="Statistical likelihood for the fit.",
    )
    parser.add_argument(
        "--nu",
        type=float,
        default=4.0,
        help="Degrees of freedom for Student-t likelihood (used when --likelihood studentt).",
    )
    parser.add_argument(
        "--no-kepler",
        action="store_true",
        help="Disable Kepler propagation and force the full ODE integrator.",
    )
    parser.add_argument(
        "--estimate-site-scales",
        action="store_true",
        help="Estimate per-site sigma scaling factors via a preliminary fit and re-fit with scaled sigmas.",
    )
    parser.add_argument(
        "--max-kappa",
        type=float,
        default=10.0,
        help="Maximum allowed per-site kappa when estimating site scales.",
    )
    parser.add_argument(
        "--estimate-site-scales-method",
        choices=["mad", "chi2", "iterative", "studentt_em"],
        default="mad",
        help="Method to estimate per-site scale factors: 'mad', 'chi2', 'iterative', or 'studentt_em' (EM Student-t).",
    )
    parser.add_argument(
        "--estimate-site-scales-iters",
        type=int,
        default=5,
        help="Maximum iterations for the 'iterative' site-scale estimator (default 5).",
    )
    parser.add_argument(
        "--estimate-site-scales-alpha",
        type=float,
        default=0.4,
        help="Damping factor (0..1) applied to kappa updates in iterative estimators (default 0.4).",
    )
    parser.add_argument(
        "--estimate-site-scales-tol",
        type=float,
        default=1e-3,
        help="Convergence tolerance for per-site kappa changes (default 1e-3).",
    )
    parser.add_argument(
        "--sigma-floor",
        type=float,
        default=0.0,
        help="Minimum allowed per-observation sigma (arcsec) after scaling (default 0.0).",
    )
    parser.add_argument(
        "--allow-unknown-site",
        action="store_true",
        help="Allow unknown or misconfigured observatory sites (fallback to default Earth).",
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
        "likelihood": args.likelihood,
        "nu": args.nu,
        "allow_unknown_site": args.allow_unknown_site,
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
            likelihood=args.likelihood,
            nu=args.nu,
            estimate_site_scales=args.estimate_site_scales,
            max_kappa=args.max_kappa,
            estimate_site_scales_method=args.estimate_site_scales_method,
            estimate_site_scales_iters=args.estimate_site_scales_iters,
            estimate_site_scales_alpha=args.estimate_site_scales_alpha,
            estimate_site_scales_tol=args.estimate_site_scales_tol,
            sigma_floor=args.sigma_floor,
            allow_unknown_site=args.allow_unknown_site,
            use_kepler=not args.no_kepler,
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
        use_kepler=not args.no_kepler,
        allow_unknown_site=args.allow_unknown_site,
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
        "fit_scale": float(getattr(posterior, "fit_scale", 1.0)),
        "nu": float(getattr(posterior, "nu", 4.0)),
    }
    if getattr(posterior, "site_kappas", None):
        posterior_json["fit"]["site_kappas"] = posterior.site_kappas
        posterior_json["site_kappas"] = posterior.site_kappas

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
        fit_scale=float(getattr(posterior, "fit_scale", 1.0)),
        nu=float(getattr(posterior, "nu", 4.0)),
        site_kappas=json.dumps(getattr(posterior, "site_kappas", {})),
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
