from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from astropy.time import Time

from .fit import load_posterior, sample_replicas
from .propagate import predict_radec_batch
from .fit_cli import load_observations
from .ranging import add_local_spread_parallel, sample_ranged_replicas


def _radec_chunk_worker(payload):
    chunk_states, epoch_str = payload
    epoch = Time(epoch_str, scale="utc")
    epochs = [epoch] * chunk_states.shape[0]
    ra, dec = predict_radec_batch(chunk_states, epochs)
    return ra, dec


def main() -> int:
    parser = argparse.ArgumentParser(description="Sample replicas from an existing orbit posterior.")
    parser.add_argument("--posterior", type=Path, required=True, help="Path to posterior .npz artifact.")
    parser.add_argument("--n", type=int, default=500, help="Number of replicas to sample.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--output", type=Path, default=Path("replicas.csv"), help="Output path for replica table.")
    parser.add_argument("--perturbers", nargs="+", default=["earth", "mars", "jupiter"], help="Perturbers for propagation.")
    parser.add_argument(
        "--method",
        choices=["multit", "gaussian"],
        default="multit",
        help="Sampling method ('multit'=Student-t, 'gaussian'=Gaussian).",
    )
    parser.add_argument("--nu", type=float, default=4.0, help="Degrees of freedom when using Student-t sampling.")
    parser.add_argument(
        "--ranged",
        action="store_true",
        help="Use ranged/importance sampling to respect distance degeneracy.",
    )
    parser.add_argument("--obs", type=Path, default=None, help="Observation CSV for ranged sampling.")
    parser.add_argument("--n-proposals", type=int, default=50000, help="Number of ranged proposals.")
    parser.add_argument("--rho-min-au", type=float, default=1e-4, help="Min rho (AU).")
    parser.add_argument("--rho-max-au", type=float, default=5.0, help="Max rho (AU).")
    parser.add_argument("--rhodot-max-kms", type=float, default=50.0, help="Max |rho_dot| (km/s).")
    parser.add_argument(
        "--range-profile",
        choices=["neo", "main-belt", "jupiter-trojan", "tno", "comet", "wide", "main", "mba"],
        default=None,
        help=(
            "Preset rho/rhodot ranges for common object classes. "
            "Choices:\n"
            "  neo             (near-earth objects)      : rho=[1e-4,2.0] AU, rhodot<=100 km/s\n"
            "  main-belt (mba) (main-belt asteroids)    : rho=[1.8,4.5] AU, rhodot<=20 km/s\n"
            "  jupiter-trojan  (Trojan asteroids)       : rho=[4.5,5.7] AU, rhodot<=10 km/s\n"
            "  tno              (trans-Neptunian objects): rho=[20,100] AU, rhodot<=5 km/s\n"
            "  comet            (comets, wide eccentric) : rho=[0.01,50] AU, rhodot<=200 km/s\n"
            "  wide             (very wide exploratory)  : rho=[1e-4,100] AU, rhodot<=100 km/s\n"
            "Aliases: 'main' or 'mba' -> 'main-belt'."
        ),
    )
    parser.add_argument(
        "--rho-prior",
        choices=["volume", "log", "uniform"],
        default="log",
        help="Prior for rho proposals: volume (~rho^2), log (1/rho), or uniform.",
    )
    parser.add_argument(
        "--rho-prior-power",
        type=float,
        default=2.0,
        help="Power for rho prior (log weight term = power * log(rho)).",
    )
    parser.add_argument(
        "--scoring-mode",
        choices=["kepler", "nbody"],
        default="kepler",
        help="Scoring mode for ranged sampling: kepler prefilter or nbody only.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Chunk size for ranged proposal scoring (defaults to proposals//workers or 128).",
    )
    parser.add_argument(
        "--radec-chunk-size",
        type=int,
        default=2048,
        help="Chunk size for RA/Dec computation.",
    )
    parser.add_argument(
        "--top-k-nbody",
        type=int,
        default=2000,
        help="Top-K proposals to rescore with n-body after Kepler prefilter.",
    )
    parser.add_argument("--n-workers", type=int, default=8, help="Workers for ranged scoring.")
    parser.add_argument(
        "--emit-debug",
        action="store_true",
        help="If set, write ranging_debug.npz with proposals/weights/top-candidates for post-mortem.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=500,
        help="Log progress every N ranged proposals (0 disables).",
    )
    parser.add_argument("--max-step", type=float, default=3600.0, help="Max step (seconds) for propagation.")
    parser.add_argument("--no-kepler", action="store_true", help="Disable Kepler propagation for ranged scoring.")
    parser.add_argument(
        "--local-spread-n",
        type=int,
        default=0,
        help="Per-state tangent jitter count (0 disables).",
    )
    parser.add_argument(
        "--local-spread-sigma-arcsec",
        type=float,
        default=0.5,
        help="Per-state tangent jitter sigma (arcsec).",
    )
    parser.add_argument(
        "--local-spread-mode",
        choices=("tangent", "multit", "attributable"),
        default="tangent",
        help=(
            "Which local-spread generator to use (tangent=existing, multit=multivariate-t, "
            "attributable=attributable-space)."
        ),
    )
    parser.add_argument(
        "--local-spread-vel-scale",
        type=float,
        default=1.0,
        help="velocity scale for local spread (multit/tangent modes).",
    )
    parser.add_argument(
        "--local-spread-df",
        type=float,
        default=4.0,
        help="degrees-of-freedom for multivariate-t.",
    )
    args = parser.parse_args()
    # Apply range-profile overrides (explicit profile wins)
    if args.range_profile is not None:
        prof = args.range_profile
        if prof in ("main", "mba"):
            prof = "main-belt"
        if prof == "neo":
            args.rho_min_au = 1e-4
            args.rho_max_au = 2.0
            args.rhodot_max_kms = 100.0
        elif prof == "main-belt":
            args.rho_min_au = 1.8
            args.rho_max_au = 4.5
            args.rhodot_max_kms = 20.0
        elif prof == "jupiter-trojan":
            args.rho_min_au = 4.5
            args.rho_max_au = 5.7
            args.rhodot_max_kms = 10.0
        elif prof == "tno":
            args.rho_min_au = 20.0
            args.rho_max_au = 100.0
            args.rhodot_max_kms = 5.0
        elif prof == "comet":
            args.rho_min_au = 0.01
            args.rho_max_au = 50.0
            args.rhodot_max_kms = 200.0
        elif prof == "wide":
            args.rho_min_au = 1e-4
            args.rho_max_au = 100.0
            args.rhodot_max_kms = 100.0
        print(
            f"[replica_cli] range_profile={args.range_profile} -> "
            f"rho=[{args.rho_min_au},{args.rho_max_au}] AU, rhodot_max={args.rhodot_max_kms} km/s"
        )

    posterior = load_posterior(args.posterior)
    nu_val = posterior.nu if posterior.nu is not None else args.nu
    if args.ranged:
        if args.obs is None:
            raise SystemExit("--ranged requires --obs (observation CSV).")
        observations = load_observations(args.obs, None)
        scoring_mode = "nbody" if args.no_kepler else args.scoring_mode
        ranged = sample_ranged_replicas(
            observations=observations,
            epoch=posterior.epoch,
            n_replicas=args.n,
            n_proposals=args.n_proposals,
            rho_min_au=args.rho_min_au,
            rho_max_au=args.rho_max_au,
            rhodot_max_kms=args.rhodot_max_kms,
            perturbers=tuple(args.perturbers),
            max_step=args.max_step,
            nu=nu_val,
            site_kappas=getattr(posterior, "site_kappas", {}),
            seed=args.seed,
            log_every=args.log_every,
            scoring_mode=scoring_mode,
            n_workers=args.n_workers,
            chunk_size=args.chunk_size,
            top_k_nbody=args.top_k_nbody,
            rho_prior_power=args.rho_prior_power,
            rho_prior_mode=args.rho_prior,
        )
        weights = ranged["weights"]
        states = ranged["states"]
        if args.emit_debug:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            diag_path = args.output.parent / "ranging_debug.npz"
            top_k = min(len(weights), max(1, args.top_k_nbody))
            top_idx = np.argsort(-weights)[:top_k]
            try:
                np.savez(
                    diag_path,
                    rhos=ranged.get("rhos"),
                    rhodots=ranged.get("rhodots"),
                    weights=weights,
                    top_idx=top_idx,
                    top_rhos=ranged.get("rhos")[top_idx],
                    top_rhodots=ranged.get("rhodots")[top_idx],
                    top_weights=weights[top_idx],
                    top_states=states[top_idx],
                )
                print("Wrote diagnostics to", diag_path)
            except Exception as exc:
                print("Failed to write debug npz:", exc)
        ess = 1.0 / np.sum(weights**2)
        if ess < max(50, 0.05 * len(weights)):
            from .ranging import stratified_resample

            replicas = stratified_resample(
                states,
                weights,
                nrep=args.n,
                n_clusters=12,
                jitter_scale=1e-6,
                nu=nu_val,
                seed=args.seed,
            ).T
        else:
            idx = np.random.default_rng(int(args.seed)).choice(
                len(states), size=args.n, replace=True, p=weights
            )
            replicas = states[idx].T
    else:
        replicas = sample_replicas(posterior, args.n, seed=args.seed, method=args.method, nu=nu_val)

    # replicas currently shaped (6, N)
    # Optionally expand each sampled state with local tangent-plane jitter to create thickness
    if args.local_spread_n and args.local_spread_n > 0:
        if args.ranged:
            obs_for_jitter = observations
        else:
            if args.obs is None:
                raise SystemExit("--local-spread requires --obs when not using --ranged")
            obs_for_jitter = args.obs

        states = replicas.T.copy()
        jittered = add_local_spread_parallel(
            states,
            obs_for_jitter,
            posterior,
            mode=args.local_spread_mode,
            n_per_state=int(args.local_spread_n),
            sigma_arcsec=float(args.local_spread_sigma_arcsec),
            fit_scale=float(getattr(posterior, "fit_scale", 1.0)),
            site_kappas=getattr(posterior, "site_kappas", {}),
            vel_scale_factor=float(args.local_spread_vel_scale),
            df=float(args.local_spread_df),
            n_workers=int(args.n_workers),
            chunk_size=int(args.chunk_size) if args.chunk_size is not None else None,
            seed=int(args.seed) if args.seed is not None else None,
        )
        combined = np.vstack([states, jittered])
        rng = np.random.default_rng(int(args.seed))
        if combined.shape[0] >= args.n:
            idxs = rng.choice(combined.shape[0], size=args.n, replace=False)
        else:
            idxs = rng.choice(combined.shape[0], size=args.n, replace=True)
        chosen = combined[idxs]
        replicas = chosen.T

    args.output.parent.mkdir(parents=True, exist_ok=True)
    meta_path = args.output.with_name(args.output.stem + "_meta.json")
    with meta_path.open("w") as fh:
        json.dump(
            {
                "epoch_utc": posterior.epoch.isot,
                "n": args.n,
                "seed": args.seed,
                "posterior": str(args.posterior),
                "perturbers": args.perturbers,
                "method": args.method,
                "nu": nu_val,
                "fit_scale": float(getattr(posterior, "fit_scale", 1.0)),
                "ranged": args.ranged,
                "obs": str(args.obs) if args.obs else None,
                "n_proposals": args.n_proposals,
                "rho_min_au": args.rho_min_au,
                "rho_max_au": args.rho_max_au,
                "rhodot_max_kms": args.rhodot_max_kms,
                "rho_prior": args.rho_prior,
                "rho_prior_power": args.rho_prior_power,
                "scoring_mode": "nbody" if args.no_kepler else args.scoring_mode,
                "chunk_size": args.chunk_size,
                "top_k_nbody": args.top_k_nbody,
                "n_workers": args.n_workers,
                "log_every": args.log_every,
            },
            fh,
            indent=2,
        )

    # Compute RA/Dec in parallel batches to avoid a slow serial loop.
    epoch_isot = posterior.epoch.isot
    states = replicas.T.copy()
    total = states.shape[0]
    radec_chunk = max(1, int(args.radec_chunk_size))

    schedule = []
    for start in range(0, total, radec_chunk):
        schedule.append((states[start : start + radec_chunk], epoch_isot))

    max_workers = max(1, int(args.n_workers or 1))
    actual_workers = min(max_workers, len(schedule))

    with args.output.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "replica_id",
                "x_km",
                "y_km",
                "z_km",
                "vx_km_s",
                "vy_km_s",
                "vz_km_s",
                "ra_deg",
                "dec_deg",
            ]
        )
        idx_offset = 0
        if actual_workers <= 1 or len(schedule) == 1:
            for chunk_states, epoch_str in schedule:
                ra, dec = _radec_chunk_worker((chunk_states, epoch_str))
                for i in range(chunk_states.shape[0]):
                    state = chunk_states[i]
                    writer.writerow(
                        [
                            idx_offset + i,
                            *(f"{val:.6f}" for val in state),
                            f"{ra[i]:.6f}",
                            f"{dec[i]:.6f}",
                        ]
                    )
                idx_offset += chunk_states.shape[0]
        else:
            with ProcessPoolExecutor(max_workers=actual_workers) as executor:
                for (chunk_states, _), (ra, dec) in zip(
                    schedule, executor.map(_radec_chunk_worker, schedule)
                ):
                    for i in range(chunk_states.shape[0]):
                        state = chunk_states[i]
                        writer.writerow(
                            [
                                idx_offset + i,
                                *(f"{val:.6f}" for val in state),
                                f"{ra[i]:.6f}",
                                f"{dec[i]:.6f}",
                            ]
                        )
                    idx_offset += chunk_states.shape[0]

    print(f"Wrote {args.n} replicas to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
