from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .fit import load_posterior, sample_replicas
from .propagate import predict_radec
from .fit_cli import load_observations
from .ranging import sample_ranged_replicas


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
    args = parser.parse_args()

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
        for idx in range(replicas.shape[1]):
            state = replicas[:, idx]
            ra, dec = predict_radec(state, posterior.epoch)
            writer.writerow(
                [idx, *(f"{val:.6f}" for val in state), f"{ra:.6f}", f"{dec:.6f}"]
            )

    print(f"Wrote {args.n} replicas to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
