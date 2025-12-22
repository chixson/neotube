from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .fit import load_posterior, sample_replicas
from .propagate import predict_radec


def main() -> int:
    parser = argparse.ArgumentParser(description="Sample replicas from an existing orbit posterior.")
    parser.add_argument("--posterior", type=Path, required=True, help="Path to posterior .npz artifact.")
    parser.add_argument("--n", type=int, default=500, help="Number of replicas to sample.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--output", type=Path, default=Path("replicas.csv"), help="Output path for replica table.")
    parser.add_argument("--perturbers", nargs="+", default=["earth", "mars", "jupiter"], help="Perturbers for propagation.")
    args = parser.parse_args()

    posterior = load_posterior(args.posterior)
    replicas = sample_replicas(posterior, args.n, seed=args.seed)

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
