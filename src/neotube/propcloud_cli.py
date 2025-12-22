from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from astropy.time import Time

from .propagate import ReplicaCloud, propagate_replicas, predict_radec


def load_times(path: Path) -> list[Time]:
    results: list[Time] = []
    with path.open() as fh:
        reader = csv.DictReader(fh)
        if "time_utc" not in reader.fieldnames:
            raise ValueError("times file requires a 'time_utc' column.")
        for row in reader:
            results.append(Time(row["time_utc"], scale="utc"))
    if not results:
        raise ValueError("No times loaded.")
    return results


def load_replicas(path: Path) -> tuple[np.ndarray, list[int]]:
    states = []
    ids = []
    with path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            ids.append(int(row["replica_id"]))
            states.append(
                [
                    float(row["x_km"]),
                    float(row["y_km"]),
                    float(row["z_km"]),
                    float(row["vx_km_s"]),
                    float(row["vy_km_s"]),
                    float(row["vz_km_s"]),
                ]
            )
    if not states:
        raise ValueError("No replicas found.")
    return np.array(states).T, ids


def main() -> int:
    parser = argparse.ArgumentParser(description="Propagate replica states to requested epochs.")
    parser.add_argument("--replicas", type=Path, required=True, help="CSV from neotube-replicas.")
    parser.add_argument("--meta", type=Path, required=True, help="JSON metadata describing replica epoch.")
    parser.add_argument("--times", type=Path, required=True, help="CSV with column time_utc.")
    parser.add_argument("--perturbers", nargs="+", default=["earth", "mars", "jupiter"], help="Perturbers for propagation.")
    parser.add_argument("--output", type=Path, default=Path("cloud.csv"), help="Output cloud CSV path.")
    parser.add_argument("--workers", type=int, default=None, help="ProcessPool workers for propagation.")
    parser.add_argument("--batch-size", type=int, default=50, help="Number of replicas per batch.")
    parser.add_argument("--max-step", type=float, default=300.0, help="Max step size (seconds) for propagate_state.")
    args = parser.parse_args()

    with args.meta.open() as fh:
        meta = json.load(fh)
    epoch = Time(meta["epoch_utc"], scale="utc")

    states, ids = load_replicas(args.replicas)
    times = load_times(args.times)
    cloud = ReplicaCloud(epoch=epoch, states=states)
    propagated = propagate_replicas(
        cloud,
        times,
        tuple(args.perturbers),
        max_step=args.max_step,
        workers=args.workers,
        batch_size=args.batch_size,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["time_utc", "replica_id", "ra_deg", "dec_deg"])
        for t, states_at in zip(times, propagated):
            time_iso = t.isot
            for rid in range(states_at.shape[1]):
                ra, dec = predict_radec(states_at[:, rid], t)
                writer.writerow([time_iso, rid, f"{ra:.6f}", f"{dec:.6f}"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
