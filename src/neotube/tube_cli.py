from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_cloud(path: Path) -> dict[str, list[tuple[float, float]]]:
    groups: dict[str, list[tuple[float, float]]] = defaultdict(list)
    with path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            ra = float(row["ra_deg"])
            dec = float(row["dec_deg"])
            groups[row["time_utc"]].append((ra, dec))
    if not groups:
        raise ValueError("Cloud file is empty.")
    return groups


def circular_mean(angles_deg: np.ndarray) -> float:
    radians = np.deg2rad(angles_deg)
    return float(np.rad2deg(np.arctan2(np.sum(np.sin(radians)), np.sum(np.cos(radians)))) % 360.0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compress replica clouds into tube nodes.")
    parser.add_argument("--cloud", type=Path, required=True, help="CSV from neotube-propcloud.")
    parser.add_argument("--cred", type=float, default=0.99, help="Credible level for radius.")
    parser.add_argument("--margin-arcsec", type=float, default=5.0, help="Additive margin.")
    parser.add_argument("--output", type=Path, default=Path("tube_nodes.csv"), help="Output CSV path.")
    args = parser.parse_args()

    groups = load_cloud(args.cloud)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["time_utc", "center_ra_deg", "center_dec_deg", "radius_arcsec", "cred"])
        for time_iso, points in sorted(groups.items()):
            ras = np.array([p[0] for p in points])
            decs = np.array([p[1] for p in points])
            center_ra = circular_mean(ras)
            center_dec = float(np.mean(decs))
            delta_ra = ((ras - center_ra + 180.0) % 360.0) - 180.0
            distances = np.hypot(delta_ra * np.cos(np.deg2rad(center_dec)), decs - center_dec) * 3600.0
            radius = float(np.quantile(distances, args.cred)) + args.margin_arcsec
            writer.writerow([time_iso, f"{center_ra:.6f}", f"{center_dec:.6f}", f"{radius:.3f}", f"{args.cred:.3f}"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
