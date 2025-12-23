#!/usr/bin/env python3
"""Plot replica spreads in RA/Dec and PCA space for debugging."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_replicas(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "ra_deg" not in df.columns or "dec_deg" not in df.columns:
        raise ValueError("replica CSV must contain 'ra_deg' and 'dec_deg' columns")
    return df


def tangent_offsets(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float, float]:
    ra0 = float(df["ra_deg"].mean())
    dec0 = float(df["dec_deg"].mean())
    cosd = np.cos(np.deg2rad(dec0))
    dra = (df["ra_deg"].to_numpy(dtype=float) - ra0) * cosd * 3600.0
    ddec = (df["dec_deg"].to_numpy(dtype=float) - dec0) * 3600.0
    return dra, ddec, ra0, dec0


def pca_components(dra: np.ndarray, ddec: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.vstack([dra, ddec])
    cov = np.cov(X)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]
    projected = eigvecs.T @ X
    return projected[0], projected[1], eigvecs


def plot_ra_dec(
    dra: np.ndarray,
    ddec: np.ndarray,
    out_path: Path,
    jpl_offset: tuple[float, float] | None,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(dra, ddec, s=6, alpha=0.3, label="replicas")
    if jpl_offset is not None:
        ax.plot(jpl_offset[0], jpl_offset[1], "r+", markersize=12, mew=2, label="JPL")
    ax.set_xlabel("ΔRA cosδ (arcsec)")
    ax.set_ylabel("ΔDec (arcsec)")
    ax.set_title("Replica cloud (tangent plane)")
    ax.grid(True, lw=0.5, color="#dddddd")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)


def plot_pca(
    pc1: np.ndarray,
    pc2: np.ndarray,
    out_path: Path,
    jpl_pc: tuple[float, float] | None,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(pc1, pc2, s=6, alpha=0.3)
    if jpl_pc is not None:
        ax.plot(jpl_pc[0], jpl_pc[1], "r+", markersize=12, mew=2, label="JPL")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Replica cloud (PCA axes)")
    ax.grid(True, lw=0.5, color="#dddddd")
    ax.axhline(0, lw=1, color="#bbbbbb")
    ax.axvline(0, lw=1, color="#bbbbbb")
    if jpl_pc is not None:
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot RA/Dec and PCA spread of replicas.")
    parser.add_argument("--replicas", type=Path, required=True, help="CSV with ra_deg/dec_deg columns.")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("replica_cloud"),
        help="Output prefix for PNG files.",
    )
    parser.add_argument("--jpl-ra", type=float, default=None, help="Optional JPL RA (deg).")
    parser.add_argument("--jpl-dec", type=float, default=None, help="Optional JPL Dec (deg).")
    args = parser.parse_args()

    df = load_replicas(args.replicas)
    dra, ddec, ra0, dec0 = tangent_offsets(df)
    jpl_offset = None
    if args.jpl_ra is not None and args.jpl_dec is not None:
        cosd = np.cos(np.deg2rad(dec0))
        dra_j = (args.jpl_ra - ra0) * cosd * 3600.0
        ddec_j = (args.jpl_dec - dec0) * 3600.0
        jpl_offset = (dra_j, ddec_j)

    base = args.output_prefix
    radec_path = base.parent / f"{base.name}_radec.png"
    pca_path = base.parent / f"{base.name}_pca.png"
    plot_ra_dec(dra, ddec, radec_path, jpl_offset)
    pc1, pc2, eigvecs = pca_components(dra, ddec)
    jpl_pc = None
    if jpl_offset is not None:
        jpl_pc = tuple((eigvecs.T @ np.array(jpl_offset)).tolist())
    plot_pca(pc1, pc2, pca_path, jpl_pc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
