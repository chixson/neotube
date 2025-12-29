#!/usr/bin/env python3
"""Replica forensics: plots + numeric diagnostics in one composite."""

from __future__ import annotations

import argparse
from pathlib import Path
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.time import Time
from astropy.coordinates import get_body_barycentric_posvel, SkyCoord
import astropy.units as u
from astroquery.jplhorizons import Horizons

from neotube.fit_cli import load_observations
from neotube.fit import _site_offset
from neotube.propagate import predict_radec_batch

AU_KM = 149597870.7


def _load_epoch_utc(replicas: Path, epoch_arg: str | None) -> str:
    if epoch_arg:
        return epoch_arg
    meta_path = replicas.with_name(replicas.stem + "_meta.json")
    if meta_path.exists():
        try:
            import json

            meta = json.loads(meta_path.read_text())
            if meta.get("epoch_utc"):
                return meta["epoch_utc"]
        except Exception:
            pass
    raise RuntimeError("epoch_utc not provided and not found in replicas meta json.")


def _topocentric_vectors(pos_km: np.ndarray, epoch_utc: str, site_offset_km: np.ndarray) -> np.ndarray:
    time_val = Time(epoch_utc, scale="utc")
    earth_pos, _ = get_body_barycentric_posvel("earth", time_val)
    sun_pos, _ = get_body_barycentric_posvel("sun", time_val)
    earth_helio = (earth_pos.xyz - sun_pos.xyz).to(u.km).value
    return pos_km - earth_helio - site_offset_km


def _tangent_offsets(ra_deg: np.ndarray, dec_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    ra0 = float(np.mean(ra_deg))
    dec0 = float(np.mean(dec_deg))
    cosd = np.cos(np.deg2rad(dec0))
    dra = (ra_deg - ra0) * cosd * 3600.0
    ddec = (dec_deg - dec0) * 3600.0
    return dra, ddec, ra0, dec0


def _pca_2d(dra: np.ndarray, ddec: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.vstack([dra, ddec])
    cov = np.cov(X)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]
    proj = eigvecs.T @ X
    return proj[0], proj[1], eigvecs


def _pca_3d(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = vectors.mean(axis=0)
    centered = vectors - mean
    _, _, v_t = np.linalg.svd(centered, full_matrices=False)
    comps = v_t[:3]
    proj = centered @ comps.T
    return proj, comps, mean


def _fetch_jpl_helio_state(target: str, epoch_utc: str) -> tuple[np.ndarray, np.ndarray]:
    t = Time(epoch_utc, scale="utc")
    obj = Horizons(id=target, location="@sun", epochs=t.jd, id_type="smallbody")
    vec = obj.vectors(refplane="earth")
    x = float(vec["x"][0])
    y = float(vec["y"][0])
    z = float(vec["z"][0])
    vx = float(vec["vx"][0])
    vy = float(vec["vy"][0])
    vz = float(vec["vz"][0])
    pos_km = np.array([x, y, z], dtype=float) * AU_KM
    vel_km_s = np.array([vx, vy, vz], dtype=float) * AU_KM / 86400.0
    return pos_km, vel_km_s


def _compute_metrics(
    dra: np.ndarray,
    ddec: np.ndarray,
    jpl_dra: float,
    jpl_ddec: float,
    topo_vec: np.ndarray,
    jpl_topo: np.ndarray,
) -> dict[str, float]:
    dist_arcsec = np.hypot(dra - jpl_dra, ddec - jpl_ddec)
    jpl_dist = float(np.hypot(jpl_dra, jpl_ddec))
    quantile = float((np.sum(dist_arcsec < jpl_dist) + 1) / (len(dist_arcsec) + 1))

    cov = np.cov(np.vstack([dra, ddec]))
    invcov = np.linalg.pinv(cov)
    delta = np.array([jpl_dra, jpl_ddec]) - np.array([np.mean(dra), np.mean(ddec)])
    maha2_2d = float(delta.T @ invcov @ delta)

    cov3 = np.cov(topo_vec.T)
    invcov3 = np.linalg.pinv(cov3)
    delta3 = jpl_topo - topo_vec.mean(axis=0)
    maha2_3d = float(delta3.T @ invcov3 @ delta3)

    nearest_idx = int(np.argmin(np.linalg.norm(topo_vec - jpl_topo, axis=1)))
    nearest_km = float(np.linalg.norm(topo_vec[nearest_idx] - jpl_topo))
    nearest_au = nearest_km / AU_KM

    return {
        "quantile_arcsec": quantile,
        "jpl_arcsec_offset": jpl_dist,
        "maha2_2d": maha2_2d,
        "maha2_3d": maha2_3d,
        "nearest_topo_km": nearest_km,
        "nearest_topo_au": nearest_au,
    }


def _save_scatter(path: Path, x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str, title: str, jpl_xy=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x, y, s=8, alpha=0.5)
    if jpl_xy is not None:
        ax.plot(jpl_xy[0], jpl_xy[1], "r+", markersize=12, mew=2, label="JPL")
        ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, lw=0.5, color="#dddddd")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Replica forensics with plots + numeric summary.")
    parser.add_argument("--replicas", type=Path, required=True)
    parser.add_argument("--obs", type=Path, required=True)
    parser.add_argument("--jpl-target", type=str, required=True)
    parser.add_argument("--epoch-utc", type=str, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/ceres/forensics"))
    args = parser.parse_args()

    df = pd.read_csv(args.replicas)
    if not {"x_km", "y_km", "z_km", "vx_km_s", "vy_km_s", "vz_km_s"}.issubset(df.columns):
        raise RuntimeError("replicas.csv must include state columns.")

    epoch_utc = _load_epoch_utc(args.replicas, args.epoch_utc)
    obs = load_observations(args.obs, None)
    if not obs:
        raise RuntimeError("No observations loaded.")
    site_code = obs[0].site
    time_val = Time(epoch_utc, scale="utc")

    states = df[["x_km", "y_km", "z_km", "vx_km_s", "vy_km_s", "vz_km_s"]].to_numpy(float)
    epochs = [time_val] * len(states)
    site_codes = [site_code] * len(states)
    ra, dec = predict_radec_batch(states, epochs, site_codes=site_codes, allow_unknown_site=True)

    dra, ddec, ra0, dec0 = _tangent_offsets(ra, dec)
    pc1, pc2, eigvecs = _pca_2d(dra, ddec)

    pos = df[["x_km", "y_km", "z_km"]].to_numpy(float)
    site_offset = _site_offset(obs[0], allow_unknown_site=True)
    topo_vec = _topocentric_vectors(pos, epoch_utc, site_offset)
    topo_proj, topo_comps, topo_mean = _pca_3d(topo_vec)

    jpl_pos, jpl_vel = _fetch_jpl_helio_state(args.jpl_target, epoch_utc)
    jpl_state = np.hstack([jpl_pos, jpl_vel])
    jpl_ra, jpl_dec = predict_radec_batch(
        jpl_state[None, :], [time_val], site_codes=[site_code], allow_unknown_site=True
    )
    cosd = np.cos(np.deg2rad(dec0))
    jpl_dra = (float(jpl_ra[0]) - ra0) * cosd * 3600.0
    jpl_ddec = (float(jpl_dec[0]) - dec0) * 3600.0
    jpl_pc = eigvecs.T @ np.array([jpl_dra, jpl_ddec])

    jpl_topo = _topocentric_vectors(jpl_pos[None, :], epoch_utc, site_offset)[0]
    jpl_topo_proj = (jpl_topo - topo_mean) @ topo_comps.T

    r_au = np.linalg.norm(pos, axis=1) / AU_KM
    topo_au = np.linalg.norm(topo_vec, axis=1) / AU_KM
    jpl_r_au = float(np.linalg.norm(jpl_pos) / AU_KM)
    jpl_topo_au = float(np.linalg.norm(jpl_topo) / AU_KM)

    metrics = _compute_metrics(dra, ddec, jpl_dra, jpl_ddec, topo_vec, jpl_topo)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    paths["radec"] = out_dir / "forensics_radec.png"
    paths["pca"] = out_dir / "forensics_pca.png"
    paths["pca1_r"] = out_dir / "forensics_pca1_rAU.png"
    paths["pca1_topo"] = out_dir / "forensics_pca1_topoAU.png"
    paths["topo_pca"] = out_dir / "forensics_topo_pca.png"
    paths["topo_pca23"] = out_dir / "forensics_topo_pca23.png"

    _save_scatter(
        paths["radec"],
        dra,
        ddec,
        "ΔRA cosδ (arcsec)",
        "ΔDec (arcsec)",
        "Replica cloud (tangent plane)",
        (jpl_dra, jpl_ddec),
    )
    _save_scatter(
        paths["pca"],
        pc1,
        pc2,
        "PC1",
        "PC2",
        "Replica cloud (PCA)",
        (jpl_pc[0], jpl_pc[1]),
    )
    _save_scatter(
        paths["pca1_r"],
        pc1,
        r_au,
        "PC1",
        "Heliocentric distance (AU)",
        "PC1 vs heliocentric distance",
        (jpl_pc[0], jpl_r_au),
    )
    _save_scatter(
        paths["pca1_topo"],
        pc1,
        topo_au,
        "PC1",
        "Topocentric distance (AU)",
        "PC1 vs topocentric distance",
        (jpl_pc[0], jpl_topo_au),
    )
    _save_scatter(
        paths["topo_pca"],
        topo_proj[:, 0],
        topo_proj[:, 1],
        "Topo PC1",
        "Topo PC2",
        "Topocentric PCA (PC1 vs PC2)",
        (jpl_topo_proj[0], jpl_topo_proj[1]),
    )
    _save_scatter(
        paths["topo_pca23"],
        topo_proj[:, 1],
        topo_proj[:, 2],
        "Topo PC2",
        "Topo PC3",
        "Topocentric PCA (PC2 vs PC3)",
        (jpl_topo_proj[1], jpl_topo_proj[2]),
    )

    text_lines = [
        f"Epoch UTC: {epoch_utc}",
        f"Site: {site_code}",
        f"Replicas: {len(states)}",
        f"JPL target: {args.jpl_target}",
        "",
        f"JPL sky offset (arcsec): {metrics['jpl_arcsec_offset']:.3f}",
        f"Quantile (sky offset): {metrics['quantile_arcsec']:.3f}",
        f"Mahalanobis^2 (2D): {metrics['maha2_2d']:.3f}",
        f"Mahalanobis^2 (3D): {metrics['maha2_3d']:.3f}",
        f"Nearest topo dist (AU): {metrics['nearest_topo_au']:.6f}",
    ]
    text_block = "\n".join(text_lines)

    # Composite image with text box.
    fig, axs = plt.subplots(3, 3, figsize=(13, 13))
    axs = axs.ravel()
    img_paths = [
        paths["radec"],
        paths["pca"],
        paths["pca1_r"],
        paths["pca1_topo"],
        paths["topo_pca"],
        paths["topo_pca23"],
    ]
    for idx, ax in enumerate(axs):
        if idx < len(img_paths):
            img = plt.imread(img_paths[idx])
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(img_paths[idx].name, fontsize=9)
        elif idx == len(img_paths):
            ax.axis("off")
            ax.text(
                0.01,
                0.99,
                textwrap.dedent(text_block),
                va="top",
                ha="left",
                fontsize=10,
                family="monospace",
                transform=ax.transAxes,
            )
        else:
            ax.axis("off")
    fig.tight_layout()
    composite_path = out_dir / "forensics_composite.png"
    fig.savefig(composite_path, dpi=150)
    plt.close(fig)

    print(text_block)
    print("Wrote plots to", out_dir)
    print("Composite:", composite_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
