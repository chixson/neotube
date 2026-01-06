#!/usr/bin/env python3
"""Plot replica spreads in RA/Dec and PCA space for debugging."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import get_body_barycentric_posvel
from astropy.time import Time
import astropy.units as u
from astroquery.jplhorizons import Horizons


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


def pca_components_3d(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = vectors.mean(axis=0)
    centered = vectors - mean
    _, _, v_t = np.linalg.svd(centered, full_matrices=False)
    comps = v_t[:3]
    projected = centered @ comps.T
    return projected, comps, mean


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


def plot_pca23(
    pc2: np.ndarray,
    pc3: np.ndarray,
    out_path: Path,
    jpl_pc: tuple[float, float] | None,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(pc2, pc3, s=14, alpha=0.6)
    if jpl_pc is not None:
        ax.plot(jpl_pc[0], jpl_pc[1], "r+", markersize=12, mew=2, label="JPL")
        ax.legend()
    ax.set_xlabel("PC2")
    ax.set_ylabel("PC3")
    ax.set_title(title)
    ax.grid(True, lw=0.5, color="#dddddd")
    ax.axhline(0, lw=1, color="#bbbbbb")
    ax.axvline(0, lw=1, color="#bbbbbb")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)

def plot_pca1_r_au(
    pc1: np.ndarray,
    r_au: np.ndarray,
    out_path: Path,
    jpl_pc1: float | None,
    jpl_r_au: float | None,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(pc1, r_au, s=6, alpha=0.3)
    if jpl_pc1 is not None and jpl_r_au is not None:
        ax.plot(jpl_pc1, jpl_r_au, "r+", markersize=12, mew=2, label="JPL")
        ax.legend()
    ax.set_xlabel("PC1 (arcsec)")
    ax.set_ylabel("Heliocentric distance (AU)")
    ax.set_title("Replica cloud (PC1 vs heliocentric distance)")
    ax.grid(True, lw=0.5, color="#dddddd")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)

def plot_pca1_topo_au(
    pc1: np.ndarray,
    topo_au: np.ndarray,
    out_path: Path,
    jpl_pc1: float | None,
    jpl_topo_au: float | None,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(pc1, topo_au, s=6, alpha=0.3)
    if jpl_pc1 is not None and jpl_topo_au is not None:
        ax.plot(jpl_pc1, jpl_topo_au, "r+", markersize=12, mew=2, label="JPL")
        ax.legend()
    ax.set_xlabel("PC1 (arcsec)")
    ax.set_ylabel("Topocentric distance (AU)")
    ax.set_title("Replica cloud (PC1 vs topocentric distance)")
    ax.grid(True, lw=0.5, color="#dddddd")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)


def _load_epoch_utc(meta_path: Path) -> str | None:
    if not meta_path.exists():
        return None
    try:
        with meta_path.open() as fh:
            meta = json.load(fh)
    except Exception:
        return None
    return meta.get("epoch_utc")


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
    pos_km = np.array([x, y, z], dtype=float) * 149597870.7
    vel_km_s = np.array([vx, vy, vz], dtype=float) * 149597870.7 / 86400.0
    return pos_km, vel_km_s


def _topocentric_distance_au(
    pos_km: np.ndarray, epoch_utc: str, site_offset_km: np.ndarray
) -> np.ndarray:
    time_array = Time([epoch_utc] * len(pos_km), scale="utc")
    earth_pos, _ = get_body_barycentric_posvel("earth", time_array)
    sun_pos, _ = get_body_barycentric_posvel("sun", time_array)
    earth_helio = (earth_pos.xyz - sun_pos.xyz).to(u.km).value.T
    vectors = pos_km - earth_helio - site_offset_km[None, :]
    return np.linalg.norm(vectors, axis=1) / 149597870.7


def _topocentric_vectors(
    pos_km: np.ndarray, epoch_utc: str, site_offset_km: np.ndarray
) -> np.ndarray:
    time_val = Time(epoch_utc, scale="utc")
    earth_pos, _ = get_body_barycentric_posvel("earth", time_val)
    sun_pos, _ = get_body_barycentric_posvel("sun", time_val)
    earth_helio = (earth_pos.xyz - sun_pos.xyz).to(u.km).value
    return pos_km - earth_helio - site_offset_km


def mirror_to_data_dir(*paths: Path) -> None:
    data_dir = Path(__file__).resolve().parents[1] / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        if path.exists():
            shutil.copy2(path, data_dir / path.name)


def write_composite(out_path: Path, image_paths: list[Path], cols: int = 3) -> None:
    existing = [p for p in image_paths if p.exists()]
    if not existing:
        return
    rows = (len(existing) + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 4.5))
    axs = np.atleast_1d(axs).ravel()
    for ax in axs[len(existing) :]:
        ax.axis("off")
    for ax, img_path in zip(axs, existing):
        img = plt.imread(img_path)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(img_path.name, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


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
    parser.add_argument("--jpl-r-au", type=float, default=None, help="Optional JPL heliocentric distance (AU).")
    parser.add_argument("--jpl-topo-au", type=float, default=None, help="Optional JPL topocentric distance (AU).")
    parser.add_argument(
        "--jpl-helio-pos-km",
        type=float,
        nargs=3,
        default=None,
        help="Optional JPL heliocentric position (x y z km) for topo PCA crosshair.",
    )
    parser.add_argument(
        "--jpl-target",
        type=str,
        default=None,
        help="Optional JPL/Horizons target name for auto crosshair (e.g., Ceres).",
    )
    parser.add_argument(
        "--base-topo",
        action="store_true",
        help="When obs/epoch are available, build base RA/Dec PCA plots from topocentric RA/Dec.",
    )
    parser.add_argument(
        "--epoch-utc",
        type=str,
        default=None,
        help="UTC epoch for topocentric distances; falls back to replicas _meta.json if present.",
    )
    parser.add_argument(
        "--obs",
        type=Path,
        default=None,
        help="Observations CSV for topocentric plots (site code). Falls back to replicas _meta.json if present.",
    )
    args = parser.parse_args()

    df = load_replicas(args.replicas)
    if {"x_km", "y_km", "z_km", "vx_km_s", "vy_km_s", "vz_km_s"}.issubset(df.columns):
        states = df[["x_km", "y_km", "z_km", "vx_km_s", "vy_km_s", "vz_km_s"]].to_numpy(
            dtype=float
        )
    else:
        states = None
    dra = ddec = ra0 = dec0 = None
    r_au = None
    topo_au = None
    obs_list = None
    site_offset_km = None
    if {"x_km", "y_km", "z_km"}.issubset(df.columns):
        pos = df[["x_km", "y_km", "z_km"]].to_numpy(dtype=float)
        r_au = np.linalg.norm(pos, axis=1) / 149597870.7
        epoch_utc = args.epoch_utc or _load_epoch_utc(
            args.replicas.with_name(args.replicas.stem + "_meta.json")
        )
        if epoch_utc is not None:
            topo_au = None
    else:
        print("replica CSV missing x_km/y_km/z_km; skipping PC1 vs r(AU) plot")

    jpl_offset = None
    jpl_state_km = None

    base = args.output_prefix
    radec_path = base.parent / f"{base.name}_radec.png"
    pca_path = base.parent / f"{base.name}_pca.png"
    pca1_r_path = base.parent / f"{base.name}_pca1_rAU.png"
    pca1_topo_path = base.parent / f"{base.name}_pca1_topoAU.png"
    topo_radec_path = base.parent / f"{base.name}_topo_radec.png"
    topo_pca_path = base.parent / f"{base.name}_topo_pca.png"
    topo_pca23_path = base.parent / f"{base.name}_topo_pca23.png"
    composite_path = base.parent / f"{base.name}_composite.png"
    # Optional topocentric plots (require obs + epoch + positions).
    epoch_utc = args.epoch_utc or _load_epoch_utc(
        args.replicas.with_name(args.replicas.stem + "_meta.json")
    )
    obs_path = args.obs
    if obs_path is None:
        meta_path = args.replicas.with_name(args.replicas.stem + "_meta.json")
        if meta_path.exists():
            try:
                with meta_path.open() as fh:
                    meta = json.load(fh)
                if meta.get("obs"):
                    obs_path = Path(meta["obs"])
            except Exception:
                obs_path = None
    topo_df = None
    if epoch_utc is not None and obs_path is not None and states is not None:
        from neotube.fit_cli import load_observations
        from neotube.propagate import predict_radec_batch

        obs_list = load_observations(obs_path, None)
        site_code = obs_list[0].site if obs_list else None
        time_val = Time(epoch_utc, scale="utc")
        epochs = [time_val] * len(states)
        site_codes = [site_code] * len(states)
        topo_ra, topo_dec = predict_radec_batch(
            states,
            epochs,
            site_codes=site_codes,
            allow_unknown_site=True,
        )
        topo_df = df.copy()
        topo_df["ra_deg"] = topo_ra
        topo_df["dec_deg"] = topo_dec
        topo_dra, topo_ddec, topo_ra0, topo_dec0 = tangent_offsets(topo_df)

        # Resolve JPL heliocentric state if possible.
        if jpl_state_km is None and args.jpl_helio_pos_km is not None:
            jpl_state_km = np.hstack([np.array(args.jpl_helio_pos_km, dtype=float), np.zeros(3)])
        if jpl_state_km is None and args.jpl_target is not None:
            try:
                jpl_pos, jpl_vel = _fetch_jpl_helio_state(args.jpl_target, epoch_utc)
                jpl_state_km = np.hstack([jpl_pos, jpl_vel])
            except Exception as exc:
                print(f"Failed to fetch JPL target '{args.jpl_target}': {exc}")
                jpl_state_km = None

        jpl_topo_offset = None
        if jpl_state_km is not None:
            jpl_ra, jpl_dec = predict_radec_batch(
                jpl_state_km[None, :],
                [time_val],
                site_codes=[site_code],
                allow_unknown_site=True,
            )
            cosd = np.cos(np.deg2rad(topo_dec0))
            dra_j = (float(jpl_ra[0]) - topo_ra0) * cosd * 3600.0
            ddec_j = (float(jpl_dec[0]) - topo_dec0) * 3600.0
            jpl_topo_offset = (float(dra_j), float(ddec_j))
            if args.jpl_ra is None or args.jpl_dec is None:
                args.jpl_ra = float(jpl_ra[0])
                args.jpl_dec = float(jpl_dec[0])
        elif args.jpl_ra is not None and args.jpl_dec is not None:
            cosd = np.cos(np.deg2rad(topo_dec0))
            dra_j = (args.jpl_ra - topo_ra0) * cosd * 3600.0
            ddec_j = (args.jpl_dec - topo_dec0) * 3600.0
            jpl_topo_offset = (dra_j, ddec_j)

        plot_ra_dec(topo_dra, topo_ddec, topo_radec_path, jpl_topo_offset)

        site_offset_km = None
        if obs_list:
            from neotube.fit import _site_offset

            site_offset_km = _site_offset(obs_list[0], allow_unknown_site=True)
        else:
            site_offset_km = np.zeros(3, dtype=float)
        topo_vec = _topocentric_vectors(pos, epoch_utc, site_offset_km)
        proj, comps, mean = pca_components_3d(topo_vec)
        jpl_pc_topo = None
        if jpl_state_km is not None:
            jpl_helio = jpl_state_km[:3]
            jpl_topo = _topocentric_vectors(jpl_helio[None, :], epoch_utc, site_offset_km)[0]
            jpl_proj = (jpl_topo - mean) @ comps.T
            jpl_pc_topo = (float(jpl_proj[0]), float(jpl_proj[1]))
            jpl_pc23 = (float(jpl_proj[1]), float(jpl_proj[2]))
        else:
            jpl_pc23 = None

        plot_pca(proj[:, 0], proj[:, 1], topo_pca_path, jpl_pc_topo)
        plot_pca23(proj[:, 1], proj[:, 2], topo_pca23_path, jpl_pc23, "Replica cloud (topocentric PC2 vs PC3)")
        mirror_to_data_dir(topo_radec_path, topo_pca_path, topo_pca23_path)

    df_base = topo_df if (args.base_topo and topo_df is not None) else df
    dra, ddec, ra0, dec0 = tangent_offsets(df_base)
    pc1, pc2, eigvecs = pca_components(dra, ddec)

    if args.jpl_ra is not None and args.jpl_dec is not None:
        cosd = np.cos(np.deg2rad(dec0))
        dra_j = (args.jpl_ra - ra0) * cosd * 3600.0
        ddec_j = (args.jpl_dec - dec0) * 3600.0
        jpl_offset = (dra_j, ddec_j)
    jpl_pc = None
    jpl_pc1 = None
    if jpl_offset is not None:
        jpl_pc = tuple((eigvecs.T @ np.array(jpl_offset)).tolist())
        jpl_pc1 = float(jpl_pc[0])
    plot_ra_dec(dra, ddec, radec_path, jpl_offset)
    plot_pca(pc1, pc2, pca_path, jpl_pc)

    if r_au is not None:
        if topo_au is None and epoch_utc is not None and site_offset_km is not None:
            try:
                topo_au = _topocentric_distance_au(pos, epoch_utc, site_offset_km)
            except Exception:
                topo_au = None
        if args.jpl_r_au is None and jpl_state_km is not None:
            args.jpl_r_au = float(np.linalg.norm(jpl_state_km[:3]) / 149597870.7)
        if args.jpl_topo_au is None and topo_au is not None and jpl_state_km is not None:
            jpl_topo = _topocentric_vectors(jpl_state_km[:3][None, :], epoch_utc, site_offset_km)[0]
            args.jpl_topo_au = float(np.linalg.norm(jpl_topo) / 149597870.7)
        plot_pca1_r_au(pc1, r_au, pca1_r_path, jpl_pc1, args.jpl_r_au)
        if topo_au is not None:
            plot_pca1_topo_au(pc1, topo_au, pca1_topo_path, jpl_pc1, args.jpl_topo_au)
            mirror_to_data_dir(radec_path, pca_path, pca1_r_path, pca1_topo_path)
        else:
            mirror_to_data_dir(radec_path, pca_path, pca1_r_path)
    else:
        mirror_to_data_dir(radec_path, pca_path)
    write_composite(
        composite_path,
        [
            radec_path,
            pca_path,
            pca1_r_path,
            pca1_topo_path,
            topo_radec_path,
            topo_pca_path,
            topo_pca23_path,
        ],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
