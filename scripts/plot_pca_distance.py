#!/usr/bin/env python3
"""
Generate topocentric RA/Dec and PCA views plus a PC1 vs topocentric rho plot.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

try:
    from astroquery.jplhorizons import Horizons
    from astropy.time import Time
    from astropy import units as u
    from astropy.coordinates import (
        CartesianRepresentation,
        CartesianDifferential,
        HeliocentricTrueEcliptic,
        ICRS,
    )

    ASTROQUERY = True
except Exception:
    ASTROQUERY = False

try:
    from sklearn.decomposition import PCA as SKPCA

    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

AU_KM = 149597870.7


def load_replicas_csv(path: str | Path) -> np.ndarray:
    rows: list[dict[str, str]] = []
    with open(path, newline="") as fh:
        rdr = csv.DictReader(fh)
        for r in rdr:
            rows.append(r)
    X = []
    for r in rows:
        x = float(r.get("x_km", r.get("x", 0.0)))
        y = float(r.get("y_km", r.get("y", 0.0)))
        z = float(r.get("z_km", r.get("z", 0.0)))
        vx = float(r.get("vx_km_s", r.get("vx", 0.0)))
        vy = float(r.get("vy_km_s", r.get("vy", 0.0)))
        vz = float(r.get("vz_km_s", r.get("vz", 0.0)))
        X.append([x, y, z, vx, vy, vz])
    return np.array(X)


def load_target_from_file(path: str | Path) -> np.ndarray:
    s = Path(path).read_text().strip()
    for sep in (",", " "):
        if sep in s:
            toks = s.replace(",", " ").split()
            if len(toks) >= 6:
                vals = [float(toks[i]) for i in range(6)]
                return np.array(vals)
    raise ValueError("Unable to parse target state file: need 6 floats")


def fetch_horizons_state(obj_id: str, epoch_iso: str) -> np.ndarray:
    if not ASTROQUERY:
        raise RuntimeError("astroquery not available; cannot fetch Horizons")
    t = Time(epoch_iso)
    obj = Horizons(id=obj_id, location="@sun", epochs=t.jd, id_type="smallbody")
    vec = obj.vectors()
    x = float(vec["x"][0])
    y = float(vec["y"][0])
    z = float(vec["z"][0])
    vx = float(vec["vx"][0])
    vy = float(vec["vy"][0])
    vz = float(vec["vz"][0])
    rep = CartesianRepresentation(x * u.AU, y * u.AU, z * u.AU)
    diff = CartesianDifferential(vx * u.AU / u.day, vy * u.AU / u.day, vz * u.AU / u.day)
    rep_w = rep.with_differentials(diff)
    hce = HeliocentricTrueEcliptic(rep_w, obstime=t)
    icrs = hce.transform_to(ICRS())
    p_km = icrs.cartesian.xyz.to(u.km).value
    v_kms = icrs.cartesian.differentials["s"].d_xyz.to(u.km / u.s).value
    return np.hstack([p_km, v_kms])


def pca_components(dra: np.ndarray, ddec: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.vstack([dra, ddec])
    cov = np.cov(X)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]
    projected = eigvecs.T @ X
    return projected[0], projected[1], eigvecs


def pca_pc1_6d(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if _HAS_SKLEARN:
        pca = SKPCA(n_components=3)
        Z = pca.fit_transform(X)
        return Z[:, 0], pca.mean_, pca.components_
    mean = X.mean(axis=0)
    C = X - mean
    _, _, Vt = np.linalg.svd(C, full_matrices=False)
    components = Vt[:3]
    Z = C @ components.T
    return Z[:, 0], mean, components


def radec_from_vectors(vecs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = vecs[:, 0]
    y = vecs[:, 1]
    z = vecs[:, 2]
    ra = np.unwrap(np.arctan2(y, x))
    r_xy = np.hypot(x, y)
    dec = np.arctan2(z, r_xy)
    return np.degrees(ra), np.degrees(dec)


def tangent_offsets(ra_deg: np.ndarray, dec_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    ra0 = float(np.mean(ra_deg))
    dec0 = float(np.mean(dec_deg))
    cosd = np.cos(np.deg2rad(dec0))
    dra = (ra_deg - ra0) * cosd * 3600.0
    ddec = (dec_deg - dec0) * 3600.0
    return dra, ddec, ra0, dec0


def resolve_target(args: argparse.Namespace) -> np.ndarray | None:
    if args.target_file:
        return load_target_from_file(args.target_file)
    if args.horizons_id:
        if args.posterior is None:
            raise SystemExit("ERROR: --posterior is required to get epoch for Horizons target.")
        from neotube.fit import load_posterior

        post = load_posterior(args.posterior)
        return fetch_horizons_state(args.horizons_id, post.epoch.isot)
    if args.posterior:
        from neotube.fit import load_posterior

        post = load_posterior(args.posterior)
        return np.array(post.state, dtype=float)
    return None


def main() -> int:
    p = argparse.ArgumentParser(description="Plot topocentric RA/Dec, PCA, and PC1 vs rho.")
    p.add_argument("--replicas", required=True, help="CSV of replicas with x_km,y_km,z_km,vx_km_s,...")
    p.add_argument("--obs", required=True, help="Observations CSV (used for site topocentric offset).")
    p.add_argument("--posterior", default=None, help="Optional posterior.npz path (used for target epoch).")
    p.add_argument("--target-file", default=None, help="Text file with 6 floats (x,y,z,vx,vy,vz) in km and km/s.")
    p.add_argument("--horizons-id", default=None, help="If set, fetch this id from Horizons at posterior epoch.")
    p.add_argument("--out-dir", default=None, help="Output directory (defaults to replicas directory).")
    args = p.parse_args()

    X = load_replicas_csv(args.replicas)
    if X.size == 0:
        raise SystemExit("No replicas found in CSV.")

    from neotube.fit_cli import load_observations
    from neotube.fit import _site_offset

    obs = load_observations(Path(args.obs), None)
    site_off = _site_offset(obs[0])
    topo_vecs = X[:, :3] - site_off
    topo_rho = np.linalg.norm(topo_vecs, axis=1) / AU_KM

    ra_deg, dec_deg = radec_from_vectors(topo_vecs)
    dra, ddec, ra0, dec0 = tangent_offsets(ra_deg, dec_deg)
    pc1_off, pc2_off, eigvecs = pca_components(dra, ddec)

    target = resolve_target(args)
    target_pc1 = None
    target_rho = None
    target_dra = None
    target_ddec = None
    target_pc1_off = None
    target_pc2_off = None
    if target is not None:
        t_topo = target[:3] - site_off
        target_rho = float(np.linalg.norm(t_topo) / AU_KM)
        t_ra, t_dec = radec_from_vectors(t_topo.reshape(1, 3))
        cosd = np.cos(np.deg2rad(dec0))
        target_dra = float((t_ra[0] - ra0) * cosd * 3600.0)
        target_ddec = float((t_dec[0] - dec0) * 3600.0)
        target_pc = eigvecs.T @ np.array([target_dra, target_ddec], dtype=float)
        target_pc1_off = float(target_pc[0])
        target_pc2_off = float(target_pc[1])

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.replicas).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    pc1_6d, mean_6d, comps_6d = pca_pc1_6d(X)
    target_pc1 = None
    if target is not None:
        target_pc1 = float((target - mean_6d).dot(comps_6d[0]))

    if target is not None:
        dpos = np.linalg.norm(X[:, :3] - target[:3], axis=1)
        imin = int(np.argmin(dpos))
        print(
            "nearest replica index:",
            imin,
            "dist_km:",
            dpos[imin],
            "dist_AU:",
            dpos[imin] / AU_KM,
        )
        jpl_coord = SkyCoord(
            x=target[0] * u.km,
            y=target[1] * u.km,
            z=target[2] * u.km,
            representation_type="cartesian",
            frame="icrs",
        )
        rep_coord = SkyCoord(ra=ra_deg[imin] * u.deg, dec=dec_deg[imin] * u.deg, frame="icrs")
        print("angular sep of nearest replica to target (arcsec):", rep_coord.separation(jpl_coord).arcsecond)
        print(
            "Median replica heliocentric radius (AU):",
            np.median(np.linalg.norm(X[:, :3], axis=1) / AU_KM),
        )

        Mpos = X[:, :3]
        pca2 = SKPCA(n_components=2) if _HAS_SKLEARN else None
        if pca2 is not None:
            Z = pca2.fit_transform(Mpos)
            jpl_pc = (target[:3] - pca2.mean_).dot(pca2.components_.T)
        else:
            mean_pos = Mpos.mean(axis=0)
            C = Mpos - mean_pos
            _, _, Vt = np.linalg.svd(C, full_matrices=False)
            comps = Vt[:2]
            Z = C @ comps.T
            jpl_pc = (target[:3] - mean_pos).dot(comps.T)
        cov_pc = np.cov(Z.T)
        invcov_pc = np.linalg.pinv(cov_pc)
        maha2_pc = float(jpl_pc.T.dot(invcov_pc).dot(jpl_pc))
        print("Mahalanobis^2 in PC1/2:", maha2_pc, "Mahalanobis:", math.sqrt(maha2_pc))

        mean6 = X.mean(axis=0)
        cov6 = np.cov(X.T)
        d6 = target - mean6
        inv_cov6 = np.linalg.pinv(cov6)
        maha2_6 = float(d6.T.dot(inv_cov6).dot(d6))
        print("Mahalanobis^2 in 6D:", maha2_6, "Mahalanobis:", math.sqrt(maha2_6))

        th = math.sqrt(maha2_pc)
        count = 0
        z_mean = Z.mean(axis=0)
        for z in Z:
            m2 = float((z - z_mean).T.dot(invcov_pc).dot(z - z_mean))
            if math.sqrt(m2) <= th:
                count += 1
        print("Fraction of replicas inside target PC1/2 Mahalanobis radius:", count / len(Z))

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(pc1_6d, topo_rho, s=20, alpha=0.6, label="replicas")
    if target_pc1 is not None and target_rho is not None:
        ax.scatter([target_pc1], [target_rho], color="red", s=120, label="JPL")
    ax.set_xlabel("PC1")
    ax.set_ylabel("Topocentric rho (AU)")
    ax.grid(True)
    ax.legend()
    diag_path = out_dir / "diag_pc1_vs_topo_rho.png"
    fig.tight_layout()
    fig.savefig(diag_path, dpi=100)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(dra, ddec, s=20, alpha=0.6, label="replicas")
    if target_dra is not None and target_ddec is not None:
        ax.scatter([target_dra], [target_ddec], color="red", s=120, label="JPL")
    ax.set_xlabel("ΔRA cosδ (arcsec)")
    ax.set_ylabel("ΔDec (arcsec)")
    ax.grid(True)
    ax.legend()
    radec_path = out_dir / "topo_radec.png"
    fig.tight_layout()
    fig.savefig(radec_path, dpi=150)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(pc1_off, pc2_off, s=20, alpha=0.6, label="replicas")
    if target_pc1_off is not None and target_pc2_off is not None:
        ax.scatter([target_pc1_off], [target_pc2_off], color="red", s=120, label="JPL")
    ax.set_xlabel("PC1 (arcsec)")
    ax.set_ylabel("PC2 (arcsec)")
    ax.grid(True)
    ax.legend()
    pca_path = out_dir / "topo_pca.png"
    fig.tight_layout()
    fig.savefig(pca_path, dpi=150)

    print("Wrote", diag_path)
    print("Wrote", radec_path)
    print("Wrote", pca_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
