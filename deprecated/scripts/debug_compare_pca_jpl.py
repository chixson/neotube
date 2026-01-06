#!/usr/bin/env python3
"""
Debug PCA / JPL mismatch. Produces numeric diagnostics and PC2xPC3 overlays
for both heliocentric and topocentric PCA bases.
"""
from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import csv
import sys

import numpy as np
import matplotlib.pyplot as plt

AU_KM = 149597870.7

try:
    from astroquery.jplhorizons import Horizons
    from astropy.time import Time
    from astropy import units as u
    from astropy.coordinates import CartesianRepresentation, CartesianDifferential, HeliocentricTrueEcliptic, ICRS

    HAVE_AQ = True
except Exception:
    HAVE_AQ = False


def pca_numpy(M: np.ndarray, k: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = M.mean(axis=0)
    C = M - mean
    _, S, Vt = np.linalg.svd(C, full_matrices=False)
    comps = Vt[:k]
    var = (S**2) / max(1, (M.shape[0] - 1))
    if var.sum() > 0:
        evr = var[: len(comps)] / var.sum()
    else:
        evr = np.zeros(len(comps))
    return mean, comps, evr


def load_replicas_csv(path: str) -> np.ndarray:
    rows = []
    with open(path) as fh:
        rdr = csv.DictReader(fh)
        for r in rdr:
            rows.append(r)
    X = []
    for r in rows:
        X.append(
            [
                float(r.get("x_km", r.get("x", 0.0))),
                float(r.get("y_km", r.get("y", 0.0))),
                float(r.get("z_km", r.get("z", 0.0))),
                float(r.get("vx_km_s", r.get("vx", 0.0))),
                float(r.get("vy_km_s", r.get("vy", 0.0))),
                float(r.get("vz_km_s", r.get("vz", 0.0))),
            ]
        )
    return np.array(X)


def load_jpl_from_horizons(obj: str, epoch_iso: str) -> np.ndarray:
    if not HAVE_AQ:
        raise RuntimeError("astroquery/astropy not available")
    t = Time(epoch_iso)
    objh = Horizons(id=obj, location="@sun", epochs=t.jd, id_type="smallbody")
    v = objh.vectors()
    x = float(v["x"][0])
    y = float(v["y"][0])
    z = float(v["z"][0])
    vx = float(v["vx"][0])
    vy = float(v["vy"][0])
    vz = float(v["vz"][0])
    rep = CartesianRepresentation(x * u.AU, y * u.AU, z * u.AU)
    diff = CartesianDifferential(vx * u.AU / u.day, vy * u.AU / u.day, vz * u.AU / u.day)
    rep_w = rep.with_differentials(diff)
    hce = HeliocentricTrueEcliptic(rep_w, obstime=t)
    icrs = hce.transform_to(ICRS())
    pos_km = icrs.cartesian.xyz.to(u.km).value
    vel_kms = icrs.cartesian.differentials["s"].d_xyz.to(u.km / u.s).value
    return np.hstack([pos_km, vel_kms])


def site_offset_from_obs(obs_path: str) -> np.ndarray:
    try:
        from neotube.fit_cli import load_observations
        from neotube.fit import _site_offset

        obs = load_observations(Path(obs_path), None)
        return _site_offset(obs[0])
    except Exception:
        return np.array([AU_KM, 0.0, 0.0])


def read_posterior_epoch(posterior_path: str) -> tuple[str | None, np.ndarray | None]:
    try:
        from neotube.fit import load_posterior

        p = load_posterior(posterior_path)
        return p.epoch.isot, np.array(p.state, float)
    except Exception:
        return None, None


def write_pc23_plot(M: np.ndarray, mean: np.ndarray, comps: np.ndarray, jpl_proj: np.ndarray, title: str, fname: Path) -> None:
    Z = (M - mean).dot(comps.T)
    plt.figure(figsize=(6, 5))
    plt.scatter(Z[:, 1], Z[:, 2], s=8, alpha=0.6, label="replicas")
    plt.scatter([jpl_proj[1]], [jpl_proj[2]], color="red", s=80, label="JPL")
    plt.xlabel("PC2")
    plt.ylabel("PC3")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(str(fname), dpi=150)
    plt.close()


def main() -> int:
    p = ArgumentParser()
    p.add_argument("--replicas", required=True)
    p.add_argument("--obs", required=True)
    p.add_argument("--posterior", required=True)
    p.add_argument("--horizons-id", default="Ceres")
    p.add_argument("--out-dir", default="runs/ceres/debug_compare")
    args = p.parse_args()

    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    X = load_replicas_csv(args.replicas)
    if X.shape[0] == 0:
        print("No replicas found")
        return 1

    pos = X[:, :3].astype(float)
    print("replicas count:", pos.shape[0])

    site_off = site_offset_from_obs(args.obs)
    print("site_off (heliocentric km):", site_off, "norm AU:", np.linalg.norm(site_off) / AU_KM)

    pos_topo = pos - site_off

    mean_hel, comps_hel, evr_hel = pca_numpy(pos, 3)
    mean_topo, comps_topo, evr_topo = pca_numpy(pos_topo, 3)
    print("evr heliocentric:", evr_hel)
    print("evr topocentric:", evr_topo)

    epoch_iso, post_state = read_posterior_epoch(args.posterior)
    if epoch_iso is None:
        print("Unable to read posterior epoch")
        return 1
    print("posterior epoch:", epoch_iso)

    jpl_state = None
    if HAVE_AQ:
        try:
            jpl_state = load_jpl_from_horizons(args.horizons_id, epoch_iso)
            print("Fetched JPL heliocentric norm (AU):", np.linalg.norm(jpl_state[:3]) / AU_KM)
        except Exception as exc:
            print("Horizons fetch failed:", exc)
    if jpl_state is None:
        jpl_state = post_state
        print("Using posterior.state as proxy JPL; norm AU:", np.linalg.norm(jpl_state[:3]) / AU_KM)

    jpl_hel = jpl_state[:3].astype(float)
    jpl_topo = jpl_hel - site_off

    jpl_pc_hel = (jpl_hel - mean_hel).dot(comps_hel.T)
    jpl_pc_topo = (jpl_topo - mean_topo).dot(comps_topo.T)

    print("JPL pc heliocentric:", jpl_pc_hel[:3])
    print("JPL pc topocentric:", jpl_pc_topo[:3])

    Z_topo = (pos_topo - mean_topo).dot(comps_topo.T)
    pc2 = Z_topo[:, 1]
    pc3 = Z_topo[:, 2]
    print("replica PC2 range (topo):", pc2.min(), pc2.max())
    print("replica PC3 range (topo):", pc3.min(), pc3.max())

    dpc23 = np.sqrt((pc2 - jpl_pc_topo[1]) ** 2 + (pc3 - jpl_pc_topo[2]) ** 2)
    imin = int(np.argmin(dpc23))
    print("nearest replica by PC23 index:", imin, "dpc23:", dpc23[imin])
    phys_dist_km = np.linalg.norm(pos_topo[imin] - jpl_topo)
    print("phys separation km:", phys_dist_km, "AU:", phys_dist_km / AU_KM)

    write_pc23_plot(pos_topo, mean_topo, comps_topo, jpl_pc_topo, "Topocentric PC2 vs PC3", outdir / "pc23_topo.png")
    write_pc23_plot(pos, mean_hel, comps_hel, jpl_pc_hel, "Heliocentric PC2 vs PC3", outdir / "pc23_helio.png")

    print("Wrote plots to", outdir)
    print("Replica median helio r (AU):", np.median(np.linalg.norm(pos, axis=1) / AU_KM))
    print("JPL helio r (AU):", np.linalg.norm(jpl_hel) / AU_KM)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
