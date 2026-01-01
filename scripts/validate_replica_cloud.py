#!/usr/bin/env python3
"""
Validate and visualize a replica cloud against observations (and optional JPL Horizons).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from astropy.time import Time

from neotube.fit_cli import load_observations
from neotube.propagate import predict_radec_from_epoch

try:
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except Exception:
    plt = None
    _HAS_MPL = False

try:
    from astroquery.jplhorizons import Horizons

    _HAS_HORIZONS = True
except Exception:
    Horizons = None
    _HAS_HORIZONS = False

AU_KM = 149597870.7
GM_SUN = 1.32712440018e11


def _load_replicas_csv(path: Path) -> np.ndarray:
    states = []
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
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
    return np.array(states, dtype=float)


def _predict_chunk(args: tuple) -> tuple[np.ndarray, np.ndarray]:
    states_chunk, epoch, obs, perturbers, max_step, use_kepler, full_physics = args
    ra_out = np.zeros((len(states_chunk), len(obs)), dtype=float)
    dec_out = np.zeros((len(states_chunk), len(obs)), dtype=float)
    for i, st in enumerate(states_chunk):
        try:
            ra_i, dec_i = predict_radec_from_epoch(
                st,
                epoch,
                obs,
                perturbers,
                max_step,
                use_kepler=use_kepler,
                allow_unknown_site=True,
                light_time_iters=2,
                full_physics=full_physics,
            )
            ra_out[i, :] = ra_i
            dec_out[i, :] = dec_i
        except Exception:
            ra_out[i, :] = np.nan
            dec_out[i, :] = np.nan
    return ra_out, dec_out


def _load_epoch(meta_path: Path) -> Time:
    with meta_path.open() as fh:
        meta = json.load(fh)
    epoch_utc = meta.get("epoch_utc")
    if not epoch_utc:
        raise ValueError("epoch_utc missing in meta json")
    return Time(epoch_utc)


def _compute_elements(states: np.ndarray) -> dict[str, np.ndarray]:
    r = states[:, :3]
    v = states[:, 3:]
    r_norm = np.linalg.norm(r, axis=1)
    v2 = np.sum(v * v, axis=1)
    h = np.cross(r, v)
    h_norm = np.linalg.norm(h, axis=1)
    e_vec = (np.cross(v, h) / GM_SUN) - (r / r_norm[:, None])
    e = np.linalg.norm(e_vec, axis=1)
    a = 1.0 / (2.0 / r_norm - v2 / GM_SUN)
    inc = np.degrees(np.arccos(np.clip(h[:, 2] / np.maximum(h_norm, 1e-12), -1.0, 1.0)))
    return {"a_au": a / AU_KM, "e": e, "inc_deg": inc, "r_au": r_norm / AU_KM}


def _residuals_summary(residuals_ra: np.ndarray, residuals_dec: np.ndarray) -> dict[str, float]:
    rms_ra = float(np.sqrt(np.mean(residuals_ra**2)))
    rms_dec = float(np.sqrt(np.mean(residuals_dec**2)))
    rms_total = float(np.sqrt(np.mean(residuals_ra**2 + residuals_dec**2)))
    return {
        "rms_ra_arcsec": rms_ra,
        "rms_dec_arcsec": rms_dec,
        "rms_total_arcsec": rms_total,
    }


def _plot_residuals(times: np.ndarray, ra_res: np.ndarray, dec_res: np.ndarray, out_dir: Path) -> None:
    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax[0].plot(times, ra_res, "o", ms=4, alpha=0.8)
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_ylabel("RA residual (arcsec)")
    ax[1].plot(times, dec_res, "o", ms=4, alpha=0.8)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_ylabel("Dec residual (arcsec)")
    ax[1].set_xlabel("Time (MJD)")
    fig.tight_layout()
    fig.savefig(out_dir / "residuals_timeseries.png", dpi=150)
    plt.close(fig)


def _plot_residual_hist(residuals_ra: np.ndarray, residuals_dec: np.ndarray, out_dir: Path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].hist(residuals_ra, bins=40, alpha=0.7)
    ax[0].set_title("RA residuals (arcsec)")
    ax[1].hist(residuals_dec, bins=40, alpha=0.7)
    ax[1].set_title("Dec residuals (arcsec)")
    fig.tight_layout()
    fig.savefig(out_dir / "residuals_hist.png", dpi=150)
    plt.close(fig)


def _plot_elements(elements: dict[str, np.ndarray], out_dir: Path) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].hist(elements["a_au"], bins=50)
    ax[0].set_title("a (AU)")
    ax[1].hist(elements["e"], bins=50)
    ax[1].set_title("e")
    ax[2].hist(elements["inc_deg"], bins=50)
    ax[2].set_title("i (deg)")
    fig.tight_layout()
    fig.savefig(out_dir / "elements_hist.png", dpi=150)
    plt.close(fig)


def _plot_rho(elements: dict[str, np.ndarray], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(elements["r_au"], bins=50)
    ax.set_title("Heliocentric distance r (AU)")
    fig.tight_layout()
    fig.savefig(out_dir / "r_au_hist.png", dpi=150)
    plt.close(fig)


def _plot_sky(obs_ra: np.ndarray, obs_dec: np.ndarray, pred_ra: np.ndarray, pred_dec: np.ndarray, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(obs_ra, obs_dec, c="k", s=20, label="obs")
    ax.scatter(pred_ra, pred_dec, c="tab:blue", s=8, alpha=0.4, label="replicas")
    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Dec (deg)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "sky_scatter.png", dpi=150)
    plt.close(fig)


def _plot_composite(
    times: np.ndarray,
    ra_res: np.ndarray,
    dec_res: np.ndarray,
    ra_res_flat: np.ndarray,
    dec_res_flat: np.ndarray,
    elements: dict[str, np.ndarray],
    obs_ra: np.ndarray,
    obs_dec: np.ndarray,
    pred_ra: np.ndarray,
    pred_dec: np.ndarray,
    pred_ra_mean: np.ndarray,
    pred_dec_mean: np.ndarray,
    horizons_ra: np.ndarray | None,
    horizons_dec: np.ndarray | None,
    out_dir: Path,
) -> None:
    fig, ax = plt.subplots(3, 2, figsize=(12, 14))

    ax[0, 0].plot(times, ra_res, "o", ms=4, alpha=0.8)
    ax[0, 0].axhline(0, color="k", lw=0.8)
    ax[0, 0].set_ylabel("RA residual (arcsec)")
    ax[0, 0].set_xlabel("Time (MJD)")

    ax[0, 1].plot(times, dec_res, "o", ms=4, alpha=0.8)
    ax[0, 1].axhline(0, color="k", lw=0.8)
    ax[0, 1].set_ylabel("Dec residual (arcsec)")
    ax[0, 1].set_xlabel("Time (MJD)")

    ax[1, 0].hist(ra_res_flat, bins=40, alpha=0.7)
    ax[1, 0].set_title("RA residuals (arcsec)")
    ax[1, 1].hist(dec_res_flat, bins=40, alpha=0.7)
    ax[1, 1].set_title("Dec residuals (arcsec)")

    ax[2, 0].hist(elements["a_au"], bins=50)
    ax[2, 0].set_title("a (AU)")
    ax[2, 0].set_xlabel("a (AU)")
    ax[2, 0].set_ylabel("count")

    ax[2, 1].scatter(obs_ra, obs_dec, c="k", s=20, label="obs")
    ax[2, 1].scatter(pred_ra, pred_dec, c="tab:blue", s=8, alpha=0.4, label="replicas")
    ax[2, 1].plot(pred_ra_mean, pred_dec_mean, color="tab:orange", lw=1.0, label="replica mean")
    if horizons_ra is not None and horizons_dec is not None:
        ax[2, 1].scatter(
            horizons_ra,
            horizons_dec,
            c="tab:red",
            s=36,
            marker="*",
            label="Horizons",
        )
    ax[2, 1].set_xlabel("RA (deg)")
    ax[2, 1].set_ylabel("Dec (deg)")
    ax[2, 1].legend()

    fig.tight_layout()
    fig.savefig(out_dir / "summary_composite.png", dpi=150)
    plt.close(fig)


def _plot_cloud_planes(
    states: np.ndarray,
    jpl_xyz: tuple[float, float, float] | None,
    out_dir: Path,
) -> None:
    r_au = states[:, :3] / AU_KM
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].scatter(r_au[:, 0], r_au[:, 1], s=6, alpha=0.3, label="replicas")
    ax[0].set_xlabel("x (AU)")
    ax[0].set_ylabel("y (AU)")
    ax[0].set_title("XY")
    ax[1].scatter(r_au[:, 0], r_au[:, 2], s=6, alpha=0.3, label="replicas")
    ax[1].set_xlabel("x (AU)")
    ax[1].set_ylabel("z (AU)")
    ax[1].set_title("XZ")
    ax[2].scatter(r_au[:, 1], r_au[:, 2], s=6, alpha=0.3, label="replicas")
    ax[2].set_xlabel("y (AU)")
    ax[2].set_ylabel("z (AU)")
    ax[2].set_title("YZ")
    if jpl_xyz is not None:
        xj, yj, zj = (np.array(jpl_xyz) / AU_KM).tolist()
        ax[0].scatter([xj], [yj], s=60, c="tab:red", marker="*", label="Horizons")
        ax[1].scatter([xj], [zj], s=60, c="tab:red", marker="*", label="Horizons")
        ax[2].scatter([yj], [zj], s=60, c="tab:red", marker="*", label="Horizons")
    for a in ax:
        a.legend()
        a.axis("equal")
    fig.tight_layout()
    fig.savefig(out_dir / "cloud_xyz_planes.png", dpi=150)
    plt.close(fig)


def _fetch_horizons(
    target: str, site: str, times: Time, id_type: str | None = None
) -> tuple[np.ndarray, np.ndarray]:
    if not _HAS_HORIZONS:
        raise RuntimeError("astroquery.jplhorizons is not available")
    kwargs = {}
    if id_type:
        kwargs["id_type"] = id_type
    obj = Horizons(id=target, location=site, epochs=times.tdb.jd, **kwargs)
    eph = obj.ephemerides()
    return np.array(eph["RA"], dtype=float), np.array(eph["DEC"], dtype=float)


def _fetch_horizons_any(
    target: str, site: str, times: Time, id_type: str | None = None
) -> tuple[np.ndarray, np.ndarray]:
    target_stripped = target.strip()
    numeric_id = target_stripped.isdigit()
    use_smallbody = not numeric_id
    attempts: list[tuple[str, str | None]] = []
    attempts.append((target, id_type))
    if use_smallbody:
        attempts.append((target, "smallbody"))
    attempts.append((target, "majorbody"))
    parts = target.strip().split()
    if len(parts) > 1:
        attempts.append((parts[0], "smallbody"))
        attempts.append((parts[-1], "smallbody"))
    if "ceres" in target.lower():
        attempts.append(("1", "smallbody"))
        attempts.append(("Ceres", "smallbody"))
    last_exc = None
    for tgt, idt in attempts:
        try:
            return _fetch_horizons(tgt, site, times, idt)
        except Exception as exc:
            last_exc = exc
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Horizons query failed")


def _fetch_horizons_vector_any(
    target: str, epoch: Time, id_type: str | None = None
) -> tuple[float, float, float]:
    if not _HAS_HORIZONS:
        raise RuntimeError("astroquery.jplhorizons is not available")
    target_stripped = target.strip()
    numeric_id = target_stripped.isdigit()
    use_smallbody = not numeric_id
    attempts: list[tuple[str, str | None]] = []
    attempts.append((target, id_type))
    if use_smallbody:
        attempts.append((target, "smallbody"))
    attempts.append((target, "majorbody"))
    parts = target.strip().split()
    if len(parts) > 1:
        attempts.append((parts[0], "smallbody"))
        attempts.append((parts[-1], "smallbody"))
    if "ceres" in target.lower():
        attempts.append(("2000001", "smallbody"))
        attempts.append(("1", "smallbody"))
        attempts.append(("Ceres", "smallbody"))
    last_exc = None
    for tgt, idt in attempts:
        try:
            kwargs = {}
            if idt:
                kwargs["id_type"] = idt
            obj = Horizons(id=tgt, location="@sun", epochs=epoch.tdb.jd, **kwargs)
            vec = obj.vectors()
            x = float(vec["x"][0])
            y = float(vec["y"][0])
            z = float(vec["z"][0])
            return x, y, z
        except Exception as exc:
            last_exc = exc
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Horizons vector query failed")


def main() -> None:
    p = argparse.ArgumentParser(description="Validate replica cloud and generate plots.")
    p.add_argument("--replicas", required=True, type=Path)
    p.add_argument("--obs", required=True, type=Path)
    p.add_argument("--meta", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--max-particles", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--use-kepler", action="store_true")
    p.add_argument("--no-full-physics", action="store_true")
    p.add_argument("--perturbers", nargs="*", default=["earth", "mars", "jupiter"])
    p.add_argument("--max-step", type=float, default=3600.0)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--chunk-size", type=int, default=0)
    p.add_argument("--horizons-target", type=str, default=None)
    p.add_argument("--horizons-id-type", type=str, default=None)
    p.add_argument(
        "--cloud-only",
        action="store_true",
        help="Skip ephemeris/residual calculations; only plot cloud/element projections.",
    )
    args = p.parse_args()

    if not _HAS_MPL:
        raise SystemExit("matplotlib is required for plotting")

    obs = load_observations(args.obs, None)
    if args.meta is None:
        meta_path = args.replicas.with_name(args.replicas.stem + "_meta.json")
    else:
        meta_path = args.meta
    epoch = _load_epoch(meta_path)

    states = _load_replicas_csv(args.replicas)
    rng = np.random.default_rng(int(args.seed))
    if len(states) > args.max_particles:
        idx = rng.choice(len(states), size=args.max_particles, replace=False)
        states = states[idx]

    obs_times = Time([o.time for o in obs])
    obs_ra = np.array([o.ra_deg for o in obs], dtype=float)
    obs_dec = np.array([o.dec_deg for o in obs], dtype=float)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    elements = _compute_elements(states)
    _plot_elements(elements, out_dir)
    _plot_rho(elements, out_dir)

    horizons_ra = None
    horizons_dec = None
    horizons_xyz = None
    if args.horizons_target:
        try:
            site = obs[0].site or "500"
            horizons_ra, horizons_dec = _fetch_horizons_any(
                args.horizons_target, site, obs_times, args.horizons_id_type
            )
        except Exception:
            try:
                horizons_ra, horizons_dec = _fetch_horizons_any(
                    args.horizons_target, "500", obs_times, args.horizons_id_type
                )
            except Exception as exc:
                print(f"[warn] Horizons overlay failed: {exc}")
        try:
            horizons_xyz = _fetch_horizons_vector_any(
                args.horizons_target, epoch, args.horizons_id_type
            )
        except Exception as exc:
            print(f"[warn] Horizons vector failed: {exc}")

    if not args.cloud_only:
        pred_ra = np.zeros((len(states), len(obs)), dtype=float)
        pred_dec = np.zeros((len(states), len(obs)), dtype=float)
        if args.workers <= 1 or len(states) == 0:
            for i, st in enumerate(states):
                ra_i, dec_i = predict_radec_from_epoch(
                    st,
                    epoch,
                    obs,
                    tuple(args.perturbers),
                    args.max_step,
                    use_kepler=args.use_kepler,
                    allow_unknown_site=True,
                    light_time_iters=2,
                    full_physics=not args.no_full_physics,
                )
                pred_ra[i, :] = ra_i
                pred_dec[i, :] = dec_i
        else:
            chunk = args.chunk_size
            if chunk <= 0:
                chunk = max(1, int(math.ceil(len(states) / max(1, args.workers * 4))))
            tasks = []
            for start in range(0, len(states), chunk):
                tasks.append(
                    (
                        states[start : start + chunk],
                        epoch,
                        obs,
                        tuple(args.perturbers),
                        args.max_step,
                        args.use_kepler,
                        not args.no_full_physics,
                    )
                )
            offset = 0
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                for ra_chunk, dec_chunk in executor.map(_predict_chunk, tasks):
                    pred_ra[offset : offset + len(ra_chunk), :] = ra_chunk
                    pred_dec[offset : offset + len(dec_chunk), :] = dec_chunk
                    offset += len(ra_chunk)

        ra_res = ((pred_ra - obs_ra[None, :] + 180.0) % 360.0 - 180.0) * 3600.0
        dec_res = (pred_dec - obs_dec[None, :]) * 3600.0
        ra_res_mean = np.mean(ra_res, axis=0)
        dec_res_mean = np.mean(dec_res, axis=0)
        ra_res_flat = ra_res.ravel()
        dec_res_flat = dec_res.ravel()
        summary = _residuals_summary(ra_res_flat, dec_res_flat)

        _plot_residuals(obs_times.mjd, ra_res_mean, dec_res_mean, out_dir)
        _plot_residual_hist(ra_res_flat, dec_res_flat, out_dir)

        pred_ra_flat = pred_ra.ravel()
        pred_dec_flat = pred_dec.ravel()
        pred_ra_mean = np.mean(pred_ra, axis=0)
        pred_dec_mean = np.mean(pred_dec, axis=0)
        _plot_sky(obs_ra, obs_dec, pred_ra_flat, pred_dec_flat, out_dir)

        _plot_composite(
            obs_times.mjd,
            ra_res_mean,
            dec_res_mean,
            ra_res_flat,
            dec_res_flat,
            elements,
            obs_ra,
            obs_dec,
            pred_ra_flat,
            pred_dec_flat,
            pred_ra_mean,
            pred_dec_mean,
            horizons_ra,
            horizons_dec,
            out_dir,
        )

        if horizons_ra is not None and horizons_dec is not None:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(obs_ra, obs_dec, c="k", s=20, label="obs")
            ax.scatter(horizons_ra, horizons_dec, c="tab:red", s=30, label="Horizons")
            ax.plot(pred_ra_mean, pred_dec_mean, color="tab:orange", lw=1.0, label="replica mean")
            ax.set_xlabel("RA (deg)")
            ax.set_ylabel("Dec (deg)")
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_dir / "sky_obs_vs_horizons.png", dpi=150)
            plt.close(fig)
    else:
        summary = {
            "n_states": int(states.shape[0]),
            "epoch_utc": epoch.isot,
        }

    _plot_cloud_planes(states, horizons_xyz, out_dir)

    summary_path = out_dir / "summary.json"
    with summary_path.open("w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)

    print(f"Saved plots and summary to {out_dir}")


if __name__ == "__main__":
    main()
