from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from scipy.signal import fftconvolve

from .fit import OrbitPosterior, load_posterior_json
from .propagate import ReplicaCloud, propagate_replicas, predict_radec


def _logaddexp(a: float, b: float) -> float:
    return float(np.logaddexp(a, b))


def _logsumexp(x: np.ndarray) -> float:
    if x.size == 0:
        return float("-inf")
    m = float(np.max(x))
    if not np.isfinite(m):
        return float("-inf")
    return m + float(np.log(np.sum(np.exp(x - m))))


def _robust_sigma(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size < 10:
        s = float(np.std(x)) if x.size else 1.0
        return s if np.isfinite(s) and s > 0 else 1.0
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    s = 1.4826 * mad if mad > 0 else float(np.std(x))
    return s if np.isfinite(s) and s > 0 else 1.0


def estimate_background_and_sigma(img: np.ndarray, *, mask_center_px: int = 9) -> tuple[float, float]:
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    half = mask_center_px / 2.0
    m = (np.abs(xx - cx) > half) | (np.abs(yy - cy) > half)
    sample = img[m]
    finite = np.isfinite(sample)
    bg = float(np.median(sample[finite])) if finite.any() else 0.0
    sig = _robust_sigma(sample - bg)
    return bg, sig


def gaussian_psf_kernel(fwhm_px: float) -> np.ndarray:
    sigma = float(fwhm_px) / 2.35482004503
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0
    size = int(max(7, math.ceil(6 * sigma)))
    if size % 2 == 0:
        size += 1
    r = size // 2
    y, x = np.mgrid[-r : r + 1, -r : r + 1]
    k = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    return k.astype(np.float64)


def matched_filter_snr_map(img: np.ndarray, *, fwhm_px: float) -> np.ndarray:
    bg, sigma_pix = estimate_background_and_sigma(img)
    im = (img - bg).astype(np.float64)
    k = gaussian_psf_kernel(fwhm_px)
    denom = float(np.sqrt(np.sum(k * k)))
    if denom <= 0 or not np.isfinite(denom):
        denom = 1.0
    conv = fftconvolve(im, k[::-1, ::-1], mode="same")
    snr = conv / (sigma_pix * denom)
    return snr.astype(np.float64)


def bilinear_interp(a: np.ndarray, x: float, y: float) -> float:
    h, w = a.shape
    if not (0 <= x < w - 1 and 0 <= y < h - 1):
        return float("nan")
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    dx = x - x0
    dy = y - y0
    v00 = float(a[y0, x0])
    v10 = float(a[y0, x0 + 1])
    v01 = float(a[y0 + 1, x0])
    v11 = float(a[y0 + 1, x0 + 1])
    return (v00 * (1 - dx) * (1 - dy) + v10 * dx * (1 - dy) + v01 * (1 - dx) * dy + v11 * dx * dy)


def ess(weights: np.ndarray) -> float:
    w = weights.astype(np.float64)
    s = float(np.sum(w))
    if s <= 0 or not np.isfinite(s):
        return 0.0
    w = w / s
    return float(1.0 / np.sum(w * w))


def normalize_logw(logw: np.ndarray) -> np.ndarray:
    m = float(np.max(logw))
    ww = np.exp(logw - m)
    s = float(np.sum(ww))
    if s <= 0 or not np.isfinite(s):
        ww = np.full_like(ww, 1.0 / len(ww))
    else:
        ww = ww / s
    logw[:] = np.log(ww)
    return ww


def systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = weights.size
    positions = (rng.random() + np.arange(n)) / n
    cumulative = np.cumsum(weights)
    idx = np.searchsorted(cumulative, positions, side="left")
    idx[idx == n] = n - 1
    return idx


def resample_states(
    states: np.ndarray,
    logw: np.ndarray,
    *,
    rng: np.random.Generator,
    jitter_pos_km: float,
    jitter_vel_km_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    weights = normalize_logw(logw.copy())
    idx = systematic_resample(weights, rng)
    out = states[:, idx].copy()
    if jitter_pos_km > 0:
        out[0:3, :] += rng.normal(0.0, jitter_pos_km, size=(3, out.shape[1]))
    if jitter_vel_km_s > 0:
        out[3:6, :] += rng.normal(0.0, jitter_vel_km_s, size=(3, out.shape[1]))
    out_logw = np.full(out.shape[1], -math.log(max(out.shape[1], 1)), dtype=np.float64)
    return out, out_logw


@dataclass(frozen=True)
class Exposure:
    exposure_id: str
    t: Time
    cutout_path: Path
    planned_center_ra: float | None = None
    planned_center_dec: float | None = None


def _iter_cutouts_index(path: Path) -> Iterator[dict]:
    with path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            yield row


def load_exposures_from_cutouts_index(path: Path) -> list[Exposure]:
    out: list[Exposure] = []
    for row in _iter_cutouts_index(path):
        status = (row.get("status") or "").strip().lower()
        cutout_path = row.get("cutout_path")
        if status != "ok" or not cutout_path:
            continue
        obsjd = row.get("obsjd")
        if not obsjd:
            continue
        t = Time(float(obsjd), format="jd", scale="utc")
        exposure_id = row.get("exposure_id")
        if not exposure_id:
            # best-effort ID
            exposure_id = f"ztf_{row.get('filefracday','unknown')}_f{row.get('field','')}_c{row.get('ccdid','')}_q{row.get('qid','')}"
        planned_ra = float(row["planned_center_ra"]) if row.get("planned_center_ra") else None
        planned_dec = float(row["planned_center_dec"]) if row.get("planned_center_dec") else None
        out.append(
            Exposure(
                exposure_id=exposure_id,
                t=t,
                cutout_path=Path(cutout_path),
                planned_center_ra=planned_ra,
                planned_center_dec=planned_dec,
            )
        )
    out.sort(key=lambda e: e.t.jd)
    return out


def load_replicas_states_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids: list[int] = []
    states: list[list[float]] = []
    logw: list[float] = []
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
            if "logw" in row and row["logw"]:
                logw.append(float(row["logw"]))
            elif "w" in row and row["w"]:
                w = float(row["w"])
                logw.append(float("-inf") if w <= 0 else math.log(w))
    st = np.array(states, dtype=np.float64).T
    if logw:
        lw = np.array(logw, dtype=np.float64)
    else:
        lw = np.full(st.shape[1], -math.log(max(st.shape[1], 1)), dtype=np.float64)
    return np.array(ids, dtype=int), st, lw


def write_replicas_weighted_csv(
    path: Path,
    *,
    replica_ids: np.ndarray,
    states: np.ndarray,
    logw: np.ndarray,
    epoch: Time,
) -> None:
    w = normalize_logw(logw.copy())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
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
                "logw",
                "w",
                "epoch_utc",
            ]
        )
        for i in range(states.shape[1]):
            writer.writerow(
                [
                    int(replica_ids[i]),
                    f"{states[0, i]:.6f}",
                    f"{states[1, i]:.6f}",
                    f"{states[2, i]:.6f}",
                    f"{states[3, i]:.9f}",
                    f"{states[4, i]:.9f}",
                    f"{states[5, i]:.9f}",
                    f"{float(logw[i]):.12e}",
                    f"{float(w[i]):.12e}",
                    epoch.isot,
                ]
            )


def _pixel_scale_arcsec(wcs: WCS) -> float:
    scales = proj_plane_pixel_scales(wcs)  # degrees/pixel
    scale = float(np.nanmedian(scales)) * 3600.0
    if not np.isfinite(scale) or scale <= 0:
        return 1.0
    return scale


@dataclass(frozen=True)
class InferConfig:
    fwhm_arcsec: float
    snr_max: float
    temperature: float
    search_radius_px: int
    pos_sigma_px: float
    pdet: float
    miss_logl: float


def per_replica_logl(
    *,
    snr_map: np.ndarray,
    pred_x: float,
    pred_y: float,
    cfg: InferConfig,
) -> float:
    # Always include predicted-position evidence.
    snr0 = bilinear_interp(snr_map, pred_x, pred_y)
    best_snr = snr0
    best_dist = 0.0

    if np.isfinite(pred_x) and np.isfinite(pred_y):
        r = int(cfg.search_radius_px)
        x0 = int(max(0, math.floor(pred_x) - r))
        x1 = int(min(snr_map.shape[1] - 1, math.floor(pred_x) + r))
        y0 = int(max(0, math.floor(pred_y) - r))
        y1 = int(min(snr_map.shape[0] - 1, math.floor(pred_y) + r))
        sub = snr_map[y0 : y1 + 1, x0 : x1 + 1]
        if sub.size:
            j, i = np.unravel_index(np.nanargmax(sub), sub.shape)
            peak_snr = float(sub[j, i])
            px = x0 + int(i)
            py = y0 + int(j)
            dist = math.hypot(px - pred_x, py - pred_y)
            if np.isfinite(peak_snr) and (not np.isfinite(best_snr) or peak_snr > best_snr):
                best_snr = peak_snr
                best_dist = dist

    if not np.isfinite(best_snr):
        log_hit = float("-inf")
    else:
        snr_c = max(0.0, min(cfg.snr_max, float(best_snr)))
        log_flux = 0.5 * (snr_c**2) / max(cfg.temperature, 1e-6)
        log_pos = -0.5 * (best_dist / max(cfg.pos_sigma_px, 1e-6)) ** 2
        log_hit = log_flux + log_pos

    pdet = float(np.clip(cfg.pdet, 1e-6, 1 - 1e-6))
    a = math.log(1.0 - pdet) + cfg.miss_logl
    b = math.log(pdet) + log_hit
    return _logaddexp(a, b)


def infer_cutouts(
    *,
    posterior: OrbitPosterior,
    replica_ids: np.ndarray,
    replica_states: np.ndarray,
    logw: np.ndarray,
    cutouts_index: Path,
    out_dir: Path,
    perturbers: tuple[str, ...],
    workers: int | None,
    batch_size: int | None,
    max_step: float,
    fwhm_arcsec: float,
    snr_max: float,
    temperature: float,
    search_radius_px: int,
    pos_sigma_px: float | None,
    pdet: float,
    miss_logl: float,
    resample_ess_frac: float,
    jitter_pos_km: float,
    jitter_vel_km_s: float,
    seed: int,
    max_exposures: int | None,
    exposure_id: str | None,
) -> tuple[Path, Path]:
    exposures = load_exposures_from_cutouts_index(cutouts_index)
    if exposure_id:
        exposures = [e for e in exposures if e.exposure_id == exposure_id]
    if max_exposures is not None:
        exposures = exposures[:max_exposures]
    if not exposures:
        raise ValueError("No usable cutouts found in cutouts_index.csv (status=ok, cutout_path present).")

    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = out_dir / "evidence.csv"
    replicas_out_path = out_dir / "replicas_weighted.csv"

    if not evidence_path.exists():
        with evidence_path.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["exposure_id", "time_utc", "jd", "n", "ess", "maxw"])

    states = replica_states
    lw = logw

    for idx, exp in enumerate(exposures):
        with fits.open(exp.cutout_path) as hdul:
            hdu = hdul[0]
            img = np.asarray(hdu.data, dtype=np.float64)
            if img.ndim == 3 and img.shape[0] == 1:
                img = img[0]
            if img.ndim != 2:
                raise ValueError(f"{exp.cutout_path} expected 2D image; got {img.shape}")
            wcs = WCS(hdu.header)
        pixscale = _pixel_scale_arcsec(wcs)
        fwhm_px = max(0.5, float(fwhm_arcsec) / pixscale)
        cfg = InferConfig(
            fwhm_arcsec=fwhm_arcsec,
            snr_max=snr_max,
            temperature=temperature,
            search_radius_px=int(search_radius_px),
            pos_sigma_px=float(pos_sigma_px) if pos_sigma_px is not None else max(1.0, 0.5 * fwhm_px),
            pdet=pdet,
            miss_logl=miss_logl,
        )

        snr_map = matched_filter_snr_map(img, fwhm_px=fwhm_px)

        cloud = ReplicaCloud(epoch=posterior.epoch, states=states)
        propagated = propagate_replicas(
            cloud,
            [exp.t],
            perturbers,
            max_step=max_step,
            workers=workers,
            batch_size=batch_size,
        )[0]

        # Update weights per replica.
        for i in range(propagated.shape[1]):
            ra, dec = predict_radec(propagated[:, i], exp.t)
            coord = SkyCoord(ra=ra, dec=dec, unit="deg", frame="icrs")
            x, y = wcs.world_to_pixel(coord)
            x = float(np.asarray(x))
            y = float(np.asarray(y))
            lw[i] = float(lw[i] + per_replica_logl(snr_map=snr_map, pred_x=x, pred_y=y, cfg=cfg))

        wnorm = normalize_logw(lw.copy())
        e = ess(wnorm)

        if resample_ess_frac > 0 and e < resample_ess_frac * len(wnorm):
            states, lw = resample_states(
                states,
                lw,
                rng=rng,
                jitter_pos_km=jitter_pos_km,
                jitter_vel_km_s=jitter_vel_km_s,
            )
            # replica IDs are no longer meaningful after resample; reindex.
            replica_ids = np.arange(states.shape[1], dtype=int)
            wnorm = normalize_logw(lw.copy())
            e = ess(wnorm)

        write_replicas_weighted_csv(replicas_out_path, replica_ids=replica_ids, states=states, logw=lw, epoch=posterior.epoch)
        with evidence_path.open("a", newline="") as fh:
            w = csv.writer(fh)
            w.writerow([exp.exposure_id, exp.t.isot, f"{exp.t.jd:.8f}", len(wnorm), f"{e:.2f}", f"{float(np.max(wnorm)):.6f}"])

        print(f"[infer] {idx+1}/{len(exposures)} exp={exp.exposure_id} jd={exp.t.jd:.5f} ESS={e:.1f}/{len(wnorm)} maxw={float(np.max(wnorm)):.4f}")

    return replicas_out_path, evidence_path

