from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales


def _robust_sigma(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size < 10:
        s = float(np.std(x)) if x.size else 1.0
        return s if np.isfinite(s) and s > 0 else 1.0
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    s = 1.4826 * mad if mad > 0 else float(np.std(x))
    return s if np.isfinite(s) and s > 0 else 1.0


def estimate_background_and_sigma(img: np.ndarray, *, mask_center_px: int = 9) -> Tuple[float, float]:
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
    s = float(np.sum(k))
    if s == 0 or not np.isfinite(s):
        return k.astype(np.float64)
    return (k / s).astype(np.float64)


def _pixel_scale_arcsec_from_header(header) -> float:
    try:
        wcs = WCS(header)
        scales = proj_plane_pixel_scales(wcs)
        scales = np.array(scales) * (180.0 / math.pi) * 3600.0
        pixscale_arcsec = float(np.nanmedian(scales))
        if not np.isfinite(pixscale_arcsec) or pixscale_arcsec <= 0:
            return 1.0
        return pixscale_arcsec
    except Exception:
        return 1.0


def _pad_psf_to_image(psf: np.ndarray, img_shape: Tuple[int, int]) -> np.ndarray:
    ny, nx = img_shape
    out = np.zeros((ny, nx), dtype=np.float64)
    hy, wx = psf.shape[0] // 2, psf.shape[1] // 2
    cy, cx = ny // 2, nx // 2
    y0 = cy - hy
    x0 = cx - wx
    y1 = y0 + psf.shape[0]
    x1 = x0 + psf.shape[1]
    yy0 = max(0, y0)
    xx0 = max(0, x0)
    yy1 = min(ny, y1)
    xx1 = min(nx, x1)
    py0 = yy0 - y0
    px0 = xx0 - x0
    py1 = py0 + (yy1 - yy0)
    px1 = px0 + (xx1 - xx0)
    out[yy0:yy1, xx0:xx1] = psf[py0:py1, px0:px1]
    return out


def _ensure_normalized(psf: np.ndarray) -> np.ndarray:
    s = np.sum(psf)
    if s == 0 or not np.isfinite(s):
        return psf
    return psf / s


def zogy_subtract(
    sci,
    ref,
    *,
    psf_sci: Optional[np.ndarray] = None,
    psf_ref: Optional[np.ndarray] = None,
    fwhm_sci_arcsec: Optional[float] = None,
    fwhm_ref_arcsec: Optional[float] = None,
    out_diff: Optional[str | Path] = None,
    out_s: Optional[str | Path] = None,
    out_pd: Optional[str | Path] = None,
    estimate_noise: bool = True,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(sci, (str, Path)):
        with fits.open(str(sci), memmap=False) as hd:
            sci_img = hd[0].data.astype(np.float64)
            sci_hdr = hd[0].header
    else:
        sci_img = np.array(sci, dtype=np.float64)
        sci_hdr = None
    if isinstance(ref, (str, Path)):
        with fits.open(str(ref), memmap=False) as hd:
            ref_img = hd[0].data.astype(np.float64)
            ref_hdr = hd[0].header
    else:
        ref_img = np.array(ref, dtype=np.float64)
        ref_hdr = None
    if sci_img.shape != ref_img.shape:
        raise ValueError("science and reference images must have same shape")
    ny, nx = sci_img.shape
    if estimate_noise:
        bg_s, sig_s = estimate_background_and_sigma(sci_img)
        bg_r, sig_r = estimate_background_and_sigma(ref_img)
    else:
        bg_s = bg_r = 0.0
        sig_s = sig_r = 1.0
    S = sci_img - bg_s
    R = ref_img - bg_r
    var_s = float(sig_s**2) if sig_s > 0 else 1.0
    var_r = float(sig_r**2) if sig_r > 0 else 1.0
    if psf_sci is None:
        if fwhm_sci_arcsec is None and sci_hdr is not None:
            pix = _pixel_scale_arcsec_from_header(sci_hdr)
            fwhm_px = 2.0 / pix
        elif fwhm_sci_arcsec is not None and sci_hdr is not None:
            pix = _pixel_scale_arcsec_from_header(sci_hdr)
            fwhm_px = fwhm_sci_arcsec / max(pix, 1e-6)
        elif fwhm_sci_arcsec is not None:
            fwhm_px = fwhm_sci_arcsec
        else:
            fwhm_px = 2.0
        psf_sci = gaussian_psf_kernel(fwhm_px)
    if psf_ref is None:
        if fwhm_ref_arcsec is None and ref_hdr is not None:
            pix = _pixel_scale_arcsec_from_header(ref_hdr)
            fwhm_px = 2.0 / pix
        elif fwhm_ref_arcsec is not None and ref_hdr is not None:
            pix = _pixel_scale_arcsec_from_header(ref_hdr)
            fwhm_px = fwhm_ref_arcsec / max(pix, 1e-6)
        elif fwhm_ref_arcsec is not None:
            fwhm_px = fwhm_ref_arcsec
        else:
            fwhm_px = 2.0
        psf_ref = gaussian_psf_kernel(fwhm_px)
    psf_sci = _ensure_normalized(np.array(psf_sci, dtype=np.float64))
    psf_ref = _ensure_normalized(np.array(psf_ref, dtype=np.float64))
    psf_sci_pad = _pad_psf_to_image(psf_sci, (ny, nx))
    psf_ref_pad = _pad_psf_to_image(psf_ref, (ny, nx))
    Fs = np.fft.fft2(S)
    Fr = np.fft.fft2(R)
    Ps = np.fft.fft2(psf_sci_pad)
    Pr = np.fft.fft2(psf_ref_pad)
    absPr2 = np.abs(Pr) ** 2
    absPs2 = np.abs(Ps) ** 2
    denom = absPr2 / var_s + absPs2 / var_r
    denom = np.maximum(denom, eps)
    numerator = (np.conj(Pr) * Fs) / var_s - (np.conj(Ps) * Fr) / var_r
    D_hat = numerator / denom
    D = np.fft.ifft2(D_hat).real
    P_D_hat = (Ps * np.conj(Pr)) / np.sqrt(denom)
    Var_D_hat = 1.0 / denom
    Var_S_hat = Var_D_hat * (np.abs(P_D_hat) ** 2)
    sqrtVar_S_hat = np.sqrt(np.maximum(Var_S_hat, eps))
    S_corr_hat = D_hat * np.conj(P_D_hat) / sqrtVar_S_hat
    S_corr = np.fft.ifft2(S_corr_hat).real
    P_D = np.fft.ifft2(P_D_hat).real
    s = np.sum(P_D)
    if s != 0 and np.isfinite(s):
        P_D = P_D / s
    header = None
    if sci_hdr is not None:
        header = sci_hdr.copy()
    elif ref_hdr is not None:
        header = ref_hdr.copy()
    if out_diff is not None:
        hdu = fits.PrimaryHDU(data=np.array(D, dtype=np.float32), header=header)
        hdu.writeto(str(out_diff), overwrite=True)
    if out_s is not None:
        hdu = fits.PrimaryHDU(data=np.array(S_corr, dtype=np.float32), header=header)
        hdu.writeto(str(out_s), overwrite=True)
    if out_pd is not None:
        hdu = fits.PrimaryHDU(data=np.array(P_D, dtype=np.float32), header=header)
        hdu.writeto(str(out_pd), overwrite=True)
    return D.astype(np.float64), S_corr.astype(np.float64), P_D.astype(np.float64)

__all__ = ["zogy_subtract"]
