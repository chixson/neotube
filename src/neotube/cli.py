from __future__ import annotations

import argparse
import csv
import logging
import os
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote_plus, urlencode

import numpy as np
import requests
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.nddata.utils import NoOverlapError
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

from .zogy import zogy_subtract

IBE_SCI_BASE = "https://irsa.ipac.caltech.edu/ibe/search/ztf/products/sci"
IBE_DATA_ROOT = "https://irsa.ipac.caltech.edu/ibe/data/ztf/products/sci"

DEFAULT_COLUMNS = [
    "obsjd",
    "obsdate",
    "ra",
    "dec",
    "filefracday",
    "field",
    "ccdid",
    "qid",
    "filtercode",
    "imgtypecode",
    "maglimit",
    "seeing",
    "airmass",
]


@dataclass(frozen=True)
class Exposure:
    obsjd: float
    obsdate: str
    ra: float
    dec: float
    filefracday: str
    field: int
    ccdid: int
    qid: int
    filtercode: str
    imgtypecode: str
    maglimit: Optional[float]
    seeing: Optional[float]
    airmass: Optional[float]
    planned_center_ra: float | None = None
    planned_center_dec: float | None = None
    planned_radius_arcsec: float | None = None
    node_time_utc: str | None = None


def pad_field(field: int) -> str:
    return f"{field:06d}"


def pad_ccd(ccdid: int) -> str:
    return f"{ccdid:02d}"


def exposure_unique_id(exp: Exposure) -> str:
    return f"ztf_{exp.filefracday}_f{pad_field(exp.field)}_{exp.filtercode}_c{pad_ccd(exp.ccdid)}_{exp.imgtypecode}_q{exp.qid}"


def sci_image_url(exp: Exposure, suffix: str = "sciimg.fits") -> str:
    filefracday = exp.filefracday
    year = filefracday[0:4]
    month = filefracday[4:6]
    day = filefracday[6:8]
    fracday = filefracday[8:14]

    return (
        f"{IBE_DATA_ROOT}/{year}/{month}{day}/{fracday}/"
        f"ztf_{filefracday}_{pad_field(exp.field)}_{exp.filtercode}_"
        f"c{pad_ccd(exp.ccdid)}_{exp.imgtypecode}_q{exp.qid}_{suffix}"
    )


class GlobalRateLimiter:
    def __init__(self, max_rps: float):
        self.min_interval = 1.0 / max_rps if max_rps > 0 else 0.0
        self._lock = threading.Lock()
        self._next_time = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.time()
            if now < self._next_time:
                time.sleep(self._next_time - now)
            jitter = random.uniform(0.02, 0.1)
            self._next_time = time.time() + self.min_interval + jitter


def _log_response(resp: requests.Response, payload_size: int) -> None:
    headers = resp.headers
    request_id = headers.get("X-Request-Id") or headers.get("Request-Id")
    server_timing = headers.get("Server-Timing")
    logging.info(
        "%s %s -> %s | payload=%dB request_id=%s server_timing=%s",
        resp.request.method,
        resp.url,
        resp.status_code,
        payload_size,
        request_id or "-",
        server_timing or "-",
    )


def _payload_size_from_request(req: requests.PreparedRequest) -> int:
    body = req.body
    if body is None:
        return 0
    if isinstance(body, bytes):
        return len(body)
    if isinstance(body, str):
        return len(body.encode("utf-8"))
    return 0


def request_with_backoff(
    session: requests.Session,
    url: str,
    *,
    headers: Dict[str, str],
    stream: bool = False,
    timeout: Tuple[int, int] = (10, 60),
    max_tries: int = 8,
    limiter: Optional[GlobalRateLimiter] = None,
) -> requests.Response:
    backoff = 1.0
    for attempt in range(1, max_tries + 1):
        if limiter:
            limiter.wait()

        try:
            resp = session.get(url, headers=headers, stream=stream, timeout=timeout)
        except requests.RequestException:
            if attempt == max_tries:
                raise
            sleep_s = backoff + random.uniform(0, 0.5)
            time.sleep(sleep_s)
            backoff = min(backoff * 2.0, 60.0)
            continue

        payload_size = _payload_size_from_request(resp.request)
        _log_response(resp, payload_size)

        if resp.status_code == 200:
            return resp

        if resp.status_code in (429, 500, 502, 503, 504):
            retry_after = resp.headers.get("Retry-After")
            try:
                sleep_s = float(retry_after) if retry_after else backoff
            except ValueError:
                sleep_s = backoff
            resp.close()
            if attempt == max_tries:
                raise RuntimeError(
                    f"Request to {url} failed after {max_tries} attempts; last status={resp.status_code}"
                )
            time.sleep(sleep_s + random.uniform(0, 0.5))
            backoff = min(backoff * 2.0, 120.0)
            continue

        body_snip = ""
        try:
            body_snip = resp.text[:300]
        except Exception:
            pass
        resp.close()
        raise RuntimeError(f"Non-retryable HTTP {resp.status_code} for {url}. Body: {body_snip}")

    raise RuntimeError("request_with_backoff reached unreachable code")


def query_exposures(
    session: requests.Session,
    *,
    ra: float,
    dec: float,
    jd_start: float,
    jd_end: float,
    columns: List[str],
    headers: Dict[str, str],
    limiter: Optional[GlobalRateLimiter],
    size_deg: float = 0.0,
    filtercode: Optional[str] = None,
) -> List[Exposure]:
    where = f"obsjd BETWEEN {jd_start} AND {jd_end}"
    if filtercode:
        where += f" AND filtercode='{filtercode}'"

    params = {
        "POS": f"{ra:.8f},{dec:.8f}",
        "CT": "csv",
        "COLUMNS": ",".join(columns),
        "WHERE": where,
    }
    if size_deg > 0:
        params["SIZE"] = str(size_deg)

    url = IBE_SCI_BASE + "?" + urlencode(params, quote_via=quote_plus)
    resp = request_with_backoff(session, url, headers=headers, limiter=limiter)
    text = resp.text
    resp.close()

    lines = text.splitlines()
    reader = csv.DictReader(lines)
    if not reader.fieldnames or "obsjd" not in reader.fieldnames:
        snippet = "\n".join(lines[:20])
        raise RuntimeError(f"Unexpected IRSA CSV response (missing obsjd header). First lines:\n{snippet}")
    exposures: List[Exposure] = []
    for row in reader:
        def parse_float(key: str) -> Optional[float]:
            value = row.get(key)
            if not value:
                return None
            try:
                return float(value)
            except ValueError:
                return None

        exposures.append(
            Exposure(
                obsjd=float(row["obsjd"]),
                obsdate=row.get("obsdate", ""),
                ra=float(row["ra"]) if row.get("ra") else ra,
                dec=float(row["dec"]) if row.get("dec") else dec,
                filefracday=row["filefracday"],
                field=int(row["field"]),
                ccdid=int(row["ccdid"]),
                qid=int(row["qid"]),
                filtercode=row["filtercode"],
                imgtypecode=row["imgtypecode"],
                maglimit=parse_float("maglimit"),
                seeing=parse_float("seeing"),
                airmass=parse_float("airmass"),
            )
        )

    exposures.sort(key=lambda exp: exp.obsjd)
    return exposures


def load_plan_exposures(path: Path) -> list[Exposure]:
    exposures: list[Exposure] = []
    with path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            exposures.append(
                Exposure(
                    obsjd=float(row["obsjd"]),
                    obsdate=row.get("obsdate", ""),
                    ra=float(row["ra"]),
                    dec=float(row["dec"]),
                    filefracday=row["filefracday"],
                    field=int(row["field"]),
                    ccdid=int(row["ccdid"]),
                    qid=int(row["qid"]),
                    filtercode=row["filtercode"],
                    imgtypecode=row["imgtypecode"],
                    maglimit=float(row["maglimit"]) if row.get("maglimit") else None,
                    seeing=float(row["seeing"]) if row.get("seeing") else None,
                    airmass=float(row["airmass"]) if row.get("airmass") else None,
                    planned_center_ra=float(row["planned_center_ra"]) if row.get("planned_center_ra") else None,
                    planned_center_dec=float(row["planned_center_dec"]) if row.get("planned_center_dec") else None,
                    planned_radius_arcsec=float(row["planned_radius_arcsec"]) if row.get("planned_radius_arcsec") else None,
                    node_time_utc=row.get("node_time_utc"),
                )
            )
    return exposures


def download_cutout(
    session: requests.Session,
    exposure: Exposure,
    *,
    center_ra: float,
    center_dec: float,
    size_arcsec: float,
    out_dir: str,
    suffix: str,
    headers: Dict[str, str],
    limiter: Optional[GlobalRateLimiter],
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    base_url = sci_image_url(exposure, suffix=suffix)
    cutout_url = (
        f"{base_url}?center={center_ra:.8f},{center_dec:.8f}"
        f"&size={size_arcsec:.1f}arcsec&gzip=false"
    )

    filename = (
        f"ztf_{exposure.filefracday}_f{pad_field(exposure.field)}_{exposure.filtercode}_"
        f"c{pad_ccd(exposure.ccdid)}_q{exposure.qid}_{suffix}_"
        f"ra{center_ra:.5f}_dec{center_dec:.5f}_s{int(size_arcsec)}.fits"
    )
    path = os.path.join(out_dir, filename)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        logging.debug("Cutout already exists: %s", path)
        return path

    try:
        resp = request_with_backoff(
            session,
            cutout_url,
            headers=headers,
            stream=True,
            max_tries=2,
            limiter=limiter,
        )

        with open(path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1024 * 256):
                if chunk:
                    fh.write(chunk)
        resp.close()
        return path
    except RuntimeError as exc:
        # IRSA ZTF products sometimes return HTTP 500 for cutout-style requests.
        # Fall back to downloading the full frame and extracting the cutout locally.
        msg = str(exc)
        if "status=500" not in msg and "HTTP 500" not in msg and "last status=500" not in msg:
            raise
        logging.warning("Remote cutout failed (HTTP 500); downloading full image and cutting locally.")

    full_dir = os.path.join(out_dir, "_full")
    os.makedirs(full_dir, exist_ok=True)
    full_path = os.path.join(full_dir, os.path.basename(base_url))
    if not (os.path.exists(full_path) and os.path.getsize(full_path) > 0):
        resp = request_with_backoff(
            session,
            base_url,
            headers=headers,
            stream=True,
            max_tries=5,
            limiter=limiter,
        )
        with open(full_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)
        resp.close()

    with fits.open(full_path, memmap=False) as hdul:
        hdu = hdul[0]
        wcs = WCS(hdu.header)
        # Estimate pixel scale (arcsec/pixel) and derive pixel cutout size.
        scales = proj_plane_pixel_scales(wcs) * u.deg
        scale_arcsec = float(np.mean(scales.to(u.arcsec)).value)
        size_pix = max(5, int(np.ceil(size_arcsec / max(scale_arcsec, 1e-6))))
        position = SkyCoord(center_ra * u.deg, center_dec * u.deg, frame="icrs")
        try:
            cutout = Cutout2D(
                hdu.data,
                position=position,
                size=(size_pix, size_pix),
                wcs=wcs,
                mode="partial",
                fill_value=np.nan,
            )
        except NoOverlapError as exc:
            raise RuntimeError("NoOverlap") from exc
        new_header = hdu.header.copy()
        new_header.update(cutout.wcs.to_header())
        fits.PrimaryHDU(data=cutout.data, header=new_header).writeto(path, overwrite=True)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Polite metadata + cutout downloader for ZTF science exposures."
    )
    parser.add_argument("--ra", type=float, default=None, help="ICRS RA in degrees (required unless --plan is used)")
    parser.add_argument("--dec", type=float, default=None, help="ICRS Dec in degrees (required unless --plan is used)")
    parser.add_argument("--jd-start", type=float, default=None, help="JD start (required unless --plan is used)")
    parser.add_argument("--jd-end", type=float, default=None, help="JD end (required unless --plan is used)")
    parser.add_argument("--filter", type=str, default=None, help="Optional filter suffix (zg, zr, zi)")
    parser.add_argument("--size-arcsec", type=float, default=50.0, help="Cutout size (square) in arcseconds")
    parser.add_argument(
        "--search-size-deg",
        type=float,
        default=0.0,
        help="Optional search box around the coordinates (degrees)",
    )
    parser.add_argument("--suffix", type=str, default="sciimg.fits", help="ZTF product suffix")
    parser.add_argument("--out", type=str, default="cutouts", help="Output directory for cutouts")
    parser.add_argument("--index-csv", type=str, default="cutouts_index.csv", help="Index CSV path")
    parser.add_argument("--max-rps", type=float, default=0.8, help="Max requests per second")
    parser.add_argument("--max-cutouts", type=int, default=0, help="Stop after downloading this many cutouts")
    parser.add_argument("--user-agent", type=str, default=None, help="Custom User-Agent header")
    parser.add_argument("--ref-suffix", type=str, default="refimg.fits", help="Suffix for reference image cutouts (empty to disable).")
    parser.add_argument("--run-zogy", action="store_true", help="Run ZOGY on science + reference cutouts.")
    parser.add_argument("--zogy-fwhm-sci-arcsec", type=float, default=None, help="Fallback PSF FWHM for science cutouts (arcsec).")
    parser.add_argument("--zogy-fwhm-ref-arcsec", type=float, default=None, help="Fallback PSF FWHM for reference cutouts (arcsec).")
    parser.add_argument("--plan", type=Path, default=None, help="Plan CSV describing exposures + centers.")
    parser.add_argument("--plan-margin-arcsec", type=float, default=5.0, help="Additional margin when using plan centers.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG|INFO|WARN)")

    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(message)s")

    headers = {"User-Agent": args.user_agent or "neotube-ztf/0.1 (contact: chixson@fourshadows.org)"}
    limiter = GlobalRateLimiter(args.max_rps)

    exposures: list[Exposure]
    with requests.Session() as session:
        if args.plan:
            exposures = load_plan_exposures(args.plan)
            logging.info("Loaded %d exposures from plan %s", len(exposures), args.plan)
        else:
            if args.ra is None or args.dec is None or args.jd_start is None or args.jd_end is None:
                raise SystemExit("Must provide --ra/--dec/--jd-start/--jd-end unless using --plan.")
            exposures = query_exposures(
                session,
                ra=args.ra,
                dec=args.dec,
                jd_start=args.jd_start,
                jd_end=args.jd_end,
                columns=DEFAULT_COLUMNS,
                headers=headers,
                limiter=limiter,
                size_deg=args.search_size_deg,
                filtercode=args.filter,
            )
            logging.info(
                "Found %d exposures for %.5f, %.5f between JD %.2f and %.2f",
                len(exposures),
                args.ra,
                args.dec,
                args.jd_start,
                args.jd_end,
            )

        if not exposures:
            return 0

        downloaded = 0
        os.makedirs(os.path.dirname(args.index_csv) or ".", exist_ok=True)
        with open(args.index_csv, "w", newline="") as idx_fh:
            writer = csv.writer(idx_fh)
            writer.writerow(
                [
                    "obsjd",
                    "obsdate",
                    "filefracday",
                    "filtercode",
                    "field",
                    "ccdid",
                    "qid",
                    "planned_center_ra",
                    "planned_center_dec",
                    "planned_radius_arcsec",
                    "status",
                    "error",
                    "cutout_path",
                    "ref_cutout_path",
                    "zogy_diff_path",
                    "zogy_s_path",
                    "zogy_pd_path",
                    "zogy_status",
                    "zogy_error",
                ]
            )

            for exposure in exposures:
                if args.max_cutouts and downloaded >= args.max_cutouts:
                    break

                if exposure.planned_center_ra is not None and exposure.planned_center_dec is not None:
                    center_ra = exposure.planned_center_ra
                    center_dec = exposure.planned_center_dec
                    radius = exposure.planned_radius_arcsec or 0.0
                    size_arcsec = max(args.size_arcsec, radius * 2.0 + args.plan_margin_arcsec)
                else:
                    center_ra = args.ra
                    center_dec = args.dec
                    size_arcsec = args.size_arcsec
                status = "ok"
                err = ""
                path = ""
                try:
                    path = download_cutout(
                        session,
                        exposure,
                        center_ra=center_ra,
                        center_dec=center_dec,
                        size_arcsec=size_arcsec,
                        out_dir=args.out,
                        suffix=args.suffix,
                        headers=headers,
                        limiter=limiter,
                    )
                except Exception as exc:
                    status = "error"
                    err = str(exc)
                    logging.warning("Failed exposure %s: %s", exposure_unique_id(exposure), err)
                    if args.log_level.upper() != "DEBUG":
                        raise
                ref_path = ""
                zogy_diff_path = ""
                zogy_s_path = ""
                zogy_pd_path = ""
                zogy_status = ""
                zogy_err = ""
                if args.ref_suffix and status == "ok":
                    try:
                        ref_path = download_cutout(
                            session,
                            exposure,
                            center_ra=center_ra,
                            center_dec=center_dec,
                            size_arcsec=size_arcsec,
                            out_dir=args.out,
                            suffix=args.ref_suffix,
                            headers=headers,
                            limiter=limiter,
                        )
                    except Exception as exc:
                        logging.warning("Failed reference cutout %s: %s", exposure_unique_id(exposure), str(exc))
                if args.run_zogy and path and ref_path:
                    try:
                        sci_p = Path(path)
                        d_out = sci_p.with_name(sci_p.stem + "_zogy_D.fits")
                        s_out = sci_p.with_name(sci_p.stem + "_zogy_S.fits")
                        pd_out = sci_p.with_name(sci_p.stem + "_zogy_Pd.fits")
                        zogy_subtract(
                            sci_p,
                            Path(ref_path),
                            fwhm_sci_arcsec=args.zogy_fwhm_sci_arcsec,
                            fwhm_ref_arcsec=args.zogy_fwhm_ref_arcsec,
                            out_diff=d_out,
                            out_s=s_out,
                            out_pd=pd_out,
                        )
                        zogy_diff_path = str(d_out)
                        zogy_s_path = str(s_out)
                        zogy_pd_path = str(pd_out)
                        zogy_status = "ok"
                    except Exception as exc:
                        zogy_status = "error"
                        zogy_err = str(exc)
                        logging.warning("ZOGY failed for %s: %s", exposure_unique_id(exposure), zogy_err)
                writer.writerow(
                    [
                        exposure.obsjd,
                        exposure.obsdate,
                        exposure.filefracday,
                        exposure.filtercode,
                        exposure.field,
                        exposure.ccdid,
                        exposure.qid,
                        exposure.planned_center_ra or "",
                        exposure.planned_center_dec or "",
                        exposure.planned_radius_arcsec or "",
                        status,
                        err,
                        path,
                        ref_path,
                        zogy_diff_path,
                        zogy_s_path,
                        zogy_pd_path,
                        zogy_status,
                        zogy_err,
                    ]
                )
                if status == "ok":
                    downloaded += 1
                    logging.info("Downloaded %s", path)

        logging.info("Downloaded %d cutouts into %s", downloaded, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
