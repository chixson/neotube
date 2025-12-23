from __future__ import annotations

import argparse
import csv
import random
import re
import subprocess
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS


def _load_cloud_rows(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    ras: list[float] = []
    decs: list[float] = []
    with path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if not row.get("ra_deg") or not row.get("dec_deg"):
                continue
            ras.append(float(row["ra_deg"]))
            decs.append(float(row["dec_deg"]))
    if not ras:
        raise ValueError(f"No RA/Dec rows found in {path}")
    return np.asarray(ras, dtype=float), np.asarray(decs, dtype=float)


_CENTER_RE = re.compile(r"_ra(?P<ra>-?\d+(?:\.\d+)?)_dec(?P<dec>-?\d+(?:\.\d+)?)_")


def _parse_requested_center_from_filename(path: Path) -> Tuple[float | None, float | None]:
    match = _CENTER_RE.search(path.name)
    if not match:
        return None, None
    return float(match.group("ra")), float(match.group("dec"))


def _circular_mean_deg(angles_deg: np.ndarray) -> float:
    ang = np.deg2rad(angles_deg)
    x = np.nanmean(np.cos(ang))
    y = np.nanmean(np.sin(ang))
    return float(np.rad2deg(np.arctan2(y, x)) % 360.0)


def _plot_cutout_overlay(
    *,
    cutout_fits: Path,
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    output: Path,
    title: str,
    max_points: int,
) -> Path:
    import matplotlib.pyplot as plt

    with fits.open(cutout_fits) as hdul:
        hdu = hdul[0]
        data = hdu.data
        header = hdu.header

    if data is None:
        raise ValueError(f"{cutout_fits} has no image data")
    if data.ndim != 2:
        raise ValueError(f"{cutout_fits} expected 2D image; got shape {data.shape}")

    wcs = WCS(header)
    ny, nx = data.shape

    rng = random.Random(0)
    idx = np.arange(len(ra_deg))
    if max_points > 0 and len(idx) > max_points:
        idx = np.asarray(rng.sample(list(idx), k=max_points), dtype=int)

    coords = SkyCoord(ra=ra_deg[idx], dec=dec_deg[idx], unit="deg", frame="icrs")
    x, y = wcs.world_to_pixel(coords)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]

    in_bounds = (x >= 0) & (x < nx) & (y >= 0) & (y < ny)
    xb = x[in_bounds]
    yb = y[in_bounds]

    fit_center_ra = _circular_mean_deg(ra_deg)
    fit_center_dec = float(np.nanmedian(dec_deg))
    fit_center = SkyCoord(ra=fit_center_ra, dec=fit_center_dec, unit="deg", frame="icrs")
    cx, cy = wcs.world_to_pixel(fit_center)

    req_ra, req_dec = _parse_requested_center_from_filename(cutout_fits)
    if req_ra is not None and req_dec is not None:
        req_center = SkyCoord(ra=req_ra, dec=req_dec, unit="deg", frame="icrs")
        rx, ry = wcs.world_to_pixel(req_center)
    else:
        rx = ry = None

    v = data.astype(float)
    finite = np.isfinite(v)
    if finite.any():
        lo, hi = np.percentile(v[finite], [1.0, 99.5])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = np.nanmin(v[finite]), np.nanmax(v[finite])
    else:
        lo, hi = 0.0, 1.0

    fig = plt.figure(figsize=(7, 7), dpi=160)
    ax = fig.add_subplot(111, projection=wcs)
    ax.imshow(v, origin="lower", cmap="gray", vmin=lo, vmax=hi)
    ax.set_title(title)
    ax.set_xlabel("RA")
    ax.set_ylabel("Dec")

    if len(xb):
        ax.scatter(xb, yb, s=6, c="#00a2ff", alpha=0.25, lw=0, transform=ax.get_transform("pixel"), label="replicas (in-bounds)")
    ax.scatter([cx], [cy], s=80, c="yellow", marker="x", linewidths=2.0, transform=ax.get_transform("pixel"), label="replica mean (fit)")
    if rx is not None and ry is not None:
        ax.scatter([rx], [ry], s=90, c="red", marker="+", linewidths=2.0, transform=ax.get_transform("pixel"), label="requested center")

    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return output


def _write_ds9_region(
    *,
    output: Path,
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    max_points: int,
    fit_center_ra: float,
    fit_center_dec: float,
    requested_center: Tuple[float | None, float | None],
) -> Path:
    rng = random.Random(0)
    idx = np.arange(len(ra_deg))
    if max_points > 0 and len(idx) > max_points:
        idx = np.asarray(rng.sample(list(idx), k=max_points), dtype=int)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as fh:
        fh.write("# Region file format: DS9 version 4.1\n")
        fh.write('global color=cyan dashlist=8 3 width=1 font="helvetica 10 normal" ')
        fh.write("select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n")
        fh.write("fk5\n")
        # Replicas
        for ra, dec in zip(ra_deg[idx], dec_deg[idx], strict=False):
            fh.write(f"point({ra:.8f},{dec:.8f}) # point=circle color=cyan\n")
        # Fit center marker
        fh.write(f"point({fit_center_ra:.8f},{fit_center_dec:.8f}) # point=x color=yellow text={{fit_mean}}\n")
        req_ra, req_dec = requested_center
        if req_ra is not None and req_dec is not None:
            fh.write(f"point({req_ra:.8f},{req_dec:.8f}) # point=box color=red text={{requested_center}}\n")
    return output


def _plot_radec_and_pca(
    *,
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    output: Path,
    title: str,
) -> Path:
    import matplotlib.pyplot as plt

    ra0 = _circular_mean_deg(ra_deg)
    dec0 = float(np.nanmedian(dec_deg))

    dra = ((ra_deg - ra0 + 180.0) % 360.0) - 180.0
    dra_cosdec = dra * np.cos(np.deg2rad(dec0))
    ddec = dec_deg - dec0

    x = dra_cosdec * 3600.0
    y = ddec * 3600.0
    pts = np.column_stack([x, y])
    pts = pts[np.all(np.isfinite(pts), axis=1)]

    cov = np.cov(pts.T)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    rot = pts @ vecs
    x1 = rot[:, 0]
    x2 = rot[:, 1]

    fig = plt.figure(figsize=(7, 7), dpi=160)
    ax = fig.add_subplot(111)
    ax.scatter(pts[:, 0], pts[:, 1], s=6, alpha=0.25, color="#00a2ff", lw=0)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.axvline(0, color="0.6", lw=0.8)
    ax.set_title(title)
    ax.set_xlabel("ΔRA cosδ (arcsec)")
    ax.set_ylabel("ΔDec (arcsec)")
    ax.set_aspect("equal", adjustable="box")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)

    # Also emit a PCA-plane plot for easier discussion.
    pca_out = output.with_name(output.stem + "_pca_plane.png")
    fig2 = plt.figure(figsize=(7, 7), dpi=160)
    ax2 = fig2.add_subplot(111)
    ax2.scatter(x1, x2, s=6, alpha=0.25, color="#00a2ff", lw=0)
    ax2.axhline(0, color="0.6", lw=0.8)
    ax2.axvline(0, color="0.6", lw=0.8)
    ax2.set_title(title + " (PCA plane)")
    ax2.set_xlabel("PC1 (arcsec)")
    ax2.set_ylabel("PC2 (arcsec)")
    ax2.set_aspect("equal", adjustable="box")
    fig2.tight_layout()
    fig2.savefig(pca_out)
    plt.close(fig2)

    return output


def _open_with_wslview(paths: Iterable[Path]) -> None:
    for p in paths:
        try:
            subprocess.Popen(["wslview", str(p)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            raise SystemExit("wslview not found. Install wslu or open the PNGs manually.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot replica cloud overlays on cutouts.")
    parser.add_argument("--cloud", type=Path, required=True, help="Propagated cloud CSV (time_utc, replica_id, ra_deg, dec_deg).")
    parser.add_argument("--cutout", type=Path, required=True, help="Cutout FITS file to overlay on (must contain WCS).")
    parser.add_argument("--out", type=Path, default=Path("replica_overlay.png"), help="Output PNG path.")
    parser.add_argument(
        "--ds9-region",
        type=Path,
        default=None,
        help="Optional DS9 region file to write (fk5) containing replica points and centers.",
    )
    parser.add_argument("--max-points", type=int, default=2000, help="Max replica points to plot (subsamples deterministically).")
    parser.add_argument("--title", type=str, default="", help="Optional plot title override.")
    parser.add_argument("--open", action="store_true", help="Open output(s) via wslview.")
    args = parser.parse_args()

    ra_deg, dec_deg = _load_cloud_rows(args.cloud)
    title = args.title or f"Replica cloud overlay: {args.cutout.name}"

    fit_center_ra = _circular_mean_deg(ra_deg)
    fit_center_dec = float(np.nanmedian(dec_deg))
    req_ra, req_dec = _parse_requested_center_from_filename(args.cutout)

    out1 = args.out
    out2 = args.out.with_name(args.out.stem + "_radec.png")
    out3 = args.out.with_name(args.out.stem + "_radec_pca_plane.png")

    _plot_cutout_overlay(
        cutout_fits=args.cutout,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        output=out1,
        title=title,
        max_points=args.max_points,
    )
    _plot_radec_and_pca(ra_deg=ra_deg, dec_deg=dec_deg, output=out2, title=title)

    if args.ds9_region:
        _write_ds9_region(
            output=args.ds9_region,
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            max_points=args.max_points,
            fit_center_ra=fit_center_ra,
            fit_center_dec=fit_center_dec,
            requested_center=(req_ra, req_dec),
        )
        print(f"Wrote {args.ds9_region}")

    if args.open:
        _open_with_wslview([out1, out2, out3])

    print(f"Wrote {out1}")
    print(f"Wrote {out2}")
    print(f"Wrote {out3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
