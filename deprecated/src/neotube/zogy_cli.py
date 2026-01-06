from __future__ import annotations

import csv
import logging
from pathlib import Path

from .zogy import zogy_subtract


def process_pairs_csv(pairs_csv: Path, *, fwhm_sci: float | None = None, fwhm_ref: float | None = None) -> Path:
    pairs_csv = Path(pairs_csv)
    out_rows: list[dict[str, str]] = []
    with pairs_csv.open() as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        for row in reader:
            sci_path = row.get("cutout_path") or row.get("sci")
            ref_path = row.get("ref_path") or row.get("ref")
            if not sci_path or not ref_path:
                logging.warning("Skipping row without sci/ref: %s", row)
                out_rows.append(row)
                continue
            sci_p = Path(sci_path)
            ref_p = Path(ref_path)
            if not sci_p.exists() or not ref_p.exists():
                logging.warning("Missing files: %s or %s", sci_p, ref_p)
                out_rows.append(row)
                continue
            try:
                d_out = sci_p.with_name(sci_p.stem + "_zogy_D.fits")
                s_out = sci_p.with_name(sci_p.stem + "_zogy_S.fits")
                pd_out = sci_p.with_name(sci_p.stem + "_zogy_Pd.fits")
                zogy_subtract(
                    sci_p,
                    ref_p,
                    fwhm_sci_arcsec=fwhm_sci,
                    fwhm_ref_arcsec=fwhm_ref,
                    out_diff=d_out,
                    out_s=s_out,
                    out_pd=pd_out,
                )
                new_row = dict(row)
                new_row["cutout_path"] = str(d_out)
                out_rows.append(new_row)
            except Exception:
                logging.exception("ZOGY failed for %s / %s", sci_p, ref_p)
                out_rows.append(row)
    out_csv = pairs_csv.with_name(pairs_csv.stem + ".zogy.csv")
    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in out_rows:
            writer.writerow(row)
    return out_csv


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run ZOGY subtraction for science/reference cutout pairs.")
    parser.add_argument("--pairs-csv", type=Path, required=True, help="CSV with columns `cutout_path` and `ref_path`.")
    parser.add_argument("--fwhm-sci-arcsec", type=float, default=None, help="Fallback FWHM for science.")
    parser.add_argument("--fwhm-ref-arcsec", type=float, default=None, help="Fallback FWHM for reference.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    process_pairs_csv(args.pairs_csv, fwhm_sci=args.fwhm_sci_arcsec, fwhm_ref=args.fwhm_ref_arcsec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
