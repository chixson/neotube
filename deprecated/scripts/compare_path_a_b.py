#!/usr/bin/env python3
import argparse
import json
import math
import os
import time
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astroquery.jplhorizons import Horizons
from astropy.time import Time


AU_KM = 149597870.7
DAY_S = 86400.0


def ang_sep_arcsec(ra1, dec1, ra2, dec2):
    ra1r, ra2r = math.radians(ra1), math.radians(ra2)
    d1, d2 = math.radians(dec1), math.radians(dec2)
    cosang = math.sin(d1) * math.sin(d2) + math.cos(d1) * math.cos(d2) * math.cos(ra1r - ra2r)
    cosang = max(-1.0, min(1.0, cosang))
    return math.degrees(math.acos(cosang)) * 3600.0


def signed_deltas(ra1, dec1, ra2, dec2):
    ra1 = np.deg2rad(np.asarray(ra1))
    ra2 = np.deg2rad(np.asarray(ra2))
    dec1 = np.deg2rad(np.asarray(dec1))
    dec2 = np.deg2rad(np.asarray(dec2))
    dra = ra1 - ra2
    dra = (dra + np.pi) % (2 * np.pi) - np.pi
    dx = dra * np.cos(dec1) * 206265.0
    dy = (dec1 - dec2) * 206265.0
    return dx, dy


def normalize_target(target: str) -> Tuple[str, str]:
    target = target.strip()
    if target.isdigit() and int(target) >= 2000000:
        return f"DES={target}", "smallbody"
    return target, "smallbody"


def load_cache(path: str) -> Dict[str, Dict[str, Tuple[float, float]]]:
    if not os.path.exists(path):
        return {}
    with open(path, "r") as fh:
        return json.load(fh)


def save_cache(path: str, data: Dict[str, Dict[str, Tuple[float, float]]]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)


def fetch_horizons_quantities(
    target: str, id_type: str, site: str, t_obs_iso: str
) -> Dict[str, Tuple[float, float]]:
    t_obs = Time(t_obs_iso, scale="utc")
    hz = Horizons(id=target, location=site, epochs=t_obs.jd, id_type=id_type)
    eph_q1 = hz.ephemerides(quantities=1, extra_precision=True)
    eph_q2 = hz.ephemerides(quantities=2, extra_precision=True)
    return {
        "q1": (float(eph_q1["RA"][0]), float(eph_q1["DEC"][0])),
        "q2": (float(eph_q2["RA_app"][0]), float(eph_q2["DEC_app"][0])),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="runs/ceres/geom_validator_out.csv")
    p.add_argument("--target", default="1")
    p.add_argument("--cache", default="runs/ceres/horizons_apparent_cache.json")
    p.add_argument("--max-rps", type=float, default=2.0)
    args = p.parse_args()

    target, id_type = normalize_target(args.target)
    df = pd.read_csv(args.csv)
    df["site"] = df["site"].astype(str).str.strip().str.upper()

    cache = load_cache(args.cache)
    min_interval = 1.0 / args.max_rps if args.max_rps > 0 else 0.0
    last_request = 0.0

    hz_ra_q1 = []
    hz_dec_q1 = []
    hz_ra_q2 = []
    hz_dec_q2 = []
    for _, row in df.iterrows():
        site = row["site"]
        t_obs = row["time_utc"]
        key = f"{target}|{site}|{t_obs}"
        cached = cache.get(key)
        if isinstance(cached, dict) and "q1" in cached and "q2" in cached:
            ra1, dec1 = cached["q1"]
            ra2, dec2 = cached["q2"]
        else:
            now = time.time()
            wait = last_request + min_interval - now
            if wait > 0:
                time.sleep(wait)
            q = fetch_horizons_quantities(target, id_type, site, t_obs)
            ra1, dec1 = q["q1"]
            ra2, dec2 = q["q2"]
            cache[key] = {"q1": [ra1, dec1], "q2": [ra2, dec2]}
            last_request = time.time()
        hz_ra_q1.append(ra1)
        hz_dec_q1.append(dec1)
        hz_ra_q2.append(ra2)
        hz_dec_q2.append(dec2)

    save_cache(args.cache, cache)

    df["hz_ra_q1"] = hz_ra_q1
    df["hz_dec_q1"] = hz_dec_q1
    df["hz_ra_q2"] = hz_ra_q2
    df["hz_dec_q2"] = hz_dec_q2

    df["sep_a_q1"] = [
        ang_sep_arcsec(r1, d1, r2, d2)
        for r1, d1, r2, d2 in zip(df["pred_ra"], df["pred_dec"], df["hz_ra_q1"], df["hz_dec_q1"])
    ]
    df["sep_a_q2"] = [
        ang_sep_arcsec(r1, d1, r2, d2)
        for r1, d1, r2, d2 in zip(df["pred_ra"], df["pred_dec"], df["hz_ra_q2"], df["hz_dec_q2"])
    ]

    if "pred_ra_b" in df.columns and df["pred_ra_b"].notna().any():
        df["sep_b_q1"] = [
            ang_sep_arcsec(r1, d1, r2, d2)
            for r1, d1, r2, d2 in zip(df["pred_ra_b"], df["pred_dec_b"], df["hz_ra_q1"], df["hz_dec_q1"])
        ]
        df["sep_b_q2"] = [
            ang_sep_arcsec(r1, d1, r2, d2)
            for r1, d1, r2, d2 in zip(df["pred_ra_b"], df["pred_dec_b"], df["hz_ra_q2"], df["hz_dec_q2"])
        ]
    else:
        df["sep_b_q1"] = np.nan
        df["sep_b_q2"] = np.nan

    df["dx_a_q1"], df["dy_a_q1"] = signed_deltas(
        df["pred_ra"], df["pred_dec"], df["hz_ra_q1"], df["hz_dec_q1"]
    )
    df["dx_a_q2"], df["dy_a_q2"] = signed_deltas(
        df["pred_ra"], df["pred_dec"], df["hz_ra_q2"], df["hz_dec_q2"]
    )
    if "pred_ra_b" in df.columns:
        df["dx_b_q1"], df["dy_b_q1"] = signed_deltas(
            df["pred_ra_b"], df["pred_dec_b"], df["hz_ra_q1"], df["hz_dec_q1"]
        )
        df["dx_b_q2"], df["dy_b_q2"] = signed_deltas(
            df["pred_ra_b"], df["pred_dec_b"], df["hz_ra_q2"], df["hz_dec_q2"]
        )
    else:
        df["dx_b_q1"], df["dy_b_q1"] = (np.nan, np.nan)
        df["dx_b_q2"], df["dy_b_q2"] = (np.nan, np.nan)

    per_site = []
    for site, g in df.groupby("site"):
        dx_a_q1 = g["dx_a_q1"].values
        dy_a_q1 = g["dy_a_q1"].values
        sep_a_q1 = g["sep_a_q1"].values
        rms_a_q1 = float(np.sqrt(np.mean(sep_a_q1 ** 2)))
        med_dx_a_q1 = float(np.median(dx_a_q1))
        med_dy_a_q1 = float(np.median(dy_a_q1))

        dx_a_q2 = g["dx_a_q2"].values
        dy_a_q2 = g["dy_a_q2"].values
        sep_a_q2 = g["sep_a_q2"].values
        rms_a_q2 = float(np.sqrt(np.mean(sep_a_q2 ** 2)))
        med_dx_a_q2 = float(np.median(dx_a_q2))
        med_dy_a_q2 = float(np.median(dy_a_q2))

        if g["sep_b_q1"].notna().any():
            dx_b_q1 = g["dx_b_q1"].values
            dy_b_q1 = g["dy_b_q1"].values
            sep_b_q1 = g["sep_b_q1"].values
            rms_b_q1 = float(np.sqrt(np.mean(sep_b_q1 ** 2)))
            med_dx_b_q1 = float(np.median(dx_b_q1))
            med_dy_b_q1 = float(np.median(dy_b_q1))

            dx_b_q2 = g["dx_b_q2"].values
            dy_b_q2 = g["dy_b_q2"].values
            sep_b_q2 = g["sep_b_q2"].values
            rms_b_q2 = float(np.sqrt(np.mean(sep_b_q2 ** 2)))
            med_dx_b_q2 = float(np.median(dx_b_q2))
            med_dy_b_q2 = float(np.median(dy_b_q2))
        else:
            rms_b_q1 = float("nan")
            med_dx_b_q1 = float("nan")
            med_dy_b_q1 = float("nan")
            rms_b_q2 = float("nan")
            med_dx_b_q2 = float("nan")
            med_dy_b_q2 = float("nan")

        per_site.append(
            {
                "site": site,
                "n": int(len(g)),
                "median_dx_a_q1": med_dx_a_q1,
                "median_dy_a_q1": med_dy_a_q1,
                "rms_a_q1": rms_a_q1,
                "median_dx_a_q2": med_dx_a_q2,
                "median_dy_a_q2": med_dy_a_q2,
                "rms_a_q2": rms_a_q2,
                "median_dx_b_q1": med_dx_b_q1,
                "median_dy_b_q1": med_dy_b_q1,
                "rms_b_q1": rms_b_q1,
                "median_dx_b_q2": med_dx_b_q2,
                "median_dy_b_q2": med_dy_b_q2,
                "rms_b_q2": rms_b_q2,
            }
        )

    per_site_df = pd.DataFrame(per_site).sort_values("n", ascending=False)
    per_site_df.to_csv("runs/ceres/compare_ab_per_site.csv", index=False)

    better_b_q1 = int((df["sep_b_q1"] < df["sep_a_q1"]).sum()) if df["sep_b_q1"].notna().any() else 0
    better_b_q2 = int((df["sep_b_q2"] < df["sep_a_q2"]).sum()) if df["sep_b_q2"].notna().any() else 0
    total = int(len(df))
    summary_lines = [
        f"n={total}",
        f"median_sep_a_q1_arcsec={np.median(df['sep_a_q1']):.4f}",
        f"median_sep_b_q1_arcsec={np.median(df['sep_b_q1']) if df['sep_b_q1'].notna().any() else float('nan'):.4f}",
        f"rms_sep_a_q1_arcsec={math.sqrt(np.mean(df['sep_a_q1']**2)):.4f}",
        f"rms_sep_b_q1_arcsec={math.sqrt(np.mean(df['sep_b_q1']**2)) if df['sep_b_q1'].notna().any() else float('nan'):.4f}",
        f"b_better_q1_count={better_b_q1}",
        f"median_sep_a_q2_arcsec={np.median(df['sep_a_q2']):.4f}",
        f"median_sep_b_q2_arcsec={np.median(df['sep_b_q2']) if df['sep_b_q2'].notna().any() else float('nan'):.4f}",
        f"rms_sep_a_q2_arcsec={math.sqrt(np.mean(df['sep_a_q2']**2)):.4f}",
        f"rms_sep_b_q2_arcsec={math.sqrt(np.mean(df['sep_b_q2']**2)) if df['sep_b_q2'].notna().any() else float('nan'):.4f}",
        f"b_better_q2_count={better_b_q2}",
    ]
    with open("runs/ceres/compare_ab_summary.txt", "w") as fh:
        fh.write("\n".join(summary_lines) + "\n")

    plt.figure(figsize=(8, 4))
    plt.hist(df["sep_a_q1"], bins=40, alpha=0.6, label="A (Q1)")
    if df["sep_b_q1"].notna().any():
        plt.hist(df["sep_b_q1"], bins=40, alpha=0.6, label="B (Q1)")
    plt.xlabel("Separation vs Horizons (arcsec)")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig("runs/ceres/compare_ab_sep_hist_q1.png", dpi=180)

    plt.figure(figsize=(8, 4))
    plt.hist(df["sep_a_q2"], bins=40, alpha=0.6, label="A (Q2)")
    if df["sep_b_q2"].notna().any():
        plt.hist(df["sep_b_q2"], bins=40, alpha=0.6, label="B (Q2)")
    plt.xlabel("Separation vs Horizons (arcsec)")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig("runs/ceres/compare_ab_sep_hist_q2.png", dpi=180)

    plt.figure(figsize=(6, 6))
    plt.scatter(per_site_df["median_dx_a_q1"], per_site_df["median_dy_a_q1"], label="A (Q1)", alpha=0.8)
    if per_site_df["median_dx_b_q1"].notna().any():
        plt.scatter(per_site_df["median_dx_b_q1"], per_site_df["median_dy_b_q1"], label="B (Q1)", alpha=0.8)
    plt.axhline(0, color="k", linewidth=0.5)
    plt.axvline(0, color="k", linewidth=0.5)
    plt.xlabel("Median dx (arcsec)")
    plt.ylabel("Median dy (arcsec)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("runs/ceres/compare_ab_site_bias_q1.png", dpi=180)

    plt.figure(figsize=(6, 6))
    plt.scatter(per_site_df["median_dx_a_q2"], per_site_df["median_dy_a_q2"], label="A (Q2)", alpha=0.8)
    if per_site_df["median_dx_b_q2"].notna().any():
        plt.scatter(per_site_df["median_dx_b_q2"], per_site_df["median_dy_b_q2"], label="B (Q2)", alpha=0.8)
    plt.axhline(0, color="k", linewidth=0.5)
    plt.axvline(0, color="k", linewidth=0.5)
    plt.xlabel("Median dx (arcsec)")
    plt.ylabel("Median dy (arcsec)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("runs/ceres/compare_ab_site_bias_q2.png", dpi=180)

    print("Wrote runs/ceres/compare_ab_per_site.csv")
    print("Wrote runs/ceres/compare_ab_summary.txt")
    print("Wrote runs/ceres/compare_ab_sep_hist_q1.png")
    print("Wrote runs/ceres/compare_ab_sep_hist_q2.png")
    print("Wrote runs/ceres/compare_ab_site_bias_q1.png")
    print("Wrote runs/ceres/compare_ab_site_bias_q2.png")


if __name__ == "__main__":
    main()
