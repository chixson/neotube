#!/usr/bin/env python3
"""
wise_astrometry_diagnose.py

Usage:
  PYTHONPATH=neotube/src python wise_astrometry_diagnose.py --diag runs/ceres/fit_diag.csv --site C51

This script:
 - groups obs into visits (same site, times within dt seconds),
 - computes per-visit mean residual vector and per-visit scatter,
 - estimates per-visit excess variance relative to reported sigma,
 - prints summary statistics and top offending visits/sites.
"""
from __future__ import annotations
import argparse, csv, math
from collections import defaultdict, namedtuple
from datetime import datetime, timedelta
from astropy.time import Time
import numpy as np
import json

Record = namedtuple("Record", ["time", "site", "res_ra", "res_dec", "sigma"])

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--diag", required=True, help="fit_diagnostics CSV (residuals.csv/fit_diag.csv)")
    p.add_argument("--dt", type=float, default=120.0, help="visit grouping window (seconds)")
    p.add_argument("--min-count", type=int, default=2, help="min obs per visit to compute stats")
    p.add_argument("--topn", type=int, default=20, help="number of worst visits to print")
    return p.parse_args()

def read_diag(path):
    rows = []
    with open(path) as fh:
        rdr = csv.DictReader(fh)
        for r in rdr:
            try:
                t = Time(r.get("time_utc") or r.get("time") or r.get("time_utc_obs") , scale="utc")
            except Exception:
                # try alternatives
                t = Time(r.get("time_utc") or r.get("time") , scale="utc")
            site = (r.get("site") or r.get("observatory") or "").strip().upper()
            # prefer normalized residuals if present, else compute from res_ columns
            if r.get("res_ra_arcsec") is not None and r.get("res_dec_arcsec") is not None:
                res_ra = float(r["res_ra_arcsec"])
                res_dec = float(r["res_dec_arcsec"])
            elif r.get("normed_ra") is not None and r.get("normed_dec") is not None:
                # normed RA/Dec are already normalized by sigma; recover by multiplying sigma
                res_ra = float(r["normed_ra"]) * float(r.get("sigma_arcsec", 1.0))
                res_dec = float(r["normed_dec"]) * float(r.get("sigma_arcsec", 1.0))
            else:
                # fallback
                continue
            sigma = float(r.get("sigma_arcsec", 0.5))
            rows.append(Record(time=t, site=site, res_ra=res_ra, res_dec=res_dec, sigma=sigma))
    return rows

def group_visits(rows, dt_seconds=120.0):
    # group by site and contiguous times within dt
    rows_sorted = sorted(rows, key=lambda r: (r.site, r.time.jd))
    visits = []
    cur_site = None
    cur_group = []
    prev_t = None
    for r in rows_sorted:
        if cur_site is None or r.site != cur_site:
            if cur_group:
                visits.append((cur_site, cur_group))
            cur_site = r.site
            cur_group = [r]
            prev_t = r.time
            continue
        # same site
        delta = abs((r.time - prev_t).to_value('sec'))
        if delta <= dt_seconds:
            cur_group.append(r)
        else:
            visits.append((cur_site, cur_group))
            cur_group = [r]
        prev_t = r.time
    if cur_group:
        visits.append((cur_site, cur_group))
    return visits

def stats_for_group(group):
    # compute per-observation residual vectors and norms
    ras = np.array([r.res_ra for r in group], dtype=float)
    decs = np.array([r.res_dec for r in group], dtype=float)
    sigs = np.array([r.sigma for r in group], dtype=float)
    n = len(group)
    mean_ra = ras.mean()
    mean_dec = decs.mean()
    # sample std (observed scatter)
    s_ra = ras.std(ddof=1) if n>1 else 0.0
    s_dec = decs.std(ddof=1) if n>1 else 0.0
    # combined squared-residuals
    chi = np.sum((ras/sigs)**2 + (decs/sigs)**2)
    # effective multiplicative kappa (method-of-moments)
    # expected chi ~ 2*n if sigs correct; so kappa^2 ~ chi/(2n)
    kappa = math.sqrt(chi / (2.0*n)) if n>0 else None
    # additive extra variance estimate (method-of-moments)
    var_ra = max(0.0, ras.var(ddof=1) - (sigs**2).mean()) if n>1 else 0.0
    var_dec = max(0.0, decs.var(ddof=1) - (sigs**2).mean()) if n>1 else 0.0
    return {
        "n": n, "mean_ra": mean_ra, "mean_dec": mean_dec,
        "std_ra": s_ra, "std_dec": s_dec,
        "chi": chi, "kappa": kappa,
        "addvar_ra": var_ra, "addvar_dec": var_dec
    }

def main():
    args = parse_args()
    rows = read_diag(args.diag)
    print("Read", len(rows), "diag rows from", args.diag)
    visits = group_visits(rows, dt_seconds=args.dt)
    print("Grouped into", len(visits), "visits (dt=", args.dt, "s)")
    visit_stats = []
    for site, group in visits:
        if len(group) < args.min_count:
            continue
        s = stats_for_group(group)
        visit_stats.append({"site": site, "start": group[0].time.isot, "n": s["n"], **s})
    # sort by absolute mean offset
    visit_stats_sorted = sorted(visit_stats, key=lambda v: math.hypot(v["mean_ra"], v["mean_dec"]), reverse=True)
    print("\nTop %d visits by mean absolute bias:" % args.topn)
    for v in visit_stats_sorted[:args.topn]:
        mean_abs = math.hypot(v["mean_ra"], v["mean_dec"])
        print(f"site={v['site']} start={v['start']} n={v['n']} mean_abs={mean_abs:.3f}\" mean_ra={v['mean_ra']:.3f}\" mean_dec={v['mean_dec']:.3f}\" kappa={v['kappa']:.3f} addvar_ra={v['addvar_ra']:.4f} addvar_dec={v['addvar_dec']:.4f}")

    # per-site summary
    per_site = defaultdict(list)
    for v in visit_stats:
        per_site[v["site"]].append(v)
    print("\nPer-site summary (median mean_abs, median kappa):")
    for site, lst in per_site.items():
        med_mean_abs = np.median([math.hypot(x["mean_ra"], x["mean_dec"]) for x in lst])
        med_kappa = np.median([x["kappa"] for x in lst if x["kappa"] is not None])
        print(f" {site}  visits={len(lst)}  med_mean_abs={med_mean_abs:.3f}\"  med_kappa={med_kappa:.3f}")

    # global diagnostics
    all_kappa = [v["kappa"] for v in visit_stats if v["kappa"] is not None]
    print("\nGlobal med kappa across visits (approx scale multiplier):", np.median(all_kappa) if all_kappa else None)
    print("Fraction of visits with mean_abs > 1.0\":", sum(1 for v in visit_stats if math.hypot(v["mean_ra"], v["mean_dec"])>1.0)/len(visit_stats) if visit_stats else 0)

if __name__ == "__main__":
    main()
