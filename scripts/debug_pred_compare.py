#!/usr/bin/env python3
"""
Compare RA/Dec predictions from geom-validator (Q1) vs
propagate.predict_radec_from_epoch / predict_radec_with_contract
for a canonical Horizons state.

Usage:
  python scripts/debug_pred_compare.py \
      --obs runs/ceres-ground-test/obs.csv \
      --geom runs/ceres-ground-test/geom_validator_out.csv
"""

from __future__ import annotations
import argparse
import csv
import math
import numpy as np
from astropy.time import Time

# Import repo modules
from pathlib import Path

from neotube.fit import _initial_state_from_horizons
from neotube.fit_cli import load_observations
from neotube.propagate import (
    predict_radec_from_epoch,
    predict_radec_with_contract,
    default_propagation_ladder,
    PropagationConfig,
)


def load_geom_validator(path):
    """Load geom_validator_out.csv (expects time_utc, pred_ra, pred_dec, site)."""
    rows = []
    with open(path) as fh:
        r = csv.DictReader(fh)
        for row in r:
            rows.append(row)
    return rows


def match_geom_for_obs(obs, geom_rows):
    # returns dict mapping iso -> geom row
    gmap = {r["time_utc"]: r for r in geom_rows}
    out = []
    for o in obs:
        iso = o.time.isot
        if iso in gmap:
            out.append(gmap[iso])
        else:
            # try matching by nearest time (if validator truncated ms)
            found = None
            for r in geom_rows:
                if r["time_utc"].startswith(iso[:19]):  # up to seconds
                    found = r
                    break
            out.append(found or {})
    return out


def signed_deltas(ra1_deg, dec1_deg, ra2_deg, dec2_deg):
    ra1 = np.deg2rad(np.asarray(ra1_deg, dtype=float))
    ra2 = np.deg2rad(np.asarray(ra2_deg, dtype=float))
    dec1 = np.deg2rad(np.asarray(dec1_deg, dtype=float))
    dec2 = np.deg2rad(np.asarray(dec2_deg, dtype=float))
    dra = ra1 - ra2
    dra = (dra + math.pi) % (2.0 * math.pi) - math.pi
    dx_arcsec = dra * np.cos(dec1) * 206265.0
    dy_arcsec = (dec1 - dec2) * 206265.0
    return dx_arcsec, dy_arcsec


def summarize_by_site(sites, deltas):
    # sites: list of site codes per obs
    # deltas: Nx2 array of dra_arcsec, ddec_arcsec (pred - geom)
    res = {}
    for s in set(sites):
        idx = [i for i, ss in enumerate(sites) if ss == s or (ss is None and s == "UNK")]
        if not idx:
            continue
        arr = deltas[idx, :]
        med = np.median(np.hypot(arr[:, 0], arr[:, 1]))
        med_ra = np.median(arr[:, 0])
        med_dec = np.median(arr[:, 1])
        res[s] = dict(n=len(idx), med_total_arcsec=float(med), med_ra=float(med_ra), med_dec=float(med_dec))
    return res


def run_compare(obs, geom_rows, state, epoch, label, ra_pred, dec_pred):
    # geom predictions from file
    geom_matched = match_geom_for_obs(obs, geom_rows)
    geom_ra = []
    geom_dec = []
    sites = []
    for o, g in zip(obs, geom_matched):
        sites.append((o.site or "UNK").strip().upper())
        if g and g.get("pred_ra"):
            geom_ra.append(float(g["pred_ra"]))
            geom_dec.append(float(g["pred_dec"]))
        else:
            geom_ra.append(o.ra_deg)  # fallback
            geom_dec.append(o.dec_deg)
    ra_pred = np.atleast_2d(ra_pred)
    dec_pred = np.atleast_2d(dec_pred)
    # Use the *median* of replicas if multiple rows (for sampling) -> here expecting single-state -> shape (1,N)
    ra_m = np.median(ra_pred, axis=0)
    dec_m = np.median(dec_pred, axis=0)
    dra, ddec = signed_deltas(ra_m, dec_m, geom_ra, geom_dec)  # pred - geom
    deltas = np.vstack([dra, ddec]).T
    summary = summarize_by_site(sites, deltas)
    print("\n=== %s ===" % label)
    for s, v in sorted(summary.items()):
        print(f"{s:4s} n={v['n']:2d} med_total={v['med_total_arcsec']:8.3f} med_ra={v['med_ra']:8.3f} med_dec={v['med_dec']:8.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--obs", required=True)
    p.add_argument("--geom", required=True)
    p.add_argument("--target", default="1")  # Ceres as default
    args = p.parse_args()

    obs = load_observations(Path(args.obs), None)
    geom_rows = load_geom_validator(args.geom)

    # Choose an epoch: use obs middle-time (same as fit_orbit)
    epoch = obs[len(obs) // 2].time

    # Get a canonical state from Horizons (@sun, like _initial_state_from_horizons does)
    state = _initial_state_from_horizons(args.target, epoch)  # shape (6,)

    # 1) Predict with predict_radec_from_epoch (varying options)
    combos = []
    combos.append(("kepler_fullphys_False_lt2_refraction_False", dict(use_kepler=True, full_physics=False, light_time_iters=2, include_refraction=False)))
    combos.append(("kepler_fullphys_True_lt2_refraction_False", dict(use_kepler=True, full_physics=True, light_time_iters=2, include_refraction=False)))
    combos.append(("nbody_fullphys_True_lt2_refraction_False", dict(use_kepler=False, full_physics=True, light_time_iters=2, include_refraction=False)))
    combos.append(("nbody_fullphys_True_lt0_refraction_False", dict(use_kepler=False, full_physics=True, light_time_iters=0, include_refraction=False)))
    combos.append(("contract_q1", dict(contract_q1=True)))

    for label, opts in combos:
        if label == "contract_q1":
            # use predict_radec_with_contract (Q1-style ladder)
            ladder = default_propagation_ladder(max_step=3600.0)
            ra_pred, dec_pred, _, _ = predict_radec_with_contract(
                np.atleast_2d(state),
                epoch,
                obs,
                epsilon_ast_arcsec=0.25,
                allow_unknown_site=True,
                configs=ladder,
            )
            run_compare(obs, geom_rows, state, epoch, label, ra_pred, dec_pred)
        else:
            # call predict_radec_from_epoch; note its signature has several args
            use_kepler = opts["use_kepler"]
            full_phys = opts["full_physics"]
            lt = opts["light_time_iters"]
            ra_pred, dec_pred = predict_radec_from_epoch(
                np.atleast_1d(state),
                epoch,
                obs,
                perturbers=("earth", "mars", "jupiter"),
                max_step=3600.0,
                use_kepler=use_kepler,
                allow_unknown_site=True,
                light_time_iters=lt,
                full_physics=full_phys,
                include_refraction=opts.get("include_refraction", False),
            )
            run_compare(obs, geom_rows, state, epoch, label, ra_pred, dec_pred)


if __name__ == "__main__":
    main()
