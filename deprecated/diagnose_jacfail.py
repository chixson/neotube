#!/usr/bin/env python3
"""
diagnose_jacfail.py

Recompute H_star/J for samples flagged jacobian_fail and print diagnostics.
"""
import argparse
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--debug", required=True, help="path to null_debug.npz")
parser.add_argument("--obs", required=True, help="path to obs.csv (used to form t0/site)")
parser.add_argument("--nshow", type=int, default=20)
args = parser.parse_args()

P = Path(args.debug)
if not P.exists():
    raise SystemExit(f"missing debug file: {P}")
D = np.load(str(P), allow_pickle=True)

rejects = np.asarray(D.get("reject_reason", np.array([], dtype=object)), dtype=object)
theta_star_arr = np.asarray(D.get("theta_star", np.array([], dtype=object)), dtype=object)
theta_lin_arr = np.asarray(D.get("theta_lin", np.array([], dtype=object)), dtype=object)
z_arr = np.asarray(D.get("z", np.array([], dtype=object)), dtype=object)
eps_arr = np.asarray(D.get("eps", np.array([], dtype=object)), dtype=object)
chi2_lin_arr = np.asarray(D.get("chi2_lin", np.array([], dtype=object)), dtype=object)
logabs_arr = np.asarray(D.get("logabsdet", np.array([], dtype=object)), dtype=object)

mask = rejects.astype(str) == "jacobian_fail"
idxs = np.where(mask)[0]
print("Total jacobian_fail count:", len(idxs))
if len(idxs) == 0:
    raise SystemExit(0)

import scripts.hybrid_sampler_neotube as hs

obs_list = hs.load_observations(Path(args.obs), None)
if not obs_list:
    raise SystemExit(f"No observations in {args.obs}")

t0_jd = float(np.median([ob.time.tdb.jd for ob in obs_list]))
t0 = hs.Time(t0_jd, format="jd", scale="tdb")
obs_ref = obs_list[0]

earth_bary, earth_bary_vel = hs._body_posvel_km_single("earth", t0)
sun_bary, sun_bary_vel = hs._body_posvel_km_single("sun", t0)
earth_helio = earth_bary - sun_bary
earth_vel_helio = earth_bary_vel - sun_bary_vel
site_pos, site_vel = hs._site_states(
    [t0],
    [obs_ref.site],
    observer_positions_km=[obs_ref.observer_pos_km],
    observer_velocities_km_s=None,
    allow_unknown_site=True,
)
cached_frame = (earth_helio, earth_vel_helio, site_pos[0], site_vel[0])

nshow = min(args.nshow, len(idxs))
print(f"Showing diagnostics for first {nshow} jacobian_fail samples...")

for j in idxs[:nshow]:
    print("=== sample idx", j, "===")
    theta_star = None
    try:
        theta_star = np.asarray(theta_star_arr[j], dtype=float)
    except Exception:
        theta_star = None
    theta_lin = None
    try:
        theta_lin = np.asarray(theta_lin_arr[j], dtype=float)
    except Exception:
        theta_lin = None
    z = z_arr[j] if j < len(z_arr) else None
    epsv = eps_arr[j] if j < len(eps_arr) else None
    chi2_lin = chi2_lin_arr[j] if j < len(chi2_lin_arr) else None
    logabs = logabs_arr[j] if j < len(logabs_arr) else None

    print("theta_lin:", theta_lin)
    print("theta_star:", theta_star)
    print("z:", z, "eps:", epsv, "chi2_lin:", chi2_lin)
    print("logabsdet recorded:", logabs)

    if theta_star is None or np.any(~np.isfinite(theta_star)):
        print("theta_star is None/NaN; skipping recompute.")
        print()
        continue

    try:
        H_star = hs.compute_H(theta_star, t0, obs_ref, cached_frame=cached_frame)
        U, S, Vt = np.linalg.svd(H_star, full_matrices=True)
        V = Vt.T
        print("H_star singular values:", S)
        if V.shape[1] >= 6:
            N_star = V[:, 4:6]
        else:
            from scipy.linalg import null_space

            N_star = null_space(H_star)

        NtH = N_star.T @ H_star.T
        print("max(|N^T H|):", np.max(np.abs(NtH)))
        print(
            "max(|N^T N - I|):",
            np.max(np.abs(N_star.T @ N_star - np.eye(N_star.shape[1]))),
        )

        smax = float(S[0]) if S.size > 0 else 1.0
        eps_reg = smax * 1e-3
        S_reg = np.maximum(S, eps_reg)
        H_plus = Vt.T[:, :H_star.shape[0]] @ np.diag(1.0 / S_reg) @ U.T
        J = np.hstack([-H_plus, N_star])
        condJ = np.linalg.cond(J)
        sign, lad = np.linalg.slogdet(J)
        print("cond(J): {:.3e}, slogdet: ({}, {:.6e})".format(condJ, sign, lad))
    except Exception as e:
        print("Recompute diagnostic failed:", repr(e))
    print()
