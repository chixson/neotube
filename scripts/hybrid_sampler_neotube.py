#!/usr/bin/env python3
"""
hybrid_sampler_neotube.py

Compare three proposal families for short-arc orbit inference:
  - JPL jitter control (seed from Horizons by MPC id)
  - Attributable + ranging (rho, rhodot)
  - Nullspace orbital-element proposals (analytic Jacobian J = [-H^+, N]) with Newton projection

Outputs separate weighted ensembles and diagnostics for each family.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
from numpy.linalg import slogdet
from scipy import linalg
from scipy.optimize import least_squares
from scipy.stats import multivariate_normal, norm

from astropy.time import Time
from astropy import units as u

from neotube.fit import _initial_state_from_horizons
from neotube.fit_cli import load_observations
from neotube.propagate import (
    _body_posvel_km_single,
    _prepare_obs_cache,
    propagate_state_kepler,
)
from neotube.geometry import unit_to_radec
from neotube.ranging import (
    Attributable,
    build_attributable_vector_fit,
    build_state_from_ranging,
    s_and_sdot,
    _attrib_rho_from_state,
)

try:
    from poliastro.twobody.orbit import Orbit
    from poliastro.bodies import Sun
    from poliastro.twobody import angles as tw_angles

    HAS_POLIASTRO = True
except Exception:
    HAS_POLIASTRO = False

AU_KM = 149597870.7
MU_SUN = 1.32712440018e11  # km^3 / s^2
C_KM_S = 299792.458


def angle_diff_deg(a: float, b: float) -> float:
    """Return wrapped difference a-b in degrees, in [-180, 180)."""
    return (a - b + 180.0) % 360.0 - 180.0


def state_from_elements(theta: np.ndarray, epoch: Time) -> tuple[np.ndarray, np.ndarray]:
    """Return (r_km, v_km_s) from elements using poliastro."""
    if not HAS_POLIASTRO:
        raise RuntimeError("poliastro required for element/state conversion.")
    a_au, e, inc, raan, argp, M0 = theta
    nu = tw_angles.M_to_nu(M0, e)
    orb = Orbit.from_classical(
        Sun,
        a_au * u.AU,
        e * u.one,
        inc * u.rad,
        raan * u.rad,
        argp * u.rad,
        (nu * u.rad),
        epoch=epoch,
    )
    r = orb.r.to(u.km).value
    v = orb.v.to(u.km / u.s).value
    return r, v


def elements_from_state(r_km: np.ndarray, v_km_s: np.ndarray, epoch: Time) -> np.ndarray:
    """Return [a_AU, e, i_rad, raan, argp, M0] from state using poliastro."""
    if not HAS_POLIASTRO:
        raise RuntimeError("poliastro required for element/state conversion.")
    orb = Orbit.from_vectors(Sun, r_km * u.km, v_km_s * u.km / u.s, epoch=epoch)
    M0 = tw_angles.nu_to_M(orb.nu.to(u.rad).value, orb.ecc.value)
    return np.array(
        [
            orb.a.to(u.AU).value,
            orb.ecc.value,
            orb.inc.to(u.rad).value,
            orb.raan.to(u.rad).value,
            orb.argp.to(u.rad).value,
            M0,
        ],
        dtype=float,
    )


def compute_loglik(
    state: np.ndarray, epoch: Time, obs_list, obs_cache
) -> float:
    """Gaussian log-likelihood in RA/Dec (arcsec) for a single state (kepler-only)."""
    total = 0.0
    for i, ob in enumerate(obs_list):
        t_obs = obs_cache.times_tdb[i]
        t_emit = t_obs
        try:
            for _ in range(2):
                st_em = propagate_state_kepler(state, epoch, (t_emit,))[0]
                sun_bary, _ = _body_posvel_km_single("sun", t_emit)
                obj_bary = st_em[:3] + sun_bary
                site_bary = obs_cache.earth_bary_km[i] + obs_cache.site_pos_km[i]
                rho = float(np.linalg.norm(obj_bary - site_bary))
                t_emit = t_obs - (rho / C_KM_S) * u.s
            topovec = obj_bary - site_bary
            topounit = topovec / max(1e-12, np.linalg.norm(topovec))
            ra_pred, dec_pred = unit_to_radec(topounit)
        except Exception:
            return float("-inf")
        dra = angle_diff_deg(ra_pred, ob.ra_deg) * math.cos(math.radians(ob.dec_deg))
        ddec = dec_pred - ob.dec_deg
        dra_arcsec = dra * 3600.0
        ddec_arcsec = ddec * 3600.0
        sigma = float(ob.sigma_arcsec)
        total += -0.5 * ((dra_arcsec / sigma) ** 2 + (ddec_arcsec / sigma) ** 2)
        total += -math.log(2.0 * math.pi * sigma * sigma)
    return float(total)


def compute_attributable(obs_list, epoch: Time) -> tuple[Attributable, np.ndarray]:
    attrib, cov = build_attributable_vector_fit(
        obs_list, epoch, robust=True, return_cov=True
    )
    return attrib, cov


def predict_attributable_from_state(state: np.ndarray, obs_ref, epoch: Time) -> np.ndarray:
    attrib, _, _ = _attrib_rho_from_state(state, obs_ref, epoch)
    return np.array(
        [
            attrib.ra_deg,
            attrib.dec_deg,
            attrib.ra_dot_deg_per_day,
            attrib.dec_dot_deg_per_day,
        ],
        dtype=float,
    )


def compute_H(theta: np.ndarray, epoch: Time, obs_ref, h_steps: np.ndarray | None = None) -> np.ndarray:
    """Centered finite-difference Jacobian dy/dtheta for attributable."""
    if h_steps is None:
        h_steps = np.array([1e-6, 1e-6, 1e-7, 1e-7, 1e-7, 1e-6], dtype=float)
    r0, v0 = state_from_elements(theta, epoch)
    y0 = predict_attributable_from_state(np.hstack([r0, v0]), obs_ref, epoch)
    H = np.zeros((4, 6), dtype=float)
    for k in range(6):
        d = np.zeros(6, dtype=float)
        d[k] = h_steps[k]
        rp, vp = state_from_elements(theta + d, epoch)
        rm, vm = state_from_elements(theta - d, epoch)
        yp = predict_attributable_from_state(np.hstack([rp, vp]), obs_ref, epoch)
        ym = predict_attributable_from_state(np.hstack([rm, vm]), obs_ref, epoch)
        H[:, k] = (yp - ym) / (2.0 * d[k])
    return H


def generate_nullspace_samples(
    seed_theta: np.ndarray,
    n_samples: int,
    attrib: Attributable,
    cov_attrib: np.ndarray,
    epoch: Time,
    obs_ref,
    obs_list,
    obs_cache,
) -> list[dict]:
    """Generate nullspace proposals anchored at seed_theta."""
    samples: list[dict] = []
    H0 = compute_H(seed_theta, epoch, obs_ref)
    U, S, Vt = linalg.svd(H0, full_matrices=False)
    rank = np.sum(S > (1e-12 * S[0] if S.size > 0 else 1e-12))
    S_inv = np.array([1 / s if i < rank else 0.0 for i, s in enumerate(S)])
    H0_plus = Vt.T @ np.diag(S_inv) @ U.T
    N0 = linalg.null_space(H0)
    if N0.shape[1] < 2:
        raise RuntimeError("Nullspace dimension < 2 at seed.")

    r_seed, v_seed = state_from_elements(seed_theta, epoch)
    y_seed = predict_attributable_from_state(np.hstack([r_seed, v_seed]), obs_ref, epoch)

    y_obs = np.array(
        [
            attrib.ra_deg,
            attrib.dec_deg,
            attrib.ra_dot_deg_per_day,
            attrib.dec_dot_deg_per_day,
        ],
        dtype=float,
    )

    cov_eps = np.array(cov_attrib, dtype=float)
    qz_mean = np.zeros(2)
    qz_cov = np.diag([0.1, 0.1])

    for _ in range(n_samples):
        eps = np.random.multivariate_normal(np.zeros(4), cov_eps)
        z = np.random.multivariate_normal(qz_mean, qz_cov)
        theta_lin = seed_theta + H0_plus @ (y_obs - y_seed - eps) + N0 @ z

        def fun(theta_param):
            try:
                r_km, v_km_s = state_from_elements(theta_param, epoch)
                y = predict_attributable_from_state(
                    np.hstack([r_km, v_km_s]), obs_ref, epoch
                )
                return y - (y_obs - eps)
            except Exception:
                return np.ones(4) * 1e6

        try:
            sol = least_squares(
                fun, theta_lin, xtol=1e-8, ftol=1e-8, gtol=1e-8, max_nfev=60
            )
            theta_star = sol.x
            success = sol.success
        except Exception:
            theta_star = theta_lin
            success = False

        try:
            H_star = compute_H(theta_star, epoch, obs_ref)
            U2, S2, Vt2 = linalg.svd(H_star, full_matrices=False)
            rank2 = np.sum(S2 > (1e-12 * S2[0] if S2.size > 0 else 1e-12))
            S2_inv = np.array([1 / s if i < rank2 else 0.0 for i, s in enumerate(S2)])
            H_plus = Vt2.T @ np.diag(S2_inv) @ U2.T
            N = linalg.null_space(H_star)
            if N.shape[1] < 2:
                N = np.zeros((6, 2), dtype=float)
            J = np.hstack([-H_plus, N])
            sign, logabsdet = slogdet(J)
            if sign <= 0:
                logabsdet = -np.inf
        except Exception:
            logabsdet = -np.inf

        log_q_eps = multivariate_normal.logpdf(eps, mean=np.zeros(4), cov=cov_eps)
        log_q_z = multivariate_normal.logpdf(z, mean=qz_mean, cov=qz_cov)
        log_q = log_q_eps + log_q_z - (logabsdet if np.isfinite(logabsdet) else 1e300)

        try:
            r_km, v_km_s = state_from_elements(theta_star, epoch)
            state = np.hstack([r_km, v_km_s])
            loglik = compute_loglik(state, epoch, obs_list, obs_cache)
        except Exception:
            success = False
            loglik = -np.inf
            state = np.full(6, np.nan)

        samples.append(
            {
                "theta": theta_star,
                "state": state,
                "log_q": log_q,
                "loglik": loglik,
                "success": success,
            }
        )
    return samples


def generate_attrib_samples(
    attrib: Attributable,
    cov_attrib: np.ndarray,
    n_samples: int,
    obs_ref,
    epoch: Time,
    obs_list,
    obs_cache,
    rho_min_au: float,
    rho_max_au: float,
    rhodot_max_kms: float,
) -> list[dict]:
    samples: list[dict] = []
    for _ in range(n_samples):
        delta_a = np.random.multivariate_normal(np.zeros(4), cov_attrib)
        attrib_draw = Attributable(
            ra_deg=attrib.ra_deg + delta_a[0],
            dec_deg=attrib.dec_deg + delta_a[1],
            ra_dot_deg_per_day=attrib.ra_dot_deg_per_day + delta_a[2],
            dec_dot_deg_per_day=attrib.dec_dot_deg_per_day + delta_a[3],
        )
        rho_au = np.random.uniform(rho_min_au, rho_max_au)
        rhodot = np.random.uniform(-rhodot_max_kms, rhodot_max_kms)
        state = build_state_from_ranging(
            obs_ref, epoch, attrib_draw, float(rho_au * AU_KM), float(rhodot)
        )
        log_q_da = multivariate_normal.logpdf(delta_a, mean=np.zeros(4), cov=cov_attrib)
        log_q_rho = -math.log(max(1e-12, rho_max_au - rho_min_au))
        log_q_rhodot = -math.log(max(1e-12, 2.0 * rhodot_max_kms))
        log_q = float(log_q_da + log_q_rho + log_q_rhodot)
        loglik = compute_loglik(state, epoch, obs_list, obs_cache)
        samples.append(
            {"theta": None, "state": state, "log_q": log_q, "loglik": loglik}
        )
    return samples


def generate_jpl_jitter(
    seed_state: np.ndarray,
    n_samples: int,
    sigma_r_km: float,
    sigma_v_km_s: float,
    epoch: Time,
    obs_list,
    obs_cache,
) -> list[dict]:
    samples: list[dict] = []
    r_seed = seed_state[:3]
    v_seed = seed_state[3:]
    for _ in range(n_samples):
        dr = np.random.normal(0.0, sigma_r_km, size=3)
        dv = np.random.normal(0.0, sigma_v_km_s, size=3)
        state = np.hstack([r_seed + dr, v_seed + dv])
        log_q = float(np.sum(norm.logpdf(dr, 0.0, sigma_r_km)) + np.sum(norm.logpdf(dv, 0.0, sigma_v_km_s)))
        loglik = compute_loglik(state, epoch, obs_list, obs_cache)
        samples.append({"theta": None, "state": state, "log_q": log_q, "loglik": loglik})
    return samples


def finalize_family(samples: list[dict], out_prefix: Path) -> dict:
    logq = np.array([d.get("log_q", -np.inf) for d in samples], dtype=float)
    loglik = np.array([d.get("loglik", -np.inf) for d in samples], dtype=float)
    states = np.array([d.get("state", np.full(6, np.nan)) for d in samples], dtype=float)
    logw = loglik - logq
    finite = np.isfinite(logw)
    if not np.any(finite):
        return {"n": len(samples), "ess": 0.0}
    maxlw = np.max(logw[finite])
    w = np.exp(logw - maxlw)
    w[~np.isfinite(w)] = 0.0
    w /= np.sum(w) + 1e-300
    ess = 1.0 / np.sum(w ** 2)
    energies = 0.5 * np.sum(states[:, 3:] ** 2, axis=1) - MU_SUN / np.linalg.norm(
        states[:, :3], axis=1
    )
    diag = {
        "n": int(len(samples)),
        "ess": float(ess),
        "elliptic_mass": float(np.sum(w[energies < 0])),
        "hyperbolic_mass": float(np.sum(w[energies > 0])),
    }
    np.savez(
        out_prefix.with_suffix(".npz"),
        state=states,
        logq=logq,
        loglik=loglik,
        logw=logw,
        w=w,
    )
    diag_path = out_prefix.parent / f"{out_prefix.name}_diag.json"
    with open(diag_path, "w") as fh:
        json.dump(diag, fh, indent=2)
    return diag


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--obs", required=True, help="Observation CSV")
    parser.add_argument("--mpc-id", required=True, type=str)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--n-attrib", type=int, default=30000)
    parser.add_argument("--n-null", type=int, default=5000)
    parser.add_argument("--n-jpl", type=int, default=2000)
    parser.add_argument("--rho-min-au", type=float, default=1e-3)
    parser.add_argument("--rho-max-au", type=float, default=5.0)
    parser.add_argument("--rhodot-max", type=float, default=120.0)
    args = parser.parse_args()

    obs_path = Path(args.obs)
    obs_list = load_observations(obs_path, None)
    if not obs_list:
        raise SystemExit(f"No observations in {obs_path}")
    obs_cache = _prepare_obs_cache(obs_list, allow_unknown_site=True)

    t0_jd = float(np.median([ob.time.tdb.jd for ob in obs_list]))
    t0 = Time(t0_jd, format="jd", scale="tdb")
    attrib, cov_attrib = compute_attributable(obs_list, t0)
    obs_ref = obs_list[0]

    if not HAS_POLIASTRO and args.n_null > 0:
        raise SystemExit("poliastro required for nullspace element proposals.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Using obs={obs_path} (n={len(obs_list)}), t0={t0.isot}")
    print(
        "Attributable:",
        attrib.ra_deg,
        attrib.dec_deg,
        attrib.ra_dot_deg_per_day,
        attrib.dec_dot_deg_per_day,
    )

    jpl_state = _initial_state_from_horizons(str(args.mpc_id), t0)

    print("Generating JPL jitter proposals...")
    jpl_samples = generate_jpl_jitter(
        jpl_state,
        args.n_jpl,
        sigma_r_km=50.0,
        sigma_v_km_s=0.05,
        epoch=t0,
        obs_list=obs_list,
        obs_cache=obs_cache,
    )
    diag_jpl = finalize_family(jpl_samples, outdir / "samples_jpl")
    print("JPL diag:", diag_jpl)

    print("Generating attributable proposals...")
    attrib_samples = generate_attrib_samples(
        attrib,
        cov_attrib,
        args.n_attrib,
        obs_ref,
        t0,
        obs_list,
        obs_cache,
        args.rho_min_au,
        args.rho_max_au,
        args.rhodot_max,
    )
    diag_attrib = finalize_family(attrib_samples, outdir / "samples_attrib")
    print("Attrib diag:", diag_attrib)

    if args.n_null > 0:
        print("Generating nullspace proposals...")
        seed_theta = elements_from_state(jpl_state[:3], jpl_state[3:], t0)
        null_samples = generate_nullspace_samples(
            seed_theta,
            args.n_null,
            attrib,
            cov_attrib,
            t0,
            obs_ref,
            obs_list,
            obs_cache,
        )
        diag_null = finalize_family(null_samples, outdir / "samples_null")
        print("Null diag:", diag_null)


if __name__ == "__main__":
    main()
