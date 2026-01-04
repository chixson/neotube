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
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
from numpy.linalg import slogdet
from scipy import linalg
from scipy.optimize import least_squares
from scipy.special import logsumexp
from scipy.stats import multivariate_normal, norm

from astropy.time import Time
from astropy import units as u

from neotube.fit import _initial_state_from_horizons
from neotube.fit_cli import load_observations
from neotube.propagate import (
    _body_posvel_km_single,
    _prepare_obs_cache,
    _site_states,
    propagate_state_kepler,
)
from neotube.geometry import unit_to_radec
from neotube.ranging import (
    Attributable,
    DAY_S,
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
                # Ensure object and site are in the same barycentric frame.
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


def predict_attributable_from_state_cached(
    state: np.ndarray,
    earth_helio: np.ndarray,
    earth_vel_helio: np.ndarray,
    site_offset: np.ndarray,
    site_vel: np.ndarray,
) -> np.ndarray:
    r_helio = state[:3].astype(float)
    v_helio = state[3:].astype(float)
    r_geo = r_helio - earth_helio
    v_geo = v_helio - earth_vel_helio - site_vel
    r_topo = r_geo - site_offset
    rho = float(np.linalg.norm(r_topo))
    if rho <= 0.0:
        raise RuntimeError("Non-positive rho in attributable conversion.")
    s = r_topo / rho
    rhodot = float(np.dot(v_geo, s))
    sdot = (v_geo - rhodot * s) / max(rho, 1e-12)

    x, y, z = s
    xd, yd, zd = sdot
    rxy2 = max(x * x + y * y, 1e-12)
    ra = math.atan2(y, x)
    dec = math.asin(np.clip(z, -1.0, 1.0))
    ra_dot = (x * yd - y * xd) / rxy2
    cosdec = max(math.sqrt(rxy2), 1e-12)
    dec_dot = (zd - z * (x * xd + y * yd) / rxy2) / cosdec
    return np.array(
        [
            float(math.degrees(ra) % 360.0),
            float(math.degrees(dec)),
            float(math.degrees(ra_dot) * DAY_S),
            float(math.degrees(dec_dot) * DAY_S),
        ],
        dtype=float,
    )


def compute_H(
    theta: np.ndarray,
    epoch: Time,
    obs_ref,
    h_steps: np.ndarray | None = None,
    cached_frame: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Centered finite-difference Jacobian dy/dtheta for attributable."""
    if h_steps is None:
        h_steps = np.array([1e-6, 1e-6, 1e-7, 1e-7, 1e-7, 1e-6], dtype=float)
    r0, v0 = state_from_elements(theta, epoch)
    if cached_frame is None:
        y0 = predict_attributable_from_state(np.hstack([r0, v0]), obs_ref, epoch)
    else:
        earth_helio, earth_vel_helio, site_offset, site_vel = cached_frame
        y0 = predict_attributable_from_state_cached(
            np.hstack([r0, v0]), earth_helio, earth_vel_helio, site_offset, site_vel
        )
    H = np.zeros((4, 6), dtype=float)
    for k in range(6):
        d = np.zeros(6, dtype=float)
        d[k] = h_steps[k]
        rp, vp = state_from_elements(theta + d, epoch)
        rm, vm = state_from_elements(theta - d, epoch)
        if cached_frame is None:
            yp = predict_attributable_from_state(np.hstack([rp, vp]), obs_ref, epoch)
            ym = predict_attributable_from_state(np.hstack([rm, vm]), obs_ref, epoch)
        else:
            earth_helio, earth_vel_helio, site_offset, site_vel = cached_frame
            yp = predict_attributable_from_state_cached(
                np.hstack([rp, vp]), earth_helio, earth_vel_helio, site_offset, site_vel
            )
            ym = predict_attributable_from_state_cached(
                np.hstack([rm, vm]), earth_helio, earth_vel_helio, site_offset, site_vel
            )
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
    cached_frame: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    newton_mode: str,
    debug_path: Path | None = None,
    debug_enabled: bool = False,
) -> tuple[list[dict], dict, dict | None]:
    """Generate nullspace proposals anchored at seed_theta."""
    samples: list[dict] = []
    stats = {
        "attempted": 0,
        "project_fail": 0,
        "sanity_reject": 0,
        "jacobian_fail": 0,
        "success": 0,
    }
    H0 = compute_H(seed_theta, epoch, obs_ref, cached_frame=cached_frame)
    U, S, Vt = linalg.svd(H0, full_matrices=False)
    rank = np.sum(S > (1e-12 * S[0] if S.size > 0 else 1e-12))
    s_floor = (S[0] if S.size > 0 else 1.0) * 1e-12
    S_reg = np.where(S < s_floor, s_floor, S)
    S_inv = np.array([1 / s if i < rank else 0.0 for i, s in enumerate(S_reg)])
    H0_plus = Vt.T @ np.diag(S_inv) @ U.T
    N0 = linalg.null_space(H0)
    if N0.shape[1] < 2:
        raise RuntimeError("Nullspace dimension < 2 at seed.")

    r_seed, v_seed = state_from_elements(seed_theta, epoch)
    y_seed = predict_attributable_from_state_cached(
        np.hstack([r_seed, v_seed]), *cached_frame
    )

    y_obs_vec = np.array(
        [
            attrib.ra_deg,
            attrib.dec_deg,
            attrib.ra_dot_deg_per_day,
            attrib.dec_dot_deg_per_day,
        ],
        dtype=float,
    )
    assert y_obs_vec.shape == (4,), "y_obs_vec must be 4-vector (alpha,delta,alpha_dot,delta_dot)"

    cov_eps = np.array(cov_attrib, dtype=float)
    qz_mean = np.zeros(2)
    qz_tight_cov = np.diag([0.005, 0.005])
    qz_wide_cov = np.diag([0.05, 0.05])
    qz_tight_prob = 0.8
    lambda_reg = 1e-4
    chi2_thresh = 25.0
    try:
        cov_inv = linalg.inv(cov_eps)
    except Exception:
        cov_inv = None

    debug = None
    if debug_path is not None or debug_enabled:
        debug = {
            "theta_lin": [],
            "z": [],
            "eps": [],
            "chi2_lin": [],
            "used_newton": [],
            "theta_star": [],
            "chi2_star": [],
            "logabsdet": [],
            "H_svals_seed": [],
            "H_svals_lin": [],
            "H_svals_star": [],
            "reject_reason": [],
            "state": [],
            "loglik": [],
            "logq": [],
        }

    def _debug_append(
        theta_lin_val,
        z_val,
        eps_val,
        chi2_lin_val,
        used_newton_val,
        theta_star_val,
        chi2_star_val,
        logabsdet_val,
        H_svals_lin_val,
        H_svals_star_val,
        reject_reason_val,
        state_val,
        loglik_val,
        logq_val,
    ) -> None:
        if debug is None:
            return
        debug["theta_lin"].append(theta_lin_val.copy())
        debug["z"].append(z_val.copy())
        debug["eps"].append(eps_val.copy())
        debug["chi2_lin"].append(chi2_lin_val)
        debug["used_newton"].append(used_newton_val)
        debug["theta_star"].append(theta_star_val.copy())
        debug["chi2_star"].append(chi2_star_val)
        debug["logabsdet"].append(logabsdet_val)
        debug["H_svals_seed"].append(S.copy())
        debug["H_svals_lin"].append(H_svals_lin_val.copy())
        debug["H_svals_star"].append(H_svals_star_val.copy())
        debug["reject_reason"].append(reject_reason_val)
        debug["state"].append(state_val.copy())
        debug["loglik"].append(loglik_val)
        debug["logq"].append(logq_val)

    max_attempts = max(10 * n_samples, n_samples + 1)
    attempts = 0
    while len(samples) < n_samples and attempts < max_attempts:
        attempts += 1
        stats["attempted"] += 1
        eps = np.random.multivariate_normal(np.zeros(4), cov_eps)
        if np.random.rand() < qz_tight_prob:
            z = np.random.multivariate_normal(qz_mean, qz_tight_cov)
        else:
            z = np.random.multivariate_normal(qz_mean, qz_wide_cov)
        theta_lin = seed_theta + H0_plus @ (y_obs_vec - y_seed - eps) + N0 @ z
        # enforce bounds on the initial guess to avoid invalid elements
        theta_lower = np.array([1e-6, 0.0, 0.0, -2.0 * math.pi, -2.0 * math.pi, -2.0 * math.pi])
        theta_upper = np.array([1e6, 0.9999, math.pi, 2.0 * math.pi, 2.0 * math.pi, 2.0 * math.pi])
        theta_lin = np.minimum(np.maximum(theta_lin, theta_lower), theta_upper)

        reject_reason = ""
        chi2_lin = float("nan")
        chi2_star = float("nan")
        logabsdet = float("nan")
        H_svals_lin = np.full(1, np.nan)
        H_svals_star = np.full(1, np.nan)
        theta_star = theta_lin
        used_newton = False

        try:
            H_lin = compute_H(theta_lin, epoch, obs_ref, cached_frame=cached_frame)
            _, S_lin, _ = linalg.svd(H_lin, full_matrices=False)
            H_svals_lin = S_lin.copy()
            r_km_lin, v_km_lin = state_from_elements(theta_lin, epoch)
            y_lin = predict_attributable_from_state_cached(
                np.hstack([r_km_lin, v_km_lin]), *cached_frame
            )
            r_vec = y_lin - (y_obs_vec - eps)
            if cov_inv is not None:
                chi2_lin = float(r_vec.T @ cov_inv @ r_vec)
            else:
                chi2_lin = float(np.dot(r_vec, r_vec))
        except Exception:
            reject_reason = "chi2_lin_fail"

        def fun(theta_param):
            try:
                a_val = float(theta_param[0])
                e_val = float(theta_param[1])
                if (a_val <= 0.0) or (e_val < 0.0) or (e_val >= 1.0):
                    return np.ones(4) * 1e6
                r_km, v_km_s = state_from_elements(theta_param, epoch)
                if not (np.isfinite(r_km).all() and np.isfinite(v_km_s).all()):
                    return np.ones(4) * 1e6
                y = predict_attributable_from_state_cached(
                    np.hstack([r_km, v_km_s]), *cached_frame
                )
                return y - (y_obs_vec - eps)
            except Exception:
                return np.ones(4) * 1e6

        use_newton = newton_mode == "on"
        if newton_mode == "auto" and np.isfinite(chi2_lin):
            use_newton = chi2_lin > chi2_thresh

        if use_newton:
            N_mat = N0

            def fun_aug(theta_param):
                res = fun(theta_param)
                null_res = math.sqrt(lambda_reg) * (N_mat.T.dot(theta_param - theta_lin) - z)
                return np.concatenate([res, null_res])

            try:
                sol = least_squares(
                    fun_aug,
                    theta_lin,
                    bounds=(theta_lower, theta_upper),
                    xtol=1e-8,
                    ftol=1e-8,
                    gtol=1e-8,
                    max_nfev=30,
                )
                theta_star = sol.x
                success = sol.success
            except Exception:
                theta_star = theta_lin
                success = False
                stats["project_fail"] += 1
                reject_reason = "project_exception"
                _debug_append(
                    theta_lin,
                    z,
                    eps,
                    chi2_lin,
                    True,
                    np.full_like(theta_lin, np.nan),
                    chi2_star,
                    float("nan"),
                    H_svals_lin,
                    H_svals_star,
                    reject_reason,
                    np.full(6, np.nan),
                    float("-inf"),
                    float("-inf"),
                )
                continue
            if not success:
                stats["project_fail"] += 1
                reject_reason = "project_fail"
                _debug_append(
                    theta_lin,
                    z,
                    eps,
                    chi2_lin,
                    True,
                    np.full_like(theta_lin, np.nan),
                    chi2_star,
                    float("nan"),
                    H_svals_lin,
                    H_svals_star,
                    reject_reason,
                    np.full(6, np.nan),
                    float("-inf"),
                    float("-inf"),
                )
                continue
            used_newton = True
        else:
            theta_star = theta_lin

        if not used_newton and newton_mode == "auto" and np.isfinite(chi2_lin):
            if chi2_lin > chi2_thresh:
                reject_reason = "chi2_too_large_no_newton"
                stats["sanity_reject"] += 1
                _debug_append(
                    theta_lin,
                    z,
                    eps,
                    chi2_lin,
                    False,
                    np.full_like(theta_lin, np.nan),
                    float("nan"),
                    float("nan"),
                    H_svals_lin,
                    H_svals_star,
                    reject_reason,
                    np.full(6, np.nan),
                    float("-inf"),
                    float("-inf"),
                )
                continue

        a_au, e_val, inc_rad = float(theta_star[0]), float(theta_star[1]), float(theta_star[2])
        if not np.isfinite(a_au) or not np.isfinite(e_val) or not np.isfinite(inc_rad):
            stats["sanity_reject"] += 1
            reject_reason = "nan_elements"
            _debug_append(
                theta_lin,
                z,
                eps,
                chi2_lin,
                used_newton,
                np.full_like(theta_star, np.nan),
                chi2_star,
                logabsdet,
                H_svals_lin,
                H_svals_star,
                reject_reason,
                np.full(6, np.nan),
                float("-inf"),
                float("-inf"),
            )
            continue
        if a_au <= 0.0 or e_val >= 0.99999 or e_val < 0.0:
            stats["sanity_reject"] += 1
            reject_reason = "ecc_or_a"
            _debug_append(
                theta_lin,
                z,
                eps,
                chi2_lin,
                used_newton,
                np.full_like(theta_star, np.nan),
                chi2_star,
                logabsdet,
                H_svals_lin,
                H_svals_star,
                reject_reason,
                np.full(6, np.nan),
                float("-inf"),
                float("-inf"),
            )
            continue
        q_au = a_au * (1.0 - e_val)
        if q_au < 0.005:
            stats["sanity_reject"] += 1
            reject_reason = "q_au"
            _debug_append(
                theta_lin,
                z,
                eps,
                chi2_lin,
                used_newton,
                np.full_like(theta_star, np.nan),
                chi2_star,
                logabsdet,
                H_svals_lin,
                H_svals_star,
                reject_reason,
                np.full(6, np.nan),
                float("-inf"),
                float("-inf"),
            )
            continue
        if math.degrees(inc_rad) > 90.0:
            stats["sanity_reject"] += 1
            reject_reason = "inclination"
            _debug_append(
                theta_lin,
                z,
                eps,
                chi2_lin,
                used_newton,
                np.full_like(theta_star, np.nan),
                chi2_star,
                logabsdet,
                H_svals_lin,
                H_svals_star,
                reject_reason,
                np.full(6, np.nan),
                float("-inf"),
                float("-inf"),
            )
            continue

        try:
            H_star = compute_H(theta_star, epoch, obs_ref, cached_frame=cached_frame)
            U2, S2, Vt2 = linalg.svd(H_star, full_matrices=False)
            rank2 = np.sum(S2 > (1e-12 * S2[0] if S2.size > 0 else 1e-12))
            s2_floor = (S2[0] if S2.size > 0 else 1.0) * 1e-12
            S2_reg = np.where(S2 < s2_floor, s2_floor, S2)
            S2_inv = np.array([1 / s if i < rank2 else 0.0 for i, s in enumerate(S2_reg)])
            H_plus = Vt2.T @ np.diag(S2_inv) @ U2.T
            N = linalg.null_space(H_star)
            if N.shape[1] < 2:
                N = np.zeros((6, 2), dtype=float)
            J = np.hstack([-H_plus, N])
            sign, logabsdet = slogdet(J)
            if sign <= 0 or not np.isfinite(logabsdet):
                stats["jacobian_fail"] += 1
                reject_reason = "jacobian_fail"
                _debug_append(
                    theta_lin,
                    z,
                    eps,
                    chi2_lin,
                    used_newton,
                    theta_star,
                    chi2_star,
                    float("nan"),
                    H_svals_lin,
                    H_svals_star,
                    reject_reason,
                    np.full(6, np.nan),
                    float("-inf"),
                    float("-inf"),
                )
                continue
            H_svals_star = S2.copy()
        except Exception:
            stats["jacobian_fail"] += 1
            reject_reason = "jacobian_exception"
            _debug_append(
                theta_lin,
                z,
                eps,
                chi2_lin,
                used_newton,
                theta_star,
                chi2_star,
                float("nan"),
                H_svals_lin,
                H_svals_star,
                reject_reason,
                np.full(6, np.nan),
                float("-inf"),
                float("-inf"),
            )
            continue

        log_q_eps = multivariate_normal.logpdf(eps, mean=np.zeros(4), cov=cov_eps)
        log_q_z = logsumexp(
            [
                math.log(qz_tight_prob)
                + multivariate_normal.logpdf(z, mean=qz_mean, cov=qz_tight_cov),
                math.log(1.0 - qz_tight_prob)
                + multivariate_normal.logpdf(z, mean=qz_mean, cov=qz_wide_cov),
            ]
        )
        log_q = log_q_eps + log_q_z - (logabsdet if np.isfinite(logabsdet) else 1e300)

        try:
            r_km, v_km_s = state_from_elements(theta_star, epoch)
            state = np.hstack([r_km, v_km_s])
            loglik = compute_loglik(state, epoch, obs_list, obs_cache)
        except Exception:
            loglik = -np.inf
            state = np.full(6, np.nan)

        samples.append(
            {
                "theta": theta_star,
                "state": state,
                "log_q": log_q,
                "loglik": loglik,
                "success": True,
            }
        )
        stats["success"] += 1
        _debug_append(
            theta_lin,
            z,
            eps,
            chi2_lin,
            used_newton,
            theta_star if used_newton else np.full_like(theta_star, np.nan),
            chi2_star,
            float(logabsdet),
            H_svals_lin,
            H_svals_star,
            reject_reason,
            state,
            float(loglik),
            float(log_q),
        )
    if debug is not None and debug_path is not None:
        np.savez(
            debug_path,
            theta_lin=np.array(debug["theta_lin"]),
            z=np.array(debug["z"]),
            eps=np.array(debug["eps"]),
            chi2_lin=np.array(debug["chi2_lin"]),
            used_newton=np.array(debug["used_newton"]),
            theta_star=np.array(debug["theta_star"]),
            chi2_star=np.array(debug["chi2_star"]),
            logabsdet=np.array(debug["logabsdet"]),
            H_svals_seed=np.array(debug["H_svals_seed"], dtype=object),
            H_svals_lin=np.array(debug["H_svals_lin"], dtype=object),
            H_svals_star=np.array(debug["H_svals_star"], dtype=object),
            reject_reason=np.array(debug["reject_reason"], dtype=object),
            state=np.array(debug["state"]),
            loglik=np.array(debug["loglik"]),
            logq=np.array(debug["logq"]),
        )
    return samples, stats, debug


def _merge_null_debug(debug_list: list[dict] | None) -> dict | None:
    if not debug_list:
        return None
    merged: dict[str, list] = {}
    for dbg in debug_list:
        if dbg is None:
            continue
        for key, val in dbg.items():
            merged.setdefault(key, []).extend(list(val))
    if not merged:
        return None
    return merged


def _save_null_debug(debug: dict, path: Path) -> None:
    np.savez(
        path,
        theta_lin=np.array(debug["theta_lin"]),
        z=np.array(debug["z"]),
        eps=np.array(debug["eps"]),
        chi2_lin=np.array(debug["chi2_lin"]),
        used_newton=np.array(debug["used_newton"]),
        theta_star=np.array(debug["theta_star"]),
        chi2_star=np.array(debug["chi2_star"]),
        logabsdet=np.array(debug["logabsdet"]),
        H_svals_seed=np.array(debug["H_svals_seed"], dtype=object),
        H_svals_lin=np.array(debug["H_svals_lin"], dtype=object),
        H_svals_star=np.array(debug["H_svals_star"], dtype=object),
        reject_reason=np.array(debug["reject_reason"], dtype=object),
        state=np.array(debug["state"]),
        loglik=np.array(debug["loglik"]),
        logq=np.array(debug["logq"]),
    )


def _nullspace_worker(payload: tuple) -> tuple[list[dict], dict, dict | None]:
    (
        seed,
        seed_theta,
        n_samples,
        attrib,
        cov_attrib,
        epoch,
        obs_ref,
        obs_list,
        obs_cache,
        cached_frame,
        newton_mode,
        enable_debug,
    ) = payload
    if seed is not None:
        np.random.seed(int(seed))
    return generate_nullspace_samples(
        seed_theta,
        n_samples,
        attrib,
        cov_attrib,
        epoch,
        obs_ref,
        obs_list,
        obs_cache,
        cached_frame,
        newton_mode,
        None,
        enable_debug,
    )


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
    stats = {"attempted": 0, "prop_fail": 0, "sanity_reject": 0}
    for _ in range(n_samples):
        stats["attempted"] += 1
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
        r_norm = float(np.linalg.norm(state[:3]))
        v_norm = float(np.linalg.norm(state[3:]))
        if r_norm <= 0.0 or not np.isfinite(r_norm) or not np.isfinite(v_norm):
            stats["sanity_reject"] += 1
            loglik = -np.inf
        else:
            energy = 0.5 * v_norm * v_norm - MU_SUN / r_norm
            if energy >= 0.0:
                stats["sanity_reject"] += 1
                loglik = -np.inf
            else:
                loglik = compute_loglik(state, epoch, obs_list, obs_cache)
                if not np.isfinite(loglik):
                    stats["prop_fail"] += 1
        log_q_da = multivariate_normal.logpdf(delta_a, mean=np.zeros(4), cov=cov_attrib)
        log_q_rho = -math.log(max(1e-12, rho_max_au - rho_min_au))
        log_q_rhodot = -math.log(max(1e-12, 2.0 * rhodot_max_kms))
        log_q = float(log_q_da + log_q_rho + log_q_rhodot)
        samples.append(
            {"theta": None, "state": state, "log_q": log_q, "loglik": loglik}
        )
    return samples, stats


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
    stats = {"attempted": 0, "prop_fail": 0}
    r_seed = seed_state[:3]
    v_seed = seed_state[3:]
    for _ in range(n_samples):
        stats["attempted"] += 1
        dr = np.random.normal(0.0, sigma_r_km, size=3)
        dv = np.random.normal(0.0, sigma_v_km_s, size=3)
        state = np.hstack([r_seed + dr, v_seed + dv])
        log_q = float(np.sum(norm.logpdf(dr, 0.0, sigma_r_km)) + np.sum(norm.logpdf(dv, 0.0, sigma_v_km_s)))
        loglik = compute_loglik(state, epoch, obs_list, obs_cache)
        if not np.isfinite(loglik):
            stats["prop_fail"] += 1
        samples.append({"theta": None, "state": state, "log_q": log_q, "loglik": loglik})
    return samples, stats


def finalize_family(samples: list[dict], out_prefix: Path, extra_diag: dict | None = None) -> dict:
    logq = np.array([d.get("log_q", -np.inf) for d in samples], dtype=float)
    loglik = np.array([d.get("loglik", -np.inf) for d in samples], dtype=float)
    states = np.array([d.get("state", np.full(6, np.nan)) for d in samples], dtype=float)
    logw = loglik - logq
    finite = np.isfinite(logw)
    if not np.any(finite):
        diag = {"n": int(len(samples)), "ess": 0.0}
        if extra_diag:
            diag.update(extra_diag)
        return diag
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
    if extra_diag:
        diag.update(extra_diag)
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
    parser.add_argument(
        "--null-newton",
        choices=["off", "on", "auto"],
        default="off",
        help="Newton refinement for nullspace proposals.",
    )
    parser.add_argument("--null-workers", type=int, default=1, help="Process count for nullspace sampling.")
    parser.add_argument("--null-seed", type=int, default=None, help="Seed for nullspace RNG.")
    parser.add_argument(
        "--null-debug",
        action="store_true",
        help="Write per-sample nullspace diagnostics to null_debug.npz.",
    )
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
    earth_bary, earth_bary_vel = _body_posvel_km_single("earth", t0)
    sun_bary, sun_bary_vel = _body_posvel_km_single("sun", t0)
    earth_helio = earth_bary - sun_bary
    earth_vel_helio = earth_bary_vel - sun_bary_vel
    site_pos, site_vel = _site_states(
        [t0],
        [obs_ref.site],
        observer_positions_km=[obs_ref.observer_pos_km],
        observer_velocities_km_s=None,
        allow_unknown_site=True,
    )
    cached_frame = (earth_helio, earth_vel_helio, site_pos[0], site_vel[0])

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
    jpl_samples, jpl_stats = generate_jpl_jitter(
        jpl_state,
        args.n_jpl,
        sigma_r_km=50.0,
        sigma_v_km_s=0.05,
        epoch=t0,
        obs_list=obs_list,
        obs_cache=obs_cache,
    )
    diag_jpl = finalize_family(jpl_samples, outdir / "samples_jpl", extra_diag=jpl_stats)
    print("JPL diag:", diag_jpl)

    print("Generating attributable proposals...")
    attrib_samples, attrib_stats = generate_attrib_samples(
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
    diag_attrib = finalize_family(
        attrib_samples, outdir / "samples_attrib", extra_diag=attrib_stats
    )
    print("Attrib diag:", diag_attrib)

    if args.n_null > 0:
        print("Generating nullspace proposals...")
        seed_theta = elements_from_state(jpl_state[:3], jpl_state[3:], t0)
        debug_path = outdir / "null_debug.npz" if args.null_debug else None
        if args.null_workers > 1 and args.n_null > 1:
            workers = min(args.null_workers, args.n_null)
            base = args.n_null // workers
            extra = args.n_null % workers
            counts = [base + (1 if i < extra else 0) for i in range(workers)]
            seed_seq = np.random.SeedSequence(args.null_seed)
            child_seeds = seed_seq.spawn(workers)
            seeds = [int(s.generate_state(1)[0]) for s in child_seeds]
            payloads = []
            for idx, n_chunk in enumerate(counts):
                if n_chunk <= 0:
                    continue
                payloads.append(
                    (
                        seeds[idx],
                        seed_theta,
                        n_chunk,
                        attrib,
                        cov_attrib,
                        t0,
                        obs_ref,
                        obs_list,
                        obs_cache,
                        cached_frame,
                        args.null_newton,
                        args.null_debug,
                    )
                )
            ctx = mp.get_context("fork")
            with ctx.Pool(processes=workers) as pool:
                results = pool.map(_nullspace_worker, payloads)
            null_samples = []
            null_stats = {"attempted": 0, "project_fail": 0, "sanity_reject": 0, "jacobian_fail": 0, "success": 0}
            debug_list = []
            for samples_i, stats_i, debug_i in results:
                null_samples.extend(samples_i)
                for key in null_stats:
                    null_stats[key] += int(stats_i.get(key, 0))
                if debug_i is not None:
                    debug_list.append(debug_i)
            merged_debug = _merge_null_debug(debug_list)
            if merged_debug is not None and debug_path is not None:
                _save_null_debug(merged_debug, debug_path)
        else:
            null_samples, null_stats, debug = generate_nullspace_samples(
                seed_theta,
                args.n_null,
                attrib,
                cov_attrib,
                t0,
                obs_ref,
                obs_list,
                obs_cache,
                cached_frame,
                args.null_newton,
                debug_path,
                args.null_debug,
            )
        diag_null = finalize_family(
            null_samples, outdir / "samples_null", extra_diag=null_stats
        )
        print("Null diag:", diag_null)


if __name__ == "__main__":
    main()
