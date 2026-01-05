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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def state_physicality(state: np.ndarray, mu: float = MU_SUN) -> tuple[bool, float, float, float]:
    """Return (ok, eps, a, e) for heliocentric state."""
    try:
        r = np.asarray(state[:3], dtype=float)
        v = np.asarray(state[3:], dtype=float)
        rnorm = float(np.linalg.norm(r))
        vnorm = float(np.linalg.norm(v))
        if not np.isfinite(rnorm) or not np.isfinite(vnorm) or rnorm <= 0.0:
            return False, float("nan"), float("nan"), float("nan")
        eps = 0.5 * vnorm * vnorm - mu / rnorm
        a = float("inf")
        if eps < 0.0:
            a = -mu / (2.0 * eps)
        h = np.cross(r, v)
        evec = (np.cross(v, h) / mu) - (r / rnorm)
        e = float(np.linalg.norm(evec))
        ok = np.isfinite(e) and np.isfinite(eps) and eps < 0.0 and e < 1.0
        return bool(ok), float(eps), float(a), float(e)
    except Exception:
        return False, float("nan"), float("nan"), float("nan")


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
    """Generate nullspace proposals anchored at seed_theta.

    Hardened implementation:
      - local (H evaluated at theta_guess) Tikhonov linear solve
      - nullspace-aware alpha scaling (uses nz = N0 @ z)
      - TRF Newton with null penalty and Huber loss
      - robust regularization and step-halving fallback
    """
    samples: list[dict] = []
    stats = {
        "attempted": 0,
        "project_fail": 0,
        "sanity_reject": 0,
        "jacobian_fail": 0,
        "success": 0,
    }
    H0 = compute_H(seed_theta, epoch, obs_ref, cached_frame=cached_frame)
    _, S0, _ = linalg.svd(H0, full_matrices=False)
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
    chi2_thresh = 25.0
    alpha_min = 1e-3
    safety_alpha = 0.9
    reg_frac = 1e-4
    lambda_null = 1e-3
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
            "log_null": [],
            "logq_eff": [],
            "rho_null_bounds": [],
            "log_null_max": [],
        }

    try:
        cov_inv = linalg.inv(cov_eps)
    except Exception:
        cov_inv = None

    def log_null_prior(rho_au: float, rho_min_au: float, rho_max_au: float, alpha: float = 0.1) -> float:
        span = max(1e-12, float(rho_max_au) - float(rho_min_au))
        tau = max(1e-12, alpha * span)
        if rho_au < rho_min_au:
            return - (rho_min_au - rho_au) / tau
        if rho_au > rho_max_au:
            return - (rho_au - rho_max_au) / tau
        return 0.0

    try:
        zmax = 0.5
        nz = 21
        zs = np.linspace(-zmax, zmax, nz)
        rhos_grid = []
        earth_helio, earth_vel_helio, site_offset, site_vel = cached_frame
        for z1 in zs:
            for z2 in zs:
                zvec = np.array([z1, z2], dtype=float)
                try:
                    theta_lin = seed_theta + (N0 @ zvec)
                    r_km, v_km = state_from_elements(theta_lin, epoch)
                    r_helio = r_km
                    r_geo = r_helio - earth_helio
                    r_topo = r_geo - site_offset
                    rho_km = float(np.linalg.norm(r_topo))
                    if np.isfinite(rho_km) and rho_km > 0.0:
                        rhos_grid.append(rho_km / AU_KM)
                except Exception:
                    continue
        if rhos_grid:
            rho_min_au = float(np.min(rhos_grid))
            rho_max_au = float(np.max(rhos_grid))
        else:
            seed_rho_au = float(np.linalg.norm(r_seed) / AU_KM)
            rho_min_au = max(1e-6, seed_rho_au / 100.0)
            rho_max_au = max(1e-6, seed_rho_au * 100.0)
        if rhos_grid:
            log_null_max = max(log_null_prior(rho, rho_min_au, rho_max_au, alpha=0.005) for rho in rhos_grid)
        else:
            log_null_max = 0.0
    except Exception:
        seed_rho_au = float(np.linalg.norm(r_seed) / AU_KM)
        rho_min_au = max(1e-6, seed_rho_au / 100.0)
        rho_max_au = max(1e-6, seed_rho_au * 100.0)
        log_null_max = 0.0

    if debug is not None:
        debug["rho_null_bounds"] = [(rho_min_au, rho_max_au)]
        debug["log_null_max"] = [float(log_null_max)]

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
        log_null_val=None,
        logq_eff_val=None,
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
        debug["H_svals_seed"].append(S0.copy())
        debug["H_svals_lin"].append(H_svals_lin_val.copy())
        debug["H_svals_star"].append(H_svals_star_val.copy())
        debug["reject_reason"].append(reject_reason_val)
        debug["state"].append(state_val.copy())
        debug["loglik"].append(loglik_val)
        debug["logq"].append(logq_val)
        if log_null_val is None:
            log_null_val = float("nan")
        debug["log_null"].append(log_null_val)
        if logq_eff_val is None:
            logq_eff_val = float("nan")
        debug["logq_eff"].append(logq_eff_val)

    def tikhonov_solve(H, r, reg_strength):
        """Solve min ||H d - r||^2 + reg_strength ||d||^2."""
        U, S, Vt = linalg.svd(H, full_matrices=False)
        smax = float(S[0]) if S.size > 0 else 1.0
        eps_reg = smax * max(1e-12, reg_strength)
        filt = S / (S * S + eps_reg)
        delta = Vt.T @ (filt * (U.T @ r))
        return delta, S

    def _perihelion(a_val, e_val):
        return a_val * (1.0 - e_val)

    e_min = 1e-6
    e_max = 0.9999
    a_min = 1e-6
    q_min = 0.01

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
        if np.isfinite(log_null_max):
            accepted_flag = False
            for _ in range(200):
                if np.random.rand() < qz_tight_prob:
                    z = np.random.multivariate_normal(qz_mean, qz_tight_cov)
                else:
                    z = np.random.multivariate_normal(qz_mean, qz_wide_cov)
                try:
                    theta_lin_tmp = seed_theta + (N0 @ z)
                    r_km_tmp, v_km_tmp = state_from_elements(theta_lin_tmp, epoch)
                    r_geo_tmp = r_km_tmp - earth_helio
                    r_topo_tmp = r_geo_tmp - site_offset
                    rho_km_tmp = float(np.linalg.norm(r_topo_tmp))
                    if not (np.isfinite(rho_km_tmp) and rho_km_tmp > 0.0):
                        continue
                    rho_au_tmp = rho_km_tmp / AU_KM
                    log_null_tmp = log_null_prior(rho_au_tmp, rho_min_au, rho_max_au, alpha=0.005)
                    if math.log(np.random.rand()) <= (log_null_tmp - log_null_max):
                        accepted_flag = True
                        break
                except Exception:
                    continue
            if not accepted_flag:
                stats["project_fail"] += 1
                continue
        reject_reason = ""
        chi2_lin = float("nan")
        chi2_star = float("nan")
        logabsdet = float("nan")
        H_svals_lin = np.full(1, np.nan)
        H_svals_star = np.full(1, np.nan)
        theta_star = np.full_like(seed_theta, np.nan)
        used_newton = False
        theta_lower = np.array(
            [1e-6, 0.0, 0.0, -2.0 * math.pi, -2.0 * math.pi, -2.0 * math.pi]
        )
        theta_upper = np.array(
            [1e6, 0.9999, math.pi, 2.0 * math.pi, 2.0 * math.pi, 2.0 * math.pi]
        )

        nz = N0 @ z
        res_for_lin = y_obs_vec - y_seed - eps
        theta_lin_guess = seed_theta + nz
        try:
            H_loc = compute_H(theta_lin_guess, epoch, obs_ref, cached_frame=cached_frame)
            delta_theta, S_loc = tikhonov_solve(H_loc, res_for_lin, reg_frac)
            if np.linalg.norm(delta_theta) > 5.0:
                delta_theta, S_loc = tikhonov_solve(H_loc, res_for_lin, 1e-3)
        except Exception:
            stats["project_fail"] += 1
            reject_reason = "H_loc_fail"
            _debug_append(
                np.full_like(seed_theta, np.nan),
                z,
                eps,
                chi2_lin,
                False,
                np.full_like(seed_theta, np.nan),
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

        theta_unclamped = seed_theta + delta_theta + nz
        a_un, e_un = float(theta_unclamped[0]), float(theta_unclamped[1])
        if (
            a_un > a_min
            and e_min <= e_un < e_max
            and _perihelion(a_un, e_un) >= q_min
            and math.degrees(float(theta_unclamped[2])) <= 90.0
        ):
            theta_lin = theta_unclamped
        else:
            seed_a = float(seed_theta[0])
            seed_e = float(seed_theta[1])
            nz_a = float(nz[0])
            nz_e = float(nz[1])
            delta_a = float(delta_theta[0])
            delta_e = float(delta_theta[1])
            alpha_max = 1.0
            if delta_e < 0.0:
                numer = (seed_e + nz_e - e_min)
                if numer <= 0.0:
                    alpha_max = 0.0
                else:
                    alpha_max = min(alpha_max, numer / (-delta_e))
            elif delta_e > 0.0:
                alpha_max = min(alpha_max, (e_max - (seed_e + nz_e)) / delta_e)

            if delta_a < 0.0:
                numer = (seed_a + nz_a - a_min)
                if numer <= 0.0:
                    alpha_max = 0.0
                else:
                    alpha_max = min(alpha_max, numer / (-delta_a))

            seed_q = _perihelion(seed_a + nz_a, seed_e + nz_e)
            denom = (delta_a * (1.0 - (seed_e + nz_e)) - (seed_a + nz_a) * delta_e)
            if denom < 0.0:
                numer_q = (seed_q - q_min)
                if numer_q <= 0.0:
                    alpha_max = 0.0
                else:
                    alpha_max = min(alpha_max, numer_q / (-denom))

            alpha_max = max(0.0, min(1.0, safety_alpha * alpha_max))
            if alpha_max <= alpha_min:
                stats["project_fail"] += 1
                reject_reason = "linear_step_infeasible"
                _debug_append(
                    np.full_like(seed_theta, np.nan),
                    z,
                    eps,
                    chi2_lin,
                    False,
                    np.full_like(seed_theta, np.nan),
                    chi2_star,
                    logabsdet,
                    S_loc if S_loc.size > 0 else H_svals_lin,
                    H_svals_star,
                    reject_reason,
                    np.full(6, np.nan),
                    float("-inf"),
                    float("-inf"),
                )
                continue

            theta_lin = seed_theta + alpha_max * delta_theta + nz

        if not np.isfinite(theta_lin).all():
            stats["project_fail"] += 1
            reject_reason = "nan_theta_lin"
            _debug_append(
                np.full_like(seed_theta, np.nan),
                z,
                eps,
                chi2_lin,
                False,
                np.full_like(seed_theta, np.nan),
                chi2_star,
                logabsdet,
                S_loc if S_loc.size > 0 else H_svals_lin,
                H_svals_star,
                reject_reason,
                np.full(6, np.nan),
                float("-inf"),
                float("-inf"),
            )
            continue

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
            stats["project_fail"] += 1
            reject_reason = "chi2_lin_fail"
            _debug_append(
                theta_lin,
                z,
                eps,
                chi2_lin,
                False,
                np.full_like(seed_theta, np.nan),
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
                null_res = math.sqrt(lambda_null) * (N_mat.T.dot(theta_param - theta_lin) - z)
                return np.concatenate([res, null_res])

            try:
                sol = least_squares(
                    fun_aug,
                    theta_lin,
                    bounds=(theta_lower, theta_upper),
                    xtol=1e-10,
                    ftol=1e-10,
                    gtol=1e-10,
                    max_nfev=1000,
                    method="trf",
                    loss="huber",
                )
                theta_star = sol.x
                success = sol.success
                if not success:
                    ok = False
                    for scale in [0.5, 0.25, 0.1]:
                        theta_try = theta_lin + scale * (theta_star - theta_lin)
                        sol2 = least_squares(
                            fun_aug,
                            theta_try,
                            bounds=(theta_lower, theta_upper),
                            xtol=1e-10,
                            ftol=1e-10,
                            gtol=1e-10,
                            max_nfev=500,
                            method="trf",
                            loss="huber",
                        )
                        if sol2.success:
                            theta_star = sol2.x
                            success = True
                            ok = True
                            break
                    if not ok:
                        success = False
            except Exception:
                theta_star = theta_lin
                success = False

            if not success:
                stats["project_fail"] += 1
                reject_reason = "project_fail_newton"
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
            if sign == 0 or not np.isfinite(logabsdet):
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

        log_null = float("nan")
        log_q_eff = log_q
        try:
            r_km, v_km_s = state_from_elements(theta_star, epoch)
            state = np.hstack([r_km, v_km_s])
            y_star = predict_attributable_from_state_cached(state, *cached_frame)
            r_vec = y_star - (y_obs_vec - eps)
            if cov_inv is not None:
                chi2_star = float(r_vec.T @ cov_inv @ r_vec)
            else:
                chi2_star = float(np.dot(r_vec, r_vec))
            ok_state, eps_state, a_state, e_state = state_physicality(state)
            if not ok_state:
                stats["sanity_reject"] += 1
                reject_reason = "state_ecc"
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
                    log_null,
                )
                continue
            loglik = compute_loglik(state, epoch, obs_list, obs_cache)
            _, rho_km, _ = _attrib_rho_from_state(state, obs_ref, epoch)
            rho_au = rho_km / AU_KM
            log_null = log_null_prior(rho_au, rho_min_au, rho_max_au, alpha=0.005)
            if np.isfinite(log_null):
                log_q_eff = float(log_q + log_null)
        except Exception:
            loglik = -np.inf
            state = np.full(6, np.nan)
            log_q_eff = log_q

        samples.append(
            {
                "theta": theta_star,
                "state": state,
                "log_q": log_q_eff,
                "log_q_base": log_q,
                "log_q_eff": log_q_eff,
                "loglik": loglik,
                "success": True,
                "used_newton": used_newton,
                "log_null": log_null,
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
            log_null,
            log_q_eff,
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
            {
                "theta": None,
                "state": state,
                "log_q": log_q,
                "loglik": loglik,
                "used_newton": False,
            }
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
        samples.append(
            {
                "theta": None,
                "state": state,
                "log_q": log_q,
                "loglik": loglik,
                "used_newton": False,
            }
        )
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


def audit_family(
    samples: list[dict],
    out_prefix: Path,
    obs_ref,
    epoch: Time,
    *,
    logprior_default: float = 0.0,
) -> dict:
    states = np.array([d.get("state", np.full(6, np.nan)) for d in samples], dtype=float)
    logq = np.array(
        [d.get("log_q_eff", d.get("log_q", -np.inf)) for d in samples], dtype=float
    )
    logq_base = np.array([d.get("log_q_base", d.get("log_q", -np.inf)) for d in samples], dtype=float)
    log_null = np.array([d.get("log_null", 0.0) for d in samples], dtype=float)
    loglik = np.array([d.get("loglik", -np.inf) for d in samples], dtype=float)
    used_newton = np.array([bool(d.get("used_newton", False)) for d in samples], dtype=bool)
    accepted = np.isfinite(loglik)
    rho_topo = np.full(len(samples), np.nan, dtype=float)
    for i, st in enumerate(states):
        try:
            _, rho_km, _ = _attrib_rho_from_state(st, obs_ref, epoch)
            rho_topo[i] = rho_km / AU_KM
        except Exception:
            continue

    logprior = np.full(len(samples), float(logprior_default), dtype=float)
    loglike_astrom = loglik.copy()
    loglike_phot = np.full(len(samples), np.nan, dtype=float)
    logw = (loglike_astrom + logprior) - logq
    finite = np.isfinite(logw) & np.isfinite(rho_topo)
    if np.any(finite):
        lw = logw[finite]
        lw = lw - np.max(lw)
        w = np.exp(lw)
        w /= np.sum(w) + 1e-300
        ess = float(1.0 / np.sum(w * w))
        max_w = float(np.max(w))
        rho_f = rho_topo[finite]
        idx = np.argsort(rho_f)
        rho_sorted = rho_f[idx]
        w_sorted = w[idx]
        cdf = np.cumsum(w_sorted)
        wmed = float(rho_sorted[np.searchsorted(cdf, 0.5)])
        umed = float(np.median(rho_f))
    else:
        ess = 0.0
        max_w = float("nan")
        wmed = float("nan")
        umed = float("nan")

    audit = {
        "ess": ess,
        "max_weight": max_w,
        "rho_weighted_median": wmed,
        "rho_unweighted_median": umed,
        "n_samples": int(len(samples)),
        "n_finite": int(np.sum(finite)),
    }

    audit_npz = out_prefix.parent / f"{out_prefix.name}_audit.npz"
    audit_json = out_prefix.parent / f"{out_prefix.name}_audit.json"
    np.savez_compressed(
        audit_npz,
        rho_topo=rho_topo,
        loglike_astrom=loglike_astrom,
        loglike_phot=loglike_phot,
        logprior=logprior,
        logq=logq,
        logq_base=logq_base,
        log_null=log_null,
        logw=logw,
        accepted=accepted,
        used_newton=used_newton,
    )
    with open(audit_json, "w") as fh:
        json.dump(audit, fh, indent=2)

    if np.any(finite):
        plt.figure(figsize=(6, 4))
        plt.scatter(rho_topo[finite], logq[finite], s=6, alpha=0.6)
        plt.xlabel("rho_topo (AU)")
        plt.ylabel("logq")
        plt.title(f"{out_prefix.name} logq vs rho")
        plt.tight_layout()
        plt.savefig(out_prefix.parent / f"{out_prefix.name}_logq_vs_rho.png", dpi=150)
        plt.close()

        plt.figure(figsize=(6, 4))
        plt.scatter(rho_topo[finite], logw[finite], s=6, alpha=0.6)
        plt.xlabel("rho_topo (AU)")
        plt.ylabel("logw")
        plt.title(f"{out_prefix.name} logw vs rho")
        plt.tight_layout()
        plt.savefig(out_prefix.parent / f"{out_prefix.name}_logw_vs_rho.png", dpi=150)
        plt.close()

    return audit


def _local_mcmc_state(
    start_state: np.ndarray,
    epoch: Time,
    obs_list,
    obs_cache,
    n_steps: int = 8000,
    burn: int = 2000,
    cov_scale: float = 0.003,
    rng_seed: int | None = None,
) -> list[dict]:
    """Simple random-walk MH around a state; returns list of samples with loglik."""
    rng = np.random.default_rng(rng_seed)
    state = np.array(start_state, dtype=float)
    loglik = compute_loglik(state, epoch, obs_list, obs_cache)
    if not np.isfinite(loglik):
        return []
    scale = np.maximum(1e-6, np.abs(state))
    step_sigma = cov_scale * scale
    out = []
    for i in range(n_steps):
        proposal = state + rng.normal(scale=step_sigma, size=state.shape)
        loglik_p = compute_loglik(proposal, epoch, obs_list, obs_cache)
        if np.isfinite(loglik_p):
            if math.log(rng.random()) <= (loglik_p - loglik):
                state = proposal
                loglik = loglik_p
        if i >= burn:
            out.append({"state": state.copy(), "loglik": float(loglik)})
    return out


def auto_local_expand_and_redistribute(
    samples: list[dict],
    stats: dict,
    obs_list,
    obs_cache,
    epoch: Time,
    *,
    delta_logw_trigger: float = 10.0,
    ess_fraction: float = 0.01,
    khat_trigger: float = 0.7,
    n_local: int = 2000,
    mcmc_steps: int = 8000,
    mcmc_burn: int = 2000,
    cov_scale: float = 0.003,
    mcmc_chains: int = 4,
) -> tuple[list[dict], dict]:
    """Expand around the top-weight sample if weights collapse, then redistribute weight."""
    if not samples:
        return samples, stats
    loglik = np.array([s.get("loglik", -np.inf) for s in samples], dtype=float)
    logq = np.array([s.get("log_q", -np.inf) for s in samples], dtype=float)
    logw = loglik - logq
    if not np.isfinite(logw).any() or logw.size < 2:
        return samples, stats
    order = np.argsort(-logw)
    top = int(order[0])
    second = int(order[1])
    delta_logw = float(logw[top] - logw[second])
    lw = logw - logsumexp(logw)
    w = np.exp(lw)
    ess = float(1.0 / np.sum(w * w))
    khat = float("inf")
    try:
        import arviz as az

        khat_val = float(az.psislw(lw)[1])
        if np.isfinite(khat_val):
            khat = khat_val
    except Exception:
        khat = float("inf")
    if (delta_logw < delta_logw_trigger) and (ess >= ess_fraction * len(samples)) and (
        not np.isfinite(khat) or khat < khat_trigger
    ):
        return samples, stats
    start_state = samples[top].get("state")
    if start_state is None or not np.isfinite(start_state).all():
        return samples, stats
    n_chains = max(1, int(mcmc_chains))
    n_chains = min(n_chains, os.cpu_count() or 1)
    per_chain_steps = max(1, int(mcmc_steps))
    per_chain_burn = min(mcmc_burn, max(0, per_chain_steps - 1))
    seeds = list(range(n_chains))
    if n_chains > 1:
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=n_chains) as pool:
            results = pool.starmap(
                _local_mcmc_state,
                [
                    (
                        start_state,
                        epoch,
                        obs_list,
                        obs_cache,
                        per_chain_steps,
                        per_chain_burn,
                        cov_scale,
                        seeds[i],
                    )
                    for i in range(n_chains)
                ],
                chunksize=1,
            )
    else:
        results = [
            _local_mcmc_state(
                start_state,
                epoch,
                obs_list,
                obs_cache,
                n_steps=per_chain_steps,
                burn=per_chain_burn,
                cov_scale=cov_scale,
                rng_seed=seeds[0],
            )
        ]
    draws = []
    for chain in results:
        if chain:
            draws.extend(chain)
    if not draws:
        return samples, stats
    if len(draws) > n_local:
        draws = draws[:n_local]
    logw_top = float(logw[top])
    logw_new = logw_top - math.log(len(draws))
    new_samples = [s for i, s in enumerate(samples) if i != top]
    for d in draws:
        loglik_d = float(d["loglik"])
        logq_d = float(loglik_d - logw_new)
        new_samples.append(
            {
                "state": d["state"],
                "theta": np.full(6, np.nan),
                "log_q": logq_d,
                "log_q_base": logq_d,
                "log_q_eff": logq_d,
                "loglik": loglik_d,
                "success": True,
                "used_newton": False,
                "log_null": float("nan"),
            }
        )
    stats = dict(stats)
    stats["expanded_local"] = int(len(draws))
    stats["expanded_delta_logw"] = float(delta_logw)
    stats["expanded_ess_before"] = float(ess)
    stats["expanded_khat"] = float(khat)
    stats["expanded_chains"] = int(n_chains)
    return new_samples, stats


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
    audit_family(jpl_samples, outdir / "samples_jpl", obs_ref, t0)
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
    audit_family(attrib_samples, outdir / "samples_attrib", obs_ref, t0)
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
        null_samples, null_stats = auto_local_expand_and_redistribute(
            null_samples,
            null_stats,
            obs_list,
            obs_cache,
            t0,
            delta_logw_trigger=10.0,
            ess_fraction=0.01,
            n_local=2000,
            mcmc_steps=8000,
            mcmc_burn=2000,
            cov_scale=0.003,
        )
        diag_null = finalize_family(
            null_samples, outdir / "samples_null", extra_diag=null_stats
        )
        audit_family(null_samples, outdir / "samples_null", obs_ref, t0)
        print("Null diag:", diag_null)


if __name__ == "__main__":
    main()
