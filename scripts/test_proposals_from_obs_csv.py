#!/usr/bin/env python3
"""
test_proposals_from_obs_csv.py

Read an obs.csv, select an object (e.g. "1" or "1 Ceres"), pick two observations,
run Variant A (hierarchical) and Variant B (joint Laplace) proposals, draw N samples
and compare to JPL/Horizons truth at the sample-specific emission times.

Dependencies: numpy, scipy, pandas, astropy, astroquery, matplotlib
"""

import argparse
import os
import numpy as np
import scipy.linalg as la
from scipy.optimize import least_squares
import pandas as pd
from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.utils import iers
import matplotlib.pyplot as plt
from astroquery.jplhorizons import Horizons
from concurrent.futures import ProcessPoolExecutor

# ------------------------------
# TRY TO IMPORT NEOTUBE HELPERS
# ------------------------------
try:
    from neotube.propagate import propagate_state, propagate_state_kepler, _site_states
    print("Imported neotube propagation helpers.")
except Exception:
    propagate_state = None
    propagate_state_kepler = None
    _site_states = None
    print("Could not import neotube helpers; using astropy fallbacks where necessary.")

C = 299792.458  # km/s

iers.conf.auto_download = False
iers.conf.iers_degraded_accuracy = "warn"

USE_FULL_PHYSICS = True
MAX_WORKERS = 50
MIXTURE_WEIGHT_GAUSS = 0.95
MIXTURE_T_NU = 3.0
MIXTURE_T_SCALE = 3.0


def _chunksize(n_items, n_workers):
    return max(1, n_items // max(1, n_workers * 4))

# ------------------------------
# CSV parsing helpers
# ------------------------------
def detect_columns(df):
    """Detect columns for object id, time, ra, dec, sigma_ra, sigma_dec, site."""
    cols = {c.lower(): c for c in df.columns}

    def find_one(candidates):
        for cand in candidates:
            key = cand.lower()
            if key in cols:
                return cols[key]
        return None

    obj_col = find_one(["object", "obj_id", "target", "designation", "id"])
    time_col = find_one(["time", "datetime", "epoch", "obs_time", "t", "date", "t_utc"])
    ra_col = find_one(["ra", "ra_deg", "ra_degrees"])
    dec_col = find_one(["dec", "dec_deg", "dec_degrees"])
    sigma_ra_col = find_one(
        ["sigma_ra_arcsec", "sigma_ra", "ra_error_arcsec", "ra_sigma_arcsec"]
    )
    sigma_dec_col = find_one(
        ["sigma_dec_arcsec", "sigma_dec", "dec_error_arcsec", "dec_sigma_arcsec"]
    )
    sigma_shared = find_one(["sigma_arcsec", "sigma", "err_arcsec"])
    site_col = find_one(["site", "observatory", "obs_site", "station"])

    if sigma_ra_col is None and sigma_shared is not None:
        sigma_ra_col = sigma_shared
    if sigma_dec_col is None and sigma_shared is not None:
        sigma_dec_col = sigma_shared

    return {
        "obj": obj_col,
        "time": time_col,
        "ra": ra_col,
        "dec": dec_col,
        "sigma_ra": sigma_ra_col,
        "sigma_dec": sigma_dec_col,
        "site": site_col,
    }


def parse_ra_dec(ra_val, dec_val):
    """Return (ra_rad, dec_rad). Accept numeric degrees or sexagesimal strings."""
    try:
        raf = float(ra_val)
        decf = float(dec_val)
        return np.deg2rad(raf), np.deg2rad(decf)
    except Exception:
        sc = SkyCoord(f"{ra_val} {dec_val}", unit=(u.hourangle, u.deg), frame="icrs")
        try:
            return sc.ra.rad, sc.dec.rad
        except Exception:
            sc2 = SkyCoord(f"{ra_val} {dec_val}", unit=(u.deg, u.deg), frame="icrs")
            return sc2.ra.rad, sc2.dec.rad


# ------------------------------
# Observer / site helpers
# ------------------------------
def observer_posvel(site, t):
    """
    Return observer position and velocity in km, km/s in ICRS-like frame.
    """
    from astropy.coordinates import get_body_barycentric_posvel

    def _as_vel_kms(obj):
        if hasattr(obj, "d_xyz"):
            return obj.d_xyz.to(u.km / u.s)
        return obj.xyz.to(u.km / u.s)

    pb_earth = get_body_barycentric_posvel("earth", t.tdb)
    pb_sun = get_body_barycentric_posvel("sun", t.tdb)
    r_earth = np.array((pb_earth[0].xyz - pb_sun[0].xyz).to(u.km)).flatten()
    v_earth = np.array((_as_vel_kms(pb_earth[1]) - _as_vel_kms(pb_sun[1]))).flatten()
    if _site_states is not None:
        try:
            pos_geo, vel_geo = _site_states([t], [site], allow_unknown_site=True)
            return r_earth + pos_geo[0], v_earth + vel_geo[0]
        except Exception:
            return r_earth, v_earth
    r = r_earth
    v = v_earth
    return r, v


# ------------------------------
# Unit vector / tangent basis
# ------------------------------
def hat_u_from_radec(alpha_rad, delta_rad):
    ca = np.cos(alpha_rad)
    sa = np.sin(alpha_rad)
    cd = np.cos(delta_rad)
    sd = np.sin(delta_rad)
    return np.array([cd * ca, cd * sa, sd])


def tangent_basis(alpha_rad, delta_rad):
    ca = np.cos(alpha_rad)
    sa = np.sin(alpha_rad)
    cd = np.cos(delta_rad)
    sd = np.sin(delta_rad)
    e_alpha = np.array([-sa, ca, 0.0])
    e_delta = np.array([-ca * sd, -sa * sd, cd])
    return e_alpha, e_delta


def angle_diff(a, b):
    """Return the short signed angle difference a - b in radians in (-pi, pi]."""
    return (a - b + np.pi) % (2 * np.pi) - np.pi


def cos_dec_for_ra_div(dec, min_abs=1e-6):
    """Return a safe cos(dec) value for dividing RA tangent components."""
    c = float(np.cos(dec))
    if abs(c) < min_abs:
        return min_abs if c >= 0.0 else -min_abs
    return c


def radec_from_vector(vec):
    x, y, z = vec
    r = np.linalg.norm(vec)
    dec = np.arcsin(z / r)
    ra = np.arctan2(y, x) % (2 * np.pi)
    return ra, dec


# ------------------------------
# Light-time / emission time solvers
# ------------------------------
def solve_emission_time_for_obs(
    t_obs, rho_km, hat_u, site, tol=1e-8, maxit=30, max_step_s=3600.0
):
    """Newton solve for t_em so that photons emitted from r = r_obs(t_em)+rho*hat_u arrive at t_obs."""
    t_em = Time(t_obs.tdb + (-rho_km / C) * u.s)
    for _ in range(maxit):
        r_obs_tobs, _ = observer_posvel(site, t_obs)
        r_obs_tem, v_obs_tem = observer_posvel(site, t_em)
        r_target = r_obs_tem + rho_km * hat_u
        dvec = r_obs_tobs - r_target
        D = np.linalg.norm(dvec)
        F = (t_obs.tdb.jd - t_em.tdb.jd) * 86400.0 - D / C
        if abs(F) < tol:
            break
        dFdt = -1.0 + (np.dot(dvec, v_obs_tem) / (C * D))
        if abs(dFdt) < 1e-12:
            break
        dt = F / dFdt
        dt = float(np.clip(dt, -max_step_s, max_step_s))
        t_em = Time(t_em.tdb + (-dt) * u.s)
    return t_em, r_obs_tem, v_obs_tem


def _propagate_state_to_time(r1, v1, t0, t1, propagate_fn):
    if propagate_fn is None:
        raise RuntimeError("No propagation function available.")
    state = np.concatenate([r1, v1])
    try:
        out = propagate_fn(state, t0, [t1])
        out = np.asarray(out)
        return out[0, :3], out[0, 3:]
    except Exception:
        try:
            r_t, v_t = propagate_fn((r1, v1), t0, t1)
            return np.asarray(r_t, dtype=float), np.asarray(v_t, dtype=float)
        except Exception as exc:
            raise RuntimeError(f"Propagation failed: {exc}") from exc


def solve_emission_time_and_propagate(
    obs2, r1, v1, t_em1, site, propagate_fn, tol=1e-8, maxit=30, max_step_s=3600.0
):
    """Solve for t_em2 and propagate target from (r1,v1) at t_em1 to t_em2."""
    r_obs_tobs2, _ = observer_posvel(site, obs2)
    D0 = np.linalg.norm(r_obs_tobs2 - r1)
    t_em = Time(obs2.tdb + (-D0 / C) * u.s)
    r_t = r1
    v_t = v1
    for _ in range(maxit):
        r_t, v_t = _propagate_state_to_time(r1, v1, t_em1, t_em, propagate_fn)
        r_obs_tobs2, _ = observer_posvel(site, obs2)
        dvec = r_obs_tobs2 - r_t
        D = np.linalg.norm(dvec)
        F = (obs2.tdb.jd - t_em.tdb.jd) * 86400.0 - D / C
        if abs(F) < tol:
            break
        dFdt = -1.0 + (np.dot(dvec, v_t) / (C * D))
        if abs(dFdt) < 1e-12:
            break
        dt = F / dFdt
        dt = float(np.clip(dt, -max_step_s, max_step_s))
        t_em = Time(t_em.tdb + (-dt) * u.s)
    return t_em, r_t, v_t


# ------------------------------
# Forward model g(Gamma,theta) producing predicted RA/Dec at obs2
# ------------------------------
def forward_predict_RADEC(Gamma, theta, obs1, obs2, site1, site2, propagate_surrogate):
    """Return predicted ra,dec (rad), emission times, states at em1/em2."""
    alpha1, delta1 = Gamma
    logrho, dotrho, ve, vn = theta
    rho = float(np.exp(logrho))
    hat_u = hat_u_from_radec(alpha1, delta1)
    e_alpha, e_delta = tangent_basis(alpha1, delta1)

    t_em1, r_obs_em1, v_obs_em1 = solve_emission_time_for_obs(obs1, rho, hat_u, site1)
    r1 = r_obs_em1 + rho * hat_u
    v_topo = dotrho * hat_u + ve * e_alpha + vn * e_delta
    v1 = v_obs_em1 + v_topo

    t_em2, r2, v2 = solve_emission_time_and_propagate(
        obs2, r1, v1, t_em1, site2, propagate_surrogate
    )

    r_obs_tobs2, _ = observer_posvel(site2, obs2)
    dvec = r_obs_tobs2 - r2
    ra_pred, dec_pred = radec_from_vector(dvec)
    return ra_pred, dec_pred, t_em1, t_em2, r1, v1, r2, v2


# ------------------------------
# Finite-difference Jacobian for (ra,dec) wrt params (optionally tangent-plane RA)
# ------------------------------
def jacobian_fd(
    Gamma,
    theta,
    obs1,
    obs2,
    site1,
    site2,
    propagate_surrogate,
    obs2_dec=None,
    eps=None,
):
    p = np.concatenate((np.array(Gamma), np.array(theta)))
    n = p.size
    if eps is None:
        eps = np.maximum(1e-8, np.abs(p) * 1e-7 + 1e-10)
    J = np.zeros((2, n))
    for i in range(n):
        dp = p.copy()
        dp[i] += eps[i]
        ra_p, dec_p, *_ = forward_predict_RADEC(
            dp[0:2], dp[2:], obs1, obs2, site1, site2, propagate_surrogate
        )
        dp[i] -= 2 * eps[i]
        ra_m, dec_m, *_ = forward_predict_RADEC(
            dp[0:2], dp[2:], obs1, obs2, site1, site2, propagate_surrogate
        )
        J[0, i] = angle_diff(ra_p, ra_m) / (2 * eps[i])
        J[1, i] = (dec_p - dec_m) / (2 * eps[i])
    if obs2_dec is not None:
        J[0, :] *= np.cos(obs2_dec)
    return J


# ------------------------------
# Variant A: conditional Laplace on psi given Gamma and rho
# ------------------------------
def optimize_conditional_psi(
    Gamma,
    logrho,
    obs1,
    obs2,
    site1,
    site2,
    propagate_surrogate,
    W2,
    prior_Ppsi_inv,
    obs2_ra,
    obs2_dec,
):
    def residuals(psi):
        theta = np.concatenate(([logrho], psi))
        try:
            ra_pred, dec_pred, *_ = forward_predict_RADEC(
                Gamma, theta, obs1, obs2, site1, site2, propagate_surrogate
            )
        except Exception:
            return np.array([1e6, 1e6])
        res = np.array([
            angle_diff(ra_pred, obs2_ra) * np.cos(obs2_dec),
            (dec_pred - obs2_dec),
        ])
        Wsqrt = la.cholesky(W2)
        return Wsqrt.dot(res)

    dt = max(1.0, (obs2.tdb.jd - obs1.tdb.jd) * 86400.0)
    dalpha = angle_diff(obs2_ra, Gamma[0])
    ddelta = (obs2_dec - Gamma[1])
    d_alpha_dt = dalpha / dt
    d_delta_dt = ddelta / dt
    rho = np.exp(logrho)
    ve0 = rho * d_alpha_dt * np.cos(Gamma[1])
    vn0 = rho * d_delta_dt
    psi0 = np.array([0.0, ve0, vn0])
    res = least_squares(residuals, psi0, method="trf", xtol=1e-8, ftol=1e-8, gtol=1e-8)
    hat_psi = res.x
    theta_hat = np.concatenate(([logrho], hat_psi))
    J_full = jacobian_fd(
        Gamma,
        theta_hat,
        obs1,
        obs2,
        site1,
        site2,
        propagate_surrogate,
        obs2_dec=obs2_dec,
    )
    Jpsi = J_full[:, 3:6]
    Sigma_psi = la.inv(Jpsi.T.dot(W2).dot(Jpsi) + prior_Ppsi_inv)
    return hat_psi, Sigma_psi


def _sample_variant_a_one(payload):
    (
        seed,
        obs1,
        obs2,
        site1,
        site2,
        obs1_ra,
        obs1_dec,
        obs2_ra,
        obs2_dec,
        obs1_sigma_ra,
        obs1_sigma_dec,
        obs2_sigma_ra,
        obs2_sigma_dec,
        use_full_physics,
    ) = payload
    rng = np.random.default_rng(seed)
    S1 = np.diag([(obs1_sigma_ra / 206265.0) ** 2, (obs1_sigma_dec / 206265.0) ** 2])
    S2 = np.diag([(obs2_sigma_ra / 206265.0) ** 2, (obs2_sigma_dec / 206265.0) ** 2])
    W2 = la.inv(S2)
    sigma_v = 50.0
    prior_Ppsi_inv = np.diag([1e-8, 1.0 / (sigma_v**2), 1.0 / (sigma_v**2)])

    dtheta_tan = rng.multivariate_normal(np.zeros(2), S1)
    alpha_s = obs1_ra + dtheta_tan[0] / cos_dec_for_ra_div(obs1_dec)
    delta_s = obs1_dec + dtheta_tan[1]
    Gamma = (alpha_s, delta_s)
    rho_min = 1e3
    rho_max = 1e10
    u = rng.random()
    logrho = np.log(rho_min) + u * (np.log(rho_max) - np.log(rho_min))
    try:
        hat_psi, Sigma_psi = optimize_conditional_psi(
            Gamma,
            logrho,
            obs1,
            obs2,
            site1,
            site2,
            propagate_state_kepler,
            W2,
            prior_Ppsi_inv,
            obs2_ra,
            obs2_dec,
        )
    except Exception:
        return None
    if rng.random() < MIXTURE_WEIGHT_GAUSS:
        psi_star = rng.multivariate_normal(hat_psi, Sigma_psi)
    else:
        z = rng.multivariate_normal(np.zeros_like(hat_psi), Sigma_psi)
        g = rng.gamma(MIXTURE_T_NU / 2.0, 2.0 / MIXTURE_T_NU)
        psi_star = hat_psi + MIXTURE_T_SCALE * z / np.sqrt(g / MIXTURE_T_NU)
    theta_star = np.concatenate(([logrho], psi_star))
    eval_fn = propagate_state if use_full_physics and propagate_state is not None else propagate_state_kepler
    try:
        ra_pred, dec_pred, t_em1, t_em2, r1, v1, r2, v2 = forward_predict_RADEC(
            Gamma, theta_star, obs1, obs2, site1, site2, eval_fn
        )
    except Exception:
        return None
    return {
        "Gamma": Gamma,
        "logrho": logrho,
        "psi": psi_star,
        "theta": theta_star,
        "ra_pred": ra_pred,
        "dec_pred": dec_pred,
        "t_em1": t_em1,
        "t_em2": t_em2,
        "r1": r1,
        "v1": v1,
        "r2": r2,
        "v2": v2,
    }


def sample_variant_A(
    obs1,
    obs2,
    site1,
    site2,
    obs1_ra,
    obs1_dec,
    obs2_ra,
    obs2_dec,
    obs1_sigma_ra,
    obs1_sigma_dec,
    obs2_sigma_ra,
    obs2_sigma_dec,
    N=200,
    workers=None,
):
    rng = np.random.default_rng()
    seeds = rng.integers(0, 2**32 - 1, size=N, dtype=np.uint32)
    workers = min(MAX_WORKERS, os.cpu_count() or 1) if workers is None else min(workers, MAX_WORKERS)
    payloads = [
        (
            int(seeds[i]),
            obs1,
            obs2,
            site1,
            site2,
            obs1_ra,
            obs1_dec,
            obs2_ra,
            obs2_dec,
            obs1_sigma_ra,
            obs1_sigma_dec,
            obs2_sigma_ra,
            obs2_sigma_dec,
            USE_FULL_PHYSICS,
        )
        for i in range(N)
    ]
    samples = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for out in executor.map(_sample_variant_a_one, payloads, chunksize=_chunksize(N, workers)):
            if out is not None:
                samples.append(out)
    return samples


# ------------------------------
# Variant B: joint Laplace over all 6 params
# ------------------------------
def _eval_variant_b_one(payload):
    (
        p,
        obs1,
        obs2,
        site1,
        site2,
        use_full_physics,
    ) = payload
    Gamma = (p[0], p[1])
    theta = p[2:]
    eval_fn = propagate_state if use_full_physics and propagate_state is not None else propagate_state_kepler
    try:
        ra_pred, dec_pred, t_em1, t_em2, r1, v1, r2, v2 = forward_predict_RADEC(
            Gamma, theta, obs1, obs2, site1, site2, eval_fn
        )
    except Exception:
        return None
    return {
        "p": p,
        "Gamma": Gamma,
        "theta": theta,
        "ra_pred": ra_pred,
        "dec_pred": dec_pred,
        "t_em1": t_em1,
        "t_em2": t_em2,
        "r1": r1,
        "v1": v1,
        "r2": r2,
        "v2": v2,
    }


def optimize_joint_and_sample(
    obs1,
    obs2,
    site1,
    site2,
    obs1_ra,
    obs1_dec,
    obs2_ra,
    obs2_dec,
    obs1_sigma_ra,
    obs1_sigma_dec,
    obs2_sigma_ra,
    obs2_sigma_dec,
    N=200,
    workers=None,
):
    S1 = np.diag([(obs1_sigma_ra / 206265.0) ** 2, (obs1_sigma_dec / 206265.0) ** 2])
    W1 = la.inv(S1)
    S2 = np.diag([(obs2_sigma_ra / 206265.0) ** 2, (obs2_sigma_dec / 206265.0) ** 2])
    W2 = la.inv(S2)
    prior_mean = np.array([obs1_ra, obs1_dec, np.log(1.0 * 149597870.7), 0.0, 0.0, 0.0])
    prior_cov = np.diag([(1e-6) ** 2, (1e-6) ** 2, 10.0**2, 100.0**2, 50.0**2, 50.0**2])
    prior_inv = la.inv(prior_cov)

    alpha0 = obs1_ra
    delta0 = obs1_dec
    hat_u1 = hat_u_from_radec(alpha0, delta0)
    hat_u2 = hat_u_from_radec(obs2_ra, obs2_dec)
    r_obs_t1, _ = observer_posvel(site1, obs1)
    r_obs_t2, _ = observer_posvel(site2, obs2)
    du = hat_u1 - hat_u2
    drob = r_obs_t2 - r_obs_t1
    if np.linalg.norm(du) > 1e-12:
        rho_tri = np.linalg.norm(drob) / np.linalg.norm(du)
    else:
        rho_tri = 1.0 * 149597870.7
    logrho0 = np.log(np.clip(rho_tri, 1e3, 1e10))
    dt = max(1.0, (obs2.tdb.jd - obs1.tdb.jd) * 86400.0)
    dalpha = angle_diff(obs2_ra, alpha0)
    ddelta = (obs2_dec - delta0)
    d_alpha_dt = dalpha / dt
    d_delta_dt = ddelta / dt
    rho0 = np.exp(logrho0)
    ve0 = rho0 * d_alpha_dt * np.cos(delta0)
    vn0 = rho0 * d_delta_dt
    dotrho0 = 0.0
    p0 = np.array([alpha0, delta0, logrho0, dotrho0, ve0, vn0])

    def joint_obj(p):
        Gamma = (p[0], p[1])
        theta = p[2:]
        try:
            ra_pred, dec_pred, *_ = forward_predict_RADEC(
                Gamma, theta, obs1, obs2, site1, site2, propagate_state_kepler
            )
        except Exception:
            return np.full(10, 1e6)
        r1 = np.array([angle_diff(p[0], obs1_ra) * np.cos(obs1_dec), (p[1] - obs1_dec)])
        r2 = np.array([angle_diff(ra_pred, obs2_ra) * np.cos(obs2_dec), (dec_pred - obs2_dec)])
        W1sqrt = la.cholesky(W1)
        W2sqrt = la.cholesky(W2)
        eigvals, eigvecs = la.eigh(prior_inv)
        eigvals[eigvals < 0] = 0.0
        sqrt_prior_inv = eigvecs.dot(np.diag(np.sqrt(eigvals))).dot(eigvecs.T)
        prior_vec = sqrt_prior_inv.dot(p - prior_mean)
        return np.concatenate([W1sqrt.dot(r1), W2sqrt.dot(r2), prior_vec])

    res = least_squares(joint_obj, p0, method="lm", xtol=1e-8, ftol=1e-8, gtol=1e-8)
    p_hat = res.x
    eps = np.maximum(1e-8, np.abs(p_hat) * 1e-7 + 1e-10)
    J2 = np.zeros((2, 6))
    for i in range(6):
        dp = p_hat.copy()
        dp[i] += eps[i]
        ra_p, dec_p, *_ = forward_predict_RADEC(
            (dp[0], dp[1]), dp[2:], obs1, obs2, site1, site2, propagate_state_kepler
        )
        dp[i] -= 2 * eps[i]
        ra_m, dec_m, *_ = forward_predict_RADEC(
            (dp[0], dp[1]), dp[2:], obs1, obs2, site1, site2, propagate_state_kepler
        )
        J2[0, i] = angle_diff(ra_p, ra_m) / (2 * eps[i])
        J2[1, i] = (dec_p - dec_m) / (2 * eps[i])
    J2[0, :] *= np.cos(obs2_dec)
    J1 = np.zeros((2, 6))
    J1[0, 0] = np.cos(obs1_dec)
    J1[1, 1] = 1.0
    W1sqrt = la.cholesky(W1)
    W2sqrt = la.cholesky(W2)
    eigvals, eigvecs = la.eigh(prior_inv)
    eigvals[eigvals < 0] = 0.0
    sqrt_prior_inv = eigvecs.dot(np.diag(np.sqrt(eigvals))).dot(eigvecs.T)
    J_big = np.vstack([W1sqrt.dot(J1), W2sqrt.dot(J2), sqrt_prior_inv])
    Sigma = la.inv(J_big.T.dot(J_big))
    ps = np.random.multivariate_normal(p_hat, Sigma, size=N)
    workers = min(MAX_WORKERS, os.cpu_count() or 1) if workers is None else min(workers, MAX_WORKERS)
    payloads = [(p, obs1, obs2, site1, site2, USE_FULL_PHYSICS) for p in ps]
    samples = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for out in executor.map(_eval_variant_b_one, payloads, chunksize=_chunksize(N, workers)):
            if out is not None:
                samples.append(out)
    return samples


# ------------------------------
# JPL Horizons fetch
# ------------------------------
def fetch_jpl_state(body_id, times, center="@sun", id_type="smallbody"):
    results = {}
    for t in times:
        try:
            obj = Horizons(id=body_id, location=center, epochs=t.tdb.jd, id_type=id_type)
            vec = obj.vectors()
            x = float(vec["x"][0]) * u.au.to(u.km)
            y = float(vec["y"][0]) * u.au.to(u.km)
            z = float(vec["z"][0]) * u.au.to(u.km)
            vx = float(vec["vx"][0]) * (u.au / u.day).to(u.km / u.s)
            vy = float(vec["vy"][0]) * (u.au / u.day).to(u.km / u.s)
            vz = float(vec["vz"][0]) * (u.au / u.day).to(u.km / u.s)
            results[round(float(t.tdb.jd), 6)] = (np.array([x, y, z]), np.array([vx, vy, vz]))
        except Exception as e:
            results[round(float(t.tdb.jd), 6)] = (None, None)
            print("Horizons fetch failed for", t, ":", e)
    return results


# ------------------------------
# Metrics + plotting
# ------------------------------
def angular_sep(ra1, dec1, ra2, dec2):
    v1 = hat_u_from_radec(ra1, dec1)
    v2 = hat_u_from_radec(ra2, dec2)
    cosang = np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1)
    return np.arccos(cosang)


def summarize_and_plot(samples_A, samples_B, jpl_states, obs2_ra, obs2_dec):
    def summarize(samples, label):
        ang_res_obs2 = []
        pos_res_em1 = []
        pos_res_em2 = []
        for s in samples:
            ang = angular_sep(s["ra_pred"], s["dec_pred"], obs2_ra, obs2_dec)
            ang_res_obs2.append(ang * 206265.0)
            key1 = round(float(s["t_em1"].tdb.jd), 6)
            key2 = round(float(s["t_em2"].tdb.jd), 6)
            r1_jpl, _ = jpl_states.get(key1, (None, None))
            r2_jpl, _ = jpl_states.get(key2, (None, None))
            pos_res_em1.append(np.nan if r1_jpl is None else np.linalg.norm(s["r1"] - r1_jpl))
            pos_res_em2.append(np.nan if r2_jpl is None else np.linalg.norm(s["r2"] - r2_jpl))
        print(f"--- {label} ---")
        print(
            "Obs2 angular residual arcsec: mean %.3f med %.3f std %.3f"
            % (np.nanmean(ang_res_obs2), np.nanmedian(ang_res_obs2), np.nanstd(ang_res_obs2))
        )
        print(
            "Pos residual @ em1 km: mean %.3f med %.3f std %.3f"
            % (np.nanmean(pos_res_em1), np.nanmedian(pos_res_em1), np.nanstd(pos_res_em1))
        )
        print(
            "Pos residual @ em2 km: mean %.3f med %.3f std %.3f"
            % (np.nanmean(pos_res_em2), np.nanmedian(pos_res_em2), np.nanstd(pos_res_em2))
        )
        return np.array(ang_res_obs2), np.array(pos_res_em1), np.array(pos_res_em2)

    A_ang, A_p1, A_p2 = summarize(samples_A, "Variant A")
    B_ang, B_p1, B_p2 = summarize(samples_B, "Variant B")

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(A_ang, bins=40, alpha=0.6, label="A")
    plt.hist(B_ang, bins=40, alpha=0.6, label="B")
    plt.xlabel("Obs2 angular residual (arcsec)")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.hist(A_p2[~np.isnan(A_p2)], bins=40, alpha=0.6, label="A em2 km")
    plt.hist(B_p2[~np.isnan(B_p2)], bins=40, alpha=0.6, label="B em2 km")
    plt.xlabel("pos residual @ em2 (km)")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ------------------------------
# Main CLI
# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to obs.csv")
    parser.add_argument(
        "--object",
        required=True,
        help='Object identifier for Horizons (e.g., "1", "1 Ceres", "Ceres")',
    )
    parser.add_argument(
        "--obs1-index", type=int, default=None, help="Index among rows for chosen object (0-based)"
    )
    parser.add_argument(
        "--obs2-index", type=int, default=None, help="Index among rows for chosen object"
    )
    parser.add_argument(
        "--site",
        default=None,
        help="Fallback observer site code when obs.csv has no site column (optional)",
    )
    parser.add_argument(
        "--site1",
        default=None,
        help="Fallback observer site for obs1 when obs.csv has no site value (optional)",
    )
    parser.add_argument(
        "--site2",
        default=None,
        help="Fallback observer site for obs2 when obs.csv has no site value (optional)",
    )
    parser.add_argument("--N", type=int, default=200, help="Number of samples per variant")
    parser.add_argument("--workers", type=int, default=None, help="Max parallel workers (default: CPU count up to 50)")
    physics_group = parser.add_mutually_exclusive_group()
    physics_group.add_argument(
        "--full-physics",
        action="store_true",
        help="Use full-physics propagation (default).",
    )
    physics_group.add_argument(
        "--no-full-physics",
        action="store_true",
        help="Disable full-physics propagation.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    colmap = detect_columns(df)
    if colmap["time"] is None or colmap["ra"] is None or colmap["dec"] is None:
        raise SystemExit(
            "Could not detect required columns (time,ra,dec). Columns found: %s"
            % list(df.columns)
        )

    obj_col = colmap["obj"]
    time_col = colmap["time"]
    ra_col = colmap["ra"]
    dec_col = colmap["dec"]

    if obj_col is None:
        df_obj = df.copy()
    else:
        df_obj = df[df[obj_col].astype(str).str.contains(str(args.object), case=False, na=False)].copy()
    if df_obj.shape[0] < 2:
        raise SystemExit("Less than two observations found for object %s" % args.object)

    df_obj["parsed_time"] = pd.to_datetime(df_obj[time_col])
    df_obj.sort_values(by="parsed_time", inplace=True)

    if args.obs1_index is None:
        row1 = df_obj.iloc[0]
    else:
        row1 = df_obj.iloc[args.obs1_index]
    if args.obs2_index is None:
        row2 = df_obj.iloc[1]
    else:
        row2 = df_obj.iloc[args.obs2_index]

    obs1_ra, obs1_dec = parse_ra_dec(row1[ra_col], row1[dec_col])
    obs2_ra, obs2_dec = parse_ra_dec(row2[ra_col], row2[dec_col])
    obs1_time = Time(row1["parsed_time"].to_pydatetime(), scale="utc")
    obs2_time = Time(row2["parsed_time"].to_pydatetime(), scale="utc")

    sigma_ra_col = colmap["sigma_ra"]
    sigma_dec_col = colmap["sigma_dec"]
    obs1_sigma_ra = (
        float(row1[sigma_ra_col])
        if sigma_ra_col in row1 and not pd.isna(row1[sigma_ra_col])
        else 0.5
    )
    obs1_sigma_dec = (
        float(row1[sigma_dec_col])
        if sigma_dec_col in row1 and not pd.isna(row1[sigma_dec_col])
        else 0.5
    )
    obs2_sigma_ra = (
        float(row2[sigma_ra_col])
        if sigma_ra_col in row2 and not pd.isna(row2[sigma_ra_col])
        else 0.5
    )
    obs2_sigma_dec = (
        float(row2[sigma_dec_col])
        if sigma_dec_col in row2 and not pd.isna(row2[sigma_dec_col])
        else 0.5
    )

    site_default = args.site if args.site is not None else "500"

    def _pick_site(row, arg_site):
        if colmap["site"] and colmap["site"] in row and not pd.isna(row[colmap["site"]]):
            return row[colmap["site"]]
        if arg_site is not None:
            return arg_site
        return site_default

    site1 = _pick_site(row1, args.site1)
    site2 = _pick_site(row2, args.site2)

    obs1 = obs1_time
    obs1.meta = {"sigma_ra_arcsec": obs1_sigma_ra, "sigma_dec_arcsec": obs1_sigma_dec}
    obs2 = obs2_time
    obs2.meta = {"sigma_ra_arcsec": obs2_sigma_ra, "sigma_dec_arcsec": obs2_sigma_dec}

    print("Selected object rows:")
    print(row1.to_dict())
    print(row2.to_dict())
    print("Observer sites:", site1, site2)
    print("obs1 ra/dec (deg):", np.rad2deg(obs1_ra), np.rad2deg(obs1_dec))
    print("obs2 ra/dec (deg):", np.rad2deg(obs2_ra), np.rad2deg(obs2_dec))

    USE_FULL_PHYSICS = not args.no_full_physics

    print("Sampling Variant A ...")
    samples_A = sample_variant_A(
        obs1,
        obs2,
        site1,
        site2,
        obs1_ra,
        obs1_dec,
        obs2_ra,
        obs2_dec,
        obs1_sigma_ra,
        obs1_sigma_dec,
        obs2_sigma_ra,
        obs2_sigma_dec,
        N=args.N,
        workers=args.workers,
    )

    print("Sampling Variant B ...")
    samples_B = optimize_joint_and_sample(
        obs1,
        obs2,
        site1,
        site2,
        obs1_ra,
        obs1_dec,
        obs2_ra,
        obs2_dec,
        obs1_sigma_ra,
        obs1_sigma_dec,
        obs2_sigma_ra,
        obs2_sigma_dec,
        N=args.N,
        workers=args.workers,
    )

    times_set = {}
    for s in samples_A + samples_B:
        times_set[round(float(s["t_em1"].tdb.jd), 6)] = s["t_em1"]
        times_set[round(float(s["t_em2"].tdb.jd), 6)] = s["t_em2"]
    times_list = list(times_set.values())
    print("Querying JPL/Horizons for %d unique emission times ..." % len(times_list))
    jpl_states = fetch_jpl_state(args.object, times_list, center="@sun")
    summarize_and_plot(samples_A, samples_B, jpl_states, obs2_ra, obs2_dec)
