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
import functools
import math
import json
import time
import numpy as np
import scipy.linalg as la
from scipy.optimize import least_squares
from scipy.special import gammaln, logsumexp
import pandas as pd
from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.utils import iers
import matplotlib.pyplot as plt
from astroquery.jplhorizons import Horizons
from concurrent.futures import ProcessPoolExecutor, as_completed
try:
    import numba as _numba
except Exception:
    _numba = None

import neotube.rng as nrng

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
GM_SUN = 1.32712440018e11  # km^3/s^2
GM_EARTH = 398600.4418  # km^3/s^2
AU_KM = 149597870.7
HORIZONS_JD_DECIMALS = 6

# New defaults for rho prior / triangulation proposals and velocity scaling
WEIGHT_RHO_TRI = 0.15
WEIGHT_FLAT_RHO = 0.10
SIGMA_LOGRHO_TRI = 1.0
DEFAULT_F_SIGMA_V = 1.0
FLAT_V_F = 1.8
W_SSO_PHYS = 0.85
ENERGY_PENALTY_LOG = -400.0
PRIOR_SCALE = 0.25
NULL_KAPPA = 2.0
NULL_TAIL_DF = 3.0
NULL_S_CLIP = 6.0
CIRCULAR_WEIGHT_SSO = 0.80
CIRCULAR_WEIGHT_OTHER = 0.20

SSO_RHO_COMPONENTS = [
    ("NEO", 0.2 * AU_KM, 0.4, 0.10),
    ("EarthVicinity", 1.0 * AU_KM, 0.6, 0.30),
    ("MainBelt", 2.8 * AU_KM, 0.5, 0.35),
    ("Jupiter", 5.2 * AU_KM, 0.3, 0.05),
    ("TNO", 30.0 * AU_KM, 0.4, 0.10),
    ("Comet", 5.0 * AU_KM, 1.5, 0.05),
]


def _chunksize(n_items, n_workers):
    return max(1, n_items // max(1, n_workers * 4))


def worker_init(openblas_threads=1):
    """Initialize worker process state and limit BLAS threads."""
    os.environ["OPENBLAS_NUM_THREADS"] = str(openblas_threads)
    os.environ["OMP_NUM_THREADS"] = str(openblas_threads)
    os.environ["MKL_NUM_THREADS"] = str(openblas_threads)
    from astropy.utils import iers
    iers.conf.auto_download = False
    iers.conf.iers_degraded_accuracy = "warn"
    try:
        iers.IERS_Auto.open()
    except Exception:
        pass
    import astropy.coordinates  # warm caches
    import neotube.propagate as _prop  # noqa: F401


# ------------------------------
# Chunked execution helpers (to keep many cores busy)
# ------------------------------
def choose_workers_and_chunk(
    n_samples,
    requested_workers=None,
    max_workers=None,
    target_per_worker=200,
    batches_per_worker=4,
):
    """Pick workers/chunk size so each worker does enough full-physics work."""
    if max_workers is None:
        max_workers = min(50, os.cpu_count() or 1)
    if requested_workers and requested_workers > 0:
        workers = min(max_workers, max(1, min(int(requested_workers), n_samples)))
    else:
        workers = min(max_workers, max(1, n_samples // max(1, target_per_worker)))
    total_chunks = max(1, workers * max(1, batches_per_worker))
    chunk_size = max(1, int(math.ceil(float(n_samples) / total_chunks)))
    return workers, chunk_size


def make_chunks(seq, chunk_size):
    """Yield consecutive chunks from seq."""
    for i in range(0, len(seq), chunk_size):
        yield seq[i : i + chunk_size]


def _process_chunk_fullphysics(chunk_payload):
    """Process a chunk of payloads in one worker."""
    results = []
    for p in chunk_payload:
        try:
            out = _sample_variant_a_one(p)
        except Exception as exc:
            import traceback

            seed_info = None
            if isinstance(p, (list, tuple)) and p:
                seed_info = p[0]
            print(f"EXCEPTION in _sample_variant_a_one (seed={seed_info}): {exc}", flush=True)
            traceback.print_exc()
            out = None
        results.append(out)
    return results


# ------------------------------
# Utility: Gaussian / multivariate-t, mixture, rho-dependent prior scale
# ------------------------------
def logpdf_gauss(x, mu, Sigma):
    """Log pdf of multivariate Gaussian N(mu, Sigma)."""
    k = x.size
    cf = la.cho_factor(Sigma, lower=True)
    diff = x - mu
    sol = la.cho_solve(cf, diff)
    logdet = 2.0 * np.sum(np.log(np.diag(cf[0])))
    return -0.5 * (k * np.log(2.0 * np.pi) + logdet + diff.dot(sol))


def logpdf_mvt(x, mu, Sigma, nu):
    """Log pdf of multivariate Student-t with df=nu, location mu, scale Sigma."""
    k = x.size
    diff = x - mu
    try:
        cf = la.cho_factor(Sigma, lower=True)
        sol = la.cho_solve(cf, diff)
        quad = float(diff.dot(sol))
        logdet = 2.0 * np.sum(np.log(np.diag(cf[0])))
    except Exception:
        invS = np.linalg.inv(Sigma)
        quad = float(diff.dot(invS).dot(diff))
        logdet = np.log(np.linalg.det(Sigma))
    a = gammaln((nu + k) / 2.0) - gammaln(nu / 2.0)
    b = -0.5 * (k * np.log(nu * np.pi) + logdet)
    c = - (nu + k) / 2.0 * np.log(1.0 + quad / nu)
    return float(a + b + c)


def logpdf_t_univariate(x, nu, loc=0.0, scale=1.0):
    """Log pdf of univariate Student-t with df=nu, location loc and scale."""
    dx = (float(x) - float(loc)) / float(scale)
    a = gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0)
    b = -0.5 * (np.log(nu * np.pi) + 2.0 * np.log(float(scale)))
    c = -0.5 * (nu + 1.0) * np.log(1.0 + (dx * dx) / float(nu))
    return float(a + b + c)


def sample_mvt(hat, Sigma, nu, s=1.0, rng=None):
    """Draw from multivariate Student-t with scale s^2 * Sigma."""
    rng = nrng.ensure_rng(rng)
    z = rng.multivariate_normal(np.zeros_like(hat), Sigma)
    g = rng.gamma(nu / 2.0, 2.0 / nu)
    return hat + s * z / np.sqrt(g / nu)


def sample_mixture(
    hat_psi,
    Sigma_psi,
    weight_gauss=MIXTURE_WEIGHT_GAUSS,
    weight_heavy=1.0 - MIXTURE_WEIGHT_GAUSS,
    nu=MIXTURE_T_NU,
    s=MIXTURE_T_SCALE,
    rng=None,
):
    rng = np.random.default_rng() if rng is None else rng
    if rng.random() < weight_gauss:
        return rng.multivariate_normal(hat_psi, Sigma_psi), "G"
    return sample_mvt(hat_psi, Sigma_psi, nu, s=s, rng=rng), "T"


def log_proposal_mixture(
    x,
    hat_psi,
    Sigma_psi,
    weight_gauss=MIXTURE_WEIGHT_GAUSS,
    weight_heavy=1.0 - MIXTURE_WEIGHT_GAUSS,
    nu=MIXTURE_T_NU,
    s=MIXTURE_T_SCALE,
):
    """Log proposal density of the mixture (stable via logsumexp)."""
    log_g = np.log(weight_gauss) + logpdf_gauss(x, hat_psi, Sigma_psi)
    Sigma_t = (s**2) * Sigma_psi
    log_h = np.log(weight_heavy) + logpdf_mvt(x, hat_psi, Sigma_t, nu)
    return float(logsumexp([log_g, log_h]))


def sigma_v_from_rhelio(r_helio_km, f=1.5, r_min_km=1e5):
    """Compute sigma_v (km/s) using heliocentric radius."""
    r = max(float(r_helio_km), r_min_km)
    sigma_v = f * np.sqrt(GM_SUN / r)
    if r < 5.0 * 6378.1363:
        sigma_v = f * np.sqrt(GM_EARTH / max(r, 1.0))
    return float(sigma_v)

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
@functools.lru_cache(maxsize=4096)
def _observer_posvel_cached_jd(site, jd):
    """Cached Earth/Sun barycentric delta for (site, jd) in TDB."""
    from astropy.coordinates import get_body_barycentric_posvel

    def _as_vel_kms(obj):
        if hasattr(obj, "d_xyz"):
            return obj.d_xyz.to_value(u.km / u.s)
        return obj.xyz.to_value(u.km / u.s)

    t = Time(jd, format="jd", scale="tdb")
    pb_earth = get_body_barycentric_posvel("earth", t)
    pb_sun = get_body_barycentric_posvel("sun", t)
    r_earth = (pb_earth[0].xyz - pb_sun[0].xyz).to_value(u.km).flatten()
    v_earth = (_as_vel_kms(pb_earth[1]) - _as_vel_kms(pb_sun[1])).flatten()
    return r_earth, v_earth


@functools.lru_cache(maxsize=4096)
def _site_state_cached_jd(site, jd):
    """Cached site offset for (site, jd) in TDB."""
    if _site_states is None:
        return None
    t = Time(jd, format="jd", scale="tdb")
    pos_geo, vel_geo = _site_states([t], [site], allow_unknown_site=True)
    pos = np.asarray(pos_geo[0], dtype=float)
    vel = np.asarray(vel_geo[0], dtype=float)
    return pos, vel


def observer_posvel(site, t):
    """
    Return observer position and velocity in km, km/s in ICRS-like frame.
    """
    jd_key = round(float(t.tdb.jd), 8)
    r_earth, v_earth = _observer_posvel_cached_jd(site, jd_key)
    if _site_states is not None:
        try:
            cached = _site_state_cached_jd(site, jd_key)
            if cached is None:
                return r_earth, v_earth
            pos_geo, vel_geo = cached
            return r_earth + pos_geo, v_earth + vel_geo
        except Exception:
            return r_earth, v_earth
    return r_earth, v_earth


def observer_posvel_jd(site, jd_tdb):
    """Observer position/velocity using a TDB JD float."""
    jd_key = round(float(jd_tdb), 8)
    r_earth, v_earth = _observer_posvel_cached_jd(site, jd_key)
    if _site_states is not None:
        try:
            cached = _site_state_cached_jd(site, jd_key)
            if cached is None:
                return r_earth, v_earth
            pos_geo, vel_geo = cached
            return r_earth + pos_geo, v_earth + vel_geo
        except Exception:
            return r_earth, v_earth
    return r_earth, v_earth


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


def triangulate_rho(O1, u1, O2, u2):
    """Return (rho1, rho2) that minimize distance between two rays."""
    r = O2 - O1
    a = float(np.dot(u1, u1))
    b = float(np.dot(u1, u2))
    c = float(np.dot(u2, u2))
    A = np.array([[a, -b], [-b, c]], dtype=float)
    rhs = np.array([float(np.dot(u1, r)), -float(np.dot(u2, r))], dtype=float)
    try:
        rho1, rho2 = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        return None, None
    return float(rho1), float(rho2)


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


if _numba is not None:
    _angle_diff_nb = _numba.njit(cache=True)(angle_diff)

    @_numba.njit(cache=True)
    def _radec_from_vector_nb(vec):
        x = vec[0]
        y = vec[1]
        z = vec[2]
        r = (x * x + y * y + z * z) ** 0.5
        dec = np.arcsin(z / r)
        ra = np.arctan2(y, x) % (2 * np.pi)
        return ra, dec
else:
    _angle_diff_nb = angle_diff
    _radec_from_vector_nb = radec_from_vector


# ------------------------------
# Light-time / emission time solvers
# ------------------------------
def solve_emission_time_for_obs(
    t_obs, rho_km, hat_u, site, tol=1e-8, maxit=30, max_step_s=3600.0
):
    """Newton solve for t_em so that photons emitted from r = r_obs(t_em)+rho*hat_u arrive at t_obs."""
    t_obs_jd = float(t_obs.tdb.jd)
    t_em_jd = t_obs_jd + (-rho_km / C) / 86400.0
    for _ in range(maxit):
        r_obs_tobs, _ = observer_posvel_jd(site, t_obs_jd)
        r_obs_tem, v_obs_tem = observer_posvel_jd(site, t_em_jd)
        r_target = r_obs_tem + rho_km * hat_u
        dvec = r_obs_tobs - r_target
        D = np.linalg.norm(dvec)
        F = (t_obs_jd - t_em_jd) * 86400.0 - D / C
        if abs(F) < tol:
            break
        dFdt = -1.0 + (np.dot(dvec, v_obs_tem) / (C * D))
        if abs(dFdt) < 1e-12:
            break
        dt = F / dFdt
        dt = float(np.clip(dt, -max_step_s, max_step_s))
        t_em_jd = t_em_jd + (-dt) / 86400.0
    return Time(t_em_jd, format="jd", scale="tdb"), r_obs_tem, v_obs_tem


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
    obs2_jd = float(obs2.tdb.jd)
    r_obs_tobs2, _ = observer_posvel_jd(site, obs2_jd)
    D0 = np.linalg.norm(r_obs_tobs2 - r1)
    t_em_jd = obs2_jd + (-D0 / C) / 86400.0
    r_t = r1
    v_t = v1
    for _ in range(maxit):
        r_t, v_t = _propagate_state_to_time(
            r1, v1, t_em1, Time(t_em_jd, format="jd", scale="tdb"), propagate_fn
        )
        r_obs_tobs2, _ = observer_posvel_jd(site, obs2_jd)
        dvec = r_obs_tobs2 - r_t
        D = np.linalg.norm(dvec)
        F = (obs2_jd - t_em_jd) * 86400.0 - D / C
        if abs(F) < tol:
            break
        dFdt = -1.0 + (np.dot(dvec, v_t) / (C * D))
        if abs(dFdt) < 1e-12:
            break
        dt = F / dFdt
        dt = float(np.clip(dt, -max_step_s, max_step_s))
        t_em_jd = t_em_jd + (-dt) / 86400.0
    return Time(t_em_jd, format="jd", scale="tdb"), r_t, v_t


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
    dvec = r2 - r_obs_tobs2
    ra_pred, dec_pred = _radec_from_vector_nb(dvec)
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
    obs2_ra,
    obs2_dec,
    f_sigma_v=1.5,
    psi_prior_mean=None,
):
    if psi_prior_mean is None:
        psi_prior_mean = np.zeros(3, dtype=float)

    hat_u = hat_u_from_radec(Gamma[0], Gamma[1])
    try:
        _, r_obs_em1_tmp, _ = solve_emission_time_for_obs(
            obs1, np.exp(logrho), hat_u, site1
        )
    except Exception:
        r_obs_em1_tmp, _ = observer_posvel(site1, obs1)
    r1_tmp = r_obs_em1_tmp + np.exp(logrho) * hat_u
    r_helio = np.linalg.norm(r1_tmp)
    sigma_v = sigma_v_from_rhelio(r_helio, f=f_sigma_v)
    sigma_rdot = sigma_v
    prior_cov = np.diag([sigma_rdot**2, sigma_v**2, sigma_v**2]) * PRIOR_SCALE
    prior_Ppsi_inv = la.inv(prior_cov)
    sqrt_prior_inv = la.cholesky(prior_Ppsi_inv)
    Wsqrt = la.cholesky(W2)

    def residuals(psi):
        theta = np.concatenate(([logrho], psi))
        try:
            ra_pred, dec_pred, *_ = forward_predict_RADEC(
                Gamma, theta, obs1, obs2, site1, site2, propagate_surrogate
            )
        except Exception:
            return np.array([1e6, 1e6, 1e6, 1e6, 1e6])
        res = np.array([
            angle_diff(ra_pred, obs2_ra) * np.cos(obs2_dec),
            (dec_pred - obs2_dec),
        ])
        prior_res = sqrt_prior_inv.dot(psi - psi_prior_mean)
        return np.concatenate([Wsqrt.dot(res), prior_res])

    dt = max(1.0, (obs2.tdb.jd - obs1.tdb.jd) * 86400.0)
    dalpha = angle_diff(obs2_ra, Gamma[0])
    ddelta = (obs2_dec - Gamma[1])
    d_alpha_dt = dalpha / dt
    d_delta_dt = ddelta / dt
    rho = np.exp(logrho)
    ve0 = rho * d_alpha_dt * np.cos(Gamma[1])
    vn0 = rho * d_delta_dt
    psi0_heur = np.array([0.0, ve0, vn0])
    psi0 = 0.7 * psi_prior_mean + 0.3 * psi0_heur
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
    return hat_psi, Sigma_psi, Jpsi


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
    rng = nrng.make_rng(seed)
    S1 = np.diag([(obs1_sigma_ra / 206265.0) ** 2, (obs1_sigma_dec / 206265.0) ** 2])
    S2 = np.diag([(obs2_sigma_ra / 206265.0) ** 2, (obs2_sigma_dec / 206265.0) ** 2])
    W2 = la.inv(S2)

    dtheta_tan = rng.multivariate_normal(np.zeros(2), S1)
    alpha_s = obs1_ra + dtheta_tan[0] / cos_dec_for_ra_div(obs1_dec)
    delta_s = obs1_dec + dtheta_tan[1]
    Gamma = (alpha_s, delta_s)
    rho_min = 1e3
    rho_max = 1e10
    try:
        hat_u1 = hat_u_from_radec(obs1_ra, obs1_dec)
        hat_u2 = hat_u_from_radec(obs2_ra, obs2_dec)
        r_obs_t1, _ = observer_posvel(site1, obs1)
        r_obs_t2, _ = observer_posvel(site2, obs2)
        rho1_tri, _ = triangulate_rho(r_obs_t1, hat_u1, r_obs_t2, hat_u2)
        rho_tri = rho1_tri
    except Exception:
        rho_tri = None

    u = rng.random()
    if rho_tri is not None and u < WEIGHT_RHO_TRI:
        logrho_center = np.log(np.clip(rho_tri, rho_min, rho_max))
        logrho = float(rng.normal(loc=logrho_center, scale=SIGMA_LOGRHO_TRI))
        rho_prior_component = "tri"
    elif u < (WEIGHT_RHO_TRI + WEIGHT_FLAT_RHO):
        rho = float(rng.random() * (rho_max - rho_min) + rho_min)
        logrho = np.log(rho)
        rho_prior_component = "flat_linear"
    else:
        names = [c[0] for c in SSO_RHO_COMPONENTS]
        centers = [c[1] for c in SSO_RHO_COMPONENTS]
        sigs = [c[2] for c in SSO_RHO_COMPONENTS]
        comp_weights = np.array([c[3] for c in SSO_RHO_COMPONENTS], dtype=float)
        comp_weights /= comp_weights.sum()
        comp_idx = rng.choice(len(SSO_RHO_COMPONENTS), p=comp_weights)
        center_rho = centers[comp_idx]
        sigma_log = sigs[comp_idx]
        logrho = float(rng.normal(loc=np.log(center_rho), scale=sigma_log))
        logrho = float(np.clip(logrho, np.log(rho_min), np.log(rho_max)))
        rho_prior_component = f"sso_{names[comp_idx]}"

    if rho_prior_component.startswith("sso_"):
        circular_w = CIRCULAR_WEIGHT_SSO
    else:
        circular_w = CIRCULAR_WEIGHT_OTHER
    vel_mode = "circular" if rng.random() < circular_w else "flat"
    f_sigma_v_sample = DEFAULT_F_SIGMA_V if vel_mode == "circular" else FLAT_V_F
    hat_u = hat_u_from_radec(Gamma[0], Gamma[1])
    try:
        _, r_obs_em1_tmp, v_obs_em1_tmp = solve_emission_time_for_obs(
            obs1, np.exp(logrho), hat_u, site1
        )
    except Exception:
        r_obs_em1_tmp, v_obs_em1_tmp = observer_posvel(site1, obs1)
    r1_tmp = r_obs_em1_tmp + np.exp(logrho) * hat_u
    r_helio_tmp = np.linalg.norm(r1_tmp)
    v_circ = np.sqrt(GM_SUN / max(r_helio_tmp, 1.0))

    dt = max(1.0, (obs2.tdb.jd - obs1.tdb.jd) * 86400.0)
    dalpha = angle_diff(obs2_ra, Gamma[0])
    ddelta = (obs2_dec - Gamma[1])
    d_alpha_dt = dalpha / dt
    d_delta_dt = ddelta / dt
    rho_val = float(np.exp(logrho))
    ve0 = rho_val * d_alpha_dt * np.cos(Gamma[1])
    vn0 = rho_val * d_delta_dt
    e_alpha, e_delta = tangent_basis(Gamma[0], Gamma[1])

    v_guess_helio = v_obs_em1_tmp + ve0 * e_alpha + vn0 * e_delta
    r1_hat = r1_tmp / max(np.linalg.norm(r1_tmp), 1e-12)
    v_proj = v_guess_helio - np.dot(v_guess_helio, r1_hat) * r1_hat
    vproj_norm = np.linalg.norm(v_proj)
    if vproj_norm < 1e-8:
        zaxis = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(r1_hat, zaxis)) > 0.9:
            zaxis = np.array([1.0, 0.0, 0.0])
        v_proj = np.cross(r1_hat, zaxis)
        vproj_norm = max(np.linalg.norm(v_proj), 1e-12)
    vproj_unit = v_proj / vproj_norm
    if np.dot(vproj_unit, v_guess_helio) < 0:
        vproj_unit = -vproj_unit
    v1_circ_mean = v_circ * vproj_unit
    v_topo_mean = v1_circ_mean - v_obs_em1_tmp
    dotrho_mean = float(np.dot(v_topo_mean, hat_u))
    ve_mean = float(np.dot(v_topo_mean, e_alpha))
    vn_mean = float(np.dot(v_topo_mean, e_delta))
    psi_prior_mean = np.array([dotrho_mean, ve_mean, vn_mean], dtype=float)
    if psi_prior_mean is None:
        psi_prior_mean = np.zeros(3, dtype=float)
    if not np.isfinite(logrho):
        return None
    try:
        hat_psi, Sigma_psi, Jpsi = optimize_conditional_psi(
            Gamma,
            logrho,
            obs1,
            obs2,
            site1,
            site2,
            propagate_state_kepler,
            W2,
            obs2_ra,
            obs2_dec,
            f_sigma_v=f_sigma_v_sample,
            psi_prior_mean=psi_prior_mean,
        )
    except Exception:
        return None

    # --- Begin diagnostic logging (insert in _sample_variant_a_one after hat_psi, Sigma_psi, Jpsi are available) ---
    import csv
    import os

    DEBUG_LOGPATH = os.path.join(
        os.path.dirname(__file__), "..", "runs", "ceres-ground-test", "debug_nullspace.csv"
    )
    DEBUG_LOGPATH = os.path.abspath(DEBUG_LOGPATH)
    os.makedirs(os.path.dirname(DEBUG_LOGPATH), exist_ok=True)

    def _dump_debug_row(
        seed,
        rho_component,
        vel_mode,
        psi_prior_mean,
        hat_psi,
        Jpsi,
        Sigma_psi,
        mu_s=None,
        sigma_s2=None,
        s_phys=None,
        eps_hat=None,
    ):
        # compute data residual at hat_psi
        try:
            theta_hat = np.concatenate(([logrho], hat_psi))
            ra_pred_hat, dec_pred_hat, t_em1_hat, t_em2_hat, r1_hat, v1_hat, _, _ = (
                forward_predict_RADEC(
                    Gamma, theta_hat, obs1, obs2, site1, site2, propagate_state_kepler
                )
            )
            res_vec = np.array(
                [angle_diff(ra_pred_hat, obs2_ra) * np.cos(obs2_dec), (dec_pred_hat - obs2_dec)]
            )
            Wsqrt = la.cholesky(W2)
            data_res_norm = float(np.linalg.norm(Wsqrt.dot(res_vec)))
        except Exception:
            data_res_norm = float("nan")
        # prior residual norm
        try:
            sigma_v_tmp = sigma_v_from_rhelio(np.linalg.norm(r1_hat), f=DEFAULT_F_SIGMA_V)
            _ = np.diag([sigma_v_tmp**2, sigma_v_tmp**2, sigma_v_tmp**2])
            prior_res = hat_psi - psi_prior_mean
            prior_res_norm = float(np.linalg.norm(prior_res))
        except Exception:
            prior_res_norm = float("nan")

        # Jpsi singular values
        try:
            Uj, Svals, Vt = np.linalg.svd(Jpsi, full_matrices=True)
            s1, s2 = float(Svals[0]), float(Svals[1])
        except Exception:
            s1, s2 = float("nan"), float("nan")

        # eps at hat_psi
        try:
            eps_hat = 0.5 * np.dot(v1_hat, v1_hat) - GM_SUN / np.linalg.norm(r1_hat)
        except Exception:
            eps_hat = float("nan")

        header = [
            "seed",
            "rho_component",
            "vel_mode",
            "psi_prior_mean_dotrho",
            "psi_prior_mean_ve",
            "psi_prior_mean_vn",
            "hat_dotrho",
            "hat_ve",
            "hat_vn",
            "data_res_norm",
            "prior_res_norm",
            "prior_over_data_ratio",
            "Jpsi_s1",
            "Jpsi_s2",
            "mu_s",
            "sigma_s2",
            "s_phys",
            "eps_hat",
        ]
        newfile = not os.path.exists(DEBUG_LOGPATH)
        with open(DEBUG_LOGPATH, "a", newline="") as f:
            w = csv.writer(f)
            if newfile:
                w.writerow(header)
            row = [
                int(seed),
                rho_component,
                vel_mode,
                float(psi_prior_mean[0]),
                float(psi_prior_mean[1]),
                float(psi_prior_mean[2]),
                float(hat_psi[0]),
                float(hat_psi[1]),
                float(hat_psi[2]),
                data_res_norm,
                prior_res_norm,
                (prior_res_norm / max(1e-12, data_res_norm))
                if (not np.isnan(data_res_norm))
                else float("nan"),
                s1,
                s2,
                float(mu_s) if mu_s is not None else float("nan"),
                float(sigma_s2) if sigma_s2 is not None else float("nan"),
                float(s_phys) if s_phys is not None else float("nan"),
                float(eps_hat) if eps_hat is not None else float("nan"),
            ]
            w.writerow(row)

    _dump_debug_row(
        seed,
        rho_prior_component,
        vel_mode,
        psi_prior_mean,
        hat_psi,
        Jpsi,
        Sigma_psi,
        mu_s=locals().get("mu_s"),
        sigma_s2=locals().get("sigma_s2"),
        s_phys=locals().get("s_phys"),
    )
    # --- End diagnostic logging ---
    eval_fn = propagate_state if use_full_physics and propagate_state is not None else propagate_state_kepler

    def compute_log_target(psi):
        theta = np.concatenate(([logrho], psi))
        ra_pred, dec_pred, t_em1, t_em2, r1, v1, r2, v2 = forward_predict_RADEC(
            Gamma, theta, obs1, obs2, site1, site2, eval_fn
        )
        res = np.array([
            angle_diff(ra_pred, obs2_ra) * np.cos(obs2_dec),
            (dec_pred - obs2_dec),
        ])
        chi2 = res.T.dot(W2).dot(res)
        r_helio = np.linalg.norm(r1)
        sigma_v = sigma_v_from_rhelio(r_helio, f=f_sigma_v_sample)
        sigma_rdot = sigma_v
        prior_cov = np.diag([sigma_rdot**2, sigma_v**2, sigma_v**2]) * PRIOR_SCALE
        if vel_mode == "flat":
            log_g = logpdf_gauss(psi, psi_prior_mean, prior_cov)
            Sigma_t = (3.0**2) * prior_cov
            log_t = logpdf_mvt(psi, psi_prior_mean, Sigma_t, nu=3.0)
            log_prior = float(logsumexp([np.log(0.8) + log_g, np.log(0.2) + log_t]))
        else:
            log_prior = logpdf_gauss(psi, psi_prior_mean, prior_cov)
        eps = 0.5 * np.dot(v1, v1) - GM_SUN / np.linalg.norm(r1)
        if eps > 0.0:
            log_prior += ENERGY_PENALTY_LOG
        log_like = -0.5 * chi2
        return float(log_like + log_prior), (t_em1, t_em2, r1, v1, r2, v2, ra_pred, dec_pred)

    try:
        log_target_cur, state_cur = compute_log_target(hat_psi)
    except Exception:
        return None

    # ------------------------------
    # Nullspace-aware sampling of psi (hat_psi + V_r z + n s)
    # ------------------------------
    s_comp = None
    for attempt in range(3):
        try:
            Uj, Svals, Vt = np.linalg.svd(Jpsi, full_matrices=True)
            V = Vt.T
            V_r = V[:, :2]
            n = V[:, 2]
            U3 = np.column_stack([V_r, n])

            Sigma_w = U3.T.dot(Sigma_psi).dot(U3)
            A = Sigma_w[:2, :2].copy()
            b = Sigma_w[:2, 2].copy()
            c = float(Sigma_w[2, 2])
            A += 1e-12 * np.eye(2)

            L = la.cho_factor(A, lower=True)
            z = rng.multivariate_normal(np.zeros(2), A)
            Ainv_z = la.cho_solve(L, z)
            Ainv_b = la.cho_solve(L, b)
            mu_s = float(b.T.dot(Ainv_z))
            sigma_s2 = max(1e-12, c - float(b.T.dot(Ainv_b)))

            s_phys = float(n.dot(psi_prior_mean - hat_psi))

            if rho_prior_component.startswith("sso_"):
                w_phys = W_SSO_PHYS
                w_cond = 1.0 - W_SSO_PHYS - 0.03
                w_tail = 0.03
            elif rho_prior_component == "tri":
                w_phys, w_cond, w_tail = 0.25, 0.50, 0.25
            elif rho_prior_component == "flat_linear":
                w_phys, w_cond, w_tail = 0.20, 0.50, 0.30
            else:
                w_phys, w_cond, w_tail = 0.25, 0.50, 0.25
            if rho_prior_component == "sso_Comet":
                w_phys, w_cond, w_tail = 0.20, 0.30, 0.50
            w_sum = w_phys + w_cond + w_tail
            w_phys, w_cond, w_tail = w_phys / w_sum, w_cond / w_sum, w_tail / w_sum

            tau = max(1e-3, 0.75 * np.sqrt(sigma_s2))
            kappa = NULL_KAPPA
            nu = NULL_TAIL_DF

            u_comp = rng.random()
            if u_comp < w_cond:
                s = float(mu_s + np.sqrt(sigma_s2) * rng.normal())
                s_comp = "cond"
            elif u_comp < (w_cond + w_phys):
                s = float(s_phys + tau * rng.normal())
                s_comp = "phys"
            else:
                s = float(mu_s + np.sqrt(sigma_s2) * kappa * rng.standard_t(df=nu))
                s_comp = "tail"
            s_max = NULL_S_CLIP * np.sqrt(sigma_s2)
            if abs(s) > s_max:
                s = np.sign(s) * s_max

            w_vec = np.concatenate([z, np.array([s])])
            psi_star = hat_psi + U3.dot(w_vec)

            log_q_z = logpdf_gauss(z, np.zeros(2), A)
            log_q_cond_s = -0.5 * (
                np.log(2.0 * np.pi * sigma_s2) + ((s - mu_s) ** 2) / sigma_s2
            )
            log_q_phys_s = -0.5 * (
                np.log(2.0 * np.pi * (tau**2)) + ((s - s_phys) ** 2) / (tau**2)
            )
            log_q_tail_s = logpdf_t_univariate(
                s, nu, loc=mu_s, scale=np.sqrt(sigma_s2) * kappa
            )
            log_q_forward = float(
                logsumexp(
                    [
                        np.log(w_cond) + log_q_z + log_q_cond_s,
                        np.log(w_phys) + log_q_z + log_q_phys_s,
                        np.log(w_tail) + log_q_z + log_q_tail_s,
                    ]
                )
            )

            log_q_z0 = logpdf_gauss(np.zeros(2), np.zeros(2), A)
            log_q_cond_s0 = -0.5 * np.log(2.0 * np.pi * sigma_s2)
            log_q_phys_s0 = -0.5 * (
                np.log(2.0 * np.pi * (tau**2)) + ((0.0 - s_phys) ** 2) / (tau**2)
            )
            log_q_tail_s0 = logpdf_t_univariate(
                0.0, nu, loc=0.0, scale=np.sqrt(sigma_s2) * kappa
            )
            log_q_reverse = float(
                logsumexp(
                    [
                        np.log(w_cond) + log_q_z0 + log_q_cond_s0,
                        np.log(w_phys) + log_q_z0 + log_q_phys_s0,
                        np.log(w_tail) + log_q_z0 + log_q_tail_s0,
                    ]
                )
            )
            comp = "NullLaplace"
            break
        except Exception as exc:
            if attempt == 2:
                raise RuntimeError(f"Nullspace proposal failed: {exc}") from exc

    try:
        log_target_star, state_star = compute_log_target(psi_star)
    except Exception:
        return None

    log_acc = log_target_star + log_q_reverse - (log_target_cur + log_q_forward)
    if np.log(rng.random()) < log_acc:
        psi_use = psi_star
        state_use = state_star
        accepted = True
        accepted_comp = comp
    else:
        psi_use = hat_psi
        state_use = state_cur
        accepted = False
        accepted_comp = None

    t_em1, t_em2, r1, v1, r2, v2, ra_pred, dec_pred = state_use
    eps = 0.5 * np.dot(v1, v1) - GM_SUN / np.linalg.norm(r1)
    is_unbound_proposed = None
    try:
        _, _, r1_star, v1_star, _, _, _, _ = state_star
        eps_star = 0.5 * np.dot(v1_star, v1_star) - GM_SUN / np.linalg.norm(r1_star)
        is_unbound_proposed = bool(eps_star > 0.0)
    except Exception:
        is_unbound_proposed = None
    is_unbound_accepted = bool(eps > 0.0)

    theta_star = np.concatenate(([logrho], psi_use))
    return {
        "Gamma": Gamma,
        "logrho": logrho,
        "rho_prior_component": rho_prior_component,
        "vel_mode": vel_mode,
        "psi": psi_use,
        "s_component": s_comp,
        "theta": theta_star,
        "ra_pred": ra_pred,
        "dec_pred": dec_pred,
        "t_em1": t_em1,
        "t_em2": t_em2,
        "r1": r1,
        "v1": v1,
        "r2": r2,
        "v2": v2,
        "proposal_component": comp,
        "accepted_component": accepted_comp,
        "accepted": accepted,
        "is_unbound_proposed": is_unbound_proposed,
        "is_unbound_accepted": is_unbound_accepted,
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
    seed=50,
):
    rng = nrng.make_rng(seed)
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
    n_prop_g = 0
    n_prop_t = 0
    n_acc_g = 0
    n_acc_t = 0
    n_unbound_prop = 0
    n_unbound_acc = 0
    n_rho_tri = 0
    n_rho_flat = 0
    sso_counts = {c[0]: 0 for c in SSO_RHO_COMPONENTS}
    n_vel_circular = 0
    n_vel_flat = 0
    workers_sel, chunk_size = choose_workers_and_chunk(
        N, requested_workers=workers, max_workers=MAX_WORKERS
    )
    chunked_payloads = list(make_chunks(payloads, chunk_size))
    with ProcessPoolExecutor(
        max_workers=workers_sel, initializer=worker_init, initargs=(1,)
    ) as executor:
        futures = [executor.submit(_process_chunk_fullphysics, chunk) for chunk in chunked_payloads]
        for fut in as_completed(futures):
            for out in fut.result():
                if out is None:
                    continue
                comp = out.get("proposal_component")
                if comp == "G":
                    n_prop_g += 1
                elif comp == "T":
                    n_prop_t += 1
                if out.get("accepted") and comp == "G":
                    n_acc_g += 1
                elif out.get("accepted") and comp == "T":
                    n_acc_t += 1
                if out.get("is_unbound_proposed"):
                    n_unbound_prop += 1
                if out.get("is_unbound_accepted"):
                    n_unbound_acc += 1
                rpc = out.get("rho_prior_component")
                if rpc == "tri":
                    n_rho_tri += 1
                elif rpc == "flat_linear":
                    n_rho_flat += 1
                elif rpc and rpc.startswith("sso_"):
                    name = rpc.split("sso_")[1]
                    if name in sso_counts:
                        sso_counts[name] += 1
                vm = out.get("vel_mode")
                if vm == "circular":
                    n_vel_circular += 1
                elif vm == "flat":
                    n_vel_flat += 1
                samples.append(out)
    print("Variant A proposal diagnostics:")
    print(" n_proposed_G:", n_prop_g, "n_proposed_T:", n_prop_t)
    print(" n_accepted_G:", n_acc_g, "n_accepted_T:", n_acc_t)
    print(" n_unbound_proposed:", n_unbound_prop, "n_unbound_accepted:", n_unbound_acc)
    print(" rho prior breakdown: tri:", n_rho_tri, "flat_linear:", n_rho_flat, "sso_counts:", sso_counts)
    print(" velocity-mode breakdown: circular:", n_vel_circular, "flat:", n_vel_flat)
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
    with ProcessPoolExecutor(
        max_workers=workers, initializer=worker_init, initargs=(1,)
    ) as executor:
        for out in executor.map(_eval_variant_b_one, payloads, chunksize=_chunksize(N, workers)):
            if out is not None:
                samples.append(out)
    return samples


# ------------------------------
# JPL Horizons fetch
# ------------------------------
def fetch_jpl_state(
    body_id,
    times,
    center="@sun",
    id_type="smallbody",
    cache_path=None,
    max_retries=5,
    backoff=2.0,
    jd_decimals=HORIZONS_JD_DECIMALS,
):
    results = {}
    failed = []
    cache = {}
    if cache_path is not None and os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:
            cache = {}
    for t in times:
        jd_key = f"{round(float(t.tdb.jd), jd_decimals):.{jd_decimals}f}"
        if jd_key in cache:
            entry = cache[jd_key]
            results[float(jd_key)] = (
                np.array(entry["r"], dtype=float),
                np.array(entry["v"], dtype=float),
            )
            continue
        last_exc = None
        for attempt in range(max_retries):
            try:
                obj = Horizons(id=body_id, location=center, epochs=t.tdb.jd, id_type=id_type)
                vec = obj.vectors()
                x = float(vec["x"][0]) * u.au.to(u.km)
                y = float(vec["y"][0]) * u.au.to(u.km)
                z = float(vec["z"][0]) * u.au.to(u.km)
                vx = float(vec["vx"][0]) * (u.au / u.day).to(u.km / u.s)
                vy = float(vec["vy"][0]) * (u.au / u.day).to(u.km / u.s)
                vz = float(vec["vz"][0]) * (u.au / u.day).to(u.km / u.s)
                r = [x, y, z]
                v = [vx, vy, vz]
                results[float(jd_key)] = (np.array(r), np.array(v))
                cache[jd_key] = {"r": r, "v": v}
                break
            except Exception as e:
                last_exc = e
                if attempt < max_retries - 1:
                    time.sleep(backoff ** attempt)
        else:
            results[float(jd_key)] = (None, None)
            failed.append(jd_key)
            print("Horizons fetch failed for", t, ":", last_exc)
    if cache_path is not None:
        try:
            tmp_path = cache_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(cache, f)
            os.replace(tmp_path, cache_path)
        except Exception:
            pass
    return results, failed


# ------------------------------
# Metrics + plotting
# ------------------------------
def angular_sep(ra1, dec1, ra2, dec2):
    v1 = hat_u_from_radec(ra1, dec1)
    v2 = hat_u_from_radec(ra2, dec2)
    cosang = np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1)
    return np.arccos(cosang)


def summarize_and_plot(
    samples_A,
    samples_B,
    jpl_states,
    obs1,
    site1,
    obs2_ra,
    obs2_dec,
    output_dir=None,
    show_plots=True,
):
    def summarize(samples, label):
        ang_res_obs2 = []
        pos_res_em1 = []
        pos_res_em2 = []
        rho_samples = []
        rho_jpl_samples = []
        for s in samples:
            ang = angular_sep(s["ra_pred"], s["dec_pred"], obs2_ra, obs2_dec)
            ang_res_obs2.append(ang * 206265.0)
            key1 = round(float(s["t_em1"].tdb.jd), HORIZONS_JD_DECIMALS)
            key2 = round(float(s["t_em2"].tdb.jd), HORIZONS_JD_DECIMALS)
            r1_jpl, _ = jpl_states.get(key1, (None, None))
            r2_jpl, _ = jpl_states.get(key2, (None, None))
            pos_res_em1.append(np.nan if r1_jpl is None else np.linalg.norm(s["r1"] - r1_jpl))
            pos_res_em2.append(np.nan if r2_jpl is None else np.linalg.norm(s["r2"] - r2_jpl))
            r_obs_em1, _ = observer_posvel(site1, s["t_em1"])
            rho_samples.append(float(np.linalg.norm(s["r1"] - r_obs_em1)))
            if r1_jpl is not None:
                rho_jpl_samples.append(float(np.linalg.norm(r1_jpl - r_obs_em1)))
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
        return (
            np.array(ang_res_obs2),
            np.array(pos_res_em1),
            np.array(pos_res_em2),
            np.array(rho_samples),
            np.array(rho_jpl_samples),
        )

    A_ang, A_p1, A_p2, A_rho, A_rho_jpl = summarize(samples_A, "Variant A")
    B_ang = B_p1 = B_p2 = B_rho = B_rho_jpl = np.array([])
    if samples_B:
        B_ang, B_p1, B_p2, B_rho, B_rho_jpl = summarize(samples_B, "Variant B")

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(A_ang, bins=40, alpha=0.6, label="A")
    if B_ang.size:
        plt.hist(B_ang, bins=40, alpha=0.6, label="B")
    plt.xlabel("Obs2 angular residual (arcsec)")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.hist(A_p2[~np.isnan(A_p2)], bins=40, alpha=0.6, label="A em2 km")
    if B_p2.size:
        plt.hist(B_p2[~np.isnan(B_p2)], bins=40, alpha=0.6, label="B em2 km")
    plt.xlabel("pos residual @ em2 (km)")
    plt.legend()
    plt.tight_layout()
    if output_dir is not None:
        out_path = os.path.join(output_dir, "variantA_residuals.png")
        plt.savefig(out_path, dpi=150)
    if show_plots:
        plt.show()

    plt.figure(figsize=(8, 4))
    plt.hist(A_rho, bins=40, alpha=0.6, label="A rho")
    if A_rho_jpl.size:
        plt.hist(A_rho_jpl, bins=40, alpha=0.4, label="JPL rho (A times)")
        median_jpl = float(np.median(A_rho_jpl))
        plt.axvline(median_jpl, color="k", linestyle="--", linewidth=1.5, label="JPL median rho")
        plt.annotate(
            f"JPL median: {median_jpl:,.0f} km",
            xy=(median_jpl, plt.gca().get_ylim()[1] * 0.9),
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=8,
            rotation=90,
            va="top",
        )
    plt.xlabel("rho at em1 (km)")
    plt.legend()
    plt.tight_layout()
    if output_dir is not None:
        out_path = os.path.join(output_dir, "variantA_rho.png")
        plt.savefig(out_path, dpi=150)
    if show_plots:
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
    parser.add_argument("--seed", type=int, default=50, help="RNG seed for reproducible sampling")
    parser.add_argument("--horizons-cache", default=None, help="Path to Horizons cache JSON")
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
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display plots interactively (also saved to disk).",
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
        seed=args.seed,
    )

    print("Skipping Variant B (joint Laplace) -- focusing on Variant A only.")
    samples_B = []

    times_set = {}
    for s in samples_A + samples_B:
        times_set[round(float(s["t_em1"].tdb.jd), 6)] = s["t_em1"]
        times_set[round(float(s["t_em2"].tdb.jd), 6)] = s["t_em2"]
    times_list = list(times_set.values())
    print("Querying JPL/Horizons for %d unique emission times ..." % len(times_list))
    cache_path = args.horizons_cache
    if cache_path is None:
        cache_dir = os.path.dirname(os.path.abspath(args.csv))
        cache_path = os.path.join(cache_dir, "horizons_cache.json")
    jpl_states, failed_jds = fetch_jpl_state(
        args.object, times_list, center="@sun", cache_path=cache_path
    )
    if failed_jds:
        print("Horizons failures:", len(failed_jds))
    output_dir = os.path.dirname(os.path.abspath(args.csv))
    show_plots = args.show_plots
    summarize_and_plot(
        samples_A,
        samples_B,
        jpl_states,
        obs1,
        site1,
        obs2_ra,
        obs2_dec,
        output_dir=output_dir,
        show_plots=show_plots,
    )
