from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
from astropy import units as u
from astropy.time import Time
from astroquery.jplhorizons import Horizons

from .models import Observation, OrbitPosterior
from .propagate import predict_radec, propagate_state

__all__ = ["fit_orbit", "sample_replicas", "predict_orbit", "load_posterior"]


def _initial_state_from_horizons(target: str, epoch: Time) -> np.ndarray:
    obj = Horizons(id=target, location="@sun", epochs=epoch.jd)
    vec = obj.vectors(refplane="ecliptic")
    row = vec[0]
    km_per_au = u.au.to(u.km)
    return np.array(
        [
            row["x"] * km_per_au,
            row["y"] * km_per_au,
            row["z"] * km_per_au,
            row["vx"] * km_per_au / 86400.0,
            row["vy"] * km_per_au / 86400.0,
            row["vz"] * km_per_au / 86400.0,
        ],
        dtype=float,
    )


def _tangent_residuals(
    pred_ra: np.ndarray, pred_dec: np.ndarray, obs: list[Observation]
) -> np.ndarray:
    res = []
    for ra, dec, ob in zip(pred_ra, pred_dec, obs):
        d_ra = ((ob.ra_deg - ra + 180.0) % 360.0) - 180.0
        ra_arcsec = d_ra * np.cos(np.deg2rad(dec)) * 3600.0
        dec_arcsec = (ob.dec_deg - dec) * 3600.0
        res.extend([ra_arcsec, dec_arcsec])
    return np.array(res)


def _predict_batch(
    state: np.ndarray,
    epoch: Time,
    obs: list[Observation],
    perturbers: Sequence[str],
    max_step: float,
) -> tuple[np.ndarray, np.ndarray]:
    times = [ob.time for ob in obs]
    propagated = propagate_state(state, epoch, times, perturbers=perturbers, max_step=max_step)
    ra = []
    dec = []
    for st, t in zip(propagated, times):
        r, d = predict_radec(st, t)
        ra.append(r)
        dec.append(d)
    return np.array(ra), np.array(dec)


def predict_orbit(
    state: np.ndarray,
    epoch: Time,
    obs: list[Observation],
    perturbers: Sequence[str],
    max_step: float,
) -> tuple[np.ndarray, np.ndarray]:
    return _predict_batch(state, epoch, obs, perturbers, max_step)


def _jacobian_fd(
    base_state: np.ndarray,
    epoch: Time,
    obs: list[Observation],
    perturbers: Sequence[str],
    eps: np.ndarray,
    max_step: float,
) -> np.ndarray:
    base_ra, base_dec = _predict_batch(base_state, epoch, obs, perturbers, max_step)
    base_res = _tangent_residuals(base_ra, base_dec, obs)
    num_obs = len(obs)
    H = np.zeros((2 * num_obs, 6), dtype=float)
    for idx in range(6):
        perturbed = base_state.copy()
        perturbed[idx] += eps[idx]
        pred_ra, pred_dec = _predict_batch(perturbed, epoch, obs, perturbers, max_step)
        res = _tangent_residuals(pred_ra, pred_dec, obs)
        H[:, idx] = (res - base_res) / eps[idx]
    return H, base_res


def fit_orbit(
    target: str,
    observations: list[Observation],
    *,
    perturbers: Sequence[str] = ("earth", "mars", "jupiter"),
    max_iter: int = 6,
    tol: float = 1e-2,
    max_step: float = 3600.0,
) -> OrbitPosterior:
    epoch = observations[0].time
    state = _initial_state_from_horizons(target, epoch)
    eps = np.array([1e-3, 1e-3, 1e-3, 1e-5, 1e-5, 1e-5])
    prior_variances = np.array(
        [1e6, 1e6, 1e6, 1e-2, 1e-2, 1e-2], dtype=float
    )  # km^2, (km/s)^2
    prior_inv = np.diag(1.0 / prior_variances)
    W = np.diag(
        [
            1.0 / (obs.sigma_arcsec**2)
            for obs in observations
            for _ in range(2)
        ]
    )
    residuals = np.zeros(2 * len(observations))
    for _ in range(max_iter):
        H, residuals = _jacobian_fd(state, epoch, observations, perturbers, eps, max_step)
        A = prior_inv + H.T @ W @ H
        b = H.T @ W @ residuals
        try:
            delta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(A, b, rcond=None)[0]
        state += delta
        if np.linalg.norm(delta) < tol:
            break
    try:
        cov = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(A)
    rms = np.sqrt(np.mean(residuals**2))
    return OrbitPosterior(epoch=epoch, state=state, cov=cov, residuals=residuals, rms_arcsec=rms)


def load_posterior(path: str | Path) -> OrbitPosterior:
    data = np.load(path)
    state = np.array(data["state"])
    cov = np.array(data["cov"])
    residuals = np.array(data["residuals"])
    epoch = Time(str(data["epoch"]), scale="utc")
    rms = float(data["rms"])
    return OrbitPosterior(epoch=epoch, state=state, cov=cov, residuals=residuals, rms_arcsec=rms)


def sample_replicas(post: OrbitPosterior, n: int, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    try:
        L = np.linalg.cholesky(post.cov)
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(post.cov + np.eye(6) * 1e-6)
    noise = rng.standard_normal((6, n))
    return post.state[:, None] + L @ noise
