from __future__ import annotations

import dataclasses
import math
from typing import Sequence

import numpy as np
from astropy.time import Time

from .propagate import (
    get_horizons_state,
    propagate_kepler,
    state_to_radec,
    OrbitalState,
)
from astropy import units as u


@dataclasses.dataclass(frozen=True)
class Observation:
    time: Time
    ra_deg: float
    dec_deg: float
    sigma_arcsec: float
    obs_code: str | None = None


@dataclasses.dataclass
class OrbitFitResult:
    epoch: Time
    state: np.ndarray
    covariance: np.ndarray
    residuals: np.ndarray
    rms_arcsec: float


def _tangent_residual(ra_pred: float, dec_pred: float, obs: Observation) -> np.ndarray:
    cos_dec = math.cos(math.radians(obs.dec_deg))
    dra = (obs.ra_deg - ra_pred) * cos_dec * 3600.0
    ddec = (obs.dec_deg - dec_pred) * 3600.0
    return np.array([dra, ddec])


def _compute_residuals(state: np.ndarray, observations: Sequence[Observation], t0: Time, target: str, perturbers: Sequence[str] | None = None) -> np.ndarray:
    res = []
    for obs in observations:
        try:
            propagated = propagate_kepler(OrbitalState(epoch=t0, rv=state), obs.time, perturbers=perturbers)
        except RuntimeError:
            propagated = get_horizons_state(target, obs.time)
        ra_pred, dec_pred = state_to_radec(propagated)
        res.extend(_tangent_residual(ra_pred, dec_pred, obs))
    return np.array(res)


def _jacobian(state: np.ndarray, residuals: np.ndarray, observations: Sequence[Observation], t0: Time, target: str, perturbers: Sequence[str] | None = None) -> np.ndarray:
    n = len(residuals)
    H = np.zeros((n, 6))
    eps = np.array([1e-3, 1e-3, 1e-3, 1e-5, 1e-5, 1e-5])
    for j in range(6):
        perturbed = state.copy()
        perturbed[j] += eps[j]
        res_pert = _compute_residuals(perturbed, observations, t0, target, perturbers=perturbers)
        H[:, j] = (res_pert - residuals) / eps[j]
    return H


def fit_orbit(
    target: str,
    observations: Sequence[Observation],
    sigma_arcsec: float,
    max_iter: int = 5,
    prior_cov_scale: float = 1e6,
    perturbers: Sequence[str] | None = None,
) -> OrbitFitResult:
    if len(observations) < 6:
        raise ValueError("need at least 6 observations")
    mean_time = Time(
        np.mean([obs.time.tcb.jd for obs in observations]), format="jd", scale="tdb"
    )
    earliest = min(observations, key=lambda o: o.time).time
    margin = 0.05 * u.day
    reference_epoch = earliest - margin
    initial = get_horizons_state(target, reference_epoch)
    state = initial.rv.copy()
    t0 = initial.epoch
    prior_inv = np.eye(6) / prior_cov_scale

    residuals = _compute_residuals(state, observations, t0, target, perturbers=perturbers)
    for _ in range(max_iter):
        H = _jacobian(state, residuals, observations, t0, target, perturbers=perturbers)
        W = np.eye(len(residuals)) / sigma_arcsec**2
        A = H.T @ W @ H + prior_inv
        b = H.T @ W @ residuals
        delta = np.linalg.solve(A, b)
        state -= delta
        residuals = _compute_residuals(state, observations, t0, target, perturbers=perturbers)
        if np.linalg.norm(delta) < 1e-5:
            break

    H_final = _jacobian(state, residuals, observations, t0, target, perturbers=perturbers)
    W = np.eye(len(residuals)) / sigma_arcsec**2
    cov = np.linalg.inv(H_final.T @ W @ H_final + prior_inv)
    rms = math.sqrt(np.mean(residuals**2))
    return OrbitFitResult(epoch=t0, state=state, covariance=cov, residuals=residuals, rms_arcsec=rms)


def sample_replicas(result: OrbitFitResult, n: int, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(result.covariance)
    z = rng.standard_normal((6, n))
    return result.state[:, None] + L @ z
