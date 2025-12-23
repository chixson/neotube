from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
from astropy import units as u
from astropy.coordinates import (
    CartesianDifferential,
    CartesianRepresentation,
    HeliocentricTrueEcliptic,
    SkyCoord,
)
from astropy.time import Time
from astroquery.jplhorizons import Horizons

from .models import Observation, OrbitPosterior
from .propagate import (
    predict_radec_batch,
    propagate_state,
    propagate_state_kepler,
)

__all__ = ["fit_orbit", "sample_replicas", "predict_orbit", "load_posterior"]


def _normalize_horizons_id(raw: str) -> str:
    """Normalize common MPC-style identifiers into something Horizons accepts.

    - MPC numbered minor planets: "1" -> "2000001"
      (Horizons uses the 2,000,000 + number convention for asteroids.)
    - Otherwise, pass through unchanged (e.g., "Ceres", "2020 AB", "DES=...").
    """
    s = raw.strip()
    if s.isdigit():
        n = int(s)
        if 1 <= n < 2000000:
            return str(2000000 + n)
    return s


def _initial_state_from_horizons(target: str, epoch: Time) -> np.ndarray:
    obj = Horizons(id=_normalize_horizons_id(target), location="@sun", epochs=epoch.jd)
    # Horizons' refplane='ecliptic' vectors are stable, but must be interpreted
    # in the ecliptic-of-date frame (set obstime) before converting to ICRS.
    vec = obj.vectors(refplane="ecliptic")
    row = vec[0]

    pos = CartesianRepresentation(
        row["x"] * u.au,
        row["y"] * u.au,
        row["z"] * u.au,
    )
    vel = CartesianDifferential(
        row["vx"] * u.au / u.day,
        row["vy"] * u.au / u.day,
        row["vz"] * u.au / u.day,
    )
    coord = SkyCoord(
        pos.with_differentials(vel),
        frame=HeliocentricTrueEcliptic(obstime=epoch),
    ).icrs
    cart = coord.cartesian

    return np.array(
        [
            cart.x.to(u.km).value,
            cart.y.to(u.km).value,
            cart.z.to(u.km).value,
            cart.differentials["s"].d_x.to(u.km / u.s).value,
            cart.differentials["s"].d_y.to(u.km / u.s).value,
            cart.differentials["s"].d_z.to(u.km / u.s).value,
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


def _ensure_finite(name: str, *arrays: np.ndarray) -> None:
    for arr in arrays:
        if not np.all(np.isfinite(arr)):
            raise RuntimeError(f"{name} contains non-finite values.")


def _predict_batch(
    state: np.ndarray,
    epoch: Time,
    obs: list[Observation],
    perturbers: Sequence[str],
    max_step: float,
    use_kepler: bool,
) -> tuple[np.ndarray, np.ndarray]:
    times = [ob.time for ob in obs]
    site_codes = [ob.site for ob in obs]
    if use_kepler:
        try:
            propagated = propagate_state_kepler(state, epoch, times)
        except ValueError as exc:
            warnings.warn(
                f"Kepler propagation failed; falling back to full propagation ({exc})",
                RuntimeWarning,
            )
            propagated = propagate_state(
                state, epoch, times, perturbers=perturbers, max_step=max_step
            )
    else:
        propagated = propagate_state(state, epoch, times, perturbers=perturbers, max_step=max_step)
    ra, dec = predict_radec_batch(propagated, times, site_codes=site_codes)
    return ra, dec


def predict_orbit(
    state: np.ndarray,
    epoch: Time,
    obs: list[Observation],
    perturbers: Sequence[str],
    max_step: float,
    use_kepler: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    return _predict_batch(state, epoch, obs, perturbers, max_step, use_kepler=use_kepler)


def _jacobian_fd(
    base_state: np.ndarray,
    epoch: Time,
    obs: list[Observation],
    perturbers: Sequence[str],
    eps: np.ndarray,
    max_step: float,
    use_kepler: bool,
) -> np.ndarray:
    base_ra, base_dec = _predict_batch(
        base_state, epoch, obs, perturbers, max_step, use_kepler=use_kepler
    )
    base_res = _tangent_residuals(base_ra, base_dec, obs)
    _ensure_finite("residuals", base_res)
    num_obs = len(obs)
    H = np.zeros((2 * num_obs, 6), dtype=float)
    for idx in range(6):
        perturbed = base_state.copy()
        perturbed[idx] += eps[idx]
        pred_ra, pred_dec = _predict_batch(
            perturbed, epoch, obs, perturbers, max_step, use_kepler=use_kepler
        )
        res = _tangent_residuals(pred_ra, pred_dec, obs)
        H[:, idx] = (res - base_res) / eps[idx]
    _ensure_finite("jacobian", H)
    return H, base_res


def fit_orbit(
    target: str,
    observations: list[Observation],
    *,
    perturbers: Sequence[str] = ("earth", "mars", "jupiter"),
    max_iter: int = 6,
    tol: float = 1.0,
    max_step: float = 3600.0,
    use_kepler: bool = True,
) -> OrbitPosterior:
    observations = sorted(observations, key=lambda ob: ob.time)
    epoch = observations[len(observations) // 2].time
    state = _initial_state_from_horizons(target, epoch)
    # Finite-difference step sizes must be large enough to dominate numeric noise
    # from propagation + coordinate transforms.
    eps = np.array([10.0, 10.0, 10.0, 1e-4, 1e-4, 1e-4], dtype=float)  # km, km/s
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
    seed_rms: float | None = None
    try:
        seed_ra, seed_dec = _predict_batch(
            state, epoch, observations, perturbers, max_step, use_kepler=use_kepler
        )
        seed_residuals = _tangent_residuals(seed_ra, seed_dec, observations)
        if np.all(np.isfinite(seed_residuals)):
            seed_rms = float(np.sqrt(np.mean(seed_residuals**2)))
    except Exception:
        seed_rms = None
    converged = False
    for _ in range(max_iter):
        H, residuals = _jacobian_fd(
            state, epoch, observations, perturbers, eps, max_step, use_kepler=use_kepler
        )
        lamb = 1e-3
        A = prior_inv + H.T @ W @ H + np.eye(6) * lamb
        # Gauss-Newton / LM step solves: (H^T W H + λI) δ = -H^T W r
        b = -H.T @ W @ residuals
        _ensure_finite("normal system", A, b)
        try:
            delta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(A, b, rcond=None)[0]
        _ensure_finite("delta", delta)
        state += delta
        if np.linalg.norm(delta) < tol:
            converged = True
            break
    try:
        cov = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(A)
    _ensure_finite("residuals", residuals)
    rms = np.sqrt(np.mean(residuals**2))
    if not np.isfinite(rms):
        raise RuntimeError("Residual RMS is non-finite.")
    return OrbitPosterior(
        epoch=epoch,
        state=state,
        cov=cov,
        residuals=residuals,
        rms_arcsec=rms,
        converged=converged,
        seed_rms_arcsec=seed_rms,
    )


def load_posterior(path: str | Path) -> OrbitPosterior:
    data = np.load(path)
    state = np.array(data["state"])
    cov = np.array(data["cov"])
    residuals = np.array(data["residuals"])
    epoch = Time(str(data["epoch"]), scale="utc")
    rms = float(data["rms"])
    converged = bool(data["converged"]) if "converged" in data else True
    seed_rms = float(data["seed_rms"]) if "seed_rms" in data else None
    if seed_rms is not None and not np.isfinite(seed_rms):
        seed_rms = None
    return OrbitPosterior(
        epoch=epoch,
        state=state,
        cov=cov,
        residuals=residuals,
        rms_arcsec=rms,
        converged=converged,
        seed_rms_arcsec=seed_rms,
    )


def load_posterior_json(path: str | Path) -> OrbitPosterior:
    import json

    with open(path) as fh:
        data = json.load(fh)

    epoch = Time(str(data["epoch_utc"]), scale="utc")
    state = np.array(data["state_km"], dtype=float)
    cov = np.array(data["cov_km2"], dtype=float)

    fit = data.get("fit") or {}
    converged = bool(fit.get("converged", True))
    rms = float(fit.get("rms_arcsec", float("nan")))
    seed_rms = fit.get("seed_rms_arcsec", None)
    seed_rms = float(seed_rms) if seed_rms is not None else None

    residuals = np.array([], dtype=float)
    if not np.isfinite(rms):
        # JSON may not record residuals; keep this as informational only.
        rms = float("nan")

    return OrbitPosterior(
        epoch=epoch,
        state=state,
        cov=cov,
        residuals=residuals,
        rms_arcsec=rms,
        converged=converged,
        seed_rms_arcsec=seed_rms if (seed_rms is None or np.isfinite(seed_rms)) else None,
    )


def sample_replicas(post: OrbitPosterior, n: int, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    try:
        L = np.linalg.cholesky(post.cov)
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(post.cov + np.eye(6) * 1e-6)
    noise = rng.standard_normal((6, n))
    return post.state[:, None] + L @ noise
