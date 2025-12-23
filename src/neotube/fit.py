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
    get_body_barycentric_posvel,
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
from .sites import get_site_location

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


def _site_offset(obs: Observation) -> np.ndarray:
    if not obs.site:
        return np.zeros(3, dtype=float)
    loc = get_site_location(obs.site)
    if loc is None:
        return np.zeros(3, dtype=float)
    gcrs = loc.get_gcrs(obstime=obs.time)
    return gcrs.cartesian.xyz.to(u.km).value


def _observation_line_of_sight(obs: Observation) -> tuple[np.ndarray, np.ndarray]:
    coord = SkyCoord(
        ra=obs.ra_deg * u.deg,
        dec=obs.dec_deg * u.deg,
        distance=1.0 * u.au,
        frame="icrs",
        obstime=obs.time,
    )
    direction = coord.cartesian.xyz.to(u.km).value
    direction = direction / np.linalg.norm(direction)
    earth_pos, _ = get_body_barycentric_posvel("earth", obs.time)
    earth_km = earth_pos.xyz.to(u.km).value
    offset = _site_offset(obs)
    return earth_km + offset, direction


def _initial_state_from_two_points(observations: list[Observation]) -> np.ndarray:
    pos, _ = _observation_line_of_sight(observations[0])
    pos1, _ = _observation_line_of_sight(observations[-1])
    dt = float((observations[-1].time - observations[0].time).to(u.s).value)
    if abs(dt) < 1e-3:
        dt = 1.0
    vel = (pos1 - pos) / dt
    return np.concatenate([pos, vel])


def _initial_state_from_observations(observations: list[Observation]) -> np.ndarray:
    if len(observations) < 3:
        return _initial_state_from_two_points(observations)

    t1 = observations[0]
    t2 = observations[len(observations) // 2]
    t3 = observations[-1]
    R2, rho2 = _observation_line_of_sight(t2)

    ra1 = observations[0].ra_deg
    ra3 = observations[-1].ra_deg
    dec1 = observations[0].dec_deg
    dec3 = observations[-1].dec_deg
    delta_ra = (((ra3 - ra1 + 180.0) % 360.0) - 180.0) * np.deg2rad
    delta_dec = (dec3 - dec1) * np.deg2rad
    dt_total = float((t3.time - t1.time).to(u.s).value)
    if abs(dt_total) < 1.0:
        dt_total = 1.0

    ra_rate = delta_ra / dt_total
    dec_rate = delta_dec / dt_total

    cos_dec = np.cos(np.deg2rad(t2.dec_deg))
    ang_rate = np.hypot(ra_rate * cos_dec, dec_rate)
    typical_speed = 30.0  # km/s (approx)
    range_km = typical_speed / max(ang_rate, 1e-10)
    range_km = np.clip(range_km, 1e5, 5e8)

    ra_rad = np.deg2rad(t2.ra_deg)
    dec_rad = np.deg2rad(t2.dec_deg)
    east = np.array([-np.sin(ra_rad), np.cos(ra_rad), 0.0], dtype=float)
    north = np.array([-np.sin(dec_rad) * np.cos(ra_rad), -np.sin(dec_rad) * np.sin(ra_rad), np.cos(dec_rad)], dtype=float)

    v_tang = range_km * (ra_rate * cos_dec * east + dec_rate * north)
    return np.concatenate([R2 + range_km * rho2, v_tang])


def _initial_state_from_observations(observations: list[Observation]) -> np.ndarray:
    if len(observations) < 2:
        raise ValueError("Need at least two observations to build an initial state.")

    def _heliocentric_pos(obs: Observation) -> np.ndarray:
        coord = SkyCoord(
            ra=obs.ra_deg * u.deg,
            dec=obs.dec_deg * u.deg,
            distance=1.0 * u.au,
            frame="icrs",
            obstime=obs.time,
        )
        direction = coord.cartesian.xyz.to(u.km).value
        earth_pos, _ = get_body_barycentric_posvel("earth", obs.time)
        earth_km = earth_pos.xyz.to(u.km).value
        return earth_km + direction

    pos0 = _heliocentric_pos(observations[0])
    pos1 = _heliocentric_pos(observations[-1])
    dt = float((observations[-1].time - observations[0].time).to(u.s).value)
    if abs(dt) < 1e-2:
        dt = 1.0
    vel = (pos1 - pos0) / dt
    return np.concatenate([pos0, vel])


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


def _huber_weights(res: np.ndarray, sigma: np.ndarray, c: float = 1.345) -> np.ndarray:
    t = np.abs(res / sigma)
    w = np.ones_like(t)
    large = t > c
    w[large] = c / t[large]
    return w


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
    use_horizons_seed: bool = False,
) -> OrbitPosterior:
    observations = sorted(observations, key=lambda ob: ob.time)
    epoch = observations[len(observations) // 2].time
    state = (
        _initial_state_from_horizons(target, epoch)
        if use_horizons_seed
        else _initial_state_from_observations(observations)
    )

    base_eps = np.array([10.0, 10.0, 10.0, 1e-4, 1e-4, 1e-4], dtype=float)
    prior_variances = np.array([1e6, 1e6, 1e6, 1e-2, 1e-2, 1e-2], dtype=float)
    prior_inv = np.diag(1.0 / prior_variances)

    sigma_vec = np.array([obs.sigma_arcsec for obs in observations for _ in range(2)], dtype=float)
    sigma_vec = np.maximum(sigma_vec, 1e-3)
    W_base_diag = 1.0 / (sigma_vec**2)

    residuals = np.zeros(2 * len(observations), dtype=float)
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

    lam = 1e-3
    lam_up = 10.0
    lam_down = 0.1
    cond_J_warn = 1e12
    cond_A_warn = 1e15
    converged = False
    A = np.eye(6, dtype=float)

    for it in range(max_iter):
        eps = np.maximum(np.abs(state) * 1e-8, base_eps)
        H, residuals = _jacobian_fd(
            state, epoch, observations, perturbers, eps, max_step, use_kepler=use_kepler
        )

        try:
            cond_J = np.linalg.cond(H)
        except Exception:
            cond_J = float("inf")
        try:
            _, s_h, _ = np.linalg.svd(H, full_matrices=False)
        except Exception:
            s_h = np.array([])

        col_norm = np.linalg.norm(H, axis=0)
        col_norm = np.maximum(col_norm, 1e-12)
        scale = 1.0 / col_norm
        Hs = H * scale
        S = np.diag(scale)
        prior_inv_scaled = S @ prior_inv @ S

        w_robust = _huber_weights(residuals, sigma_vec)
        W_total_diag = W_base_diag * w_robust
        W_total = np.diag(W_total_diag)

        A = prior_inv_scaled + Hs.T @ (W_total @ Hs)
        try:
            cond_A = np.linalg.cond(A)
        except Exception:
            cond_A = float("inf")

        print(f"[fit_orbit] it={it} cond(J)={cond_J:.3e} cond(A)={cond_A:.3e} lam={lam:.3e}")
        if s_h.size > 0:
            print(f"[fit_orbit]  sv(min..): {s_h[-6:].tolist()}")
        normed = np.abs(residuals / sigma_vec)
        top_inds = np.argsort(-normed)[:6]
        print("[fit_orbit]  top residuals (idx, normed, arcsec):")
        for idx in top_inds:
            print(
                f"   {idx:3d}  {normed[idx]:6.2f}  {residuals[idx]:7.2f}\"  "
                f"(sigma={sigma_vec[idx]:.3f}\")"
            )

        if cond_J > cond_J_warn or cond_A > cond_A_warn:
            lam *= 1e3
            print(f"[fit_orbit]  condition too large, pushing lam -> {lam:.3e}")

        A_lm = A + lam * np.diag(np.diag(A))
        b = -Hs.T @ (W_total @ residuals)

        try:
            delta_prime = np.linalg.solve(A_lm, b)
        except np.linalg.LinAlgError:
            print("[fit_orbit]  normal matrix singular; switching to lstsq")
            delta_prime = np.linalg.lstsq(A_lm, b, rcond=None)[0]

        delta = delta_prime * scale
        if not np.all(np.isfinite(delta)):
            raise RuntimeError("delta contains non-finite values.")

        candidate_state = state + delta
        try:
            cand_ra, cand_dec = _predict_batch(
                candidate_state, epoch, observations, perturbers, max_step, use_kepler=use_kepler
            )
            cand_residuals = _tangent_residuals(cand_ra, cand_dec, observations)
            _ensure_finite("candidate residuals", cand_residuals)
        except Exception as exc:
            print(f"[fit_orbit]  candidate prediction failed: {exc}")
            cand_residuals = residuals.copy() + 1e6

        cost = 0.5 * float(residuals.T @ (W_total @ residuals))
        cost_new = 0.5 * float(cand_residuals.T @ (W_total @ cand_residuals))

        if cost_new < cost:
            state = candidate_state
            residuals = cand_residuals
            lam = max(lam * lam_down, 1e-16)
            print(f"[fit_orbit]  step accepted cost {cost:.3e}->{cost_new:.3e}, lam->{lam:.3e}")
        else:
            lam *= lam_up
            print(f"[fit_orbit]  step rejected cost {cost:.3e}->{cost_new:.3e}, lam->{lam:.3e}")
            if lam > 1e12:
                print("[fit_orbit]  lambda huge; using damped pseudo-inverse fallback")
                try:
                    U, svals, Vt = np.linalg.svd(A, full_matrices=False)
                    s_reg = svals / (svals**2 + (lam * 1e-6))
                    delta_prime = Vt.T @ (s_reg * (U.T @ b))
                    delta = delta_prime * scale
                    state = state + delta
                    ra, dec = _predict_batch(
                        state, epoch, observations, perturbers, max_step, use_kepler=use_kepler
                    )
                    residuals = _tangent_residuals(ra, dec, observations)
                    lam *= 10.0
                    print("[fit_orbit]  fallback step taken")
                except Exception as exc:
                    print(f"[fit_orbit]  fallback failed: {exc}")

        if np.linalg.norm(delta) < tol:
            converged = True
            print(f"[fit_orbit] convergence reached (delta norm {np.linalg.norm(delta):.3e} < tol)")
            break

    try:
        cov = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(A)

    _ensure_finite("residuals", residuals)
    rms = float(np.sqrt(np.mean(residuals**2)))
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
    data = np.load(path, allow_pickle=True)
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
