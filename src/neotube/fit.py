from __future__ import annotations

import copy
import json
import warnings
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from astropy import units as u
from astropy.coordinates import (
    CartesianDifferential,
    CartesianRepresentation,
    GCRS,
    ICRS,
    EarthLocation,
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
from .sites import get_site_location, load_observatories

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
    # Return the geocentric cartesian vector (km) for the observing site at obs.time.
    if not obs.site:
        loc = EarthLocation.from_geodetic(lon=0.0 * u.deg, lat=0.0 * u.deg, height=0.0 * u.m)
        gcrs = loc.get_gcrs(obstime=obs.time)
        return gcrs.cartesian.xyz.to(u.km).value

    loc = get_site_location(obs.site)
    if loc is None:
        try:
            _ = load_observatories()
        except Exception:
            pass
        loc = get_site_location(obs.site)
    if loc is None:
        warnings.warn(
            f"Site code '{obs.site}' not found in observatory catalog; using fallback Earth location."
        )
        loc = EarthLocation.from_geodetic(lon=0.0 * u.deg, lat=0.0 * u.deg, height=0.0 * u.m)
        gcrs = loc.get_gcrs(obstime=obs.time)
        return gcrs.cartesian.xyz.to(u.km).value

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


def _compute_attributable(observations: list[Observation], epoch: Time):
    times = np.array([(ob.time - epoch).to(u.s).value for ob in observations], dtype=float)
    ra_rad = np.unwrap(np.deg2rad(np.array([ob.ra_deg for ob in observations], dtype=float)))
    dec_rad = np.deg2rad(np.array([ob.dec_deg for ob in observations], dtype=float))

    A = np.vstack([np.ones_like(times), times]).T
    coeffs_ra, *_ = np.linalg.lstsq(A, ra_rad, rcond=None)
    coeffs_dec, *_ = np.linalg.lstsq(A, dec_rad, rcond=None)

    ra0, ra_dot = coeffs_ra
    dec0, dec_dot = coeffs_dec
    return ra0, dec0, ra_dot, dec_dot


def _s_and_sdot_from_ra_dec(ra: float, dec: float, ra_dot: float, dec_dot: float) -> tuple[np.ndarray, np.ndarray]:
    cr = np.cos(ra)
    sr = np.sin(ra)
    cd = np.cos(dec)
    sd = np.sin(dec)
    s = np.array([cd * cr, cd * sr, sd], dtype=float)
    s_dot = np.array(
        [
            -cd * sr * ra_dot - sd * cr * dec_dot,
            cd * cr * ra_dot - sd * sr * dec_dot,
            cd * dec_dot,
        ],
        dtype=float,
    )
    return s, s_dot


def _line_intersection(
    p1: np.ndarray, d1: np.ndarray, p2: np.ndarray, d2: np.ndarray
) -> np.ndarray | None:
    cross_d = np.cross(d1, d2)
    denom = np.dot(cross_d, cross_d)
    if denom < 1e-12:
        return None
    t = np.dot(np.cross(p2 - p1, d2), cross_d) / denom
    return p1 + d1 * t


def _initial_state_from_gauss(observations: list[Observation]) -> np.ndarray:
    if len(observations) < 3:
        return _initial_state_from_observations(observations)

    obs1, obs2, obs3 = observations[0], observations[len(observations) // 2], observations[-1]
    p1, d1 = _observation_line_of_sight(obs1)
    p2, d2 = _observation_line_of_sight(obs2)
    p3, d3 = _observation_line_of_sight(obs3)

    points = []
    for (p_a, d_a), (p_b, d_b) in (( (p1, d1), (p2, d2) ), ( (p1, d1), (p3, d3) ), ( (p2, d2), (p3, d3) )):
        pt = _line_intersection(p_a, d_a, p_b, d_b)
        if pt is not None:
            points.append(pt)
    if not points:
        center = p2
        start = p1
        end = p3
    else:
        center = np.mean(points, axis=0)
        start = points[0]
        end = points[-1] if len(points) > 1 else points[0]

    dt = float((obs3.time - obs1.time).to(u.s).value)
    if abs(dt) < 1e-2:
        dt = 1.0
    vel = end - start
    vel = vel / dt
    return np.concatenate([center, vel])


def _initial_state_from_attributable(
    observations: list[Observation],
    epoch: Time,
    perturbers: Sequence[str] = ("earth", "mars", "jupiter"),
    max_step: float = 3600.0,
    site_choice: str | None = None,
) -> np.ndarray:
    ra0, dec0, ra_dot, dec_dot = _compute_attributable(observations, epoch)
    s, s_dot = _s_and_sdot_from_ra_dec(ra0, dec0, ra_dot, dec_dot)

    earth_pos, earth_vel = get_body_barycentric_posvel("earth", epoch)
    sun_pos, sun_vel = get_body_barycentric_posvel("sun", epoch)
    earth_pos_helio = (earth_pos.xyz - sun_pos.xyz).to(u.km).value.flatten()
    earth_vel_helio = (earth_vel.xyz - sun_vel.xyz).to(u.km / u.s).value.flatten()

    site_code = site_choice or observations[0].site
    site_offset = np.zeros(3, dtype=float)
    if site_code:
        loc = get_site_location(site_code)
        if loc is not None:
            gcrs = loc.get_gcrs(obstime=epoch)
            icrs = gcrs.transform_to(ICRS())
            site_offset = icrs.cartesian.xyz.to(u.km).value.flatten()

    best_state = None
    best_rms = np.inf
    typical_speeds = np.logspace(np.log10(5.0), np.log10(40.0), 5)
    for speed in typical_speeds:
        rho = speed / max(np.linalg.norm(s_dot), 1e-8)
        rho = float(np.clip(rho, 1e4, 5e8))
        r_gc = site_offset + rho * s
        v_gc = rho * s_dot
        r_helio = earth_pos_helio + r_gc
        v_helio = earth_vel_helio + v_gc
        cand_state = np.concatenate([r_helio, v_helio])
        try:
            pred_ra, pred_dec = _predict_batch(cand_state, epoch, observations, perturbers, max_step, use_kepler=True)
        except Exception:
            continue
        res = _tangent_residuals(pred_ra, pred_dec, observations)
        rms = np.sqrt(np.mean(res**2))
        if rms < best_rms:
            best_rms = rms
            best_state = cand_state

    if best_state is None:
        raise RuntimeError("Could not find an attributable-based initial state.")
    return best_state


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


def _mad_scale(residuals: np.ndarray, sigma_vec: np.ndarray) -> float:
    t = residuals / sigma_vec
    t = t[np.isfinite(t)]
    if t.size == 0:
        return 1.0
    med = np.median(t)
    mad = np.median(np.abs(t - med))
    scale = mad / 0.6744898 if mad > 0 else 0.0
    return float(max(1.0, scale))


def sample_multivariate_t(
    mean: np.ndarray, cov: np.ndarray, n: int, nu: float = 4.0, seed: Optional[int] = None
) -> np.ndarray:
    rng = np.random.default_rng(None if seed is None else int(seed))
    cov = 0.5 * (cov + cov.T)
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 1e-12)
        L = eigvecs @ np.diag(np.sqrt(eigvals))
    d = mean.size
    gs = rng.standard_normal(size=(n, d))
    chis = rng.chisquare(df=max(1.0, nu), size=n) / float(max(1.0, nu))
    ys = (gs @ L.T) / np.sqrt(chis)[:, None]
    samples = ys + mean[None, :]
    return samples.T


def sample_replicas(
    posterior: OrbitPosterior,
    n: int,
    seed: Optional[int] = None,
    method: str = "multit",
    nu: float = 4.0,
) -> np.ndarray:
    mean = np.array(posterior.state, dtype=float).reshape(6)
    cov = np.array(posterior.cov, dtype=float)
    fit_scale = float(getattr(posterior, "fit_scale", 1.0))
    cov_inflated = cov * (fit_scale**2)
    if method in ("multit", "multivariate-t"):
        return sample_multivariate_t(mean, cov_inflated, n, nu=nu, seed=seed)
    if method == "gaussian":
        rng = np.random.default_rng(None if seed is None else int(seed))
        try:
            L = np.linalg.cholesky(cov_inflated)
        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.linalg.eigh(cov_inflated)
            eigvals = np.maximum(eigvals, 1e-12)
            L = eigvecs @ np.diag(np.sqrt(eigvals))
        noise = rng.standard_normal((6, n))
        samples = mean[:, None] + L @ noise
        return samples
    raise ValueError(f"Unknown sampling method: {method}")


def load_posterior(path: Path | str) -> OrbitPosterior:
    data = np.load(str(path), allow_pickle=True)
    state = np.array(data["state"], dtype=float)
    cov = np.array(data["cov"], dtype=float)
    residuals = np.array(data["residuals"], dtype=float)
    epoch_val = data["epoch"]
    if isinstance(epoch_val, (np.ndarray, list)):
        epoch_str = str(epoch_val.tolist())
    else:
        epoch_str = str(epoch_val)
    epoch = Time(epoch_str, scale="utc")
    rms = float(data.get("rms", np.sqrt(np.mean(residuals**2))))
    converged = bool(data.get("converged", True))
    seed_rms = data.get("seed_rms", np.nan)
    seed_rms_val = float(seed_rms) if np.isfinite(seed_rms) else None
    fit_scale = float(data.get("fit_scale", 1.0))
    nu = float(data.get("nu", 4.0))
    site_kappas = {}
    site_kappas_json = data.get("site_kappas", None)
    if site_kappas_json is not None:
        try:
            raw = site_kappas_json.tolist() if hasattr(site_kappas_json, "tolist") else site_kappas_json
            site_kappas = json.loads(raw)
        except Exception:
            site_kappas = {}
    return OrbitPosterior(
        epoch=epoch,
        state=state,
        cov=cov,
        residuals=residuals,
        rms_arcsec=rms,
        converged=converged,
        seed_rms_arcsec=seed_rms_val,
        fit_scale=fit_scale,
        nu=nu,
        site_kappas=site_kappas,
    )


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
        except (ValueError, OverflowError) as exc:
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
    seed_method: str | None = None,
    likelihood: str = "gaussian",
    nu: float = 4.0,
    estimate_site_scales: bool = False,
    max_kappa: float = 10.0,
    estimate_site_scales_method: str = "mad",
    estimate_site_scales_iters: int = 5,
    estimate_site_scales_alpha: float = 0.4,
    estimate_site_scales_tol: float = 1e-3,
    sigma_floor: float = 0.0,
    internal_run: bool = False,
) -> OrbitPosterior:
    observations = sorted(observations, key=lambda ob: ob.time)
    epoch = observations[len(observations) // 2].time
    if seed_method is None:
        seed_method = "attributable"
    if seed_method == "horizons":
        state = _initial_state_from_horizons(target, epoch)
    elif seed_method == "observations":
        state = _initial_state_from_observations(observations)
    elif seed_method == "gauss":
        state = _initial_state_from_gauss(observations)
    elif seed_method == "attributable":
        state = _initial_state_from_attributable(observations, epoch)
    else:
        raise ValueError(f"Unknown seed_method '{seed_method}'")

    if estimate_site_scales and not internal_run:
        def _compute_site_kappas_from_prelim(prelim, method: str) -> dict[str, float]:
            res = prelim.residuals
            site_groups: dict[str, list[float]] = {}
            for i, ob in enumerate(observations):
                site = ob.site or "UNK"
                ra_norm = float(res[2 * i] / ob.sigma_arcsec)
                dec_norm = float(res[2 * i + 1] / ob.sigma_arcsec)
                site_groups.setdefault(site, []).extend([ra_norm, dec_norm])

            kappas: dict[str, float] = {}
            for site, vals in site_groups.items():
                arr = np.array(vals, dtype=float)
                if arr.size == 0:
                    kappas[site] = 1.0
                    continue
                if method == "mad":
                    mad = float(np.median(np.abs(arr - np.median(arr))))
                    kappa = max(1.0, mad / 0.6744898)
                elif method == "chi2":
                    chi2_site = float(np.sum(arr**2))
                    dof_site = max(1, arr.size)
                    kappa = max(1.0, (chi2_site / dof_site) ** 0.5)
                else:
                    mad = float(np.median(np.abs(arr - np.median(arr))))
                    kappa = max(1.0, mad / 0.6744898)
                kappas[site] = float(np.clip(kappa, 1.0, max_kappa))
            return kappas

        method = estimate_site_scales_method
        if method not in ("mad", "chi2", "iterative", "studentt_em"):
            method = "mad"

        alpha = float(estimate_site_scales_alpha)
        tol_kappa = float(estimate_site_scales_tol)
        sigma_floor_val = float(sigma_floor)
        nu_val_local = max(1.0, float(nu))

        sites = sorted({(ob.site or "UNK") for ob in observations})
        site_kappas: dict[str, float] = {s: 1.0 for s in sites}

        if method in ("mad", "chi2"):
            try:
                prelim = fit_orbit(
                    target,
                    observations,
                    perturbers=perturbers,
                    max_iter=max(3, max_iter // 2),
                    tol=tol,
                    max_step=max_step,
                    use_kepler=use_kepler,
                    seed_method=seed_method,
                    likelihood=likelihood,
                    nu=nu,
                    estimate_site_scales=False,
                    internal_run=True,
                )
            except Exception as exc:
                warnings.warn(
                    f"Preliminary fit for site-scale estimation failed: {exc}",
                    RuntimeWarning,
                )
                prelim = None
            if prelim is not None:
                site_kappas = _compute_site_kappas_from_prelim(prelim, method=method)
        elif method == "studentt_em":
            for _ in range(max(1, estimate_site_scales_iters)):
                obs_scaled = copy.deepcopy(observations)
                for ob in obs_scaled:
                    s = ob.site or "UNK"
                    ob.sigma_arcsec = max(
                        sigma_floor_val,
                        float(ob.sigma_arcsec) * float(site_kappas.get(s, 1.0)),
                    )
                try:
                    prelim = fit_orbit(
                        target,
                        obs_scaled,
                        perturbers=perturbers,
                        max_iter=max(3, max_iter // 2),
                        tol=tol,
                        max_step=max_step,
                        use_kepler=use_kepler,
                        seed_method=seed_method,
                        likelihood="studentt",
                        nu=nu,
                        estimate_site_scales=False,
                        internal_run=True,
                    )
                except Exception as exc:
                    warnings.warn(
                        f"Student-t EM preliminary fit failed: {exc}",
                        RuntimeWarning,
                    )
                    prelim = None
                if prelim is None:
                    break

                res = prelim.residuals
                weights = np.zeros_like(res, dtype=float)
                for i, ob in enumerate(observations):
                    s = ob.site or "UNK"
                    sigma_i = max(
                        sigma_floor_val,
                        float(ob.sigma_arcsec) * float(site_kappas.get(s, 1.0)),
                    )
                    for comp in (0, 1):
                        idx = 2 * i + comp
                        tval = res[idx] / sigma_i
                        weights[idx] = (nu_val_local + 1.0) / (nu_val_local + tval * tval)

                new_kappas: dict[str, float] = {}
                for s in sites:
                    idxs = [i for i, ob in enumerate(observations) if (ob.site or "UNK") == s]
                    if not idxs:
                        new_kappas[s] = site_kappas[s]
                        continue
                    wsum = 0.0
                    rss = 0.0
                    for i in idxs:
                        for comp in (0, 1):
                            idx = 2 * i + comp
                            rss += weights[idx] * (res[idx] ** 2)
                            wsum += weights[idx]
                    if wsum <= 0:
                        new_kappas[s] = site_kappas[s]
                        continue
                    s_j = (rss / wsum) ** 0.5
                    sigma_med = float(
                        np.median(
                            [ob.sigma_arcsec for ob in observations if (ob.site or "UNK") == s]
                        )
                    )
                    k_est = max(1.0, s_j / max(1e-12, sigma_med))
                    k_est = float(np.clip(k_est, 1.0, max_kappa))
                    new_kappas[s] = (1.0 - alpha) * site_kappas[s] + alpha * k_est

                max_change = max(abs(new_kappas[s] - site_kappas[s]) for s in sites)
                site_kappas = new_kappas
                if max_change < tol_kappa:
                    break
        else:
            prev_kappas: dict[str, float] | None = None
            for _ in range(max(1, estimate_site_scales_iters)):
                try:
                    prelim = fit_orbit(
                        target,
                        observations,
                        perturbers=perturbers,
                        max_iter=max(3, max_iter // 2),
                        tol=tol,
                        max_step=max_step,
                        use_kepler=use_kepler,
                        seed_method=seed_method,
                        likelihood=likelihood,
                        nu=nu,
                        estimate_site_scales=False,
                        internal_run=True,
                    )
                except Exception as exc:
                    warnings.warn(
                        f"Iterative preliminary fit failed: {exc}",
                        RuntimeWarning,
                    )
                    prelim = None
                if prelim is None:
                    break
                computed = _compute_site_kappas_from_prelim(prelim, method="chi2")
                for s, k in computed.items():
                    computed[s] = float(np.clip(max(1.0, k), 1.0, max_kappa))
                if prev_kappas is not None:
                    all_sites = set(prev_kappas.keys()) | set(computed.keys())
                    max_change = max(abs(prev_kappas.get(s, 1.0) - computed.get(s, 1.0)) for s in all_sites)
                    if max_change < tol_kappa:
                        site_kappas = computed
                        break
                prev_kappas = computed
                for ob in observations:
                    s = ob.site or "UNK"
                    kappa = computed.get(s, 1.0)
                    ob.sigma_arcsec = max(
                        sigma_floor_val, float(ob.sigma_arcsec) * float(kappa)
                    )
            if prev_kappas is not None:
                site_kappas = prev_kappas

        if not site_kappas:
            site_kappas = {(ob.site or "UNK"): 1.0 for ob in observations}

        obs_scaled = copy.deepcopy(observations)
        for ob in obs_scaled:
            site = ob.site or "UNK"
            kappa = site_kappas.get(site, 1.0)
            ob.sigma_arcsec = max(
                sigma_floor_val, float(ob.sigma_arcsec) * float(kappa)
            )

        posterior = fit_orbit(
            target,
            obs_scaled,
            perturbers=perturbers,
            max_iter=max_iter,
            tol=tol,
            max_step=max_step,
            use_kepler=use_kepler,
            seed_method=seed_method,
            likelihood=likelihood,
            nu=nu,
            estimate_site_scales=False,
            internal_run=True,
        )
        posterior.site_kappas = site_kappas
        return posterior

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
    converged = False
    use_studentt = likelihood.lower() == "studentt"
    nu_val = max(1.0, float(nu))

    for it in range(max_iter):
        eps = np.maximum(np.abs(state) * 1e-8, base_eps)
        H, residuals = _jacobian_fd(
            state, epoch, observations, perturbers, eps, max_step, use_kepler=use_kepler
        )

        col_norm = np.linalg.norm(H, axis=0)
        col_norm = np.maximum(col_norm, 1e-12)
        scale = 1.0 / col_norm
        Hs = H * scale
        S = np.diag(scale)
        prior_inv_scaled = S @ prior_inv @ S

        W_total_diag = W_base_diag.copy()
        if use_studentt:
            t = residuals / sigma_vec
            W_total_diag *= (nu_val + 1.0) / (nu_val + t**2)
        W_total = np.diag(W_total_diag)

        A = prior_inv_scaled + Hs.T @ (W_total @ Hs)
        b = -Hs.T @ (W_total @ residuals)
        A_lm = A + lam * np.diag(np.diag(A))

        try:
            delta_prime = np.linalg.solve(A_lm, b)
        except np.linalg.LinAlgError:
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
        except Exception:
            cand_residuals = residuals.copy() + 1e6

        cost = 0.5 * float(residuals.T @ (W_total @ residuals))
        cost_new = 0.5 * float(cand_residuals.T @ (W_total @ cand_residuals))
        if cost_new < cost:
            state = candidate_state
            residuals = cand_residuals
            lam = max(lam * lam_down, 1e-16)
        else:
            lam *= lam_up

        if np.linalg.norm(delta) < tol:
            converged = True
            break

    try:
        cov_scaled = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        cov_scaled = np.linalg.pinv(A)
    cov = S.T @ cov_scaled @ S

    ndof = max(1, 2 * len(observations) - 6)
    try:
        chi2 = float(residuals.T @ (np.diag(W_base_diag) @ residuals))
        chi2_red = chi2 / ndof
        scale_chi2 = float(np.sqrt(max(1.0, chi2_red)))
    except Exception:
        scale_chi2 = 1.0
    robust_scale = _mad_scale(residuals, sigma_vec)
    fit_scale = float(max(scale_chi2, robust_scale, 1.0))
    cov = cov * (fit_scale**2)

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
        fit_scale=fit_scale,
        nu=nu_val if use_studentt else nu_val,
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
    fit_scale = float(data.get("fit_scale", fit.get("fit_scale", 1.0)))
    nu = float(data.get("nu", fit.get("nu", 4.0)))
    site_kappas = data.get("site_kappas", fit.get("site_kappas", {}))

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
        fit_scale=fit_scale,
        nu=nu,
        site_kappas=site_kappas,
    )
