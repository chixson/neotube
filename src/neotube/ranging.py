from __future__ import annotations

import math
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Any, Iterable, Sequence

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord, get_body_barycentric_posvel
from astropy.time import Time

from .fit import _predict_batch, _site_offset, _site_offset_cached
from .models import Observation

AU_KM = 149597870.7
DAY_S = 86400.0


def tangent_basis_from_unit(s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Given unit LOS s (3,), return orthonormal basis e_alpha, e_delta."""
    z = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(s, z)) > 0.9:
        z = np.array([0.0, 1.0, 0.0])
    e_alpha = np.cross(z, s)
    e_alpha /= np.linalg.norm(e_alpha)
    e_delta = np.cross(s, e_alpha)
    e_delta /= np.linalg.norm(e_delta)
    return e_alpha, e_delta


def add_tangent_jitter(
    states: np.ndarray,
    obs: Sequence[Observation] | str,
    posterior: object,
    n_per_state: int = 10,
    sigma_arcsec: float = 0.5,
    fit_scale: float | None = None,
    site_kappas: dict[str, float] | None = None,
    vel_timescale_sec: float | None = None,
    vel_scale_factor: float = 1.0,
) -> np.ndarray:
    """
    Improved tangent-plane jitter: produces local clouds around each state with
    consistent position and velocity perturbations.

    Parameters
    ----------
    states : (N,6) array
      Input candidate states (x,y,z,vx,vy,vz) in km and km/s.
    obs : observation list or obs CSV path
      Used to compute site offsets and a representative time baseline.
    posterior : posterior object (used for fit_scale, site_kappas lookup)
    n_per_state : int
      Number of jittered samples to generate per input state.
    sigma_arcsec : float
      Angular sigma for tangent jitter (arcsec); multiplied by fit_scale and site kappa.
    fit_scale : float or None
      Use posterior.fit_scale if None.
    site_kappas : dict or None
      Per-site kappa multipliers; if None read from posterior.
    vel_timescale_sec : float or None
      Timescale to map spatial jitter to velocity jitter: v_noise ~ dr / vel_timescale_sec.
      If None, use median(obs span) or 86400s default.
    vel_scale_factor : float
      Additional multiplier applied to computed velocity jitter (>=1.0 increases spread).

    Returns
    -------
    (N*n_per_state,6) np.ndarray of jittered states.
    """
    if site_kappas is None:
        site_kappas = getattr(posterior, "site_kappas", {}) or {}
    if fit_scale is None:
        fit_scale = float(getattr(posterior, "fit_scale", 1.0))

    if isinstance(obs, str):
        from neotube.fit_cli import load_observations

        obs = load_observations(obs, None)

    dt_seconds = 86400.0
    try:
        times = np.array([o.time.jd for o in obs])
        if len(times) >= 2:
            span_days = np.max(times) - np.min(times)
            dt_seconds = max(1.0, span_days * 86400.0)
    except Exception:
        dt_seconds = 86400.0
    if vel_timescale_sec is None:
        vel_timescale_sec = dt_seconds

    site = obs[0].site if len(obs) > 0 else None
    kappa = site_kappas.get(site, 1.0)
    sigma_eff_arcsec = sigma_arcsec * fit_scale * kappa
    sigma_rad = np.deg2rad(sigma_eff_arcsec / 3600.0)

    out = []
    for st in states:
        r = st[:3].astype(float)
        v = st[3:].astype(float)
        rho = np.linalg.norm(r)
        if rho <= 0:
            continue
        s = r / rho
        e_a, e_d = tangent_basis_from_unit(s)
        for _ in range(n_per_state):
            d_alpha = np.random.normal(scale=sigma_rad)
            d_delta = np.random.normal(scale=sigma_rad)
            cosdec = np.sqrt(max(0.0, 1.0 - s[2] ** 2))
            dra_km = rho * cosdec * d_alpha
            ddec_km = rho * d_delta
            dr = e_a * dra_km + e_d * ddec_km
            new_r = r + dr
            v_noise = (dr / max(1.0, vel_timescale_sec)) * vel_scale_factor
            v_noise += np.random.normal(scale=0.1, size=3) * np.linalg.norm(v_noise + 1e-12)
            new_v = v + v_noise
            out.append(np.hstack([new_r, new_v]))
    if len(out) == 0:
        return np.empty((0, 6), dtype=float)
    return np.array(out, dtype=float)


def add_local_multit_jitter(
    states: np.ndarray,
    obs: Sequence[Observation] | str,
    posterior: object,
    n_per_state: int = 10,
    sigma_arcsec: float = 0.5,
    fit_scale: float | None = None,
    site_kappas: dict | None = None,
    vel_scale_factor: float = 1.0,
    df: float = 4.0,
) -> np.ndarray:
    """For each 6D state, sample n_per_state correlated multivariate-t draws.

    - Position std (km) = rho * sigma_rad
    - Velocity std (km/s) = position_std / dt_seconds * vel_scale_factor
    - Covariance is diagonal ([pos_std^2]*3 + [vel_std^2]*3)
    Returns array (N*n_per_state,6).
    """
    from .fit_cli import load_observations

    if site_kappas is None:
        site_kappas = getattr(posterior, "site_kappas", {}) or {}
    if fit_scale is None:
        fit_scale = float(getattr(posterior, "fit_scale", 1.0))

    if isinstance(obs, str):
        obs = load_observations(obs, None)

    if len(obs) >= 2:
        times = np.array([o.time.jd for o in obs])
        dt_days = float(np.max(times) - np.min(times))
        dt_seconds = max(1.0, dt_days * 86400.0)
    else:
        dt_seconds = 86400.0

    site = obs[0].site if len(obs) > 0 else None
    kappa = site_kappas.get(site, 1.0)
    sigma_eff_arcsec = sigma_arcsec * fit_scale * kappa
    sigma_rad = np.deg2rad(sigma_eff_arcsec / 3600.0)

    out = []
    for st in states:
        r = st[:3].astype(float)
        v = st[3:].astype(float)
        rho = np.linalg.norm(r)
        if rho <= 0:
            continue
        pos_std_km = float(rho * sigma_rad)
        vel_std_km_s = float((pos_std_km / max(1.0, dt_seconds)) * vel_scale_factor)

        mean = np.hstack([r, v])
        cov = np.diag([pos_std_km**2] * 3 + [vel_std_km_s**2] * 3)
        d = 6
        for _ in range(n_per_state):
            g = np.random.gamma(df / 2.0, 2.0 / df)
            z = np.random.multivariate_normal(np.zeros(d), cov)
            sample = mean + z / np.sqrt(g)
            out.append(sample)
    if len(out) == 0:
        return np.empty((0, 6), dtype=float)
    return np.vstack(out)


def add_attributable_jitter(
    states: np.ndarray,
    obs: Sequence[Observation] | str,
    posterior: object,
    n_per_state: int = 10,
    sigma_arcsec: float = 0.5,
    fit_scale: float | None = None,
    site_kappas: dict | None = None,
) -> np.ndarray:
    """Jitter states in attributable space (ra, dec, ra_dot, dec_dot)."""
    from .fit_cli import load_observations

    if site_kappas is None:
        site_kappas = getattr(posterior, "site_kappas", {}) or {}
    if fit_scale is None:
        fit_scale = float(getattr(posterior, "fit_scale", 1.0))

    if isinstance(obs, str):
        obs = load_observations(obs, None)

    if len(obs) >= 2:
        times = np.array([o.time.jd for o in obs])
        dt_days = float(np.max(times) - np.min(times))
        dt_seconds = max(1.0, dt_days * DAY_S)
    else:
        dt_seconds = DAY_S

    site = obs[0].site if len(obs) > 0 else None
    kappa = site_kappas.get(site, 1.0)
    sigma_eff_arcsec = sigma_arcsec * fit_scale * kappa
    sigma_rad = np.deg2rad(sigma_eff_arcsec / 3600.0)

    epoch = getattr(posterior, "epoch", None)
    if epoch is None:
        raise RuntimeError("posterior.epoch is required for attributable jitter")

    earth_pos, earth_vel = get_body_barycentric_posvel("earth", epoch)
    sun_pos, sun_vel = get_body_barycentric_posvel("sun", epoch)
    earth_helio = (earth_pos.xyz - sun_pos.xyz).to(u.km).value.flatten()
    earth_vel_helio = (earth_vel.xyz - sun_vel.xyz).to(u.km / u.s).value.flatten()
    site_offset = _site_offset_cached(obs[0])

    out = []
    for st in states:
        r_helio = st[:3].astype(float)
        v_helio = st[3:].astype(float)
        r_geo = r_helio - earth_helio
        v_geo = v_helio - earth_vel_helio
        r_topo = r_geo - site_offset
        rho = np.linalg.norm(r_topo)
        if rho <= 0:
            continue
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

        for _ in range(n_per_state):
            ra_j = ra + np.random.normal(scale=sigma_rad)
            dec_j = dec + np.random.normal(scale=sigma_rad)
            dec_j = np.clip(dec_j, -0.5 * math.pi, 0.5 * math.pi)
            ra_dot_j = ra_dot + np.random.normal(scale=sigma_rad / dt_seconds)
            dec_dot_j = dec_dot + np.random.normal(scale=sigma_rad / dt_seconds)

            attrib = Attributable(
                ra_deg=float(np.degrees(ra_j) % 360.0),
                dec_deg=float(np.degrees(dec_j)),
                ra_dot_deg_per_day=float(np.degrees(ra_dot_j) * DAY_S),
                dec_dot_deg_per_day=float(np.degrees(dec_dot_j) * DAY_S),
            )
            new_state = build_state_from_ranging(obs[0], epoch, attrib, rho, rhodot)
            out.append(new_state)
    if len(out) == 0:
        return np.empty((0, 6), dtype=float)
    return np.vstack(out)


_SPREAD_WORKER_CTX: dict[str, object] = {}


def _init_spread_worker(obs: Sequence[Observation], posterior_small: dict[str, object]) -> None:
    """Initializer for local-spread workers to avoid repeated pickling."""
    global _SPREAD_WORKER_CTX
    _SPREAD_WORKER_CTX["obs"] = obs

    class PosteriorLite:
        def __init__(self, epoch_jd: float, fit_scale: float, site_kappas: dict[str, float], nu: float | None):
            self.epoch = Time(epoch_jd, format="jd")
            self.fit_scale = fit_scale
            self.site_kappas = site_kappas
            self.nu = nu

    _SPREAD_WORKER_CTX["posterior"] = PosteriorLite(
        float(posterior_small.get("epoch_jd", Time.now().jd)),
        float(posterior_small.get("fit_scale", 1.0)),
        dict(posterior_small.get("site_kappas", {})),
        posterior_small.get("nu", None),
    )


def _spread_chunk_worker(job: dict[str, object]) -> np.ndarray:
    """Worker function for local-spread chunk jobs."""
    mode = job["mode"]
    states = job["states"]
    n_per_state = int(job.get("n_per_state", 10))
    sigma_arcsec = float(job.get("sigma_arcsec", 0.5))
    vel_scale_factor = float(job.get("vel_scale_factor", 1.0))
    df = float(job.get("df", 4.0))
    posterior = _SPREAD_WORKER_CTX["posterior"]
    obs = _SPREAD_WORKER_CTX["obs"]

    if mode == "tangent":
        return add_tangent_jitter(
            states,
            obs,
            posterior,
            n_per_state=n_per_state,
            sigma_arcsec=sigma_arcsec,
            fit_scale=posterior.fit_scale,
            site_kappas=posterior.site_kappas,
            vel_timescale_sec=None,
            vel_scale_factor=vel_scale_factor,
        )
    if mode == "multit":
        return add_local_multit_jitter(
            states,
            obs,
            posterior,
            n_per_state=n_per_state,
            sigma_arcsec=sigma_arcsec,
            fit_scale=posterior.fit_scale,
            site_kappas=posterior.site_kappas,
            vel_scale_factor=vel_scale_factor,
            df=df,
        )
    if mode == "attributable":
        return add_attributable_jitter(
            states,
            obs,
            posterior,
            n_per_state=n_per_state,
            sigma_arcsec=sigma_arcsec,
            fit_scale=posterior.fit_scale,
            site_kappas=posterior.site_kappas,
        )
    raise RuntimeError(f"Unknown local-spread mode: {mode}")


def add_local_spread_parallel(
    states: np.ndarray,
    obs: Sequence[Observation] | str,
    posterior: object,
    mode: str = "tangent",
    n_per_state: int = 10,
    sigma_arcsec: float = 0.5,
    fit_scale: float | None = None,
    site_kappas: dict[str, float] | None = None,
    vel_scale_factor: float = 1.0,
    df: float = 4.0,
    n_workers: int = 8,
    chunk_size: int | None = None,
) -> np.ndarray:
    """Parallel driver for local-spread jitter using a process pool."""
    if isinstance(obs, str):
        from neotube.fit_cli import load_observations

        obs = load_observations(obs, None)

    if site_kappas is None:
        site_kappas = getattr(posterior, "site_kappas", {}) or {}
    if fit_scale is None:
        fit_scale = float(getattr(posterior, "fit_scale", 1.0))

    n_states = len(states)
    if n_states == 0:
        return np.empty((0, 6), dtype=float)

    if chunk_size is None:
        chunk_size = min(max(128, n_states // (max(1, n_workers) * 4 + 1)), 4096)

    posterior_small = {
        "epoch_jd": float(getattr(posterior, "epoch", Time.now()).jd),
        "fit_scale": fit_scale,
        "site_kappas": site_kappas,
        "nu": getattr(posterior, "nu", None),
    }

    chunks = []
    for i in range(0, n_states, chunk_size):
        chunks.append(states[i : i + chunk_size])

    jobs = []
    for chunk in chunks:
        jobs.append(
            {
                "mode": mode,
                "states": chunk,
                "n_per_state": n_per_state,
                "sigma_arcsec": sigma_arcsec,
                "vel_scale_factor": float(vel_scale_factor),
                "df": float(df),
            }
        )

    results = []
    with ProcessPoolExecutor(
        max_workers=max(1, int(n_workers)),
        initializer=_init_spread_worker,
        initargs=(obs, posterior_small),
    ) as executor:
        for res in executor.map(_spread_chunk_worker, jobs):
            if isinstance(res, np.ndarray) and res.size:
                results.append(res)

    if not results:
        return np.empty((0, 6), dtype=float)
    return np.vstack(results)


def _rho_log_prior(
    rhos_km: np.ndarray,
    rho_prior_mode: str | None,
    rho_prior_power: float | None,
) -> np.ndarray:
    if rho_prior_mode is None:
        power = 2.0 if rho_prior_power is None else float(rho_prior_power)
        return power * np.log(np.maximum(rhos_km, 1e-12))
    mode = rho_prior_mode.lower()
    if mode == "volume":
        return 2.0 * np.log(np.maximum(rhos_km, 1e-12))
    if mode == "uniform":
        return np.zeros_like(rhos_km, dtype=float)
    if mode == "log":
        return -1.0 * np.log(np.maximum(rhos_km, 1e-12))
    raise ValueError(f"Unknown rho_prior_mode={rho_prior_mode!r}")


def _stratified_indices(weights: np.ndarray, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    if n_samples <= 0:
        return np.empty(0, dtype=int)
    cum = np.cumsum(weights)
    if cum[-1] <= 0.0:
        raise ValueError("Weights must sum to > 0 for resampling.")
    positions = (rng.random(n_samples) + np.arange(n_samples)) / n_samples
    return np.searchsorted(cum, positions, side="right")


def stratified_resample(
    states: np.ndarray,
    weights: np.ndarray,
    nrep: int,
    n_clusters: int = 1,
    jitter_scale: float = 0.0,
    nu: float | None = None,
    seed: int | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    weights = np.asarray(weights, dtype=float)
    weights = weights / np.sum(weights)
    nrep = int(nrep)
    if nrep <= 0:
        return np.empty((0, states.shape[1]), dtype=float)

    labels = np.zeros(len(states), dtype=int)
    if n_clusters > 1:
        try:
            from sklearn.cluster import KMeans

            kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=seed)
            labels = kmeans.fit_predict(states[:, :3])
        except Exception:
            labels = np.zeros(len(states), dtype=int)

    unique = np.unique(labels)
    if len(unique) == 1:
        idx = _stratified_indices(weights, nrep, rng)
        reps = states[idx]
    else:
        reps = []
        cluster_weights = np.array([weights[labels == lab].sum() for lab in unique], dtype=float)
        cluster_weights = cluster_weights / np.sum(cluster_weights)
        n_per = np.floor(cluster_weights * nrep).astype(int)
        remainder = nrep - n_per.sum()
        if remainder > 0:
            extra = rng.choice(len(unique), size=remainder, replace=True, p=cluster_weights)
            for lab in extra:
                n_per[lab] += 1
        for lab, count in zip(unique, n_per):
            if count <= 0:
                continue
            mask = labels == lab
            w = weights[mask]
            w = w / np.sum(w)
            idx = _stratified_indices(w, count, rng)
            reps.append(states[mask][idx])
        reps = np.vstack(reps) if reps else np.empty((0, states.shape[1]), dtype=float)

    if jitter_scale > 0.0 and len(reps) > 0:
        std = np.std(states, axis=0)
        jitter = rng.standard_normal(reps.shape) * (jitter_scale * std)
        reps = reps + jitter
    return reps


@dataclass(frozen=True)
class Attributable:
    ra_deg: float
    dec_deg: float
    ra_dot_deg_per_day: float
    dec_dot_deg_per_day: float


def build_attributable(observations: Sequence[Observation], epoch: Time) -> Attributable:
    times = np.array([(ob.time - epoch).to(u.day).value for ob in observations], dtype=float)
    ra_deg = np.array([ob.ra_deg for ob in observations], dtype=float)
    dec_deg = np.array([ob.dec_deg for ob in observations], dtype=float)

    ra_rad = np.unwrap(np.deg2rad(ra_deg))
    dec_rad = np.deg2rad(dec_deg)

    A = np.vstack([np.ones_like(times), times]).T
    ra_coef, *_ = np.linalg.lstsq(A, ra_rad, rcond=None)
    dec_coef, *_ = np.linalg.lstsq(A, dec_rad, rcond=None)

    ra0 = float(np.rad2deg(ra_coef[0]))
    dec0 = float(np.rad2deg(dec_coef[0]))
    ra_dot = float(np.rad2deg(ra_coef[1]))
    dec_dot = float(np.rad2deg(dec_coef[1]))
    return Attributable(ra_deg=ra0, dec_deg=dec0, ra_dot_deg_per_day=ra_dot, dec_dot_deg_per_day=dec_dot)


def s_and_sdot(attrib: Attributable) -> tuple[np.ndarray, np.ndarray]:
    ra = math.radians(attrib.ra_deg)
    dec = math.radians(attrib.dec_deg)
    ra_dot = math.radians(attrib.ra_dot_deg_per_day) / DAY_S
    dec_dot = math.radians(attrib.dec_dot_deg_per_day) / DAY_S

    s = np.array(
        [math.cos(dec) * math.cos(ra), math.cos(dec) * math.sin(ra), math.sin(dec)],
        dtype=float,
    )
    sdot = np.array(
        [
            -math.sin(ra) * math.cos(dec) * ra_dot - math.cos(ra) * math.sin(dec) * dec_dot,
            math.cos(ra) * math.cos(dec) * ra_dot - math.sin(ra) * math.sin(dec) * dec_dot,
            math.cos(dec) * dec_dot,
        ],
        dtype=float,
    )
    return s, sdot


def build_state_from_ranging(
    obs: Observation,
    epoch: Time,
    attrib: Attributable,
    rho_km: float,
    rhodot_km_s: float,
) -> np.ndarray:
    s, sdot = s_and_sdot(attrib)
    earth_pos, earth_vel = get_body_barycentric_posvel("earth", epoch)
    sun_pos, sun_vel = get_body_barycentric_posvel("sun", epoch)
    earth_helio = (earth_pos.xyz - sun_pos.xyz).to(u.km).value.flatten()
    earth_vel_helio = (earth_vel.xyz - sun_vel.xyz).to(u.km / u.s).value.flatten()

    site_offset = _site_offset(obs)
    r_geo = site_offset + rho_km * s
    v_geo = rhodot_km_s * s + rho_km * sdot
    r_helio = earth_helio + r_geo
    v_helio = earth_vel_helio + v_geo
    return np.hstack([r_helio, v_helio]).astype(float)


def studentt_loglike(residuals: np.ndarray, sigma_vec: np.ndarray, nu: float) -> float:
    t = (residuals / sigma_vec) ** 2
    return float(-0.5 * np.sum((nu + 1.0) * np.log1p(t / nu)))


def score_candidate(
    state: np.ndarray,
    epoch: Time,
    observations: Sequence[Observation],
    perturbers: Sequence[str],
    max_step: float,
    nu: float,
    site_kappas: dict[str, float],
    use_kepler: bool,
) -> tuple[float, np.ndarray]:
    pred_ra, pred_dec = _predict_batch(
        state, epoch, list(observations), perturbers, max_step, use_kepler=use_kepler
    )
    residuals = []
    sigma_vec = []
    for ra, dec, ob in zip(pred_ra, pred_dec, observations):
        d_ra = ((ob.ra_deg - ra + 180.0) % 360.0) - 180.0
        ra_arcsec = d_ra * math.cos(math.radians(dec)) * 3600.0
        dec_arcsec = (ob.dec_deg - dec) * 3600.0
        residuals.extend([ra_arcsec, dec_arcsec])
        kappa = site_kappas.get(ob.site or "UNK", 1.0)
        sigma = max(1e-6, float(ob.sigma_arcsec) * float(kappa))
        sigma_vec.extend([sigma, sigma])
    res = np.array(residuals, dtype=float)
    sig = np.array(sigma_vec, dtype=float)
    return studentt_loglike(res, sig, nu), res


_RANGE_CTX: dict[str, object] = {}


def _init_worker(
    obs0: Observation,
    epoch: Time,
    attrib: Attributable,
    observations: Sequence[Observation],
    perturbers: Sequence[str],
    max_step: float,
    nu: float,
    site_kappas: dict[str, float],
    use_kepler: bool,
) -> None:
    global _RANGE_CTX
    _RANGE_CTX = {
        "obs0": obs0,
        "epoch": epoch,
        "attrib": attrib,
        "observations": observations,
        "perturbers": perturbers,
        "max_step": max_step,
        "nu": nu,
        "site_kappas": site_kappas,
        "use_kepler": use_kepler,
    }


def _score_chunk(chunk: Sequence[tuple[float, float]]) -> list[tuple[float, float, float, np.ndarray]]:
    ctx = _RANGE_CTX
    out: list[tuple[float, float, float, np.ndarray]] = []
    for rho_km, rhodot_km_s in chunk:
        state = build_state_from_ranging(
            ctx["obs0"],
            ctx["epoch"],
            ctx["attrib"],
            rho_km,
            rhodot_km_s,
        )
        try:
            ll, _ = score_candidate(
                state,
                ctx["epoch"],
                ctx["observations"],
                ctx["perturbers"],
                ctx["max_step"],
                ctx["nu"],
                ctx["site_kappas"],
                ctx["use_kepler"],
            )
        except Exception:
            continue
        out.append((rho_km, rhodot_km_s, ll, state))
    return out


def _init_worker_state(
    epoch: Time,
    observations: Sequence[Observation],
    perturbers: Sequence[str],
    max_step: float,
    nu: float,
    site_kappas: dict[str, float],
    use_kepler: bool,
) -> None:
    global _RANGE_CTX
    _RANGE_CTX = {
        "epoch": epoch,
        "observations": observations,
        "perturbers": perturbers,
        "max_step": max_step,
        "nu": nu,
        "site_kappas": site_kappas,
        "use_kepler": use_kepler,
    }


def _score_state_chunk(states: np.ndarray) -> list[tuple[float, np.ndarray]]:
    ctx = _RANGE_CTX
    out: list[tuple[float, np.ndarray]] = []
    for state in states:
        try:
            ll, _ = score_candidate(
                state,
                ctx["epoch"],
                ctx["observations"],
                ctx["perturbers"],
                ctx["max_step"],
                ctx["nu"],
                ctx["site_kappas"],
                ctx["use_kepler"],
            )
        except Exception:
            ll = -np.inf
        out.append((ll, state))
    return out


def sample_ranged_replicas(
    observations: Sequence[Observation],
    epoch: Time,
    n_replicas: int,
    n_proposals: int,
    rho_min_au: float,
    rho_max_au: float,
    rhodot_max_kms: float,
    perturbers: Sequence[str],
    max_step: float,
    nu: float,
    site_kappas: dict[str, float],
    seed: int,
    log_every: int = 0,
    scoring_mode: str = "kepler",
    n_workers: int = 1,
    chunk_size: int | None = None,
    top_k_nbody: int = 2000,
    rho_prior_power: float = 2.0,
    rho_prior_mode: str | None = "log",
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    obs0 = min(observations, key=lambda ob: abs((ob.time - epoch).to(u.s).value))
    attrib = build_attributable(observations, epoch)
    log_rho_min = math.log(max(1e-12, rho_min_au))
    log_rho_max = math.log(max(rho_min_au, rho_max_au))

    print(
        "[ranging] init n_proposals={} rho=[{:.2e},{:.2e}] AU rhodot=[-{:.1f},{:.1f}] km/s".format(
            n_proposals, rho_min_au, rho_max_au, rhodot_max_kms, rhodot_max_kms
        ),
        flush=True,
    )
    rhos = np.exp(rng.uniform(log_rho_min, log_rho_max, size=n_proposals)) * AU_KM
    rhodots = rng.uniform(-rhodot_max_kms, rhodot_max_kms, size=n_proposals)
    print("[ranging] proposals sampled; scoring...", flush=True)

    proposals = list(zip(rhos, rhodots))
    if n_workers < 1:
        n_workers = 1
    if chunk_size is None:
        chunk_size = max(128, n_proposals // max(1, n_workers))
    chunks = [proposals[i : i + chunk_size] for i in range(0, len(proposals), chunk_size)]

    use_kepler = scoring_mode == "kepler"
    t0 = time.perf_counter()
    if n_workers == 1:
        results = []
        n_fail = 0
        best_ll = -np.inf
        for i, (rho_km, rhodot_km_s) in enumerate(proposals):
            state = build_state_from_ranging(obs0, epoch, attrib, rho_km, rhodot_km_s)
            try:
                ll, _ = score_candidate(
                    state,
                    epoch,
                    observations,
                    perturbers,
                    max_step,
                    nu,
                    site_kappas,
                    use_kepler,
                )
            except Exception:
                ll = -np.inf
                n_fail += 1
            if ll > best_ll:
                best_ll = ll
            results.append((rho_km, rhodot_km_s, ll, state))
            if log_every and ((i + 1) % log_every == 0 or (i + 1) == n_proposals):
                elapsed = time.perf_counter() - t0
                rate = (i + 1) / max(elapsed, 1e-6)
                print(
                    "[ranging] scored {}/{} proposals; best_ll={:.2f}; failures={}; rate={:.1f}/s".format(
                        i + 1, n_proposals, best_ll, n_fail, rate
                    ),
                    flush=True,
                )
        t1 = time.perf_counter()
        results = [r for r in results if np.isfinite(r[2])]
    else:
        with Pool(
            processes=n_workers,
            initializer=_init_worker,
            initargs=(
                obs0,
                epoch,
                attrib,
                list(observations),
                tuple(perturbers),
                max_step,
                nu,
                site_kappas,
                use_kepler,
            ),
        ) as pool:
            chunk_results = pool.map(_score_chunk, chunks)
        t1 = time.perf_counter()
        results = [r for sub in chunk_results for r in sub]

    if len(results) == 0:
        raise RuntimeError("All ranged proposals failed scoring.")
    rhos_out = np.array([r[0] for r in results], dtype=float)
    rhodots_out = np.array([r[1] for r in results], dtype=float)
    loglikes = np.array([r[2] for r in results], dtype=float)
    states = np.array([r[3] for r in results], dtype=float)

    log_prior = _rho_log_prior(rhos_out, rho_prior_mode, rho_prior_power)
    log_w = loglikes + log_prior
    max_lw = np.max(log_w)
    weights = np.exp(log_w - max_lw)
    weights /= np.sum(weights)

    prefilter_debug: dict[str, Any] | None = None
    if scoring_mode == "kepler" and top_k_nbody > 0:
        top_k = min(top_k_nbody, len(states))
        selected_idx, prefilter_debug = kepler_prefilter_and_select(
            raw_rhos_km=rhos_out,
            kepler_ll=loglikes,
            states=states,
            top_k=top_k,
            n_bins=None,
            au_km=AU_KM,
            emit_debug=False,
            path_to_debug_npz=None,
        )
        top_states = states[selected_idx]
        top_rhos = rhos_out[selected_idx]
        top_rhodots = rhodots_out[selected_idx]
        if n_workers > 1:
            state_chunks = [
                top_states[i : i + chunk_size] for i in range(0, len(top_states), chunk_size)
            ]
            with Pool(
                processes=n_workers,
                initializer=_init_worker_state,
                initargs=(
                    epoch,
                    list(observations),
                    tuple(perturbers),
                    max_step,
                    nu,
                    site_kappas,
                    False,
                ),
            ) as pool:
                chunk_results = pool.map(_score_state_chunk, state_chunks)
            scored = [r for sub in chunk_results for r in sub]
        else:
            scored = []
            for st in top_states:
                ll, _ = score_candidate(
                    st,
                    epoch,
                    observations,
                    perturbers,
                    max_step,
                    nu,
                    site_kappas,
                    False,
                )
                scored.append((ll, st))
        lls = np.array([r[0] for r in scored], dtype=float)
        log_prior_top = _rho_log_prior(top_rhos, rho_prior_mode, rho_prior_power)
        log_w_top = lls + log_prior_top
        max_lw_top = np.max(log_w_top)
        weights = np.exp(log_w_top - max_lw_top)
        weights /= np.sum(weights)
        states = top_states
        rhos_out = top_rhos
        rhodots_out = top_rhodots

    return {
        "states": states,
        "weights": weights,
        "rhos": rhos_out,
        "rhodots": rhodots_out,
        "attrib": attrib,
        "diagnostics": prefilter_debug or {},
    }


def kepler_prefilter_and_select(
    raw_rhos_km: np.ndarray,
    kepler_ll: np.ndarray,
    states: np.ndarray,
    top_k: int,
    n_bins: int | None = None,
    au_km: float = AU_KM,
    emit_debug: bool = False,
    path_to_debug_npz: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Robust Kepler prefilter and stratified selection."""
    M = len(raw_rhos_km)
    if M == 0:
        return np.empty(0, dtype=int), {}

    kepler_ll_sane = np.asarray(kepler_ll, dtype=float).copy()
    is_bad = ~np.isfinite(kepler_ll_sane)
    kepler_invalid_count = int(np.sum(is_bad))
    if kepler_invalid_count > 0:
        finite_mask = np.isfinite(kepler_ll_sane)
        if np.any(finite_mask):
            finite_min = float(np.nanmin(kepler_ll_sane[finite_mask]))
            kepler_ll_sane[is_bad] = finite_min - 1e3
        else:
            kepler_ll_sane[:] = -1e300

    if n_bins is None:
        n_bins = min(20, max(4, int(M // 50)))
    n_bins = max(2, int(n_bins))

    rho_au = (raw_rhos_km / au_km).astype(float)
    bins = np.linspace(float(rho_au.min()), float(rho_au.max()), n_bins + 1)
    bin_idx = np.digitize(rho_au, bins) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    top_k = min(int(top_k), M)
    k_per_bin = max(1, top_k // n_bins)

    selected_indices: list[int] = []
    for b in range(n_bins):
        idxs = np.where(bin_idx == b)[0]
        if idxs.size == 0:
            continue
        ordering = np.argsort(-kepler_ll_sane[idxs])
        pick = idxs[ordering[:k_per_bin]]
        selected_indices.extend(pick.tolist())

    selected_indices = list(dict.fromkeys(selected_indices))
    if len(selected_indices) < top_k:
        needed = top_k - len(selected_indices)
        remaining = np.setdiff1d(np.arange(M), np.array(selected_indices), assume_unique=True)
        global_order = np.argsort(-kepler_ll_sane[remaining])
        extra = remaining[global_order[:needed]]
        selected_indices.extend(extra.tolist())

    selected_indices = np.unique(np.array(selected_indices, dtype=int))[:top_k]

    debug = {
        "raw_rhos": raw_rhos_km,
        "kepler_ll_raw": kepler_ll_sane,
        "prefilter_indices": selected_indices,
        "prefilter_rhos": raw_rhos_km[selected_indices],
        "prefilter_kepler_ll": kepler_ll_sane[selected_indices],
        "n_bins": n_bins,
        "k_per_bin": k_per_bin,
        "kepler_invalid_count": kepler_invalid_count,
    }

    if emit_debug and path_to_debug_npz:
        try:
            np.savez(path_to_debug_npz, **debug, allow_pickle=True)
        except Exception:
            pass

    return np.array(selected_indices, dtype=int), debug
