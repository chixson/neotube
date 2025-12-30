from __future__ import annotations

import json
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from astropy.time import Time

from .models import Observation
from .fit_adapt import AdaptiveConfig, adaptively_grow_cloud
from .propagate import (
    _body_posvel,
    _prepare_obs_cache,
    ObsCache,
    default_propagation_ladder,
    predict_radec_from_epoch,
    predict_radec_with_contract,
)
from .ranging import (
    _admissible_ok,
    _phase_func_hg,
    _ranging_reference_observation,
    Attributable,
    build_attributable_vector_fit,
    build_state_from_ranging,
)

AU_KM = 149597870.7
DAY_S = 86400.0

_SCORE_CONTEXT: dict[str, object] = {}
_SCORE_FULL_CONTEXT: dict[str, object] = {}
_PREDICT_CONTEXT: dict[str, object] = {}


def _score_states_contract_chunk(states: np.ndarray) -> np.ndarray:
    ctx = _SCORE_CONTEXT
    obs_chunk = ctx["obs_chunk"]
    ra_pred, dec_pred, _, _ = predict_radec_with_contract(
        states,
        ctx["epoch"],
        obs_chunk,
        epsilon_ast_arcsec=ctx["eps_arcsec"],
        allow_unknown_site=ctx["allow_unknown_site"],
        configs=ctx["ladder"],
    )
    loglikes_local = np.zeros(len(states), dtype=float)
    for idx, ob in enumerate(obs_chunk):
        sigma = _sigma_arcsec(ob, ctx["site_kappas"])
        for i in range(len(states)):
            loglikes_local[i] += _gaussian_loglike(
                float(ra_pred[i, idx]), float(dec_pred[i, idx]), ob, sigma
            )
    return loglikes_local


def _score_states_full_chunk(states: np.ndarray) -> np.ndarray:
    ctx = _SCORE_FULL_CONTEXT
    obs = ctx["obs"]
    loglikes_local = np.zeros(len(states), dtype=float)
    for i, state in enumerate(states):
        ra_pred, dec_pred = predict_radec_from_epoch(
            state,
            ctx["epoch"],
            obs,
            ctx["perturbers"],
            ctx["max_step"],
            use_kepler=ctx["use_kepler"],
            allow_unknown_site=ctx["allow_unknown_site"],
            light_time_iters=ctx["light_time_iters"],
            full_physics=True,
            obs_cache=ctx["obs_cache"],
        )
        ll = 0.0
        for ra, dec, ob in zip(ra_pred, dec_pred, obs):
            ll += _gaussian_loglike(float(ra), float(dec), ob, _sigma_arcsec(ob, ctx["site_kappas"]))
        loglikes_local[i] = ll
    return loglikes_local


def _predict_contract_chunk(
    states: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ctx = _PREDICT_CONTEXT
    return predict_radec_with_contract(
        states,
        ctx["epoch"],
        ctx["obs_chunk"],
        epsilon_ast_arcsec=ctx["eps_arcsec"],
        allow_unknown_site=ctx["allow_unknown_site"],
        configs=ctx["ladder"],
    )


@dataclass
class ReplicaCloud:
    states: np.ndarray
    weights: np.ndarray
    epoch: Time
    metadata: dict[str, Any]


def save_replica_cloud(cloud: ReplicaCloud, path: str) -> None:
    np.savez_compressed(
        path,
        states=cloud.states,
        weights=cloud.weights,
        epoch=cloud.epoch.isot,
        metadata=json.dumps(cloud.metadata, sort_keys=True),
    )


def _tangent_residuals(ra_pred: float, dec_pred: float, ob: Observation) -> tuple[float, float]:
    delta_ra = ((ob.ra_deg - ra_pred + 180.0) % 360.0) - 180.0
    ra_arcsec = delta_ra * math.cos(math.radians(dec_pred)) * 3600.0
    dec_arcsec = (ob.dec_deg - dec_pred) * 3600.0
    return ra_arcsec, dec_arcsec


def _sigma_arcsec(ob: Observation, site_kappas: dict[str, float] | None) -> float:
    kappa = 1.0
    if site_kappas:
        kappa = site_kappas.get(ob.site or "UNK", 1.0)
    return max(1e-6, float(ob.sigma_arcsec) * float(kappa))


def _epsilon_arcsec_for_obs(
    ob: Observation,
    *,
    epsilon_ast_arcsec: float,
    epsilon_ast_scale: float,
    epsilon_ast_floor_arcsec: float,
    epsilon_ast_ceiling_arcsec: float | None,
) -> float:
    eps = epsilon_ast_arcsec
    if epsilon_ast_scale > 0.0:
        eps = max(epsilon_ast_floor_arcsec, float(ob.sigma_arcsec) * epsilon_ast_scale)
        if epsilon_ast_ceiling_arcsec is not None:
            eps = min(eps, epsilon_ast_ceiling_arcsec)
    return eps


def _gaussian_loglike(ra_pred: float, dec_pred: float, ob: Observation, sigma_arcsec: float) -> float:
    dra, ddec = _tangent_residuals(ra_pred, dec_pred, ob)
    res2 = (dra / sigma_arcsec) ** 2 + (ddec / sigma_arcsec) ** 2
    return float(-0.5 * res2)


def _systematic_resample(states: np.ndarray, weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(weights)
    positions = (rng.random() + np.arange(n)) / n
    cumulative = np.cumsum(weights)
    out = np.empty_like(states)
    i = 0
    for j, pos in enumerate(positions):
        while pos > cumulative[i]:
            i += 1
        out[j] = states[i]
    return out


def _stratified_log_rho_samples(
    n_samples: int,
    rho_min_km: float,
    rho_max_km: float,
    rng: np.random.Generator,
    decades: int = 8,
) -> np.ndarray:
    lo = math.log10(max(1e-12, rho_min_km))
    hi = math.log10(max(rho_min_km, rho_max_km))
    span = hi - lo
    if span <= 0:
        return np.full(n_samples, rho_min_km, dtype=float)
    bins = max(1, min(decades, int(math.ceil(span))))
    per_bin = max(1, int(math.ceil(n_samples / bins)))
    out: list[float] = []
    edges = np.linspace(lo, hi, bins + 1)
    for i in range(bins):
        a = edges[i]
        b = edges[i + 1]
        count = min(per_bin, n_samples - len(out))
        if count <= 0:
            break
        logs = rng.random(count) * (b - a) + a
        out.extend((10.0 ** logs).tolist())
    while len(out) < n_samples:
        val = 10.0 ** (rng.random() * (hi - lo) + lo)
        out.append(val)
    return np.array(out[:n_samples], dtype=float)


def _photometry_loglike_marginal(
    sum_inv_var: np.ndarray,
    sum_y_inv_var: np.ndarray,
    sum_y2_inv_var: np.ndarray,
    n: int,
    sigma_h: float,
    logdet_v: float,
) -> np.ndarray:
    denom = 1.0 / (sigma_h**2) + sum_inv_var
    quad = sum_y2_inv_var - (sum_y_inv_var**2) / denom
    logterm = np.log1p((sigma_h**2) * sum_inv_var)
    return -0.5 * (quad + logdet_v + logterm + n * np.log(2.0 * np.pi))


def _photometry_accumulate(
    states: np.ndarray,
    ob: Observation,
    obs_helio: np.ndarray,
    mu_h: float,
    g: float,
    sum_inv_var: np.ndarray,
    sum_y_inv_var: np.ndarray,
    sum_y2_inv_var: np.ndarray,
) -> None:
    if ob.mag is None:
        return
    sigma_mag = float(ob.sigma_mag) if ob.sigma_mag is not None else 0.15
    r_vec = states[:, :3]
    r_norm = np.linalg.norm(r_vec, axis=1)
    topo_vec = r_vec - obs_helio
    topo_norm = np.linalg.norm(topo_vec, axis=1)
    denom = np.maximum(r_norm * topo_norm, 1e-12)
    cos_phase = np.sum(r_vec * (-topo_vec), axis=1) / denom
    cos_phase = np.clip(cos_phase, -1.0, 1.0)
    phase = np.arccos(cos_phase)
    phi = np.array([max(_phase_func_hg(float(p), g), 1e-12) for p in phase], dtype=float)
    r_au = r_norm / AU_KM
    delta_au = topo_norm / AU_KM
    b_pred = 5.0 * np.log10(np.maximum(r_au * delta_au, 1e-12)) - 2.5 * np.log10(phi)
    y = float(ob.mag) - b_pred - mu_h
    inv_var = 1.0 / max(1e-12, sigma_mag**2)
    sum_inv_var += inv_var
    sum_y_inv_var += y * inv_var
    sum_y2_inv_var += (y * y) * inv_var


def sequential_fit_replicas(
    observations: Sequence[Observation],
    *,
    n_particles: int = 2000,
    rho_min_au: float = 1e-5,
    rho_max_au: float = 100.0,
    rhodot_max_km_s: float = 120.0,
    v_max_km_s: float = 120.0,
    perturbers: Sequence[str] = ("earth", "mars", "jupiter"),
    max_step: float = 3600.0,
    use_kepler: bool = True,
    ess_threshold: float = 0.3,
    rejuvenation_scale: float = 0.05,
    light_time_iters: int = 2,
    allow_unknown_site: bool = True,
    full_physics_final: bool = True,
    mu_h: float = 20.0,
    sigma_h: float = 3.0,
    g: float = 0.15,
    epsilon_ast_arcsec: float = 0.1,
    epsilon_ast_scale: float = 0.0,
    epsilon_ast_floor_arcsec: float = 0.01,
    epsilon_ast_ceiling_arcsec: float | None = None,
    shadow_diagnostics: bool = True,
    diagnostics_path: str | None = None,
    auto_grow: bool = False,
    auto_n_max: int = 50000,
    auto_n_add: int = 3000,
    auto_ess_target: float = 500.0,
    auto_psis_khat: float = 0.7,
    auto_logrho_bins: int = 8,
    auto_min_per_decade: int = 20,
    auto_w_min_mode: float = 0.001,
    auto_min_per_mode: int = 20,
    workers: int | None = None,
    chunk_size: int = 512,
    site_kappas: dict[str, float] | None = None,
    seed: int | None = None,
    admissible_bound: bool = True,
    admissible_q_min_au: float | None = None,
    admissible_q_max_au: float | None = None,
) -> ReplicaCloud:
    """Fit an SMC replica cloud from observations and return the final weighted states."""
    if len(observations) < 3:
        raise ValueError("Need at least 3 observations to seed SMC.")

    obs = sorted(observations, key=lambda ob: ob.time)
    obs3 = obs[:3]
    epoch = obs3[1].time
    rng = np.random.default_rng(seed)
    ladder = default_propagation_ladder(max_step=max_step)
    if not use_kepler:
        ladder = [cfg for cfg in ladder if cfg.model != "kepler"]
    worker_count = max(1, int(workers or 1))
    chunk_size = max(64, int(chunk_size))

    def _score_chunks(
        pool_states: np.ndarray,
        obs_chunk: Sequence[Observation],
        eps_arcsec: float,
    ) -> np.ndarray:
        _SCORE_CONTEXT["epoch"] = epoch
        _SCORE_CONTEXT["obs_chunk"] = obs_chunk
        _SCORE_CONTEXT["eps_arcsec"] = eps_arcsec
        _SCORE_CONTEXT["allow_unknown_site"] = allow_unknown_site
        _SCORE_CONTEXT["ladder"] = ladder
        _SCORE_CONTEXT["site_kappas"] = site_kappas
        return _score_states_contract_chunk(pool_states)

    def _score_parallel(
        pool_states: np.ndarray,
        obs_chunk: Sequence[Observation],
        eps_arcsec: float,
    ) -> np.ndarray:
        _SCORE_CONTEXT["epoch"] = epoch
        _SCORE_CONTEXT["obs_chunk"] = obs_chunk
        _SCORE_CONTEXT["eps_arcsec"] = eps_arcsec
        _SCORE_CONTEXT["allow_unknown_site"] = allow_unknown_site
        _SCORE_CONTEXT["ladder"] = ladder
        _SCORE_CONTEXT["site_kappas"] = site_kappas
        if worker_count <= 1 or len(pool_states) <= chunk_size:
            return _score_states_contract_chunk(pool_states)
        tasks = []
        for start in range(0, len(pool_states), chunk_size):
            tasks.append(pool_states[start : start + chunk_size])
        out = np.empty(len(pool_states), dtype=float)
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            for offset, result in zip(
                range(0, len(pool_states), chunk_size),
                executor.map(_score_states_contract_chunk, tasks),
            ):
                out[offset : offset + len(result)] = result
        return out

    def _score_full_parallel(pool_states: np.ndarray, obs_cache_all: ObsCache) -> np.ndarray:
        _SCORE_FULL_CONTEXT["epoch"] = epoch
        _SCORE_FULL_CONTEXT["obs"] = obs
        _SCORE_FULL_CONTEXT["perturbers"] = perturbers
        _SCORE_FULL_CONTEXT["max_step"] = max_step
        _SCORE_FULL_CONTEXT["use_kepler"] = use_kepler
        _SCORE_FULL_CONTEXT["allow_unknown_site"] = allow_unknown_site
        _SCORE_FULL_CONTEXT["light_time_iters"] = light_time_iters
        _SCORE_FULL_CONTEXT["site_kappas"] = site_kappas
        _SCORE_FULL_CONTEXT["obs_cache"] = obs_cache_all
        if worker_count <= 1 or len(pool_states) <= chunk_size:
            return _score_states_full_chunk(pool_states)
        tasks = []
        for start in range(0, len(pool_states), chunk_size):
            tasks.append(pool_states[start : start + chunk_size])
        out = np.empty(len(pool_states), dtype=float)
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            for offset, result in zip(
                range(0, len(pool_states), chunk_size),
                executor.map(_score_states_full_chunk, tasks),
            ):
                out[offset : offset + len(result)] = result
        return out

    def _predict_contract_parallel(
        pool_states: np.ndarray,
        obs_chunk: Sequence[Observation],
        eps_arcsec: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        _PREDICT_CONTEXT["epoch"] = epoch
        _PREDICT_CONTEXT["obs_chunk"] = obs_chunk
        _PREDICT_CONTEXT["eps_arcsec"] = eps_arcsec
        _PREDICT_CONTEXT["allow_unknown_site"] = allow_unknown_site
        _PREDICT_CONTEXT["ladder"] = ladder
        if worker_count <= 1 or len(pool_states) <= chunk_size:
            return _predict_contract_chunk(pool_states)
        tasks = []
        for start in range(0, len(pool_states), chunk_size):
            tasks.append(pool_states[start : start + chunk_size])
        ra_parts = []
        dec_parts = []
        lvl_parts = []
        delta_parts = []
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            for result in executor.map(_predict_contract_chunk, tasks):
                ra, dec, levels, deltas = result
                ra_parts.append(ra)
                dec_parts.append(dec)
                lvl_parts.append(levels)
                delta_parts.append(deltas)
        return (
            np.vstack(ra_parts),
            np.vstack(dec_parts),
            np.concatenate(lvl_parts),
            np.concatenate(delta_parts),
        )

    attrib = build_attributable_vector_fit(obs3, epoch)
    sigma_arcsec = float(np.median([o.sigma_arcsec for o in obs3]))
    sigma_deg = sigma_arcsec / 3600.0
    dt_days = max(1e-6, float((obs3[-1].time - obs3[0].time).to_value("day")))
    sigma_rate_deg_per_day = sigma_deg / dt_days
    attrib_mean = np.array(
        [attrib.ra_deg, attrib.dec_deg, attrib.ra_dot_deg_per_day, attrib.dec_dot_deg_per_day],
        dtype=float,
    )
    attrib_cov = np.diag(
        [
            sigma_deg**2,
            sigma_deg**2,
            sigma_rate_deg_per_day**2,
            sigma_rate_deg_per_day**2,
        ]
    )
    attrib_samples = rng.multivariate_normal(attrib_mean, attrib_cov, size=n_particles)

    rho_samples = _stratified_log_rho_samples(
        n_particles, rho_min_au * AU_KM, rho_max_au * AU_KM, rng
    )
    rhodot_samples = rng.uniform(-rhodot_max_km_s, rhodot_max_km_s, size=n_particles)

    obs_ref = _ranging_reference_observation(obs3, epoch)
    states = np.zeros((n_particles, 6), dtype=float)
    valid = np.ones(n_particles, dtype=bool)
    for i in range(n_particles):
        a = attrib_samples[i]
        attrib_i = Attributable(
            ra_deg=float(a[0]),
            dec_deg=float(a[1]),
            ra_dot_deg_per_day=float(a[2]),
            dec_dot_deg_per_day=float(a[3]),
        )
        state = build_state_from_ranging(obs_ref, epoch, attrib_i, rho_samples[i], rhodot_samples[i])
        if np.linalg.norm(state[3:]) > v_max_km_s:
            valid[i] = False
            states[i] = state
            continue
        if not _admissible_ok(
            state,
            q_min_au=admissible_q_min_au,
            q_max_au=admissible_q_max_au,
            bound_only=admissible_bound,
        ):
            valid[i] = False
            states[i] = state
            continue
        states[i] = state

    states = states[valid]
    if states.size == 0:
        raise RuntimeError("No valid initial particles; widen rho/rhodot bounds.")
    if len(states) < n_particles:
        idx = rng.choice(len(states), size=n_particles, replace=True)
        states = states[idx]

    eps_seed = min(
        _epsilon_arcsec_for_obs(
            ob,
            epsilon_ast_arcsec=epsilon_ast_arcsec,
            epsilon_ast_scale=epsilon_ast_scale,
            epsilon_ast_floor_arcsec=epsilon_ast_floor_arcsec,
            epsilon_ast_ceiling_arcsec=epsilon_ast_ceiling_arcsec,
        )
        for ob in obs3
    )
    ra_pred, dec_pred, used_level, max_delta = _predict_contract_parallel(
        states,
        obs3,
        eps_seed,
    )
    loglikes = np.zeros(len(states), dtype=float)
    for idx, ob in enumerate(obs3):
        sigma = _sigma_arcsec(ob, site_kappas)
        for i in range(len(states)):
            loglikes[i] += _gaussian_loglike(float(ra_pred[i, idx]), float(dec_pred[i, idx]), ob, sigma)
    loglikes -= float(np.max(loglikes))
    weights = np.exp(loglikes)
    weights = weights / np.sum(weights)

    diag_used_levels = []
    diag_max_delta = []
    diag_step_labels: list[str] = []
    if shadow_diagnostics:
        diag_used_levels.append(used_level.copy())
        diag_max_delta.append(max_delta.copy())
        diag_step_labels.append("seed")

    for ob_idx, ob in enumerate(obs[3:], start=3):
        loglikes = np.empty(len(states), dtype=float)
        sigma = _sigma_arcsec(ob, site_kappas)
        eps_ob = _epsilon_arcsec_for_obs(
            ob,
            epsilon_ast_arcsec=epsilon_ast_arcsec,
            epsilon_ast_scale=epsilon_ast_scale,
            epsilon_ast_floor_arcsec=epsilon_ast_floor_arcsec,
            epsilon_ast_ceiling_arcsec=epsilon_ast_ceiling_arcsec,
        )
        ra_pred, dec_pred, used_level, max_delta = _predict_contract_parallel(
            states,
            [ob],
            eps_ob,
        )
        for i in range(len(states)):
            loglikes[i] = _gaussian_loglike(float(ra_pred[i, 0]), float(dec_pred[i, 0]), ob, sigma)
        logw = np.log(weights + 1e-300) + loglikes
        logw -= float(np.max(logw))
        weights = np.exp(logw)
        weights = weights / np.sum(weights)

        if shadow_diagnostics:
            diag_used_levels.append(used_level.copy())
            diag_max_delta.append(max_delta.copy())
            diag_step_labels.append(f"obs_{ob_idx}")

        ess = 1.0 / np.sum(weights**2)
        if ess < ess_threshold * len(weights):
            states = _systematic_resample(states, weights, rng)
            weights = np.full(len(states), 1.0 / len(states), dtype=float)
            if rejuvenation_scale > 0.0 and len(states) > 1:
                cov = np.cov(states.T)
                cov = cov * rejuvenation_scale
                jitter = rng.multivariate_normal(np.zeros(6), cov, size=len(states))
                states = states + jitter

    auto_grow_done = False
    auto_grow_diag: dict | None = None
    if auto_grow:
        obs_ref = _ranging_reference_observation(obs, epoch)
        attrib_mean = attrib_mean
        attrib_cov = attrib_cov

        def _score_contract(pool_states: np.ndarray) -> np.ndarray:
            eps_final = min(
                _epsilon_arcsec_for_obs(
                    ob,
                    epsilon_ast_arcsec=epsilon_ast_arcsec,
                    epsilon_ast_scale=epsilon_ast_scale,
                    epsilon_ast_floor_arcsec=epsilon_ast_floor_arcsec,
                    epsilon_ast_ceiling_arcsec=epsilon_ast_ceiling_arcsec,
                )
                for ob in obs
            )
            return _score_parallel(pool_states, obs, eps_final)

        obs_cache_all = _prepare_obs_cache(obs, allow_unknown_site=allow_unknown_site)

        def _score_full(pool_states: np.ndarray) -> np.ndarray:
            loglikes_local = _score_full_parallel(pool_states, obs_cache_all)
            sum_inv_var = np.zeros(len(pool_states), dtype=float)
            sum_y_inv_var = np.zeros(len(pool_states), dtype=float)
            sum_y2_inv_var = np.zeros(len(pool_states), dtype=float)
            phot_obs = [ob for ob in obs if ob.mag is not None]
            if phot_obs:
                for ob in phot_obs:
                    obs_cache = _prepare_obs_cache([ob], allow_unknown_site=allow_unknown_site)
                    obs_bary = obs_cache.earth_bary_km[0] + obs_cache.site_pos_km[0]
                    sun_pos, _ = _body_posvel("sun", ob.time.tdb)
                    sun_bary = sun_pos.xyz.to_value("km").flatten()
                    obs_helio = obs_bary - sun_bary
                    _photometry_accumulate(
                        pool_states,
                        ob,
                        obs_helio,
                        mu_h,
                        g,
                        sum_inv_var,
                        sum_y_inv_var,
                        sum_y2_inv_var,
                    )
                logdet_v = sum(
                    math.log(
                        max(1e-12, float(ob.sigma_mag) if ob.sigma_mag is not None else 0.15) ** 2
                    )
                    for ob in phot_obs
                )
                loglikes_local += _photometry_loglike_marginal(
                    sum_inv_var,
                    sum_y_inv_var,
                    sum_y2_inv_var,
                    len(phot_obs),
                    sigma_h,
                    logdet_v,
                )
            return loglikes_local

        cfg = AdaptiveConfig(
            n_max=auto_n_max,
            n_add=auto_n_add,
            ess_target=auto_ess_target,
            psis_khat_threshold=auto_psis_khat,
            logrho_bins=auto_logrho_bins,
            min_particles_per_decade=auto_min_per_decade,
            w_min_mode=auto_w_min_mode,
            min_particles_per_mode=auto_min_per_mode,
            rho_prior_mode="log",
            rho_prior_power=2.0,
            rhodot_max_km_s=float(rhodot_max_km_s),
            v_max_km_s=float(v_max_km_s),
        )
        states, weights, auto_grow_diag = adaptively_grow_cloud(
            states,
            obs,
            obs_ref=obs_ref,
            attrib_mean=attrib_mean,
            attrib_cov=attrib_cov,
            rho_min_au=rho_min_au,
            rho_max_au=rho_max_au,
            score_fn=_score_contract,
            final_score_fn=_score_full,
            cfg=cfg,
            rng=rng,
        )
        auto_grow_done = True

    if full_physics_final and not auto_grow_done:
        eps_final = min(
            _epsilon_arcsec_for_obs(
                ob,
                epsilon_ast_arcsec=epsilon_ast_arcsec,
                epsilon_ast_scale=epsilon_ast_scale,
                epsilon_ast_floor_arcsec=epsilon_ast_floor_arcsec,
                epsilon_ast_ceiling_arcsec=epsilon_ast_ceiling_arcsec,
            )
            for ob in obs
        )
        ra_pred, dec_pred, used_level, max_delta = _predict_contract_parallel(
            states,
            obs,
            eps_final,
        )
        obs_cache_all = _prepare_obs_cache(obs, allow_unknown_site=allow_unknown_site)
        loglikes = _score_full_parallel(states, obs_cache_all)
        if shadow_diagnostics:
            diag_used_levels.append(used_level.copy())
            diag_max_delta.append(max_delta.copy())
            diag_step_labels.append("final")

        sum_inv_var = np.zeros(len(states), dtype=float)
        sum_y_inv_var = np.zeros(len(states), dtype=float)
        sum_y2_inv_var = np.zeros(len(states), dtype=float)
        phot_obs = [ob for ob in obs if ob.mag is not None]
        if phot_obs:
            for ob in phot_obs:
                obs_cache = _prepare_obs_cache([ob], allow_unknown_site=allow_unknown_site)
                obs_bary = obs_cache.earth_bary_km[0] + obs_cache.site_pos_km[0]
                sun_pos, _ = _body_posvel("sun", ob.time.tdb)
                sun_bary = sun_pos.xyz.to_value("km").flatten()
                obs_helio = obs_bary - sun_bary
                _photometry_accumulate(
                    states, ob, obs_helio, mu_h, g, sum_inv_var, sum_y_inv_var, sum_y2_inv_var
                )
            logdet_v = sum(
                math.log(
                    max(1e-12, float(ob.sigma_mag) if ob.sigma_mag is not None else 0.15) ** 2
                )
                for ob in phot_obs
            )
            loglikes += _photometry_loglike_marginal(
                sum_inv_var, sum_y_inv_var, sum_y2_inv_var, len(phot_obs), sigma_h, logdet_v
            )

        loglikes -= float(np.max(loglikes))
        weights = np.exp(loglikes)
        weights = weights / np.sum(weights)

    diagnostics: dict[str, float] = {}
    if shadow_diagnostics and diag_max_delta:
        deltas = np.concatenate(diag_max_delta)
        diagnostics = {
            "shadow_max_arcsec": float(np.max(deltas)),
            "shadow_p95_arcsec": float(np.percentile(deltas, 95.0)),
            "shadow_mean_arcsec": float(np.mean(deltas)),
        }
        if diagnostics_path:
            levels = np.stack(diag_used_levels, axis=0)
            deltas_steps = np.stack(diag_max_delta, axis=0)
            step_labels = np.array(diag_step_labels, dtype=str)
            np.savez_compressed(
                diagnostics_path,
                used_levels=levels,
                max_delta_arcsec=deltas_steps,
                step_labels=step_labels,
            )

    metadata = {
        "n_particles": int(len(states)),
        "rho_min_au": float(rho_min_au),
        "rho_max_au": float(rho_max_au),
        "rhodot_max_km_s": float(rhodot_max_km_s),
        "v_max_km_s": float(v_max_km_s),
        "ess_threshold": float(ess_threshold),
        "rejuvenation_scale": float(rejuvenation_scale),
        "full_physics_final": bool(full_physics_final),
        "mu_h": float(mu_h),
        "sigma_h": float(sigma_h),
        "g": float(g),
        "epsilon_ast_arcsec": float(epsilon_ast_arcsec),
        "epsilon_ast_scale": float(epsilon_ast_scale),
        "epsilon_ast_floor_arcsec": float(epsilon_ast_floor_arcsec),
        "epsilon_ast_ceiling_arcsec": (
            float(epsilon_ast_ceiling_arcsec) if epsilon_ast_ceiling_arcsec is not None else None
        ),
        "auto_grow": bool(auto_grow),
        "auto_grow_diag": auto_grow_diag,
    }
    metadata.update(diagnostics)
    return ReplicaCloud(states=states, weights=weights, epoch=epoch, metadata=metadata)
