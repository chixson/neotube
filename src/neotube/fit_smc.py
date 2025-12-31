from __future__ import annotations

import json
import math
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from astropy.time import Time

from .models import Observation
from .fit_adapt import AdaptiveConfig, adaptively_grow_cloud
from .propagate import (
    _body_posvel_km_single,
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
    build_state_from_ranging_s_sdot,
    build_state_from_ranging,
)

AU_KM = 149597870.7
DAY_S = 86400.0

_SCORE_CONTEXT: dict[str, object] = {}
_SCORE_FULL_CONTEXT: dict[str, object] = {}
_PREDICT_CONTEXT: dict[str, object] = {}

try:
    from scipy.stats import qmc  # type: ignore

    _HAS_SOBOL = True
except Exception:
    qmc = None
    _HAS_SOBOL = False


def _score_states_contract_chunk_args(args: tuple[np.ndarray, dict[str, object]]) -> np.ndarray:
    states, ctx = args
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


def _score_states_full_chunk_args(args: tuple[np.ndarray, dict[str, object]]) -> np.ndarray:
    states, ctx = args
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


def _predict_contract_chunk_args(
    args: tuple[np.ndarray, dict[str, object]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    states, ctx = args
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


def sobol_logrho_rhodot(
    n_points: int,
    rho_min_km: float,
    rho_max_km: float,
    rhodot_min: float,
    rhodot_max: float,
    rng: np.random.Generator,
    *,
    scramble: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    lo = math.log10(max(1e-12, rho_min_km))
    hi = math.log10(max(rho_min_km, rho_max_km))
    if _HAS_SOBOL:
        sampler = qmc.Sobol(d=2, scramble=scramble)
        pts = sampler.random(n_points)
        u1 = pts[:, 0]
        u2 = pts[:, 1]
    else:
        u1 = np.linspace(0.0, 1.0, n_points, endpoint=False) + rng.random(n_points) / n_points
        rng.shuffle(u1)
        u2 = np.linspace(0.0, 1.0, n_points, endpoint=False) + rng.random(n_points) / n_points
        rng.shuffle(u2)
    logrho = lo + u1 * (hi - lo)
    rho = (10.0 ** logrho).astype(float)
    rhodot = (rhodot_min + u2 * (rhodot_max - rhodot_min)).astype(float)
    return rho, rhodot


def coarse_systematic_grid_scan(
    obs_ref: Observation,
    obs3: Sequence[Observation],
    attrib_hat: Attributable,
    epoch: Time,
    *,
    rho_min_km: float,
    rho_max_km: float,
    rhodot_min: float,
    rhodot_max: float,
    nx: int = 40,
    ny: int = 40,
    site_kappas: dict[str, float] | None = None,
    s_seed: np.ndarray | None = None,
    sdot_seed: np.ndarray | None = None,
) -> list[tuple[float, float, float]]:
    log_lo = math.log10(max(1e-12, rho_min_km))
    log_hi = math.log10(max(rho_min_km, rho_max_km))
    logrs = np.linspace(log_lo, log_hi, nx)
    rds = np.linspace(rhodot_min, rhodot_max, ny)
    candidates: list[tuple[float, float, float]] = []
    for lr in logrs:
        rho_km = 10.0 ** lr
        for rd in rds:
            try:
                if s_seed is None or sdot_seed is None:
                    state = build_state_from_ranging(obs_ref, epoch, attrib_hat, rho_km, float(rd))
                else:
                    state = build_state_from_ranging_s_sdot(
                        obs_ref,
                        epoch,
                        s_seed,
                        sdot_seed,
                        rho_km,
                        float(rd),
                    )
                ra_pred, dec_pred = predict_radec_from_epoch(
                    state,
                    epoch,
                    obs3,
                    perturbers=("earth", "mars", "jupiter"),
                    max_step=3600.0,
                    use_kepler=True,
                    allow_unknown_site=True,
                    light_time_iters=1,
                    full_physics=False,
                )
                score = 0.0
                for ra, dec, ob in zip(ra_pred, dec_pred, obs3):
                    sigma = _sigma_arcsec(ob, site_kappas)
                    dra, ddec = _tangent_residuals(float(ra), float(dec), ob)
                    score -= (dra / sigma) ** 2 + (ddec / sigma) ** 2
                candidates.append((float(rho_km), float(rd), float(score)))
            except Exception:
                continue
    if not candidates:
        return []
    candidates.sort(key=lambda x: -x[2])
    keep = max(8, min(64, int(len(candidates) * 0.01)))
    return candidates[:keep]


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
    log_every_obs: int = 1,
    verbose: bool = True,
    checkpoint_path: str | None = None,
    resume: bool = False,
    checkpoint_every_obs: int = 1,
    halt_before_seed_score: bool = False,
    seed_score_limit: int | None = None,
    seed_score_log_every: int = 10,
) -> ReplicaCloud:
    """Fit an SMC replica cloud from observations and return the final weighted states."""
    if len(observations) < 3:
        raise ValueError("Need at least 3 observations to seed SMC.")

    t_start = time.perf_counter()

    def _log(msg: str) -> None:
        if not verbose:
            return
        elapsed = time.perf_counter() - t_start
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[fit_smc {stamp} +{elapsed:8.1f}s] {msg}", flush=True)

    def _save_checkpoint(
        *,
        states: np.ndarray,
        weights: np.ndarray,
        next_obs_index: int,
        stage: str,
        metadata: dict[str, object] | None = None,
    ) -> None:
        if not checkpoint_path:
            return
        ckpt_path = Path(checkpoint_path).expanduser().resolve()
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "next_obs_index": int(next_obs_index),
            "stage": stage,
            "epoch_isot": epoch.isot,
            "metadata": json.dumps(metadata or {}, sort_keys=True),
        }
        rng_state = rng.bit_generator.state
        np.savez_compressed(
            ckpt_path,
            states=states,
            weights=weights,
            rng_state=np.array([rng_state], dtype=object),
            **payload,
        )

    def _load_checkpoint() -> tuple[np.ndarray, np.ndarray, int, str, dict[str, object] | None]:
        if not checkpoint_path:
            raise RuntimeError("checkpoint_path is required for resume.")
        ckpt_path = Path(checkpoint_path).expanduser().resolve()
        data = np.load(ckpt_path, allow_pickle=True)
        states = np.array(data["states"], dtype=float)
        weights = np.array(data["weights"], dtype=float)
        next_obs_index = int(data["next_obs_index"])
        stage = str(data["stage"])
        epoch_isot = str(data["epoch_isot"])
        rng_state = data["rng_state"].item()
        rng.bit_generator.state = rng_state
        meta_raw = str(data.get("metadata", "{}"))
        meta = json.loads(meta_raw) if meta_raw else {}
        return states, weights, next_obs_index, stage, meta, epoch_isot

    obs = sorted(observations, key=lambda ob: ob.time)
    obs3 = obs[:3]
    epoch = obs3[1].time
    rng = np.random.default_rng(seed)
    ladder = default_propagation_ladder(max_step=max_step)
    if not use_kepler:
        ladder = [cfg for cfg in ladder if cfg.model != "kepler"]
    worker_count = max(1, int(workers or 1))
    chunk_size = max(64, int(chunk_size))
    log_every_obs = max(1, int(log_every_obs))

    _log(
        "start n_particles={} obs={} use_kepler={} full_physics_final={} auto_grow={} "
        "workers={} chunk_size={}".format(
            n_particles,
            len(obs),
            bool(use_kepler),
            bool(full_physics_final),
            bool(auto_grow),
            worker_count,
            chunk_size,
        )
    )

    executor: ProcessPoolExecutor | None = None
    if worker_count > 1:
        executor = ProcessPoolExecutor(max_workers=worker_count)

    def _shutdown_executor() -> None:
        if executor is not None:
            executor.shutdown(wait=True)

    checkpoint_every_obs = max(1, int(checkpoint_every_obs))

    def _choose_chunk_size(n: int, base_chunk: int, target_chunks: int = 1000) -> int:
        if n <= 0:
            return max(1, base_chunk)
        desired = int(math.ceil(n / max(1, target_chunks)))
        desired = max(1, desired)
        return min(base_chunk, desired) if base_chunk > 0 else desired

    def _score_chunks(
        pool_states: np.ndarray,
        obs_chunk: Sequence[Observation],
        eps_arcsec: float,
        executor: ProcessPoolExecutor | None,
    ) -> np.ndarray:
        ctx = {
            "epoch": epoch,
            "obs_chunk": obs_chunk,
            "eps_arcsec": eps_arcsec,
            "allow_unknown_site": allow_unknown_site,
            "ladder": ladder,
            "site_kappas": site_kappas,
        }
        if executor is None or len(pool_states) <= chunk_size:
            return _score_states_contract_chunk_args((pool_states, ctx))
        use_chunk = _choose_chunk_size(len(pool_states), chunk_size)
        tasks = []
        for start in range(0, len(pool_states), use_chunk):
            tasks.append((pool_states[start : start + use_chunk], ctx))
        out = np.empty(len(pool_states), dtype=float)
        for offset, result in zip(
            range(0, len(pool_states), use_chunk),
            executor.map(_score_states_contract_chunk_args, tasks),
        ):
            out[offset : offset + len(result)] = result
        return out

    def _score_parallel(
        pool_states: np.ndarray,
        obs_chunk: Sequence[Observation],
        eps_arcsec: float,
        executor: ProcessPoolExecutor | None,
    ) -> np.ndarray:
        return _score_chunks(pool_states, obs_chunk, eps_arcsec, executor)

    def _score_full_parallel(
        pool_states: np.ndarray,
        obs_cache_all: ObsCache,
        executor: ProcessPoolExecutor | None,
    ) -> np.ndarray:
        ctx = {
            "epoch": epoch,
            "obs": obs,
            "perturbers": perturbers,
            "max_step": max_step,
            "use_kepler": use_kepler,
            "allow_unknown_site": allow_unknown_site,
            "light_time_iters": light_time_iters,
            "site_kappas": site_kappas,
            "obs_cache": obs_cache_all,
        }
        if executor is None or len(pool_states) <= chunk_size:
            return _score_states_full_chunk_args((pool_states, ctx))
        use_chunk = _choose_chunk_size(len(pool_states), chunk_size)
        tasks = []
        for start in range(0, len(pool_states), use_chunk):
            tasks.append((pool_states[start : start + use_chunk], ctx))
        out = np.empty(len(pool_states), dtype=float)
        for offset, result in zip(
            range(0, len(pool_states), use_chunk),
            executor.map(_score_states_full_chunk_args, tasks),
        ):
            out[offset : offset + len(result)] = result
        return out

    def _predict_contract_parallel(
        pool_states: np.ndarray,
        obs_chunk: Sequence[Observation],
        eps_arcsec: float,
        executor: ProcessPoolExecutor | None,
        *,
        log_progress: bool = False,
        log_every: int = 10,
        log_label: str = "predict",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ctx = {
            "epoch": epoch,
            "obs_chunk": obs_chunk,
            "eps_arcsec": eps_arcsec,
            "allow_unknown_site": allow_unknown_site,
            "ladder": ladder,
        }
        if executor is None or len(pool_states) <= chunk_size:
            return _predict_contract_chunk_args((pool_states, ctx))
        use_chunk = _choose_chunk_size(len(pool_states), chunk_size)
        tasks = []
        for start in range(0, len(pool_states), use_chunk):
            tasks.append((pool_states[start : start + use_chunk], ctx))
        ra_parts = []
        dec_parts = []
        lvl_parts = []
        delta_parts = []
        total = len(tasks)
        for idx, result in enumerate(executor.map(_predict_contract_chunk_args, tasks), start=1):
            ra, dec, levels, deltas = result
            ra_parts.append(ra)
            dec_parts.append(dec)
            lvl_parts.append(levels)
            delta_parts.append(deltas)
            if log_progress and (idx % max(1, log_every) == 0 or idx == total):
                _log(f"{log_label} chunks {idx}/{total}")
        return (
            np.vstack(ra_parts),
            np.vstack(dec_parts),
            np.concatenate(lvl_parts),
            np.concatenate(delta_parts),
        )

    def _score_seed(states: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        score_states = states
        if seed_score_limit is not None and seed_score_limit > 0:
            score_states = states[: seed_score_limit]
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
            score_states,
            obs3,
            eps_seed,
            executor,
            log_progress=True,
            log_every=seed_score_log_every,
            log_label="seed score",
        )
        loglikes = np.zeros(len(score_states), dtype=float)
        for idx, ob in enumerate(obs3):
            sigma = _sigma_arcsec(ob, site_kappas)
            for i in range(len(score_states)):
                loglikes[i] += _gaussian_loglike(
                    float(ra_pred[i, idx]), float(dec_pred[i, idx]), ob, sigma
                )
        loglikes -= float(np.max(loglikes))
        weights = np.exp(loglikes)
        weights = weights / np.sum(weights)
        ess = 1.0 / np.sum(weights**2)
        _log(
            "seed scoring done ess={:.1f} used_level_max={} shadow_max_arcsec={:.4f} (n={})".format(
                ess,
                int(np.max(used_level)) if len(used_level) else 0,
                float(np.max(max_delta)) if len(max_delta) else 0.0,
                len(score_states),
            )
        )
        if len(score_states) < len(states):
            raise SystemExit(
                "Seed scoring sample complete (n={}); set --fit-smc-seed-score-limit=0 to score full set.".format(
                    len(score_states)
                )
            )
        _save_checkpoint(
            states=states,
            weights=weights,
            next_obs_index=3,
            stage="seeded",
        )
        return weights, used_level, max_delta

    if resume:
        states, weights, next_obs_index, stage, ckpt_meta, epoch_isot = _load_checkpoint()
        epoch = Time(epoch_isot)
        weights = weights / np.sum(weights)
        _log(
            "resumed from checkpoint stage={} next_obs_index={} n={}".format(
                stage, next_obs_index, len(states)
            )
        )
        if stage == "final":
            metadata = ckpt_meta if ckpt_meta is not None else {}
            metadata["resumed_from_checkpoint"] = str(checkpoint_path)
            metadata["checkpoint_stage"] = stage
            _shutdown_executor()
            return ReplicaCloud(states=states, weights=weights, epoch=epoch, metadata=metadata)
        if stage == "pre_seed_score":
            _log("resuming seed scoring from checkpoint")
            weights, used_level, max_delta = _score_seed(states)
            next_obs_index = 3
    else:
        next_obs_index = 3

    # Use the legacy vector fit to keep geometry consistent with the original pipeline.
    attrib, attrib_cov, s_seed, sdot_seed = build_attributable_vector_fit(
        obs3,
        epoch,
        robust=False,
        return_cov=True,
        return_s_sdot=True,
        site_kappas=site_kappas,
    )
    sigma_arcsec = float(np.median([o.sigma_arcsec for o in obs3]))
    sigma_deg = sigma_arcsec / 3600.0
    dt_days = max(1e-6, float((obs3[-1].time - obs3[0].time).to_value("day")))
    sigma_rate_deg_per_day = sigma_deg / dt_days
    attrib_mean = np.array(
        [attrib.ra_deg, attrib.dec_deg, attrib.ra_dot_deg_per_day, attrib.dec_dot_deg_per_day],
        dtype=float,
    )
    if attrib_cov is None or not np.all(np.isfinite(attrib_cov)) or not np.any(attrib_cov):
        attrib_cov = np.diag(
            [
                sigma_deg**2,
                sigma_deg**2,
                sigma_rate_deg_per_day**2,
                sigma_rate_deg_per_day**2,
            ]
        )

    used_level = np.array([], dtype=int)
    max_delta = np.array([], dtype=float)

    if not resume:
        obs_ref = _ranging_reference_observation(obs3, epoch)
        seed_states: list[np.ndarray] = []
        grid_hits = coarse_systematic_grid_scan(
            obs_ref,
            obs3,
            attrib,
            epoch,
            rho_min_km=rho_min_au * AU_KM,
            rho_max_km=rho_max_au * AU_KM,
            rhodot_min=-float(rhodot_max_km_s),
            rhodot_max=float(rhodot_max_km_s),
            nx=40,
            ny=40,
            site_kappas=site_kappas,
            s_seed=s_seed,
            sdot_seed=sdot_seed,
        )
        if grid_hits:
            grid_local = 10
            max_grid = max(0, n_particles // 2)
            if len(grid_hits) * grid_local > max_grid:
                grid_hits = grid_hits[: max(1, max_grid // grid_local)]
            for rho_km, rhodot_km_s, _ in grid_hits:
                for _ in range(grid_local):
                    a = rng.multivariate_normal(attrib_mean, attrib_cov)
                    attrib_i = Attributable(
                        ra_deg=float(a[0]),
                        dec_deg=float(a[1]),
                        ra_dot_deg_per_day=float(a[2]),
                        dec_dot_deg_per_day=float(a[3]),
                    )
                    rho_local = float(rho_km) * (1.0 + rng.normal(scale=0.01))
                    rhodot_local = float(rhodot_km_s) + rng.normal(
                        scale=max(0.1, 0.05 * abs(rhodot_km_s))
                    )
                    try:
                        state = build_state_from_ranging_s_sdot(
                            obs_ref, epoch, s_seed, sdot_seed, rho_local, rhodot_local
                        )
                        seed_states.append(state)
                    except Exception:
                        continue

        n_remaining = max(0, n_particles - len(seed_states))
        rho_sobol, rhodot_sobol = sobol_logrho_rhodot(
            n_remaining,
            rho_min_au * AU_KM,
            rho_max_au * AU_KM,
            -float(rhodot_max_km_s),
            float(rhodot_max_km_s),
            rng,
        )
        attrib_samples = rng.multivariate_normal(attrib_mean, attrib_cov, size=n_remaining)

        states_list: list[np.ndarray] = []
        valid_mask: list[bool] = []
        for state in seed_states:
            states_list.append(state)
            valid_mask.append(True)
        for i in range(n_remaining):
            state = build_state_from_ranging_s_sdot(
                obs_ref, epoch, s_seed, sdot_seed, rho_sobol[i], rhodot_sobol[i]
            )
            if np.linalg.norm(state[3:]) > v_max_km_s:
                states_list.append(state)
                valid_mask.append(False)
                continue
            if not _admissible_ok(
                state,
                q_min_au=admissible_q_min_au,
                q_max_au=admissible_q_max_au,
                bound_only=admissible_bound,
            ):
                states_list.append(state)
                valid_mask.append(False)
                continue
            states_list.append(state)
            valid_mask.append(True)

        states = np.array(states_list, dtype=float)
        valid = np.array(valid_mask, dtype=bool)
        states = states[valid]
        if states.size == 0:
            raise RuntimeError("No valid initial particles; widen rho/rhodot bounds.")
        if len(states) < n_particles:
            idx = rng.choice(len(states), size=n_particles, replace=True)
            states = states[idx]
        _log("seeded {} states (valid={})".format(len(states), int(np.sum(valid))))
        if halt_before_seed_score:
            weights = np.full(len(states), 1.0 / len(states), dtype=float)
            _save_checkpoint(
                states=states,
                weights=weights,
                next_obs_index=0,
                stage="pre_seed_score",
                metadata={"halted_before_seed_score": True},
            )
            _shutdown_executor()
            raise SystemExit(
                "Halted before seed scoring; checkpoint saved to {}".format(checkpoint_path)
            )

        weights, used_level, max_delta = _score_seed(states)

    diag_used_levels = []
    diag_max_delta = []
    diag_step_labels: list[str] = []
    if shadow_diagnostics:
        diag_used_levels.append(used_level.copy())
        diag_max_delta.append(max_delta.copy())
        diag_step_labels.append("seed")

    for ob_idx, ob in enumerate(obs[next_obs_index:], start=next_obs_index):
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
            executor,
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
        if ob_idx % log_every_obs == 0 or ob_idx == len(obs) - 1:
            _log(
                "obs {} ess={:.1f} used_level_max={} shadow_max_arcsec={:.4f}".format(
                    ob_idx,
                    ess,
                    int(np.max(used_level)) if len(used_level) else 0,
                    float(np.max(max_delta)) if len(max_delta) else 0.0,
                )
            )
        if ess < ess_threshold * len(weights):
            states = _systematic_resample(states, weights, rng)
            weights = np.full(len(states), 1.0 / len(states), dtype=float)
            if rejuvenation_scale > 0.0 and len(states) > 1:
                cov = np.cov(states.T)
                cov = cov * rejuvenation_scale
                jitter = rng.multivariate_normal(np.zeros(6), cov, size=len(states))
                states = states + jitter
            _log("resampled ess={:.1f} -> uniform weights".format(ess))

        if checkpoint_path and (ob_idx % checkpoint_every_obs == 0 or ob_idx == len(obs) - 1):
            _save_checkpoint(
                states=states,
                weights=weights,
                next_obs_index=ob_idx + 1,
                stage="assimilating",
            )

    auto_grow_done = False
    auto_grow_diag: dict | None = None
    if auto_grow:
        obs_ref = _ranging_reference_observation(obs, epoch)
        attrib_mean = attrib_mean
        attrib_cov = attrib_cov
        _log(
            "auto-grow start n={} n_max={} n_add={} ess_target={} psis_khat={}".format(
                len(states), auto_n_max, auto_n_add, auto_ess_target, auto_psis_khat
            )
        )

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
            return _score_parallel(pool_states, obs, eps_final, executor)

        obs_cache_all = _prepare_obs_cache(obs, allow_unknown_site=allow_unknown_site)

        def _score_full(pool_states: np.ndarray) -> np.ndarray:
            loglikes_local = _score_full_parallel(pool_states, obs_cache_all, executor)
            sum_inv_var = np.zeros(len(pool_states), dtype=float)
            sum_y_inv_var = np.zeros(len(pool_states), dtype=float)
            sum_y2_inv_var = np.zeros(len(pool_states), dtype=float)
            phot_obs = [ob for ob in obs if ob.mag is not None]
            if phot_obs:
                for ob in phot_obs:
                    obs_cache = _prepare_obs_cache([ob], allow_unknown_site=allow_unknown_site)
                    obs_bary = obs_cache.earth_bary_km[0] + obs_cache.site_pos_km[0]
                    sun_bary, _ = _body_posvel_km_single("sun", ob.time.tdb)
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

        def _auto_grow_checkpoint(
            states_local: np.ndarray, weights_local: np.ndarray, diag: dict[str, object]
        ) -> None:
            _save_checkpoint(
                states=states_local,
                weights=weights_local,
                next_obs_index=len(obs),
                stage="auto_grow",
                metadata={"auto_grow_diag": diag},
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
            log_fn=_log if verbose else None,
            checkpoint_fn=_auto_grow_checkpoint if checkpoint_path else None,
        )
        auto_grow_done = True
        _log("auto-grow done n={}".format(len(states)))

    if full_physics_final and not auto_grow_done:
        _log("full-physics final scoring start n={}".format(len(states)))
        _save_checkpoint(
            states=states,
            weights=weights,
            next_obs_index=len(obs),
            stage="pre_final",
        )
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
            executor,
        )
        obs_cache_all = _prepare_obs_cache(obs, allow_unknown_site=allow_unknown_site)
        loglikes = _score_full_parallel(states, obs_cache_all, executor)
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
                sun_bary, _ = _body_posvel_km_single("sun", ob.time.tdb)
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
        _log("full-physics final scoring done")
        _save_checkpoint(
            states=states,
            weights=weights,
            next_obs_index=len(obs),
            stage="final",
        )

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
    _log("done n={} ess={:.1f}".format(len(states), float(1.0 / np.sum(weights**2))))
    _shutdown_executor()
    return ReplicaCloud(states=states, weights=weights, epoch=epoch, metadata=metadata)
