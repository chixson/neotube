from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .models import Observation
from .propagate import _prepare_obs_cache
from .ranging import Attributable, build_state_from_ranging

AU_KM = 149597870.7

try:
    from scipy.stats import genpareto  # type: ignore

    _HAVE_SCIPY = True
except Exception:
    genpareto = None
    _HAVE_SCIPY = False

try:
    from sklearn.cluster import KMeans  # type: ignore

    _HAVE_SKLEARN = True
except Exception:
    KMeans = None
    _HAVE_SKLEARN = False


@dataclass(frozen=True)
class AdaptiveConfig:
    n_max: int = 50000
    n_add: int = 3000
    ess_target: float = 500.0
    psis_khat_threshold: float = 0.7
    logrho_bins: int = 8
    min_particles_per_decade: int = 20
    w_min_mode: float = 0.001
    min_particles_per_mode: int = 20
    rho_prior_mode: str = "log"
    rho_prior_power: float = 2.0
    rhodot_max_km_s: float = 120.0
    v_max_km_s: float = 120.0


def ess_from_weights(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    total = float(np.sum(w))
    if total <= 0.0:
        return 0.0
    w = w / total
    return float(1.0 / np.sum(w * w))


def _obs_barycentric(obs_ref: Observation) -> np.ndarray:
    obs_cache = _prepare_obs_cache([obs_ref], allow_unknown_site=True)
    return obs_cache.earth_bary_km[0] + obs_cache.site_pos_km[0]


def logrho_bin_coverage(
    states: np.ndarray,
    obs_ref: Observation,
    *,
    rho_min_au: float,
    rho_max_au: float,
    n_bins: int,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    obs_bary = _obs_barycentric(obs_ref)
    r_topo = np.linalg.norm(states[:, :3] - obs_bary[None, :], axis=1)
    r_au = r_topo / AU_KM
    logr = np.log10(np.clip(r_au, max(1e-6, rho_min_au), max(rho_min_au, rho_max_au)))
    edges = np.linspace(math.log10(rho_min_au), math.log10(rho_max_au), n_bins + 1)
    counts, _ = np.histogram(logr, bins=edges)
    mass = np.zeros(n_bins, dtype=float)
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        if w.sum() > 0:
            w = w / w.sum()
        bin_idx = np.clip(np.digitize(logr, edges) - 1, 0, n_bins - 1)
        for i in range(n_bins):
            mass[i] = float(np.sum(w[bin_idx == i]))
    return counts, mass, edges


def compute_psis_khat(log_weights: np.ndarray) -> float:
    lw = np.asarray(log_weights, dtype=float)
    if lw.size < 20:
        return 1.0
    lw = lw - np.max(lw)
    w = np.exp(lw)
    n = w.size
    tail_n = max(10, int(0.2 * n))
    w_sorted = np.sort(w)
    thresh = w_sorted[-tail_n]
    tail = w_sorted[-tail_n:]
    if _HAVE_SCIPY:
        try:
            excess = tail - thresh
            shape, _, _ = genpareto.fit(excess, floc=0.0)
            return float(shape)
        except Exception:
            pass
    ratios = np.clip(tail / max(thresh, 1e-300), 1.0, None)
    return float(np.mean(np.log(ratios)))


def _cluster_modes(
    states: np.ndarray,
    weights: np.ndarray,
    n_clusters: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if not _HAVE_SKLEARN or n_clusters <= 1 or len(states) < n_clusters:
        return np.zeros(len(states), dtype=int), np.array([float(weights.sum())])
    X = states.astype(float)
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std <= 0] = 1.0
    Xn = (X - mean) / std
    km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=int(rng.integers(0, 2**31 - 1)))
    labels = km.fit_predict(Xn)
    w = weights / max(1e-12, float(np.sum(weights)))
    mode_weights = np.array([float(np.sum(w[labels == i])) for i in range(n_clusters)], dtype=float)
    return labels, mode_weights


def _rho_prior_logprob(rho_au: np.ndarray, mode: str, power: float) -> np.ndarray:
    rho = np.clip(rho_au, 1e-12, None)
    if mode == "uniform":
        return np.zeros_like(rho)
    if mode == "volume":
        return power * np.log(rho)
    return -np.log(rho)


def augment_particles_stratified(
    attrib_mean: np.ndarray,
    attrib_cov: np.ndarray,
    obs_ref: Observation,
    rho_bins: Sequence[tuple[float, float]],
    n_add: int,
    *,
    rhodot_max_km_s: float,
    v_max_km_s: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if n_add <= 0 or not rho_bins:
        return np.empty((0, 6), dtype=float)
    per_bin = max(1, int(math.ceil(n_add / len(rho_bins))))
    states_out: list[np.ndarray] = []
    for lo, hi in rho_bins:
        count = min(per_bin, n_add - len(states_out))
        if count <= 0:
            break
        logs = rng.random(count) * (hi - lo) + lo
        rhos = (10.0 ** logs) * AU_KM
        attrib_samples = rng.multivariate_normal(attrib_mean, attrib_cov, size=count)
        rhodots = rng.uniform(-rhodot_max_km_s, rhodot_max_km_s, size=count)
        for a, rho_km, rhodot in zip(attrib_samples, rhos, rhodots):
            attrib = Attributable(
                ra_deg=float(a[0]),
                dec_deg=float(a[1]),
                ra_dot_deg_per_day=float(a[2]),
                dec_dot_deg_per_day=float(a[3]),
            )
            state = build_state_from_ranging(obs_ref, obs_ref.time, attrib, float(rho_km), float(rhodot))
            if np.linalg.norm(state[3:]) > v_max_km_s:
                continue
            states_out.append(state)
    if not states_out:
        return np.empty((0, 6), dtype=float)
    return np.vstack(states_out)


def adaptively_grow_cloud(
    states: np.ndarray,
    observations: Sequence[Observation],
    *,
    obs_ref: Observation,
    attrib_mean: np.ndarray,
    attrib_cov: np.ndarray,
    rho_min_au: float,
    rho_max_au: float,
    score_fn: Callable[[np.ndarray], np.ndarray],
    final_score_fn: Callable[[np.ndarray], np.ndarray] | None,
    cfg: AdaptiveConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict]:
    diagnostics: dict[str, object] = {"iterations": []}
    total_states = states.copy()
    while True:
        loglikes = score_fn(total_states)
        obs_bary = _obs_barycentric(obs_ref)
        rho_au = np.linalg.norm(total_states[:, :3] - obs_bary[None, :], axis=1) / AU_KM
        logprior = _rho_prior_logprob(rho_au, cfg.rho_prior_mode, cfg.rho_prior_power)
        logw = loglikes + logprior
        logw -= np.max(logw)
        weights = np.exp(logw)
        weights = weights / np.sum(weights)

        ess = ess_from_weights(weights)
        khat = compute_psis_khat(logw)
        counts, mass, edges = logrho_bin_coverage(
            total_states,
            obs_ref,
            rho_min_au=rho_min_au,
            rho_max_au=rho_max_au,
            n_bins=cfg.logrho_bins,
            weights=weights,
        )
        mode_labels, mode_weights = _cluster_modes(
            total_states,
            weights,
            n_clusters=min(6, max(2, cfg.logrho_bins)),
            rng=rng,
        )
        mode_counts = np.array(
            [int(np.sum(mode_labels == i)) for i in range(len(mode_weights))], dtype=int
        )
        diagnostics["iterations"].append(
            {
                "n": int(len(total_states)),
                "ess": float(ess),
                "psis_khat": float(khat),
                "logrho_counts": counts.tolist(),
                "logrho_mass": mass.tolist(),
                "mode_weights": mode_weights.tolist(),
                "mode_counts": mode_counts.tolist(),
            }
        )

        passes = True
        if ess < cfg.ess_target:
            passes = False
        if khat >= cfg.psis_khat_threshold:
            passes = False
        if np.any(counts < cfg.min_particles_per_decade):
            passes = False
        if np.any((mass > cfg.w_min_mode) & (counts < cfg.min_particles_per_mode)):
            passes = False
        if np.any((mode_weights > cfg.w_min_mode) & (mode_counts < cfg.min_particles_per_mode)):
            passes = False

        if passes or len(total_states) >= cfg.n_max:
            diagnostics["final_n"] = int(len(total_states))
            diagnostics["final_ess"] = float(ess)
            diagnostics["final_psis_khat"] = float(khat)
            diagnostics["logrho_edges"] = edges.tolist()
            if final_score_fn is not None:
                final_loglikes = final_score_fn(total_states)
                logw = final_loglikes + logprior
                logw -= np.max(logw)
                weights = np.exp(logw)
                weights = weights / np.sum(weights)
            return total_states, weights, diagnostics

        weak_bins = []
        for i in range(len(counts)):
            if counts[i] < cfg.min_particles_per_decade:
                weak_bins.append((edges[i], edges[i + 1]))
            elif mass[i] > cfg.w_min_mode and counts[i] < cfg.min_particles_per_mode:
                weak_bins.append((edges[i], edges[i + 1]))
        for i in range(len(mode_weights)):
            if mode_weights[i] > cfg.w_min_mode and mode_counts[i] < cfg.min_particles_per_mode:
                if len(counts) > 0:
                    weak_bins.append((edges[0], edges[-1]))

        new_states = augment_particles_stratified(
            attrib_mean,
            attrib_cov,
            obs_ref,
            weak_bins,
            cfg.n_add,
            rhodot_max_km_s=cfg.rhodot_max_km_s,
            v_max_km_s=cfg.v_max_km_s,
            rng=rng,
        )
        if new_states.size == 0:
            diagnostics["final_n"] = int(len(total_states))
            diagnostics["final_ess"] = float(ess)
            diagnostics["final_psis_khat"] = float(khat)
            diagnostics["logrho_edges"] = edges.tolist()
            return total_states, weights, diagnostics
        total_states = np.vstack([total_states, new_states])
