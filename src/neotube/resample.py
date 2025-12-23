from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


STATE_COLS = ("x_km", "y_km", "z_km", "vx_km_s", "vy_km_s", "vz_km_s")


def _normalize_weights(logw: np.ndarray) -> np.ndarray:
    m = float(np.max(logw))
    ww = np.exp(logw - m)
    s = float(np.sum(ww))
    if not np.isfinite(s) or s <= 0:
        return np.full_like(ww, 1.0 / max(len(ww), 1), dtype=np.float64)
    return ww / s


def systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = int(weights.size)
    if n <= 0:
        return np.zeros(0, dtype=int)
    positions = (rng.random() + np.arange(n)) / n
    cumulative = np.cumsum(weights)
    idx = np.searchsorted(cumulative, positions, side="left")
    idx[idx == n] = n - 1
    return idx


def weighted_mean_and_cov(x: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    x: (n, d)
    w: (n,) normalized to sum 1
    Returns: mean (d,), cov (d,d)
    """
    w = w.astype(np.float64)
    s = float(np.sum(w))
    if not np.isfinite(s) or s <= 0:
        w = np.full_like(w, 1.0 / max(len(w), 1), dtype=np.float64)
    else:
        w = w / s

    mean = np.sum(x * w[:, None], axis=0)
    xc = x - mean[None, :]
    cov = (xc.T * w) @ xc
    cov = 0.5 * (cov + cov.T)
    return mean, cov


@dataclass(frozen=True)
class ResampleConfig:
    liu_west_a: float = 0.99
    ridge: float = 1e-6


def liu_west_move(
    x_parent: np.ndarray,
    mean: np.ndarray,
    cov: np.ndarray,
    *,
    a: float,
    rng: np.random.Generator,
    ridge: float,
) -> np.ndarray:
    """
    Liu-West shrinkage move:
      x_new = a*x_parent + (1-a)*mean + eta
      eta ~ N(0, (1-a^2)*cov)
    """
    a = float(a)
    if not (0 < a < 1):
        raise ValueError("liu_west_a must be in (0,1)")

    d = int(cov.shape[0])
    cov_noise = (1.0 - a * a) * cov
    cov_noise = cov_noise + ridge * np.eye(d)

    try:
        L = np.linalg.cholesky(cov_noise)
    except np.linalg.LinAlgError:
        evals, evecs = np.linalg.eigh(cov_noise)
        evals = np.maximum(evals, ridge)
        L = evecs @ np.diag(np.sqrt(evals))

    z = rng.normal(size=(d,))
    eta = L @ z
    return a * x_parent + (1.0 - a) * mean + eta


def resample_replicas(
    replicas: pd.DataFrame,
    *,
    seed: int,
    cfg: ResampleConfig,
) -> pd.DataFrame:
    """
    Input: replicas DataFrame with STATE_COLS and either logw or w.
    Output: resampled replicas with uniform weights (logw,w).
    """
    missing = [c for c in STATE_COLS if c not in replicas.columns]
    if missing:
        raise ValueError(f"replicas missing required state columns: {missing}")

    if "logw" in replicas.columns:
        logw = replicas["logw"].to_numpy(np.float64)
    elif "w" in replicas.columns:
        w = replicas["w"].to_numpy(np.float64)
        w = np.clip(w, 1e-300, np.inf)
        logw = np.log(w)
    else:
        raise ValueError("replicas must contain 'logw' or 'w'")

    rng = np.random.default_rng(int(seed))
    wnorm = _normalize_weights(logw)

    x = replicas.loc[:, STATE_COLS].to_numpy(np.float64)
    mean, cov = weighted_mean_and_cov(x, wnorm)

    idx = systematic_resample(wnorm, rng)
    out = replicas.iloc[idx].reset_index(drop=True).copy()
    x_par = out.loc[:, STATE_COLS].to_numpy(np.float64)

    moved = np.empty_like(x_par)
    for i in range(moved.shape[0]):
        moved[i] = liu_west_move(
            x_par[i],
            mean,
            cov,
            a=cfg.liu_west_a,
            rng=rng,
            ridge=cfg.ridge,
        )

    out.loc[:, STATE_COLS] = moved

    n = len(out)
    if n <= 0:
        out["logw"] = []
        out["w"] = []
        return out

    out["logw"] = -math.log(n)
    out["w"] = 1.0 / n
    return out

