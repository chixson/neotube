from __future__ import annotations

from typing import Iterable

import numpy as np


DEFAULT_SEED = 50


def make_rng(seed: int | None) -> np.random.Generator:
    """Create a Generator from an optional seed.

    Use this everywhere to avoid ad-hoc reseeding and keep RNG creation centralized.
    """
    seed_value = DEFAULT_SEED if seed is None else int(seed)
    return np.random.default_rng(seed_value)


def ensure_rng(
    rng: np.random.Generator | None, seed: int | None = None
) -> np.random.Generator:
    """Return rng if provided, otherwise create one from seed."""
    return rng if rng is not None else make_rng(seed)


def spawn_rngs(rng: np.random.Generator, n: int) -> list[np.random.Generator]:
    """Spawn independent child generators from a parent RNG."""
    n = int(n)
    if n <= 0:
        return []
    seed = rng.integers(0, 2**32 - 1, dtype=np.uint32)
    ss = np.random.SeedSequence(int(seed))
    return [np.random.default_rng(child) for child in ss.spawn(n)]


def normalize_seed_sequence(seeds: Iterable[int] | None) -> list[int]:
    """Normalize a seed sequence (e.g., from CLI) into a list of ints."""
    if seeds is None:
        return []
    return [int(s) for s in seeds]
