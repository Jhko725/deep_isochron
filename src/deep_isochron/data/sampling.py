"""Samplers over a ``TimeSeriesDataSource``: how windows are drawn for training.

Trajectories start off the limit cycle and converge to it, so late windows are nearly
periodic and early (transient) windows carry most of the information about the
amplitude dynamics. Two ways to oversample them, kept side by side so they can be
compared:

``weighted_windows``
    One ``grain.MapDataset`` that draws window indices with probability proportional
    to a per-window weight — importance sampling with a known proposal (the same idea
    as torch's ``WeightedRandomSampler``). The policy is a function of the window's
    start time (``transient_weights``), so it is smooth, logged as two numbers, and
    sweepable. grain has no weighted sampler of its own; this one is ``random_map``
    over an infinite index stream, so it is deterministic given the seed and
    checkpointable like any grain dataset.

``mixed_split``
    The two-loader design: cut the source in time, shuffle each half, interleave them
    with ``grain.MapDataset.mix``. ``mix`` is a *deterministic* interleave by weights
    (``mix([range(5), range(7, 10)]) == [0, 7, 1, 8, 2, 9]``), its length is that of the
    shortest input, and the halves must be shuffled *before* mixing to keep the
    proportions random — hence ``.shuffle().repeat()`` on each. The boundary is a hard
    index cut.
"""

from collections.abc import Callable
from typing import Any

import grain
import numpy as np
from jaxtyping import Float

from .dataset import TimeSeriesDataSource


def transient_weights(
    source: TimeSeriesDataSource, *, boost: float, tau: float
) -> Float[np.ndarray, " windows"]:
    """``1 + boost * exp(-(t_start - t_0) / tau)`` per window: windows starting at the
    beginning of a trajectory are drawn ``1 + boost`` times as often as late ones."""
    if boost < 0 or tau <= 0:
        raise ValueError("boost must be >= 0 and tau > 0.")
    t = source.window_start_times()
    return 1.0 + boost * np.exp(-(t - t[0]) / tau)


def weighted_windows(
    source: TimeSeriesDataSource,
    weights: Float[np.ndarray, " windows"] | Callable[[TimeSeriesDataSource], Any],
    *,
    seed: int,
) -> grain.MapDataset:
    """Infinite stream of windows of ``source`` drawn i.i.d. with probability
    proportional to ``weights`` (an array over windows, or a function producing one).
    Uniform weights reproduce shuffled-with-replacement sampling."""
    w = np.asarray(weights(source) if callable(weights) else weights, dtype=np.float64)
    if w.shape != (len(source),) or np.any(w < 0) or not np.any(w > 0):
        raise ValueError("weights must be a non-negative vector over the windows.")
    cdf = np.cumsum(w)
    cdf /= cdf[-1]

    def draw(_, rng: np.random.Generator):
        idx = int(np.searchsorted(cdf, rng.random(), side="right"))
        return source[min(idx, len(source) - 1)]

    return grain.MapDataset.range(len(source)).repeat().seed(seed).random_map(draw)


def mixed_split(
    source: TimeSeriesDataSource,
    split_idx: int,
    *,
    weights: tuple[float, float] = (1.0, 1.0),
    seed: int,
) -> grain.MapDataset:
    """Infinite stream interleaving the transient part (``time < ts[split_idx]``) and
    the rest in the ratio ``weights``, each part shuffled independently."""
    if not source.window_size <= split_idx <= len(source.ts) - source.window_size:
        raise ValueError(
            "split_idx must leave at least one window on each side: "
            f"{source.window_size} <= split_idx <= "
            f"{len(source.ts) - source.window_size}."
        )
    early, late = source.split_time(split_idx)
    ds = [
        grain.MapDataset.source(s).seed(seed + i).shuffle().repeat()
        for i, s in enumerate((early, late))
    ]
    return grain.MapDataset.mix(ds, weights=list(weights))
