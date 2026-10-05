"""Shared pieces of the windowing package: element/batch types, start-range
validation, the weight functions and the NumPy categorical draw. Used by both the
elementwise pipeline (``elementwise``) and the batched source (``batched``)."""

from collections.abc import Callable

import numpy as np

from ..dataset import TimeSeriesDataSource


Element = dict[str, np.ndarray]
"""One window (or one trajectory): ``{"t": (L,), "u": (L, dim)}``."""
Batch = dict[str, np.ndarray]
"""A batch of windows: ``{"t": (B, L), "u": (B, L, dim)}``."""
WeightFn = Callable[[np.ndarray], np.ndarray]
"""``t_starts (n,) -> weights (n,)``, non-negative, not all zero."""


def _check_range(length: int, T: int, start_range: tuple[int, int] | None):
    if not 1 <= length <= T:
        raise ValueError(f"window length must be in [1, {T}], got {length}.")
    lo, hi = (0, T - length + 1) if start_range is None else start_range
    if not 0 <= lo < hi <= T - length + 1:
        raise ValueError(
            f"start_range must satisfy 0 <= lo < hi <= {T - length + 1}, got "
            f"{(lo, hi)}."
        )
    return lo, hi


def _cut(element: Element, k: int, length: int) -> Element:
    sel = slice(k, k + length)
    return {"t": element["t"][sel], "u": element["u"][sel]}


def categorical(rng: np.random.Generator, weights: np.ndarray) -> int:
    """One index drawn with probability ``weights / weights.sum()`` (inverse-transform
    sampling on the cumulative sum); the NumPy-``Generator`` counterpart of
    ``jax.random.categorical`` for the grain pipeline. ``weights`` must be non-negative
    with a positive sum."""
    cdf = np.cumsum(np.asarray(weights, dtype=np.float64))
    k = int(np.searchsorted(cdf, rng.random() * cdf[-1], side="right"))
    return min(k, len(cdf) - 1)  # guards u == 1 - eps rounding past the last bin


def transient_weight(boost: float, tau: float) -> WeightFn:
    """``t -> 1 + boost * exp(-(t - t[0]) / tau)``: the first window is drawn
    ``1 + boost`` times as often as late ones, decaying over ``tau``."""
    if boost < 0 or tau <= 0:
        raise ValueError("boost must be >= 0 and tau > 0.")

    def weight(t: np.ndarray) -> np.ndarray:
        return 1.0 + boost * np.exp(-(t - t[0]) / tau)

    return weight


def start_weights(weight: WeightFn, ts: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """``weight`` evaluated on the start times ``ts[lo:hi]``, validated (non-negative,
    not all zero, one value per start)."""
    w = np.asarray(weight(ts[lo:hi]), dtype=np.float64)
    if w.shape != (hi - lo,) or np.any(w < 0) or not np.any(w > 0):
        raise ValueError("weight must map the start times to a non-negative vector.")
    return w


__all__ = [
    "Batch",
    "Element",
    "TimeSeriesDataSource",
    "WeightFn",
    "categorical",
    "start_weights",
    "transient_weight",
]
