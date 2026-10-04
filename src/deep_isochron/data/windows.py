"""Windowing transforms: how training windows are cut from whole trajectories.

The source (``TimeSeriesDataSource``) yields whole trajectories ``{"t": ts, "u": u}``; a
``grain.transforms.RandomMap`` applied after ``.shuffle().repeat()`` cuts one window of
``length`` steps per visit, drawing the start index afresh every epoch (grain seeds the
per-element generator by the *global* index, so a repeated element gets a new draw).
This is the swirl-dynamics ``RandomSection`` pattern; the source carries no window
bookkeeping.

Trajectories start off the limit cycle and converge to it, so late windows are nearly
periodic and early (transient) windows carry most of the information about the amplitude
dynamics. Two ways to oversample them, kept side by side so they can be compared:

``WeightedWindow``
    draws the start index with probability proportional to a weight ``w(t_start)`` —
    importance sampling with a known proposal (torch's ``WeightedRandomSampler`` idea,
    per trajectory). ``transient_weight(boost, tau)`` is the smooth default,
    ``1 + boost·exp(-(t - t_0)/tau)``: two numbers to log and sweep.
``mixed_windows``
    the two-loader design: windows starting before ``ts[split_idx]`` and windows
    starting after it, each a ``RandomWindow`` with a restricted ``start_range``,
    interleaved by ``grain.MapDataset.mix`` in a fixed ratio. ``mix`` is a
    *deterministic* interleave by weights whose length is that of the shortest input and
    whose inputs must be shuffled beforehand — hence ``.shuffle().repeat()`` on each. A
    hard boundary, by construction.

``windows(source, length, ...)`` wires a source into either pipeline;
``validation_windows`` is the finite, deterministic counterpart for evaluation, and
``to_device`` the prefetching transfer to the accelerator for training runs.
"""

from collections.abc import Callable
from dataclasses import dataclass

import grain
import jax
import numpy as np

from .dataset import TimeSeriesDataSource


Element = dict[str, np.ndarray]
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


@dataclass(frozen=True)
class RandomWindow(grain.transforms.RandomMap):
    """One window of ``length`` steps with a uniformly random start in ``start_range``
    (``[lo, hi)`` of start indices; default every valid start)."""

    length: int
    start_range: tuple[int, int] | None = None

    def random_map(self, element: Element, rng: np.random.Generator) -> Element:
        lo, hi = _check_range(self.length, len(element["t"]), self.start_range)
        return _cut(element, int(rng.integers(lo, hi)), self.length)


@dataclass(frozen=True)
class WeightedWindow(grain.transforms.RandomMap):
    """One window of ``length`` steps whose start index ``k`` is drawn with probability
    proportional to ``weight(t[k])`` over the valid starts."""

    length: int
    weight: WeightFn

    def random_map(self, element: Element, rng: np.random.Generator) -> Element:
        lo, hi = _check_range(self.length, len(element["t"]), None)
        w = np.asarray(self.weight(element["t"][lo:hi]), dtype=np.float64)
        if w.shape != (hi - lo,) or np.any(w < 0) or not np.any(w > 0):
            raise ValueError(
                "weight must map the start times to a non-negative vector."
            )
        return _cut(element, categorical(rng, w), self.length)


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


def windows(
    source: TimeSeriesDataSource,
    length: int,
    *,
    seed: int,
    weight: WeightFn | None = None,
    start_range: tuple[int, int] | None = None,
) -> grain.MapDataset:
    """Infinite stream of windows: trajectories shuffled and repeated, one window per
    visit — uniform start (``RandomWindow``) or weighted start (``WeightedWindow``)."""
    _check_range(length, source.trajectory_length, start_range)
    if weight is not None and start_range is not None:
        raise ValueError("weight and start_range are alternatives; pass one.")
    transform = (
        RandomWindow(length, start_range)
        if weight is None
        else WeightedWindow(length, weight)
    )
    return (
        grain.MapDataset.source(source)
        .seed(seed)
        .shuffle()
        .repeat()
        .random_map(transform)
    )


def mixed_windows(
    source: TimeSeriesDataSource,
    length: int,
    split_idx: int,
    *,
    weights: tuple[float, float] = (1.0, 1.0),
    seed: int,
) -> grain.MapDataset:
    """Infinite stream interleaving windows that start before ``ts[split_idx]`` (the
    transient) and windows that start at or after it, in the ratio ``weights``."""
    T = source.trajectory_length
    if not length <= split_idx <= T - length:
        raise ValueError(
            "split_idx must leave at least one window on each side: "
            f"{length} <= split_idx <= {T - length}."
        )
    early = windows(source, length, seed=seed, start_range=(0, split_idx - length + 1))
    late = windows(
        source, length, seed=seed + 1, start_range=(split_idx, T - length + 1)
    )
    return grain.MapDataset.mix([early, late], weights=list(weights))


class _FixedWindows:
    """Every window of ``length`` starting at ``0, stride, 2·stride, …`` of every
    trajectory, in order — a finite, deterministic ``RandomAccessDataSource``."""

    def __init__(self, source: TimeSeriesDataSource, length: int, stride: int):
        _check_range(length, source.trajectory_length, None)
        if stride < 1:
            raise ValueError("stride must be positive.")
        self.source, self.length, self.stride = source, length, stride
        self.starts = range(0, source.trajectory_length - length + 1, stride)

    def __len__(self) -> int:
        return len(self.source) * len(self.starts)

    def __getitem__(self, index: int) -> Element:
        i, j = divmod(index, len(self.starts))
        return _cut(self.source[i], self.starts[j], self.length)


def validation_windows(
    source: TimeSeriesDataSource, length: int, *, stride: int | None = None
) -> grain.MapDataset:
    """A **finite, deterministic** dataset of windows for evaluation: windows of
    ``length`` at starts ``0, stride, 2·stride, …`` of every held-out trajectory, in
    trajectory-major order, no shuffle, no repeat (``stride`` defaults to ``length``,
    non-overlapping). Iterating it twice gives the same batches, so a metric computed
    on it is comparable across evaluations (ADR-0009)."""
    stride = length if stride is None else stride
    return grain.MapDataset.source(_FixedWindows(source, length, stride))


def to_device(
    dataset: grain.MapDataset | grain.IterDataset,
    device=None,
    *,
    cpu_buffer_size: int = 4,
    device_buffer_size: int = 2,
) -> grain.IterDataset:
    """grain's two-stage prefetch to the accelerator (its JAX training tutorial's
    "option C", the recommended pattern for real training): a CPU-side thread
    prepares ``cpu_buffer_size`` batches ahead while ``device_buffer_size`` batches
    already sit on ``device`` (default: JAX's default device). The transfer then
    overlaps the training step instead of blocking the loop between steps. Apply
    after ``.batch(...)``."""
    if device is None:
        device = jax.devices()[0]
    ds = (
        dataset if isinstance(dataset, grain.IterDataset) else dataset.to_iter_dataset()
    )
    return grain.experimental.device_put(
        ds,
        device,
        cpu_buffer_size=cpu_buffer_size,
        device_buffer_size=device_buffer_size,
    )
