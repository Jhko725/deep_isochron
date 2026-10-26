"""The elementwise pipeline: whole trajectories from a ``TimeSeriesDataSource``, one
window cut per element by a ``grain.transforms.RandomMap`` after ``.shuffle().repeat()``
(ADR-0008, Decision 2).

This is the reference implementation, short and what the tests check the batched
source against, and the path of ``validation_windows``; training runs use ``batched``,
which is 60–125 times faster per batch. Trajectories start off the limit cycle and
converge to it, so the late windows are nearly periodic and the early, transient
windows carry most of the information about the amplitude dynamics. Two ways to
oversample them are kept side by side so they can be compared. ``WeightedWindow`` draws
the start index with probability proportional to a weight ``w(t_start)``, importance
sampling with a known proposal, as torch's ``WeightedRandomSampler`` does per
trajectory; ``transient_weight(boost, tau)`` is the smooth default, two numbers to log
and sweep. ``mixed_windows`` is the two-loader design: windows starting before
``ts[split_idx]`` and windows starting after it, each a ``RandomWindow`` with a
restricted ``start_range``, interleaved by ``grain.MapDataset.mix`` in a fixed ratio, a
hard boundary by construction.
"""

from dataclasses import dataclass

import grain
import numpy as np

from .common import (
    _check_range,
    _cut,
    categorical,
    Element,
    start_weights,
    TimeSeriesDataSource,
    WeightFn,
)


@dataclass(frozen=True)
class RandomWindow(grain.transforms.RandomMap):
    """One window of `length` steps with a uniformly random start in `start_range`,
    the start indices ``[lo, hi)``; every valid start by default."""

    length: int
    start_range: tuple[int, int] | None = None

    def random_map(self, element: Element, rng: np.random.Generator) -> Element:
        """The window cut from `element` at a start drawn with `rng`."""
        lo, hi = _check_range(self.length, len(element["t"]), self.start_range)
        return _cut(element, int(rng.integers(lo, hi)), self.length)


@dataclass(frozen=True)
class WeightedWindow(grain.transforms.RandomMap):
    """One window of `length` steps whose start index ``k`` is drawn with probability
    proportional to ``weight(t[k])`` over the valid starts."""

    length: int
    weight: WeightFn

    def random_map(self, element: Element, rng: np.random.Generator) -> Element:
        """The window cut from `element` at a start drawn by the weights with `rng`."""
        lo, hi = _check_range(self.length, len(element["t"]), None)
        w = start_weights(self.weight, element["t"], lo, hi)
        return _cut(element, categorical(rng, w), self.length)


def windows(
    source: TimeSeriesDataSource,
    length: int,
    *,
    seed: int,
    weight: WeightFn | None = None,
    start_range: tuple[int, int] | None = None,
) -> grain.MapDataset:
    """An infinite stream of windows: the trajectories shuffled and repeated, one
    window per visit, with a uniform start (`RandomWindow`) or a weighted one
    (`WeightedWindow`).

    **Arguments:**

    - `source`: The trajectories.
    - `length`: The window length.
    - `seed`: The pipeline's seed.
    - `weight`: The start weight function, for a weighted start.
    - `start_range`: The start indices ``[lo, hi)``, for a restricted uniform start.

    **Returns:**

    The ``grain.MapDataset``.

    **Raises:**

    `ValueError` when both `weight` and `start_range` are given, or the length or range
    does not fit."""
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
    """An infinite stream interleaving windows that start before ``ts[split_idx]``,
    the transient, and windows that start at or after it, in the ratio `weights`.

    **Raises:**

    `ValueError` unless `split_idx` leaves at least one window on each side."""
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
    """Every window of `length` starting at ``0, stride, 2·stride, …`` of every
    trajectory, in order: a finite, deterministic ``RandomAccessDataSource``."""

    def __init__(self, source: TimeSeriesDataSource, length: int, stride: int):
        _check_range(length, source.trajectory_length, None)
        if stride < 1:
            raise ValueError("stride must be positive.")
        self.source, self.length, self.stride = source, length, stride
        self.starts = range(0, source.trajectory_length - length + 1, stride)

    def __len__(self) -> int:
        """Trajectories times starts."""
        return len(self.source) * len(self.starts)

    def __getitem__(self, index: int) -> Element:
        """Window `index`, in trajectory-major order."""
        i, j = divmod(index, len(self.starts))
        return _cut(self.source[i], self.starts[j], self.length)


def validation_windows(
    source: TimeSeriesDataSource, length: int, *, stride: int | None = None
) -> grain.MapDataset:
    """A finite, deterministic dataset of windows for evaluation: windows of `length`
    at starts ``0, stride, 2·stride, …`` of every trajectory of `source`, in
    trajectory-major order, with no shuffle and no repeat.

    Iterating it twice gives the same batches, so a metric computed on it is
    comparable across evaluations (ADR-0009). `stride` defaults to `length`, so the
    windows do not overlap.
    """
    stride = length if stride is None else stride
    return grain.MapDataset.source(_FixedWindows(source, length, stride))
