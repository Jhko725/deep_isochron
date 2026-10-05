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

``windows(source, length, ...)`` wires a source into either pipeline, one window per
element — the reference semantics, and what the tests check the batched source against.
**Training runs use ``window_batches`` / ``mixed_window_batches``**: a
``WindowBatchSource`` whose elements are whole batches, built by one vectorized gather,
with the same start distributions and a concrete length (epochs of every window once;
roadmap C9). ``validation_windows`` is the finite, deterministic counterpart for
evaluation, ``to_device`` the prefetching transfer to the accelerator, and
``resolve_device`` the one sanctioned way to pick the device.
"""

import os
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


Batch = dict[str, np.ndarray]


class WindowBatchSource:
    """A ``RandomAccessDataSource`` whose element ``i`` is the ``i``-th **batch** of
    training windows of the run — the training-run counterpart of the per-element
    pipeline above (roadmap C9, ADR-0008 amended).

    An *epoch* is every window of ``length`` of every trajectory once: ``num_windows =
    N · (hi − lo)`` over the valid starts ``[lo, hi)`` (``start_range``, default all).
    Batch ``i`` belongs to epoch ``e, j = divmod(i, batches_per_epoch)`` and is a pure
    function of ``(seed, i)``: in the default mode it is the ``j``-th slice of a fresh
    permutation of the epoch's windows (``rng([seed, e])``), so every window appears
    exactly once per epoch and the shuffle is new each epoch; the
    ``num_windows % batch`` windows the permutation puts last are dropped
    (``drop_remainder`` semantics — different windows each epoch, uniform batch shapes
    for ``jit``). With ``weight``, or
    with ``ranges``/``range_weights`` (the ``mixed_windows`` design), windows are drawn
    *with replacement* — a trajectory uniformly, a start from the weighted or mixed
    distribution — ``num_windows`` draws per epoch, so ``len`` means the same amount of
    data in every mode. ``len(self) = epochs · batches_per_epoch``: the loader is finite
    and the data define the run (``Trainer.train(num_steps=None)``); resuming at step
    ``s`` is ``window_batches(...)[s:]``.

    One vectorized gather per batch (``ys[traj[:, None], start[:, None] + arange(L)]``)
    instead of ``batch`` per-element ``__getitem__`` calls through grain — ≈ 1 ms
    against ≈ 50–90 ms per batch of 512 (change document 2026-10-04, round 2). Picklable
    (NumPy
    arrays and ints), so it works under ``mp_prefetch``; batches are
    ``{"t": (B, L), "u": (B, L, dim)}``, the shape ``.batch(B)`` produced."""

    def __init__(
        self,
        source: TimeSeriesDataSource,
        length: int,
        batch: int,
        *,
        seed: int,
        epochs: int,
        weight: WeightFn | None = None,
        start_range: tuple[int, int] | None = None,
        ranges: tuple[tuple[int, int], tuple[int, int]] | None = None,
        range_weights: tuple[float, float] = (1.0, 1.0),
    ):
        T = source.trajectory_length
        if sum(x is not None for x in (weight, start_range, ranges)) > 1:
            raise ValueError(
                "weight, start_range and ranges are alternatives; pass one."
            )
        if batch < 1 or epochs < 1:
            raise ValueError("batch and epochs must be positive.")
        self.source, self.length, self.batch = source, length, batch
        self.seed, self.epochs = seed, epochs
        self.ts, self.ys = np.asarray(source.ts), np.asarray(source.ys)
        if ranges is not None:
            lo0, hi0 = _check_range(length, T, ranges[0])
            lo1, hi1 = _check_range(length, T, ranges[1])
            if min(range_weights) < 0 or sum(range_weights) <= 0:
                raise ValueError("range_weights must be non-negative, not both zero.")
            self.lo, self.hi = min(lo0, lo1), max(hi0, hi1)
            self._ranges = ((lo0, hi0), (lo1, hi1))
            self._p_first = range_weights[0] / sum(range_weights)
            self.num_windows = len(source) * ((hi0 - lo0) + (hi1 - lo1))
        else:
            self.lo, self.hi = _check_range(length, T, start_range)
            self._ranges = None
            self.num_windows = len(source) * (self.hi - self.lo)
        self._cdf = None
        if weight is not None:
            w = np.asarray(weight(self.ts[self.lo : self.hi]), dtype=np.float64)
            if w.shape != (self.hi - self.lo,) or np.any(w < 0) or not np.any(w > 0):
                raise ValueError(
                    "weight must map the start times to a non-negative vector."
                )
            self._cdf = np.cumsum(w)
        self.batches_per_epoch = self.num_windows // batch
        if self.batches_per_epoch == 0:
            raise ValueError(
                f"batch {batch} exceeds the {self.num_windows} windows of an epoch."
            )
        self._perm_cache: dict[int, np.ndarray] = {}

    @property
    def with_replacement(self) -> bool:
        return self._cdf is not None or self._ranges is not None

    def for_steps(self, num_steps: int) -> "WindowBatchSource":
        """The same source with ``epochs = ceil(num_steps / batches_per_epoch)``, so
        ``len`` covers ``num_steps`` steps."""
        if num_steps < 1:
            raise ValueError("num_steps must be positive.")
        self.epochs = -(-num_steps // self.batches_per_epoch)
        return self

    def __len__(self) -> int:
        return self.epochs * self.batches_per_epoch

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if k != "_perm_cache"}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._perm_cache = {}

    def _permutation(self, epoch: int) -> np.ndarray:
        if epoch not in self._perm_cache:
            rng = np.random.default_rng([self.seed, epoch])
            self._perm_cache = {epoch: rng.permutation(self.num_windows)}
        return self._perm_cache[epoch]

    def _draw(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """``(traj, start)`` index vectors of batch ``index``."""
        e, j = divmod(index, self.batches_per_epoch)
        if not self.with_replacement:
            ids = self._permutation(e)[j * self.batch : (j + 1) * self.batch]
            traj, offset = np.divmod(ids, self.hi - self.lo)
            return traj, self.lo + offset
        rng = np.random.default_rng([self.seed, e, j])
        traj = rng.integers(0, len(self.source), self.batch)
        if self._cdf is not None:
            cdf = self._cdf
            k = np.searchsorted(cdf, rng.random(self.batch) * cdf[-1], side="right")
            return traj, self.lo + np.minimum(k, len(cdf) - 1)
        assert self._ranges is not None
        (lo0, hi0), (lo1, hi1) = self._ranges
        first = rng.random(self.batch) < self._p_first
        start = np.where(
            first,
            rng.integers(lo0, hi0, self.batch),
            rng.integers(lo1, hi1, self.batch),
        )
        return traj, start

    def __getitem__(self, index: int) -> Batch:
        if not 0 <= index < len(self):
            raise IndexError(index)
        traj, start = self._draw(index)
        idx = start[:, None] + np.arange(self.length)
        return {"t": self.ts[idx], "u": self.ys[traj[:, None], idx]}


def _sized(
    src: WindowBatchSource, epochs: int | None, num_steps: int | None
) -> grain.MapDataset:
    if (epochs is None) == (num_steps is None):
        raise ValueError("pass exactly one of epochs and num_steps.")
    if num_steps is not None:
        src.for_steps(num_steps)
    return grain.MapDataset.source(src)


def window_batches(
    source: TimeSeriesDataSource,
    length: int,
    batch: int,
    *,
    seed: int,
    epochs: int | None = None,
    num_steps: int | None = None,
    weight: WeightFn | None = None,
    start_range: tuple[int, int] | None = None,
) -> grain.MapDataset:
    """The training loader: ``grain.MapDataset.source(WindowBatchSource(...))`` — a
    finite dataset of ``epochs`` (or ``ceil(num_steps / batches_per_epoch)``) epochs of
    batched windows, uniform (every window once per epoch), weighted (``weight``) or
    restricted to ``start_range``. Follow with ``to_device(..., device)``; slice
    ``[s:]`` to resume at step ``s``."""
    src = WindowBatchSource(
        source,
        length,
        batch,
        seed=seed,
        epochs=epochs or 1,
        weight=weight,
        start_range=start_range,
    )
    return _sized(src, epochs, num_steps)


def mixed_window_batches(
    source: TimeSeriesDataSource,
    length: int,
    batch: int,
    split_idx: int,
    *,
    weights: tuple[float, float] = (1.0, 1.0),
    seed: int,
    epochs: int | None = None,
    num_steps: int | None = None,
) -> grain.MapDataset:
    """``mixed_windows`` at batch level: each window starts before ``ts[split_idx]``
    with probability ``weights[0] / sum(weights)``, otherwise at or after it, both
    uniform within their range — a per-window draw instead of ``grain.MapDataset.mix``
    of two pipelines, the same marginal ratio."""
    T = source.trajectory_length
    if not length <= split_idx <= T - length:
        raise ValueError(
            "split_idx must leave at least one window on each side: "
            f"{length} <= split_idx <= {T - length}."
        )
    ranges = ((0, split_idx - length + 1), (split_idx, T - length + 1))
    src = WindowBatchSource(
        source,
        length,
        batch,
        seed=seed,
        epochs=epochs or 1,
        ranges=ranges,
        range_weights=weights,
    )
    return _sized(src, epochs, num_steps)


def single_threaded(dataset: grain.MapDataset | grain.IterDataset) -> grain.IterDataset:
    """An ``IterDataset`` that reads with one thread and no read-ahead. Iterating a
    ``MapDataset`` directly starts grain's default reader (16 threads, 500-element
    buffer), which competes with a JAX training loop for the GIL; use this (or
    ``to_device``, which does the same) wherever a pipeline is consumed next to JAX
    work, e.g. validation batches."""
    if isinstance(dataset, grain.IterDataset):
        return dataset
    return dataset.to_iter_dataset(
        grain.ReadOptions(num_threads=1, prefetch_buffer_size=1)
    )


def to_device(
    dataset: grain.MapDataset | grain.IterDataset,
    device: jax.Device,
    *,
    cpu_buffer_size: int = 4,
    device_buffer_size: int = 2,
) -> grain.IterDataset:
    """grain's two-stage prefetch to the accelerator (its JAX training tutorial's
    "option C", the recommended pattern for real training): one CPU-side thread
    prepares ``cpu_buffer_size`` batches ahead while a second thread keeps
    ``device_buffer_size`` batches already transferred to ``device`` (exactly two
    prefetch threads, one per stage, plus one reader thread; no pool). The transfer
    then overlaps the training step instead of blocking the loop between steps.
    ``device`` is explicit — on a shared machine the caller (or the scheduler, through
    ``CUDA_VISIBLE_DEVICES``) decides which card a job uses, never this function. Apply
    after ``.batch(...)``."""
    ds = single_threaded(dataset)  # one reader; the two prefetch stages buffer
    return grain.experimental.device_put(
        ds,
        device,
        cpu_buffer_size=cpu_buffer_size,
        device_buffer_size=device_buffer_size,
    )


_VISIBLE_DEVICES_VARS = (
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
)


def resolve_device(index: int | None = None) -> jax.Device:
    """The device a run should use, chosen so that nothing ever grabs an accelerator the
    scheduler did not assign.

    - ``index`` given: ``jax.devices()[index]`` — the explicit choice always wins.
    - One device visible (a CPU-only machine, or a job the scheduler scoped to one
      card), or several visible but a ``*_VISIBLE_DEVICES`` variable set (Slurm's
      ``--gpus`` / ``--gres`` export ``CUDA_VISIBLE_DEVICES`` into the job): JAX's own
      default device, ``jax.devices()[0]`` — the first device of the default backend,
      where uncommitted arrays and jitted computations land anyway, so the loader's
      batches and the model agree without any further configuration.
    - Several accelerators visible and no such variable — a login node, an interactive
      shell on a shared machine: refuse, with the list, rather than pick one."""
    devices = jax.devices()
    if index is not None:
        return devices[index]
    if len(devices) == 1 or devices[0].platform == "cpu":
        return devices[0]
    if any(os.environ.get(v) for v in _VISIBLE_DEVICES_VARS):
        return devices[0]
    visible = [f"{d.platform}:{d.id} {d.device_kind}" for d in devices]
    raise ValueError(
        f"{len(devices)} accelerators are visible and none of {_VISIBLE_DEVICES_VARS} "
        "is set, so no scheduler scoped this process to a card; pass the device index "
        f"explicitly. Visible: {visible}"
    )
