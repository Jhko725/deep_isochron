"""The batched window source for training runs (ADR-0008 Decision 3, roadmap C9):
``WindowBatchSource`` is a ``RandomAccessDataSource`` whose element ``i`` is the
``i``-th **batch** of windows of the run, built by one vectorized gather — the
performant counterpart of ``elementwise``, with the same start distributions and a
concrete length (epochs of every window once). ``window_batches`` /
``mixed_window_batches`` wrap it in a ``grain.MapDataset``; follow with
``data.device.to_device``.
"""

import math

import grain
import numpy as np

from .common import _check_range, Batch, start_weights, TimeSeriesDataSource, WeightFn


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
        epochs: int | None = None,
        num_steps: int | None = None,
        weight: WeightFn | None = None,
        start_range: tuple[int, int] | None = None,
        ranges: tuple[tuple[int, int], tuple[int, int]] | None = None,
        range_weights: tuple[float, float] = (1.0, 1.0),
    ):
        """**Arguments:**

        - ``source``: the trajectories (``ts (T,)``, ``ys (N, T, dim)``).
        - ``length``: window length ``L``; ``batch``: windows per element ``B``.
        - ``seed``: every batch is a pure function of ``(seed, i)``.
        - ``epochs`` **or** ``num_steps`` (exactly one): the run's length, as epochs of
          every window once, or as steps — ``epochs = ceil(num_steps /
          batches_per_epoch)``, the smallest number of epochs covering ``num_steps``.
        - ``weight``, ``start_range``, ``ranges`` (at most one): the start
          distribution — ``weight(t_start)`` by inverse CDF (``WeightFn``); uniform over
          ``[lo, hi)``; or two ranges drawn with probabilities ``range_weights``
          (``mixed_windows``).
          Without any: uniform over every valid start.
        """
        T = source.trajectory_length
        if sum(x is not None for x in (weight, start_range, ranges)) > 1:
            raise ValueError(
                "weight, start_range and ranges are alternatives; pass one."
            )
        if (epochs is None) == (num_steps is None):
            raise ValueError("pass exactly one of epochs and num_steps.")
        if batch < 1:
            raise ValueError("batch must be positive.")
        self.source, self.length, self.batch, self.seed = source, length, batch, seed
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
            self._cdf = np.cumsum(start_weights(weight, self.ts, self.lo, self.hi))
        self.batches_per_epoch = self.num_windows // batch
        if self.batches_per_epoch == 0:
            raise ValueError(
                f"batch {batch} exceeds the {self.num_windows} windows of an epoch."
            )
        if epochs is not None:
            if epochs < 1:
                raise ValueError("epochs must be positive.")
            self.epochs = epochs
        else:
            assert num_steps is not None
            if num_steps < 1:
                raise ValueError("num_steps must be positive.")
            self.epochs = math.ceil(num_steps / self.batches_per_epoch)
        self._perm_cache: dict[int, np.ndarray] = {}

    @property
    def with_replacement(self) -> bool:
        return self._cdf is not None or self._ranges is not None

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
        epochs=epochs,
        num_steps=num_steps,
        weight=weight,
        start_range=start_range,
    )
    return grain.MapDataset.source(src)


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
        epochs=epochs,
        num_steps=num_steps,
        ranges=ranges,
        range_weights=weights,
    )
    return grain.MapDataset.source(src)
