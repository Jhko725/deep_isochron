"""The laws of the windowing package, ``deep_isochron.data.windows``, in both of its
forms (ADR-0008, Decisions 2 and 3).

The elementwise form: `RandomWindow` cuts contiguous slices with starts uniform in
their range, `WeightedWindow` draws starts with the requested frequencies, both
re-draw across epochs and are deterministic in the seed, `mixed_windows` interleaves
two start ranges in ratio, `validation_windows` is finite, deterministic and
complete, and the `categorical` draw and `transient_weight` obey their definitions.
The batched form, `WindowBatchSource`: batch shapes and contiguity, determinism in
the seed, an epoch that is every window exactly once and the next epoch a reshuffle,
the remainder dropped, `start_range`, `weight` and mixed marginals equal to the
elementwise laws, `epochs` exclusive with `num_steps`, slicing that resumes,
pickling, and feeding `to_device`. The sources are 3 or 4 arithmetic trajectories
whose values encode the trajectory and the start index.
"""

import collections
import pickle

import jax
import numpy as np
import pytest
from deep_isochron.data import (
    mixed_window_batches,
    mixed_windows,
    RandomWindow,
    TimeSeriesDataSource,
    to_device,
    transient_weight,
    validation_windows,
    WeightedWindow,
    window_batches,
    WindowBatchSource,
    windows,
)
from deep_isochron.data.windows import categorical
from hypothesis import given, settings, strategies as st

from tests.helpers import assert_close, TOL


def _source(n=3, T=10, dim=2):
    ts = np.linspace(0.0, 1.0, T)
    ys = np.arange(n * T * dim, dtype=float).reshape(n, T, dim)
    return TimeSeriesDataSource(ts, ys)


# ---------------------------------------------------------------- 1. elementwise ----
def _start_of(window, src):
    """Start index of a window element on ``src``'s grid (``ts = k/(T-1)``)."""
    return int(round(window["t"][0] * (src.trajectory_length - 1)))


@given(T=st.integers(2, 12), data=st.data())
def test_random_window_cuts_contiguous_slices(T, data):
    """Ten draws of a `RandomWindow` of length ``L <= T`` each have shape ``(L,)`` and
    ``(L, 2)`` and equal the slice of the trajectory at their start."""
    L = data.draw(st.integers(1, T), label="length")
    src = _source(n=2, T=T)
    rng = np.random.default_rng(0)
    tf = RandomWindow(L)
    for _ in range(10):
        w = tf.random_map(src[1], rng)
        k = _start_of(w, src)
        assert w["t"].shape == (L,) and w["u"].shape == (L, 2) and 0 <= k <= T - L
        assert_close(w["u"], src.ys[1, k : k + L], rtol=TOL["identity"])


def test_random_window_start_range_and_validation():
    """A `start_range` of ``(2, 5)`` produces the starts ``{2, 3, 4}`` over 200 draws;
    an empty, negative, overlong or reversed range and a window longer than the
    trajectory are rejected."""
    src = _source(n=1, T=10)
    rng = np.random.default_rng(0)
    tf = RandomWindow(3, (2, 5))
    starts = {_start_of(tf.random_map(src[0], rng), src) for _ in range(200)}
    assert starts == {2, 3, 4}
    for bad in ((0, 0), (-1, 3), (0, 9), (5, 3)):
        with pytest.raises(ValueError):
            RandomWindow(3, bad).random_map(src[0], rng)
    with pytest.raises(ValueError):
        RandomWindow(11).random_map(src[0], rng)


@settings(deadline=None, max_examples=5)
@given(seed=st.integers(0, 10))
def test_weighted_window_frequencies_and_determinism(seed):
    """Start-index frequencies match the weights (within sampling error over 4000
    draws); the stream is deterministic in the seed and re-draws across epochs."""
    src = _source(n=1, T=5)  # 4 valid starts for L = 2
    weight = lambda t: np.array([4.0, 2.0, 1.0, 1.0])  # noqa: E731
    ds = windows(src, 2, seed=seed, weight=weight)
    n = 4000
    counts = collections.Counter(_start_of(ds[i], src) for i in range(n))
    freq = np.array([counts[k] for k in range(4)]) / n
    assert_close(freq, np.array([0.5, 0.25, 0.125, 0.125]), atol=0.03)
    ds2 = windows(src, 2, seed=seed, weight=weight)
    assert all(np.array_equal(ds[i]["u"], ds2[i]["u"]) for i in range(50))
    assert len({_start_of(ds[i], src) for i in range(50)}) > 1  # re-drawn per visit


def test_validation_windows_are_finite_deterministic_and_complete():
    """Every window at starts 0, stride, … of every trajectory, trajectory-major; two
    passes give identical batches; ``stride`` defaults to ``length``."""
    src = _source(n=3, T=10)
    ds = validation_windows(src, 4, stride=3)  # starts 0, 3, 6
    assert len(ds) == 3 * 3
    first = [w for w in ds]
    assert np.array_equal(first[0]["u"], src.ys[0, 0:4])
    assert np.array_equal(first[1]["u"], src.ys[0, 3:7])
    assert np.array_equal(first[3]["u"], src.ys[1, 0:4])
    second = [w for w in ds]
    assert all(np.array_equal(a["u"], b["u"]) for a, b in zip(first, second))
    assert len(validation_windows(src, 4)) == 3 * 2  # non-overlapping: starts 0, 4
    batches = list(validation_windows(src, 4).batch(4))
    assert [b["u"].shape[0] for b in batches] == [4, 2]
    with pytest.raises(ValueError):
        validation_windows(src, 4, stride=0)
    with pytest.raises(ValueError):
        validation_windows(src, 11)


def test_categorical_matches_the_weights_and_covers_the_support():
    """``categorical`` draws index k with probability w_k / Σw (3 % over 20000 draws);
    zero-weight bins are never drawn; a single positive bin is always drawn."""
    rng = np.random.default_rng(0)
    w = np.array([0.0, 1.0, 3.0, 0.0, 4.0])
    draws = np.array([categorical(rng, w) for _ in range(20000)])
    freq = np.bincount(draws, minlength=5) / len(draws)
    assert np.all(freq[w == 0] == 0)
    np.testing.assert_allclose(freq, w / w.sum(), atol=0.03)
    assert all(categorical(rng, np.array([0.0, 0.0, 2.0])) == 2 for _ in range(50))


def test_transient_weight_and_weighted_window_validation():
    """`transient_weight` starts at ``1 + boost``, decreases and stays at least 1, and
    rejects a negative boost; a `WeightedWindow` rejects an all-zero weight and a
    weight of the wrong length, and `windows` rejects a weight with a start range."""
    w = transient_weight(boost=4.0, tau=0.5)(np.linspace(0.0, 1.0, 10))
    assert w[0] == pytest.approx(5.0) and np.all(np.diff(w) < 0) and np.all(w >= 1.0)
    with pytest.raises(ValueError):
        transient_weight(boost=-1.0, tau=1.0)
    src = _source(n=1, T=5)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        WeightedWindow(2, lambda t: np.zeros_like(t)).random_map(src[0], rng)
    with pytest.raises(ValueError):
        WeightedWindow(2, lambda t: np.ones(3)).random_map(src[0], rng)
    with pytest.raises(ValueError):
        windows(src, 2, seed=0, weight=transient_weight(1.0, 1.0), start_range=(0, 2))


def test_windows_pipeline_batches():
    """A batch of 8 from `windows(...).batch(8)` has shapes ``(8, 4)`` and ``(8, 4, 2)``
    and every window is a slice of one trajectory, read off the encoded values."""
    src = _source(n=3, T=10)
    batch = next(iter(windows(src, 4, seed=0).batch(8)))
    assert batch["t"].shape == (8, 4) and batch["u"].shape == (8, 4, 2)
    # each window is a slice of one trajectory: ys[i] takes values in [20 i, 20 i + 20)
    traj_ids = {int(u[0, 0]) // 20 for u in batch["u"]}
    assert traj_ids <= {0, 1, 2}


def test_mixed_windows_ratio_and_validation():
    """`mixed_windows` with weights ``(1, 2)`` draws every start from one of the two
    ranges, one third from the early range over 300 draws; a `split_idx` too close
    to either end is rejected."""
    src = _source(n=2, T=12)
    ds = mixed_windows(src, 3, 6, weights=(1.0, 2.0), seed=0)
    starts = [_start_of(ds[i], src) for i in range(300)]
    assert all(0 <= k <= 3 or 6 <= k <= 9 for k in starts)  # [0, 6-3] and [6, 12-3]
    frac_early = np.mean([k < 6 for k in starts])
    assert frac_early == pytest.approx(1 / 3, abs=0.02)  # deterministic interleave
    for bad in (2, 10):
        with pytest.raises(ValueError):
            mixed_windows(src, 3, bad, seed=0)


# ------------------------------------------------------------------- 2. batched -----
def _starts_of(batch, src):
    """The start indices of a batch's windows on ``src``'s grid."""
    return np.rint(batch["t"][:, 0] * (src.trajectory_length - 1)).astype(int)


def test_window_batches_shapes_length_and_determinism():
    """`window_batches` over 32 windows has ``32 // 8`` batches per epoch with the
    source's dtype, every window a contiguous slice of its trajectory on the shared
    grid, the same batches for the same seed and different ones for another."""
    src = _source(n=4, T=10, dim=2)  # 4 · (10 − 3 + 1) = 32 windows of length 3
    ds = window_batches(src, 3, 8, seed=0, epochs=2)
    assert len(ds) == 2 * 4  # 32 // 8 batches per epoch
    b = ds[0]
    assert b["t"].shape == (8, 3) and b["u"].shape == (8, 3, 2)
    assert b["u"].dtype == src.ys.dtype
    # each window is a contiguous slice of its trajectory, and t is the shared grid
    for k in range(8):
        traj = int(b["u"][k, 0, 0] // (10 * 2))
        start = _starts_of(b, src)[k]
        assert np.array_equal(b["u"][k], src.ys[traj, start : start + 3])
        assert np.array_equal(b["t"][k], src.ts[start : start + 3])
    ds2 = window_batches(src, 3, 8, seed=0, epochs=2)
    assert all(np.array_equal(ds[i]["u"], ds2[i]["u"]) for i in range(len(ds)))
    assert not np.array_equal(
        window_batches(src, 3, 8, seed=1, epochs=1)[0]["u"], ds[0]["u"]
    )


def test_window_batches_epoch_is_every_window_once_then_reshuffled():
    """The `_draw` calls of one epoch cover each of the 32 windows exactly once, and
    the next epoch covers the same set in a different order."""
    src = _source(n=4, T=10)
    wsrc = WindowBatchSource(src, 3, 8, seed=0, epochs=2)
    assert wsrc.num_windows == 32 and wsrc.batches_per_epoch == 4

    def windows_of(epoch):
        out = []
        for j in range(wsrc.batches_per_epoch):
            traj, start = wsrc._draw(epoch * wsrc.batches_per_epoch + j)
            out += list(zip(traj.tolist(), start.tolist()))
        return out

    first, second = windows_of(0), windows_of(1)
    all_windows = {(i, k) for i in range(4) for k in range(8)}
    assert set(first) == all_windows and len(first) == 32  # once each
    assert set(second) == all_windows and first != second  # same set, new order


def test_window_batches_drop_remainder_and_too_large_batch():
    """24 windows in batches of 10 give 2 batches of 20 distinct windows, the 4 others
    dropped; a batch larger than the window count is rejected, and an index past the
    last batch raises `IndexError`."""
    src = _source(n=3, T=10)  # 24 windows of length 3
    wsrc = WindowBatchSource(src, 3, 10, seed=0, epochs=1)
    assert wsrc.batches_per_epoch == 2 and len(wsrc) == 2  # 4 windows dropped
    seen = set()
    for j in range(2):
        traj, start = wsrc._draw(j)
        seen |= set(zip(traj.tolist(), start.tolist()))
    assert len(seen) == 20  # 20 distinct windows used, 4 dropped
    with pytest.raises(ValueError, match="exceeds"):
        WindowBatchSource(src, 3, 25, seed=0, epochs=1)
    with pytest.raises(IndexError):
        wsrc[2]


def test_window_batches_start_range_weight_and_mix():
    """`start_range`, `weight` and the mixed form give the same marginals as the
    elementwise laws: starts in the range only, the weighted frequencies over 4000
    draws with draws that differ between batches, and the early fraction of the mix;
    a weight with a start range and a bad `split_idx` are rejected."""
    src = _source(n=1, T=5)  # 4 valid starts for L = 2
    ranged = WindowBatchSource(src, 2, 2, seed=0, epochs=3, start_range=(1, 3))
    assert ranged.num_windows == 2 and len(ranged) == 3
    starts = np.concatenate([_starts_of(ranged[i], src) for i in range(len(ranged))])
    assert set(starts.tolist()) == {1, 2}
    # weighted: frequencies over 4000 draws match the weights (same law as
    # WeightedWindow); draws differ between batches
    weight = lambda t: np.array([4.0, 2.0, 1.0, 1.0])  # noqa: E731
    weighted = window_batches(src, 2, 4, seed=0, epochs=1000, weight=weight)
    starts = np.concatenate([_starts_of(weighted[i], src) for i in range(1000)])
    freq = np.bincount(starts, minlength=4) / starts.size
    assert_close(freq, np.array([0.5, 0.25, 0.125, 0.125]), atol=0.03)
    first, second = _starts_of(weighted[0], src), _starts_of(weighted[1], src)
    assert not np.array_equal(first, second)
    # mixed: the early/late ratio per window
    src2 = _source(n=2, T=12)
    mixed = mixed_window_batches(src2, 3, 16, 6, weights=(1.0, 3.0), seed=0, epochs=40)
    starts = np.concatenate([_starts_of(mixed[i], src2) for i in range(40)])
    assert abs(np.mean(starts < 6) - 0.25) < 0.06
    assert set(starts.tolist()) <= set(range(0, 10))
    with pytest.raises(ValueError, match="alternatives"):
        WindowBatchSource(
            src, 2, 2, seed=0, epochs=1, weight=weight, start_range=(0, 2)
        )
    with pytest.raises(ValueError, match="split_idx"):
        mixed_window_batches(src2, 3, 2, 1, seed=0, epochs=1)


def test_window_batches_num_steps_slicing_and_pickling():
    """``num_steps=10`` over 4 batches per epoch gives 12 batches (3 whole epochs),
    ``ds[5:]`` resumes at batch 5, and a pickled source yields the same batches."""
    src = _source(n=4, T=10)  # 4 batches of 8 per epoch
    ds = window_batches(src, 3, 8, seed=0, num_steps=10)
    assert len(ds) == 12  # ceil(10 / 4) = 3 epochs
    assert np.array_equal(ds[5:][0]["u"], ds[5]["u"])  # resume at step 5
    wsrc = WindowBatchSource(src, 3, 8, seed=0, epochs=1)
    clone = pickle.loads(pickle.dumps(wsrc))
    assert np.array_equal(clone[2]["u"], wsrc[2]["u"])


def test_window_batch_source_epochs_and_num_steps_are_exclusive():
    """Exactly one of ``epochs`` and ``num_steps``; ``num_steps`` rounds up to whole
    epochs (``ceil``)."""
    src = _source(n=4, T=10)  # 4 batches of 8 per epoch
    with pytest.raises(ValueError, match="exactly one"):
        WindowBatchSource(src, 3, 8, seed=0)
    with pytest.raises(ValueError, match="exactly one"):
        WindowBatchSource(src, 3, 8, seed=0, epochs=1, num_steps=5)
    with pytest.raises(ValueError, match="exactly one"):
        window_batches(src, 3, 8, seed=0)
    assert WindowBatchSource(src, 3, 8, seed=0, num_steps=5).epochs == 2
    assert WindowBatchSource(src, 3, 8, seed=0, num_steps=4).epochs == 1
    with pytest.raises(ValueError, match="positive"):
        WindowBatchSource(src, 3, 8, seed=0, num_steps=0)


def test_window_batches_feed_to_device():
    """The batched loader is what training runs consume: through ``to_device`` on the
    default device, shapes and values intact (the trainer's end-of-loader behavior is
    ``tests/test_training.py``)."""
    src = _source(n=4, T=10)
    ds = window_batches(src, 3, 8, seed=0, epochs=1)
    dev = list(to_device(ds, jax.devices()[0]))
    assert len(dev) == 4 and all(isinstance(b["u"], jax.Array) for b in dev)
    for i, d in enumerate(dev):
        assert np.array_equal(np.asarray(d["u"]), ds[i]["u"])
