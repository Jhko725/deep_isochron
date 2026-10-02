"""Laws of the data layer (``deep_isochron.data``).

1. windows       enumeration is trajectory-major; every window is a contiguous slice;
                 window_start_times matches __getitem__; invalid window sizes rejected
2. splits        split_time partitions the time axis; split_trajectories partitions
                 the trajectories, deterministically in the seed
3. disk          save -> load round-trips arrays and every metadata field through one
                 file; dtype mismatch is loud; missing metadata cannot be saved
4. generate      shapes, u0 == ys[:, 0], metadata fields, config_hash stable across
                 runs and sensitive to the config, failures raise with indices
5. sampling      weighted_windows draws windows with the requested frequencies and is
                 deterministic in the seed; mixed_split interleaves in ratio and
                 rejects a split that does not fit a window
"""

import collections

import diffrax as dfx
import jax.numpy as jnp
import numpy as np
import pytest
from deep_isochron.data import (
    config_hash,
    dataset_path,
    DatasetMetadata,
    generate,
    mixed_split,
    TimeSeriesDataSource,
    transient_weights,
    UniformAnnulus,
    UniformBox,
    weighted_windows,
)
from deep_isochron.systems import BautinNormalForm, FitzhughNagumo, SolverConfig
from hypothesis import given, settings, strategies as st

from tests.helpers import assert_close, TOL


def _source(n=3, T=10, dim=2, window=4):
    ts = np.linspace(0.0, 1.0, T)
    ys = np.arange(n * T * dim, dtype=float).reshape(n, T, dim)
    return TimeSeriesDataSource(ts, ys, window_size=window)


# ------------------------------------------------------------------ 1. windows -----
@given(n=st.integers(1, 4), T=st.integers(2, 12), data=st.data())
def test_window_enumeration(n, T, data):
    window = data.draw(st.integers(1, T), label="window")
    src = _source(n, T, 2, window)
    assert len(src) == n * (T - window + 1)
    starts = src.window_start_times()
    for idx in range(len(src)):
        i, k = src.window_index(idx)
        t, y = src[idx]
        assert t.shape == (window,) and y.shape == (window, 2)
        assert_close(t, src.ts[k : k + window], rtol=TOL["identity"])
        assert_close(y, src.ys[i, k : k + window], rtol=TOL["identity"])
        assert starts[idx] == t[0]
    assert src.window_index(len(src) - 1) == (n - 1, T - window)


def test_invalid_window_and_shapes():
    with pytest.raises(ValueError):
        _source(window=11)
    with pytest.raises(ValueError):
        _source(window=0)
    with pytest.raises(ValueError):
        TimeSeriesDataSource(np.zeros(5), np.zeros((2, 4, 2)))


# ------------------------------------------------------------------- 2. splits -----
def test_split_time_partitions():
    src = _source(n=2, T=10, window=3)
    a, b = src.split_time(6)
    assert a.trajectory_length == 6 and b.trajectory_length == 4
    assert_close(np.concatenate([a.ts, b.ts]), src.ts, rtol=TOL["identity"])
    assert_close(np.concatenate([a.ys, b.ys], axis=1), src.ys, rtol=TOL["identity"])
    assert a.window_size == b.window_size == 3
    a2, b2 = src.split_time(2)  # too short for the window: whole trajectories
    assert a2.window_size == 2 and b2.window_size == 3


def test_split_trajectories_partitions_and_is_seeded():
    src = _source(n=10, T=5, window=5)
    train, held = src.split_trajectories(0.3, seed=1)
    assert train.num_trajectories == 7 and held.num_trajectories == 3
    rows = {tuple(r) for r in np.concatenate([train.ys, held.ys])[:, 0, 0:1]}
    assert rows == {tuple(r) for r in src.ys[:, 0, 0:1]}
    train2, _ = src.split_trajectories(0.3, seed=1)
    assert_close(train2.ys, train.ys, rtol=TOL["identity"])
    train3, _ = src.split_trajectories(0.3, seed=2)
    assert not np.array_equal(train3.ys, train.ys)
    with pytest.raises(ValueError):
        src.split_trajectories(1.0)


# --------------------------------------------------------------------- 3. disk -----
def _meta(**kw) -> DatasetMetadata:
    base = dict(
        system="FitzhughNagumo",
        system_params={"a": 0.7, "b": 0.8, "c": 3.0, "z": -0.4},
        ic_sampler="UniformBox",
        ic_sampler_params={"lo": [-3.0, -3.0], "hi": [3.0, 3.0]},
        seed=0,
        n_trajectories=3,
        n_time=10,
        dim=2,
        t0=0.0,
        t1=1.0,
        solver="Tsit5",
        rtol=1e-6,
        atol=1e-8,
        max_steps=4096,
        extra={"note": "test", "nested": {"k": [1, 2]}},
    )
    base.update(kw)
    return DatasetMetadata(**base)


def test_save_load_round_trip(tmp_path):
    base = _source()
    src = TimeSeriesDataSource(base.ts, base.ys, window_size=4, metadata=_meta())
    path = src.save(tmp_path / "d.nc")
    assert path.exists() and len(list(tmp_path.iterdir())) == 1  # one file
    back = TimeSeriesDataSource.load(path, window_size=4)
    assert_close(back.ts, src.ts, rtol=TOL["identity"])
    assert_close(back.ys, src.ys, rtol=TOL["identity"])
    assert back.metadata == src.metadata  # every field, nested dicts included
    assert back.window_size == 4 and len(back) == len(src)


def test_load_dtype_mismatch_is_loud(tmp_path):
    base = _source()
    src = TimeSeriesDataSource(
        base.ts, base.ys.astype(np.float32), metadata=_meta(dtype="float32")
    )
    path = src.save(tmp_path / "f32.nc")
    with pytest.raises(TypeError, match="float32"):
        TimeSeriesDataSource.load(path, dtype=np.float64)
    assert TimeSeriesDataSource.load(path, dtype=np.float32).ys.dtype == np.float32


def test_save_requires_metadata(tmp_path):
    with pytest.raises(ValueError, match="metadata"):
        _source().save(tmp_path / "x.nc")


def test_metadata_attrs_round_trip():
    m = _meta()
    assert DatasetMetadata.from_attrs(m.to_attrs()) == m
    assert isinstance(m.to_attrs()["extra"], str)  # nested -> JSON string attribute


# ----------------------------------------------------------------- 4. generate -----
TIGHT = SolverConfig(solver=dfx.Dopri5(), rtol=1e-9, atol=1e-11)


def test_generate_bautin():
    nf = BautinNormalForm(1.0, 0.5, 2.0, 0.5)
    ts = jnp.linspace(0.0, 1.0, 6)
    src = generate(
        nf, UniformAnnulus(0.3, 2.0), ts, 5, seed=3, config=TIGHT, window_size=3
    )
    assert src.ys.shape == (5, 6, 2) and src.window_size == 3
    assert_close(src.u0, src.ys[:, 0], rtol=TOL["identity"])
    m = src.metadata
    assert m is not None
    assert m.system == "BautinNormalForm" and m.strategy == "RadiusSquaredIntegration"
    assert_close(m.system_params["a"], 1.0, rtol=TOL["closed_form"])
    assert_close(m.system_params["b"], 0.5, rtol=TOL["closed_form"])
    assert m.ic_sampler == "UniformAnnulus" and m.ic_sampler_params["r_max"] == 2.0
    assert m.seed == 3 and m.n_trajectories == 5 and m.n_time == 6 and m.dim == 2
    assert m.solver == "Dopri5" and m.rtol == 1e-9 and m.dtype == "float64"
    assert len(m.config_hash) == 8 and m.created
    # the trajectories are the system's flow
    ref = nf.flow(ts, jnp.asarray(src.u0[0]), config=TIGHT, strategy="cartesian")
    assert_close(src.ys[0], ref, rtol=TOL["flow"], atol=TOL["flow"])
    # same config -> same hash; different seed or tolerance -> different hash
    again = generate(nf, UniformAnnulus(0.3, 2.0), ts, 5, seed=3, config=TIGHT)
    assert again.metadata.config_hash == m.config_hash
    other = generate(nf, UniformAnnulus(0.3, 2.0), ts, 5, seed=4, config=TIGHT)
    assert other.metadata.config_hash != m.config_hash
    assert dataset_path("data", "bautin", m).name == f"bautin-{m.config_hash}.nc"
    assert config_hash(m) == m.config_hash


def test_generate_fhn_and_failure():
    ode = FitzhughNagumo()
    ts = jnp.linspace(0.0, 2.0, 5)
    src = generate(ode, UniformBox([-2.0, -1.0], [2.0, 1.0]), ts, 4, seed=0)
    assert src.ys.shape == (4, 5, 2) and src.metadata.strategy == ""
    assert np.all(np.abs(src.u0[:, 0]) <= 2.0) and np.all(np.abs(src.u0[:, 1]) <= 1.0)
    with pytest.raises(ValueError, match="strategy"):
        box = UniformBox([-1.0] * 2, [1.0] * 2)
        generate(ode, box, ts, 2, seed=0, strategy="polar")
    with pytest.raises(RuntimeError, match="integrations failed"):
        generate(
            ode,
            UniformBox([-1.0] * 2, [1.0] * 2),
            jnp.linspace(0.0, 50.0, 5),
            3,
            seed=0,
            config=SolverConfig(max_steps=4),
        )


# ----------------------------------------------------------------- 5. sampling -----
def test_transient_weights_shape_and_decay():
    src = _source(n=2, T=10, window=3)
    w = transient_weights(src, boost=4.0, tau=0.5)
    assert w.shape == (len(src),)
    per_traj = w[: src.windows_per_trajectory]
    assert per_traj[0] == pytest.approx(5.0) and np.all(np.diff(per_traj) < 0)
    assert_close(w[src.windows_per_trajectory :], per_traj, rtol=TOL["identity"])
    with pytest.raises(ValueError):
        transient_weights(src, boost=-1.0, tau=1.0)


@settings(deadline=None, max_examples=5)
@given(seed=st.integers(0, 10))
def test_weighted_windows_frequencies(seed):
    """Empirical draw frequencies match the weights (within sampling error) and the
    stream is deterministic in the seed."""
    src = _source(n=1, T=5, window=2)  # 4 windows
    weights = np.array([4.0, 2.0, 1.0, 1.0])
    ds = weighted_windows(src, weights, seed=seed)
    n = 4000
    counts = collections.Counter()
    for item in ds[:n]:
        counts[int(item[0][0] * 4 + 1e-9)] += 1  # start time -> window index (ts=k/4)
    freq = np.array([counts[k] for k in range(4)]) / n
    assert_close(freq, weights / weights.sum(), atol=0.03)
    ds2 = weighted_windows(src, weights, seed=seed)
    assert all(np.array_equal(a[1], b[1]) for a, b in zip(ds[:50], ds2[:50]))


def test_weighted_windows_validation():
    src = _source(n=1, T=5, window=2)
    with pytest.raises(ValueError):
        weighted_windows(src, np.ones(3), seed=0)
    with pytest.raises(ValueError):
        weighted_windows(src, np.zeros(4), seed=0)
    # callable weights
    fn = lambda s: transient_weights(s, boost=1.0, tau=1.0)  # noqa: E731
    ds = weighted_windows(src, fn, seed=0)
    assert ds[0][1].shape == (2, 2)


def test_mixed_split_ratio_and_validation():
    src = _source(n=2, T=12, window=3)
    ds = mixed_split(src, 6, weights=(1.0, 2.0), seed=0)
    early_cut = src.ts[6]
    starts = [item[0][0] for item in ds[:300]]
    frac_early = np.mean([t < early_cut for t in starts])
    assert frac_early == pytest.approx(1 / 3, abs=0.02)  # deterministic interleave
    with pytest.raises(ValueError):
        mixed_split(src, 2, seed=0)
    with pytest.raises(ValueError):
        mixed_split(src, 11, seed=0)
