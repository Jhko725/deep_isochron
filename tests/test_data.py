"""Laws of the data layer (``deep_isochron.data``).

1. source        a grain random-access source over whole trajectories; shape checks;
                 ``copy.replace`` splits — split_time partitions the time axis,
                 split_trajectories partitions the trajectories deterministically
2. metadata      grouped; attrs round trip through JSON; config_hash covers the four
                 generation groups and nothing in provenance/extra
3. disk          save -> load round-trips arrays and metadata through one file; dtype
                 mismatch is loud; missing metadata cannot be saved
4. windows       RandomWindow cuts contiguous slices with starts uniform in range;
                 WeightedWindow draws starts with the requested frequencies; both
                 re-draw across epochs and are deterministic in the seed;
                 mixed_windows interleaves two start ranges in ratio; validation
5. generate      shapes, u0 == ys[:, 0], metadata groups, trajectories equal the flow,
                 config_hash stable / sensitive, failures raise with indices
"""

import collections

import diffrax as dfx
import jax.numpy as jnp
import numpy as np
import pytest
from deep_isochron.data import (
    dataset_path,
    DatasetMetadata,
    generate,
    GridSpec,
    mixed_windows,
    Provenance,
    RandomWindow,
    SamplingSpec,
    SolveSpec,
    SystemSpec,
    TimeSeriesDataSource,
    transient_weight,
    UniformAnnulus,
    UniformBox,
    WeightedWindow,
    windows,
)
from deep_isochron.systems import BautinNormalForm, FitzhughNagumo, SolverConfig
from hypothesis import given, settings, strategies as st

from tests.helpers import assert_close, TOL


def _source(n=3, T=10, dim=2, metadata=None):
    ts = np.linspace(0.0, 1.0, T)
    ys = np.arange(n * T * dim, dtype=float).reshape(n, T, dim)
    return TimeSeriesDataSource(ts, ys, metadata)


def _meta(**kw) -> DatasetMetadata:
    base = dict(
        system=SystemSpec("FitzhughNagumo", {"a": 0.7, "b": 0.8, "c": 3.0, "z": -0.4}),
        sampling=SamplingSpec(
            "UniformBox", {"lo": [-3.0, -3.0], "hi": [3.0, 3.0]}, 0, 3
        ),
        grid=GridSpec(0.0, 1.0, 10),
        solve=SolveSpec("Tsit5", 1e-6, 1e-8, 4096),
        provenance=Provenance(created="2026-10-03T00:00:00+00:00", git_sha="abc"),
        extra={"note": "test", "nested": {"k": [1, 2]}},
    )
    base.update(kw)
    return DatasetMetadata(**base)


# ------------------------------------------------------------------- 1. source -----
def test_source_protocol_and_views():
    src = _source(n=3, T=10)
    assert len(src) == 3 and src.dim == 2 and src.trajectory_length == 10
    el = src[1]
    assert set(el) == {"t", "u"} and el["t"].shape == (10,) and el["u"].shape == (10, 2)
    assert_close(el["u"], src.ys[1], rtol=TOL["identity"])
    assert_close(src.u0, src.ys[:, 0], rtol=TOL["identity"])
    # the jaxtyping hook rejects the shape mismatch first; __post_init__ otherwise
    with pytest.raises((ValueError, TypeError)):
        TimeSeriesDataSource(np.zeros(5), np.zeros((2, 4, 2)))


def test_split_time_partitions():
    src = _source(n=2, T=10, metadata=_meta())
    a, b = src.split_time(6)
    assert a.trajectory_length == 6 and b.trajectory_length == 4
    assert_close(np.concatenate([a.ts, b.ts]), src.ts, rtol=TOL["identity"])
    assert_close(np.concatenate([a.ys, b.ys], axis=1), src.ys, rtol=TOL["identity"])
    assert a.metadata is src.metadata  # copy.replace keeps the other fields
    for bad in (0, 10):
        with pytest.raises(ValueError):
            src.split_time(bad)


def test_split_trajectories_partitions_and_is_seeded():
    src = _source(n=10, T=5)
    train, held = src.split_trajectories(0.3, seed=1)
    assert train.num_trajectories == 7 and held.num_trajectories == 3
    rows = {float(r) for r in np.concatenate([train.ys, held.ys])[:, 0, 0]}
    assert rows == {float(r) for r in src.ys[:, 0, 0]}
    same = src.split_trajectories(0.3, seed=1)[0]
    assert_close(same.ys, train.ys, rtol=TOL["identity"])
    assert not np.array_equal(src.split_trajectories(0.3, seed=2)[0].ys, train.ys)
    with pytest.raises(ValueError):
        src.split_trajectories(1.0)


# ----------------------------------------------------------------- 2. metadata -----
def test_metadata_attrs_round_trip_and_hash():
    m = _meta()
    attrs = m.to_attrs()
    groups = {"system", "sampling", "grid", "solve", "provenance", "extra"}
    assert set(attrs) == groups | {"config_hash"}
    assert all(isinstance(v, str) for v in attrs.values())  # flat, JSON-encoded groups
    assert DatasetMetadata.from_attrs(attrs) == m
    assert len(m.config_hash) == 8 and attrs["config_hash"] == m.config_hash
    # provenance and extra do not enter the hash; the generation groups do
    assert _meta(provenance=Provenance(created="later")).config_hash == m.config_hash
    assert _meta(extra={}).config_hash == m.config_hash
    assert _meta(grid=GridSpec(0.0, 2.0, 10)).config_hash != m.config_hash
    other_solve = SolveSpec("Tsit5", 1e-4, 1e-8, 4096)
    assert _meta(solve=other_solve).config_hash != m.config_hash
    other_sampling = SamplingSpec("UniformBox", {}, 1, 3)
    assert _meta(sampling=other_sampling).config_hash != m.config_hash


# --------------------------------------------------------------------- 3. disk -----
def test_save_load_round_trip(tmp_path):
    src = _source(metadata=_meta())
    path = src.save(tmp_path / "d.nc")
    assert path.exists() and len(list(tmp_path.iterdir())) == 1  # one file
    back = TimeSeriesDataSource.load(path)
    assert_close(back.ts, src.ts, rtol=TOL["identity"])
    assert_close(back.ys, src.ys, rtol=TOL["identity"])
    assert back.metadata == src.metadata  # every group, nested dicts included


def test_load_dtype_mismatch_is_loud(tmp_path):
    base = _source()
    src = TimeSeriesDataSource(base.ts, base.ys.astype(np.float32), _meta())
    path = src.save(tmp_path / "f32.nc")
    with pytest.raises(TypeError, match="float32"):
        TimeSeriesDataSource.load(path, dtype=np.float64)
    assert TimeSeriesDataSource.load(path, dtype=np.float32).ys.dtype == np.float32


def test_save_requires_metadata(tmp_path):
    with pytest.raises(ValueError, match="metadata"):
        _source().save(tmp_path / "x.nc")


# ------------------------------------------------------------------ 4. windows -----
def _start_of(window, src):
    """Start index of a window element on ``src``'s grid (``ts = k/(T-1)``)."""
    return int(round(window["t"][0] * (src.trajectory_length - 1)))


@given(T=st.integers(2, 12), data=st.data())
def test_random_window_cuts_contiguous_slices(T, data):
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


def test_transient_weight_and_weighted_window_validation():
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
    src = _source(n=3, T=10)
    batch = next(iter(windows(src, 4, seed=0).batch(8)))
    assert batch["t"].shape == (8, 4) and batch["u"].shape == (8, 4, 2)
    # each window is a slice of one trajectory: ys[i] takes values in [20 i, 20 i + 20)
    traj_ids = {int(u[0, 0]) // 20 for u in batch["u"]}
    assert traj_ids <= {0, 1, 2}


def test_mixed_windows_ratio_and_validation():
    src = _source(n=2, T=12)
    ds = mixed_windows(src, 3, 6, weights=(1.0, 2.0), seed=0)
    starts = [_start_of(ds[i], src) for i in range(300)]
    assert all(0 <= k <= 3 or 6 <= k <= 9 for k in starts)  # [0, 6-3] and [6, 12-3]
    frac_early = np.mean([k < 6 for k in starts])
    assert frac_early == pytest.approx(1 / 3, abs=0.02)  # deterministic interleave
    for bad in (2, 10):
        with pytest.raises(ValueError):
            mixed_windows(src, 3, bad, seed=0)


# ----------------------------------------------------------------- 5. generate -----
TIGHT = SolverConfig(solver=dfx.Dopri5(), rtol=1e-9, atol=1e-11)


def test_generate_bautin():
    nf = BautinNormalForm(1.0, 0.5, 2.0, 0.5)
    ts = jnp.linspace(0.0, 1.0, 6)
    src = generate(nf, UniformAnnulus(0.3, 2.0), ts, 5, seed=3, config=TIGHT)
    assert src.ys.shape == (5, 6, 2)
    assert_close(src.u0, src.ys[:, 0], rtol=TOL["identity"])
    m = src.metadata
    assert m is not None
    assert m.system.name == "BautinNormalForm"
    assert_close(m.system.params["a"], 1.0, rtol=TOL["closed_form"])
    assert_close(m.system.params["b"], 0.5, rtol=TOL["closed_form"])
    assert m.sampling == SamplingSpec(
        "UniformAnnulus", {"r_min": 0.3, "r_max": 2.0, "center": [0.0, 0.0]}, 3, 5
    )
    assert m.grid == GridSpec(0.0, 1.0, 6)
    assert m.solve == SolveSpec("Dopri5", 1e-9, 1e-11, 4096, "RadiusSquaredIntegration")
    assert m.provenance.dtype == "float64" and m.provenance.created
    # the trajectories are the system's flow
    ref = nf.flow(ts, jnp.asarray(src.u0[0]), config=TIGHT, integration="cartesian").ys
    assert_close(src.ys[0], ref, rtol=TOL["flow"], atol=TOL["flow"])
    # same config -> same hash; different seed or tolerance -> different hash
    again = generate(nf, UniformAnnulus(0.3, 2.0), ts, 5, seed=3, config=TIGHT)
    assert again.metadata.config_hash == m.config_hash
    other = generate(nf, UniformAnnulus(0.3, 2.0), ts, 5, seed=4, config=TIGHT)
    assert other.metadata.config_hash != m.config_hash
    assert dataset_path("data", "bautin", m).name == f"bautin-{m.config_hash}.nc"


def test_generate_fhn_and_failure():
    ode = FitzhughNagumo()
    ts = jnp.linspace(0.0, 2.0, 5)
    box = UniformBox([-2.0, -1.0], [2.0, 1.0])
    src = generate(ode, box, ts, 4, seed=0)
    assert src.ys.shape == (4, 5, 2) and src.metadata.solve.integration == ""
    assert src.metadata.system.params == {"a": 0.7, "b": 0.8, "c": 3, "z": -0.4}
    assert np.all(np.abs(src.u0[:, 0]) <= 2.0) and np.all(np.abs(src.u0[:, 1]) <= 1.0)
    with pytest.raises(ValueError, match="integration"):
        generate(ode, box, ts, 2, seed=0, integration="polar")
    with pytest.raises(RuntimeError, match="integrations failed"):
        generate(
            ode,
            box,
            jnp.linspace(0.0, 50.0, 5),
            3,
            seed=0,
            config=SolverConfig(max_steps=4),
        )
