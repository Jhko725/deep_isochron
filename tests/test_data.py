"""Laws of the data layer (``deep_isochron.data``).

1. source        a grain random-access source over whole trajectories; shape checks;
                 ``copy.replace`` splits — split_time partitions the time axis,
                 split_trajectories partitions the trajectories deterministically
2. metadata      grouped; attrs round trip through JSON; config_hash covers the four
                 generation groups and nothing in provenance/extra
3. disk          save -> load round-trips arrays and metadata through one file; dtype
                 mismatch is loud; missing metadata cannot be saved
4. device        importing the package initializes no JAX backend; to_device yields
                 device arrays in order; resolve_device picks JAX's default only when
                 the process is scoped to a card
5. generate      shapes, u0 == ys[:, 0], metadata groups, trajectories equal the flow,
                 config_hash stable / sensitive, failures raise with indices

Windowing (``data.windows``) is ``tests/test_windows.py``.
"""

import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from deep_isochron.data import (
    dataset_path,
    DatasetMetadata,
    generate,
    GridSpec,
    Provenance,
    SamplingSpec,
    SolveSpec,
    SystemSpec,
    TimeSeriesDataSource,
    to_device,
    UniformAnnulus,
    UniformBox,
    validation_windows,
)
from deep_isochron.systems import BautinNormalForm, FitzhughNagumo, SolverConfig

from tests.helpers import assert_close, SOLVERS, TOL


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


# ------------------------------------------------------------------- 4. device ------
def test_importing_the_package_does_not_initialize_a_jax_backend():
    """grain worker processes import the package to unpickle the source; if that
    initialized JAX they would each grab a CUDA context and preallocate GPU memory
    (seen on a V100: every worker OOM at import). Module-level constraint primitives
    compute their shift lazily for this reason."""
    code = (
        "import jax._src.xla_bridge as xb; import deep_isochron.data, "
        "deep_isochron.training, deep_isochron.model; print(bool(xb._backends))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "False", out.stdout + out.stderr


def test_to_device_yields_device_arrays():
    """grain's two-stage prefetch: batches arrive as JAX arrays on the device, in the
    same order and with the same values as the host pipeline."""
    src = _source(n=3, T=10)
    host = list(validation_windows(src, 4).batch(2))
    dev = list(to_device(validation_windows(src, 4).batch(2), jax.devices()[0]))
    assert len(dev) == len(host)
    for h, d in zip(host, dev):
        assert isinstance(d["u"], jax.Array) and d["u"].dtype == jnp.float64
        assert np.array_equal(np.asarray(d["u"]), h["u"])


# ----------------------------------------------------------------- 5. generate -----
def test_generate_bautin():
    nf = BautinNormalForm(1.0, 0.5, 2.0, 0.5)
    ts = jnp.linspace(0.0, 1.0, 6)
    src = generate(nf, UniformAnnulus(0.3, 2.0), ts, 5, seed=3, config=SOLVERS["data"])
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
    assert m.solve == SolveSpec("Tsit5", 1e-9, 1e-11, 4096, "ClosedFormIntegration")
    assert m.provenance.dtype == "float64" and m.provenance.created
    # the trajectories are the system's flow
    ref = nf.flow(
        ts, jnp.asarray(src.u0[0]), config=SOLVERS["data"], integration="cartesian"
    ).ys
    assert_close(src.ys[0], ref, rtol=TOL["flow"], atol=TOL["flow"])
    # same config -> same hash; different seed or tolerance -> different hash
    again = generate(
        nf, UniformAnnulus(0.3, 2.0), ts, 5, seed=3, config=SOLVERS["data"]
    )
    assert again.metadata.config_hash == m.config_hash
    other = generate(
        nf, UniformAnnulus(0.3, 2.0), ts, 5, seed=4, config=SOLVERS["data"]
    )
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


def test_resolve_device_explicit_index_and_cpu_default():
    from deep_isochron.data import resolve_device

    assert resolve_device(0) == jax.devices()[0]
    assert resolve_device() == jax.devices()[0]  # one device / CPU: JAX's default
    with pytest.raises(IndexError):
        resolve_device(len(jax.devices()))


def test_resolve_device_refuses_unassigned_accelerators(monkeypatch):
    """Several accelerators visible and no ``*_VISIBLE_DEVICES`` variable: no silent
    pick. (The fakes cannot pass the return annotation's runtime check, so the
    scheduler-scoped branch is covered by the CPU test above: one device → index 0.)"""
    from deep_isochron.data import resolve_device

    class Fake:
        platform, device_kind = "gpu", "fake"

        def __init__(self, i):
            self.id = i

    monkeypatch.setattr(jax, "devices", lambda backend=None: [Fake(0), Fake(1)])
    for var in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(ValueError, match="2 accelerators"):
        resolve_device()
