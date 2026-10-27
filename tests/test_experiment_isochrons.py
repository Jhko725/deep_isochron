"""The isochron configs, ``deep_isochron.experiment.isochrons`` with
``configs/isochrons/``: both shipped configs compose with their data config's system;
the phases come from exactly one of ``count`` and ``values``; the file's hash follows
what decides the result and ignores ``max_workers``; a reader never computes and a
writer never overwrites. With AUTO (marker ``auto``), a one-phase foliation computed
from a config loads back through the same config.
"""

import math
from pathlib import Path

import numpy as np
import pytest
from deep_isochron.analysis.ode.auto import _env
from deep_isochron.experiment import (
    compose,
    compose_group,
    compute_isochrons,
    isochron_file,
    isochron_hash,
    isochron_phases,
    isochron_settings,
    load_isochrons,
)
from omegaconf import DictConfig


def _config(name: str, *overrides: str) -> DictConfig:
    """``configs/isochrons/<name>.yaml`` with `overrides`."""
    return compose(*overrides, config_name=f"isochrons/{name}")


@pytest.mark.parametrize("name, data", [("fhn", "fhn"), ("fhn-easy", "fhn-easy")])
def test_shipped_isochron_configs_take_their_data_configs_system(name, data):
    """Each isochron config composes its data config: the same system, the same name,
    and a file under ``data/isochrons`` named after it."""
    cfg = _config(name)
    assert cfg.data == compose_group("data", data)
    path = isochron_file(cfg)
    assert path.parent == Path("data/isochrons")
    assert path.name == f"{cfg.data.name}-{isochron_hash(cfg)}.nc"
    assert isochron_settings(cfg).box_scale == 2.5


def test_phases_from_count_or_values():
    """``count`` gives ``2πk/count`` wrapped to (−π, π]; ``values`` are wrapped; setting
    both or neither, or a count below 1, raises `ValueError`."""
    phases = isochron_phases(_config("fhn", "phases.count=4"))
    assert phases == pytest.approx((0.0, math.pi / 2, math.pi, -math.pi / 2))
    listed = isochron_phases(
        _config("fhn", "phases.count=null", "phases.values=[0.5,7.0]")
    )
    assert listed == pytest.approx((0.5, 7.0 - 2 * math.pi))
    with pytest.raises(ValueError, match="exactly one"):
        isochron_phases(_config("fhn", "phases.values=[0.5]"))
    with pytest.raises(ValueError, match="exactly one"):
        isochron_phases(_config("fhn", "phases.count=null"))
    with pytest.raises(ValueError, match="positive"):
        isochron_phases(_config("fhn", "phases.count=0"))


def test_hash_follows_what_decides_the_result():
    """The hash changes with the system's parameters, the anchor, the phases and a
    setting, and not with ``max_workers``, which decides only the speed, or with the
    ``phase_field`` block, which the foliation does not use."""
    base = isochron_hash(_config("fhn"))
    assert isochron_hash(_config("fhn", "settings.max_workers=3")) == base
    assert isochron_hash(_config("fhn", "phase_field.t_end=10.0")) == base
    for override in (
        "+data.system.c=2.5",
        "cycle.anchor=[-1.7,0.4]",
        "phases.count=50",
        "settings.n_periods=4",
    ):
        assert isochron_hash(_config("fhn", override)) != base, override


def test_reader_never_computes_and_writer_never_overwrites(tmp_path):
    """`load_isochrons` without a file raises `FileNotFoundError` naming the script;
    `compute_isochrons` with a file in place raises `FileExistsError` before any
    computation."""
    cfg = _config("fhn", f"out_dir={tmp_path}")
    with pytest.raises(FileNotFoundError, match="compute_isochrons.py"):
        load_isochrons(cfg)
    isochron_file(cfg).touch()
    with pytest.raises(FileExistsError):
        compute_isochrons(cfg)


@pytest.mark.auto
def test_computed_foliation_loads_back_through_its_config(tmp_path):
    """A one-phase foliation computed from the FitzHugh–Nagumo config, with test-sized
    settings, is saved at `isochron_file` with the resolved config in its provenance,
    and `load_isochrons` rebuilds the cycle and returns the same isochron."""
    try:
        _env.load_auto()
    except _env.AutoNotInstalled as exc:
        pytest.skip(str(exc))
    cfg = _config(
        "fhn",
        f"out_dir={tmp_path}",
        "phases.count=null",
        "phases.values=[0.5]",
        "settings.n_periods=1",
        "settings.box_scale=2.0",
        "settings.max_steps=1500",
        "settings.max_workers=1",
    )
    computed = compute_isochrons(cfg)
    assert not computed.failures and isochron_file(cfg).exists()
    assert "anchor" in computed.provenance["config"]
    loaded = load_isochrons(cfg)
    (got,), (want,) = loaded.isochrons, computed.isochrons
    assert got.phase == want.phase == pytest.approx(0.5)
    np.testing.assert_array_equal(got.points, want.points)
