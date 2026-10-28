"""The isochron configs, ``deep_isochron.experiment.isochrons`` with
``configs/isochrons/`` and ``configs/cycle/``: both shipped configs compose against the
`IsochronConfig` schema with their data config's system and their cycle config; the
schema rejects an unknown key, a wrong type and a missing anchor; the phases come from
exactly one of ``count`` and ``values``; the file's hash follows what decides the
result, ignores ``max_workers``, and is pinned for the shipped configs so that saved
foliations stay findable; a reader never computes and a writer never overwrites. With
AUTO (marker ``auto``), a one-phase foliation computed from a config loads back through
the same config.
"""

import math
from pathlib import Path

import numpy as np
import pytest
from deep_isochron.analysis.ode import IsochronSettings
from deep_isochron.experiment import (
    compose,
    compose_group,
    compute_isochrons,
    CycleConfig,
    isochron_config,
    IsochronConfig,
    load_isochrons,
)
from hydra.errors import ConfigCompositionException
from omegaconf import OmegaConf


def _config(name: str, *overrides: str) -> IsochronConfig:
    """``configs/isochrons/<name>.yaml`` with `overrides`, as the schema object."""
    return isochron_config(compose(*overrides, config_name=f"isochrons/{name}"))


SHIPPED_HASHES = {"fhn": "d9ded7af", "fhn-easy": "b688cbfe"}
"""The hashes of the shipped configs as of 2026-10-28: a change renames every saved
foliation, so one here is deliberate."""


@pytest.mark.parametrize("name, data", [("fhn", "fhn"), ("fhn-easy", "fhn-easy")])
def test_shipped_isochron_configs_compose_on_the_schema(name, data):
    """Each isochron config composes its data config and its cycle config into an
    `IsochronConfig` whose `settings` is an `IsochronSettings`, the same name, and a
    file under ``data/isochrons`` named after it with the pinned hash."""
    cfg = _config(name)
    assert isinstance(cfg, IsochronConfig)
    assert cfg.data == compose_group("data", data)
    cycle = OmegaConf.to_object(compose_group("cycle", data))
    assert cfg.cycle == cycle and isinstance(cycle, CycleConfig)
    assert isinstance(cfg.settings, IsochronSettings)
    assert cfg.settings.box_scale == 2.5
    assert cfg.result_digest() == SHIPPED_HASHES[name]
    stem = f"{cfg.data['name']}-{cfg.result_digest()}"
    assert cfg.file() == Path("data/isochrons") / f"{stem}.nc"


@pytest.mark.parametrize(
    "override",
    ["settings.n_period=4", "settings.ntst=x", "cycle.anchor=null", "foo=1"],
)
def test_schema_rejects_an_unknown_key_a_wrong_type_and_a_missing_anchor(override):
    """Composition fails on a key the schema does not have, a value of the wrong type,
    and a null for the required anchor (`ConfigCompositionException`)."""
    with pytest.raises(ConfigCompositionException):
        compose(override, config_name="isochrons/fhn")


def test_settings_are_validated_on_conversion():
    """A value the schema's types allow but `IsochronSettings` refuses raises its
    `ValueError` when the config is converted."""
    with pytest.raises(ValueError, match="n_periods"):
        _config("fhn", "settings.n_periods=0")


def test_phases_from_count_or_values():
    """``count`` gives ``2πk/count`` wrapped to (−π, π]; ``values`` are wrapped; setting
    both or neither, or a count below 1, raises `ValueError`."""
    phases = _config("fhn", "phases.count=4").wrapped_phases()
    assert phases == pytest.approx((0.0, math.pi / 2, math.pi, -math.pi / 2))
    listed = _config("fhn", "phases.count=null", "phases.values=[0.5,7.0]")
    assert listed.wrapped_phases() == pytest.approx((0.5, 7.0 - 2 * math.pi))
    with pytest.raises(ValueError, match="exactly one"):
        _config("fhn", "phases.values=[0.5]").wrapped_phases()
    with pytest.raises(ValueError, match="exactly one"):
        _config("fhn", "phases.count=null").wrapped_phases()
    with pytest.raises(ValueError, match="positive"):
        _config("fhn", "phases.count=0").wrapped_phases()


def test_hash_follows_what_decides_the_result():
    """The hash changes with the system's parameters, the anchor, the phases and a
    setting, and not with ``max_workers``, which decides only the speed."""
    base = _config("fhn").result_digest()
    assert _config("fhn", "settings.max_workers=3").result_digest() == base
    for override in (
        "+data.system.c=2.5",
        "cycle.anchor=[-1.7,0.4]",
        "phases.count=50",
        "settings.n_periods=4",
    ):
        assert _config("fhn", override).result_digest() != base, override


def test_reader_never_computes_and_writer_never_overwrites(tmp_path):
    """`load_isochrons` without a file raises `FileNotFoundError` naming the script;
    `compute_isochrons` with a file in place raises `FileExistsError` before any
    computation."""
    cfg = _config("fhn", f"out_dir={tmp_path}")
    with pytest.raises(FileNotFoundError, match="compute_isochrons.py"):
        load_isochrons(cfg)
    cfg.file().touch()
    with pytest.raises(FileExistsError):
        compute_isochrons(cfg)


@pytest.mark.auto
def test_computed_foliation_loads_back_through_its_config(auto, tmp_path):
    """A one-phase foliation computed from the FitzHugh–Nagumo config, with test-sized
    settings, is saved at `file()` with the resolved config in its provenance, and
    `load_isochrons` rebuilds the cycle and returns the same isochron, from the
    composed config as from the schema object."""
    cfg = compose(
        f"out_dir={tmp_path}",
        "phases.count=null",
        "phases.values=[0.5]",
        "settings.n_periods=1",
        "settings.box_scale=2.0",
        "settings.max_steps=1500",
        "settings.max_workers=1",
        config_name="isochrons/fhn",
    )
    computed = compute_isochrons(cfg)
    assert not computed.failures and isochron_config(cfg).file().exists()
    assert "anchor" in computed.provenance["config"]
    loaded = load_isochrons(isochron_config(cfg))
    (got,), (want,) = loaded.isochrons, computed.isochrons
    assert got.phase == want.phase == pytest.approx(0.5)
    np.testing.assert_array_equal(got.points, want.points)
