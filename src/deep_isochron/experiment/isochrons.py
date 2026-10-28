r"""From an isochron config in ``configs/isochrons/`` to a saved, reloadable foliation.

An isochron config is validated against the schema `IsochronConfig` (ADR-0011 §1): it
composes a ``data`` config, whose ``system`` it traces, and a ``cycle`` config from
``configs/cycle/`` (`CycleConfig`: ``x0``, ``t_settle``, ``m``, ``anchor``), and adds
the ``phases`` (`PhasesConfig`: ``count`` or ``values``) and the ``settings``, an
`IsochronSettings` node. ``isochron_config`` turns the composed `DictConfig` into the
dataclass, whose methods derive what is a function of the fields alone: the
`wrapped_phases`, the `result_digest` and the `file`. The functions that do work take
either form: ``isochron_cycle`` finds and anchors the cycle, ``compute_isochrons``
traces the
foliation by AUTO-07p (``analysis.ode.auto``, ADR-0017) and saves it at ``file()``,
``<out_dir>/<name>-<hash>.nc``, with the resolved config in its provenance, and
``load_isochrons`` rebuilds the same cycle and reads the file back, the counterpart of
``build_source`` for datasets. ``scripts/compute_isochrons.py`` is the Hydra entry
point.

The hash covers everything that decides the result: the system and its parameters, the
cycle, the wrapped phases and the settings other than ``max_workers``, which only
decides the speed. The traced points are stored as AUTO returned them; resampling is the
reader's choice (`Isochron.resample`).
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from omegaconf import DictConfig, MISSING, OmegaConf

from ..analysis.ode import (
    find_limit_cycle,
    isochron_foliation,
    IsochronFoliation,
    IsochronSettings,
    LimitCycle,
    reanchor,
    wrap_phases,
)
from .instantiate import instantiate


@dataclass
class CycleConfig:
    """The schema of ``configs/cycle/``: how a system's cycle is found and anchored.

    **Attributes:**

    - `x0`: A point in the basin of the cycle, for `find_limit_cycle`.
    - `t_settle`: Its settling time.
    - `m`: The number of shooting points.
    - `anchor`: The point near the cycle that `reanchor` moves phase 0 to; required,
        since without it phase 0 is wherever settling ended.
    """

    x0: list[float] = MISSING
    t_settle: float = MISSING
    m: int = 8
    anchor: list[float] = MISSING


@dataclass
class PhaseFieldConfig:
    """The schema of ``configs/phase_field/``: how a system's asymptotic phase field is
    read off by `phase_grid` (route (a) of ``docs/design/phase-field.md`` §3.1).

    **Attributes:**

    - `t_end`: The integration time after which the phase is read off the cycle; long
        enough for the slowest point of the grid to settle.
    """

    t_end: float = MISSING


@dataclass
class PhasesConfig:
    """The phases of a foliation: exactly one of `count` and `values`.

    **Attributes:**

    - `count`: Evenly spaced phases $2\\pi k/$`count`, wrapped to $(-\\pi, \\pi]$.
    - `values`: An explicit list of phases, in radians.
    """

    count: int | None = None
    values: list[float] | None = None


@dataclass
class IsochronConfig:
    """The schema of ``configs/isochrons/``.

    **Attributes:**

    - `data`: The composed data config, whose `system` is traced; a `_target_` mapping.
    - `cycle`: The composed cycle config.
    - `phases`: Which phases to trace.
    - `settings`: How to trace them.
    - `name`: The file's stem, the data config's name by default.
    - `out_dir`: The directory of the file.
    """

    data: Any = MISSING
    cycle: CycleConfig = MISSING
    phases: PhasesConfig = field(default_factory=PhasesConfig)
    settings: IsochronSettings = field(default_factory=IsochronSettings)
    name: str = "${data.name}"
    out_dir: str = "data/isochrons"

    def wrapped_phases(self) -> tuple[float, ...]:
        r"""The phases, wrapped to $(-\pi, \pi]$.

        **Raises:**

        `ValueError` unless exactly one of `phases.count` and `phases.values` is set,
        and for a phase that is not finite or repeats another after wrapping.
        """
        count, values = self.phases.count, self.phases.values
        if (count is None) == (values is None):
            raise ValueError(
                "set exactly one of phases.count and phases.values; got "
                f"count={count}, values={values}."
            )
        if values is None:
            assert count is not None  # exactly one is set, checked above
            if count < 1:
                raise ValueError(f"phases.count must be positive; got {count}.")
            values = [2 * math.pi * k / count for k in range(count)]
        return wrap_phases(list(values))

    def result_identity(self) -> dict[str, Any]:
        """What decides the result, as JSON-able values: the system and its
        parameters, the cycle, the wrapped phases and the settings other than
        `max_workers`."""
        with jax.enable_x64(True):
            system = instantiate(self.data["system"])
            params = {k: float(v) for k, v in system.params().items()}
        settings = dataclasses.asdict(self.settings)
        del settings["max_workers"]  # decides the speed, not the result
        return {
            "system": {"name": type(system).__name__, "params": params},
            "cycle": dataclasses.asdict(self.cycle),
            "phases": list(self.wrapped_phases()),
            "settings": settings,
        }

    def result_digest(self) -> str:
        """The 8-character SHA-256 of `result_identity`, as the datasets'
        ``config_hash``: a stable digest across processes and machines, which the
        dataclass's own ``__hash__``, salted per process and tied to the field-wise
        ``__eq__``, is not."""
        blob = json.dumps(self.result_identity(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:8]

    def file(self) -> Path:
        """Where the foliation lives, whether or not it exists:
        ``<out_dir>/<name>-<hash>.nc``."""
        return Path(self.out_dir) / f"{self.name}-{self.result_digest()}.nc"


def isochron_config(cfg: DictConfig | IsochronConfig) -> IsochronConfig:
    """The `IsochronConfig` of a composed config, with interpolations resolved and
    `IsochronSettings` validated; an `IsochronConfig` is returned as it is.

    **Raises:**

    `omegaconf.errors.ValidationError` for a value of the wrong type or a field the
    schema requires and the config leaves out; `ValueError` from `IsochronSettings`.
    """
    if isinstance(cfg, IsochronConfig):
        return cfg
    config = OmegaConf.to_object(cfg)
    assert isinstance(config, IsochronConfig)
    return config


def isochron_cycle(cfg: DictConfig | IsochronConfig) -> LimitCycle:
    """The cycle of an isochron config: `find_limit_cycle` from ``cycle.x0`` for the
    data config's system, in float64, with its phase origin moved by `reanchor` to the
    cycle point nearest ``cycle.anchor``."""
    config = isochron_config(cfg)
    with jax.enable_x64(True):
        system = instantiate(config.data["system"])
        cycle = find_limit_cycle(
            system,
            jnp.asarray(config.cycle.x0, dtype=float),
            t_settle=config.cycle.t_settle,
            m=config.cycle.m,
        )
        return reanchor(cycle, jnp.asarray(config.cycle.anchor, dtype=float))


def compute_isochrons(cfg: DictConfig | IsochronConfig) -> IsochronFoliation:
    """Trace the foliation of an isochron config and save it at ``file()``, with the
    resolved config in its provenance under ``"config"``.

    A foliation with failed phases is saved too; the caller reads
    ``foliation.failures``.

    **Returns:**

    The `IsochronFoliation`.

    **Raises:**

    `FileExistsError` when the file exists, since the result would be the same.
    """
    config = isochron_config(cfg)
    path = config.file()
    if path.exists():
        raise FileExistsError(f"{path} exists; delete it to recompute.")
    cycle = isochron_cycle(config)
    foliation = isochron_foliation(cycle, config.wrapped_phases(), config.settings)
    resolved = json.dumps(dataclasses.asdict(config), default=str)
    foliation = copy.replace(
        foliation, provenance=foliation.provenance | {"config": resolved}
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    foliation.save(path)
    return foliation


def load_isochrons(cfg: DictConfig | IsochronConfig) -> IsochronFoliation:
    """Read the foliation of an isochron config, with its cycle rebuilt from the config.

    **Raises:**

    `FileNotFoundError`, naming the command to run, when the file does not exist; a
    reader never computes. `ValueError` from `IsochronFoliation.load` when the rebuilt
    cycle does not match the file.
    """
    config = isochron_config(cfg)
    path = config.file()
    if not path.exists():
        raise FileNotFoundError(
            f"isochrons {path} not found; run\n"
            "    uv run python scripts/compute_isochrons.py --config-name "
            "isochrons/<config>\n(with the same overrides) first."
        )
    return IsochronFoliation.load(path, isochron_cycle(config))
