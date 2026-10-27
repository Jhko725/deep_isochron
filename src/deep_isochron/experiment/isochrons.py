r"""From an isochron config in ``configs/isochrons/`` to a saved, reloadable foliation.

An isochron config composes a ``data`` config, whose ``system`` it traces, with a
``cycle`` block (``x0``, ``t_settle``, ``m``, ``anchor``), the ``phases`` (``count`` or
``values``) and the ``settings`` of `IsochronSettings`. ``compute_isochrons`` finds the
cycle, moves its phase origin to the anchor, traces the foliation by AUTO-07p
(``analysis.ode.auto``, ADR-0017) and saves it at ``isochron_file(cfg)``,
``<out_dir>/<name>-<hash>.nc``, with the resolved config in its provenance.
``load_isochrons`` rebuilds the same cycle and reads the file back: the counterpart of
``build_source`` for datasets. ``scripts/compute_isochrons.py`` is the Hydra entry
point.

The hash covers everything that decides the result: the system and its parameters, the
cycle block, the wrapped phases and the settings other than ``max_workers``, which only
decides the speed. The traced points are stored as AUTO returned them; resampling is the
reader's choice (`Isochron.resample`).
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from omegaconf import DictConfig, OmegaConf

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


def isochron_phases(cfg: DictConfig) -> tuple[float, ...]:
    r"""The phases of an isochron config, wrapped to $(-\pi, \pi]$: ``count`` evenly
    spaced phases $2\pi k/\text{count}$, or the listed ``values``.

    **Raises:**

    `ValueError` unless exactly one of ``phases.count`` and ``phases.values`` is set,
    and for a phase that is not finite or repeats another after wrapping.
    """
    count, values = cfg.phases.get("count"), cfg.phases.get("values")
    if (count is None) == (values is None):
        raise ValueError(
            "set exactly one of phases.count and phases.values; got "
            f"count={count}, values={values}."
        )
    if count is not None:
        if count < 1:
            raise ValueError(f"phases.count must be positive; got {count}.")
        values = [2 * math.pi * k / count for k in range(count)]
    return wrap_phases(list(values))


def isochron_settings(cfg: DictConfig) -> IsochronSettings:
    """The `IsochronSettings` of an isochron config.

    **Raises:**

    `TypeError` for a key that `IsochronSettings` does not have.
    """
    fields = OmegaConf.to_container(cfg.settings, resolve=True)
    assert isinstance(fields, dict)
    return IsochronSettings(**{str(k): v for k, v in fields.items()})


def isochron_cycle(cfg: DictConfig) -> LimitCycle:
    """The cycle of an isochron config: `find_limit_cycle` from ``cycle.x0`` for the
    data config's system, in float64, with its phase origin moved by `reanchor` to the
    cycle point nearest ``cycle.anchor``."""
    with jax.enable_x64(True):
        system = instantiate(cfg.data.system)
        cycle = find_limit_cycle(
            system,
            jnp.asarray(cfg.cycle.x0, dtype=float),
            t_settle=float(cfg.cycle.t_settle),
            m=int(cfg.cycle.m),
        )
        return reanchor(cycle, jnp.asarray(cfg.cycle.anchor, dtype=float))


def _identity(cfg: DictConfig) -> dict[str, Any]:
    """What decides an isochron config's result, as JSON-able values."""
    with jax.enable_x64(True):
        system = instantiate(cfg.data.system)
        params = {k: float(v) for k, v in system.params().items()}
    settings = dataclasses.asdict(isochron_settings(cfg))
    del settings["max_workers"]  # decides the speed, not the result
    return {
        "system": {"name": type(system).__name__, "params": params},
        "cycle": OmegaConf.to_container(cfg.cycle, resolve=True),
        "phases": list(isochron_phases(cfg)),
        "settings": settings,
    }


def isochron_hash(cfg: DictConfig) -> str:
    """The 8-character SHA-256 of what decides an isochron config's result, as the
    datasets' ``config_hash`` is of theirs."""
    blob = json.dumps(_identity(cfg), sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:8]


def isochron_file(cfg: DictConfig) -> Path:
    """Where the foliation of an isochron config lives, whether or not it exists:
    ``<out_dir>/<name>-<hash>.nc``."""
    return Path(cfg.out_dir) / f"{cfg.name}-{isochron_hash(cfg)}.nc"


def compute_isochrons(cfg: DictConfig) -> IsochronFoliation:
    """Trace the foliation of an isochron config and save it at ``isochron_file(cfg)``,
    with the resolved config in its provenance under ``"config"``.

    A foliation with failed phases is saved too; the caller reads
    ``foliation.failures``.

    **Returns:**

    The `IsochronFoliation`.

    **Raises:**

    `FileExistsError` when the file exists, since the result would be the same.
    """
    path = isochron_file(cfg)
    if path.exists():
        raise FileExistsError(f"{path} exists; delete it to recompute.")
    cycle = isochron_cycle(cfg)
    foliation = isochron_foliation(cycle, isochron_phases(cfg), isochron_settings(cfg))
    config = json.dumps(OmegaConf.to_container(cfg, resolve=True))
    foliation = copy.replace(
        foliation, provenance=foliation.provenance | {"config": config}
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    foliation.save(path)
    return foliation


def load_isochrons(cfg: DictConfig) -> IsochronFoliation:
    """Read the foliation of an isochron config, with its cycle rebuilt from the config.

    **Raises:**

    `FileNotFoundError`, naming the command to run, when the file does not exist; a
    reader never computes. `ValueError` from `IsochronFoliation.load` when the rebuilt
    cycle does not match the file.
    """
    path = isochron_file(cfg)
    if not path.exists():
        raise FileNotFoundError(
            f"isochrons {path} not found; run\n"
            "    uv run python scripts/compute_isochrons.py --config-name "
            "isochrons/<config>\n(with the same overrides) first."
        )
    return IsochronFoliation.load(path, isochron_cycle(cfg))
