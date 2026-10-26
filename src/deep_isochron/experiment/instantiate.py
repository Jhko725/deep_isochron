"""``instantiate``: Hydra's ``_target_`` convention with one change, YAML lists become
tuples.

The code base types fixed-size parameters as tuples (``default_weights``, ``xy_range``,
``before`` and ``after``), the test suite's beartype hook enforces those annotations at
runtime, and Hydra's own ``instantiate`` can only produce lists. Nested ``_target_``
mappings are instantiated recursively; everything else is passed through as plain
Python.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from hydra.utils import get_method
from omegaconf import DictConfig, ListConfig, OmegaConf


TARGET = "_target_"
"""The key naming the callable."""


def _convert(value: Any) -> Any:
    """`value` with nested targets instantiated and sequences made tuples."""
    if isinstance(value, Mapping):
        if TARGET in value:
            return instantiate(value)
        return {k: _convert(v) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return tuple(_convert(v) for v in value)
    return value


def instantiate(cfg: Mapping[str, Any] | DictConfig, **overrides: Any) -> Any:
    """Call ``cfg._target_`` with the remaining keys as keyword arguments.

    **Arguments:**

    - `cfg`: The mapping with a ``_target_``; lists become tuples and nested targets
        are instantiated.
    - `overrides`: Keyword arguments that take precedence, such as the per-layer
        ``key`` and ``flip`` a config cannot express.

    **Raises:** `ValueError` without a ``_target_``.
    """
    if isinstance(cfg, DictConfig | ListConfig):
        plain = OmegaConf.to_container(cfg, resolve=True)
    else:
        plain = dict(cfg)
    assert isinstance(plain, dict), "instantiate expects a mapping with _target_"
    entries = {str(k): v for k, v in plain.items()}
    if TARGET not in entries:
        raise ValueError(f"no {TARGET} in config {sorted(entries)}")
    target = get_method(entries[TARGET])
    kwargs: dict[str, Any] = {k: _convert(v) for k, v in entries.items() if k != TARGET}
    kwargs.update(overrides)
    return target(**kwargs)
