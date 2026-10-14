"""``compose(*overrides)``: the training config as ``scripts/train.py`` would see it,
for notebooks and tests — ``hydra.compose`` over the repository's ``configs/`` without
Hydra's run-directory and logging machinery.

    cfg = compose("model=autoencoder", "loss=autoencoder", "num_steps=200")
    state = train(cfg, "runs/notebook/try-1")

``compose_group(group, name, *overrides)`` composes one group's config on its own —
``compose_group("data", "fhn", "n_trajectories=500")`` is the ``data`` node a notebook
hands to ``load_or_generate_source`` / ``build_source`` / ``generate_dataset``, with the
overrides written relative to the group (no ``data.`` prefix).
"""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig


CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"
"""``<repository>/configs`` for a source checkout (the package lives in ``src/``)."""


def compose(
    *overrides: str,
    config_name: str = "train",
    config_dir: str | Path | None = None,
) -> DictConfig:
    """Compose ``configs/<config_name>.yaml`` with Hydra overrides (``"data=fhn"``,
    ``"windows.batch=4096"``); ``config_dir`` defaults to the checkout's
    ``configs/``."""
    directory = Path(config_dir) if config_dir is not None else CONFIG_DIR
    if not directory.is_dir():
        raise FileNotFoundError(f"config directory {directory} not found")
    with hydra.initialize_config_dir(version_base=None, config_dir=str(directory)):
        return hydra.compose(config_name, overrides=list(overrides))


def compose_group(
    group: str,
    name: str,
    *overrides: str,
    config_dir: str | Path | None = None,
) -> DictConfig:
    """``configs/<group>/<name>.yaml`` alone, with overrides relative to the group
    (``compose_group("data", "fhn", "n_trajectories=500")``), returned as that group's
    node — what ``cfg.<group>`` would be after ``compose(f"{group}={name}", ...)``."""
    cfg = compose(
        *(f"{group}.{o}" for o in overrides),
        config_name=f"{group}/{name}",
        config_dir=config_dir,
    )
    node = cfg[group]
    assert isinstance(node, DictConfig)
    return node
