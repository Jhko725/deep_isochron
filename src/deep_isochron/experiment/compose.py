"""``compose(*overrides)``: the training config as ``scripts/train.py`` would see it,
for notebooks and tests — ``hydra.compose`` over the repository's ``configs/`` without
Hydra's run-directory and logging machinery.

    cfg = compose("model=autoencoder", "loss=autoencoder", "num_steps=200")
    state = train(cfg, "runs/notebook/try-1")
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
