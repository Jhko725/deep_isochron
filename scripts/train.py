"""Train a model from a composed config (Phase D, ADR-0011).

    uv run python scripts/train.py            # defaults: data=bautin model=conjugacy
    uv run python scripts/train.py model=autoencoder loss=autoencoder schedule=yawata
    uv run python scripts/train.py num_steps=20000 windows.batch=4096 wandb=online
    uv run python scripts/train.py resume=runs/2026-10-05/12-00-00   # continue a run
    uv run python scripts/train.py -m seed=0,1,2   # multirun; wandb group = sweep dir

Hydra owns the run directory (``runs/<date>/<time>``, or ``runs/multirun/...`` with
``-m``); the resolved config, ``metadata.json`` and Orbax checkpoints land there. The
dataset must have been generated (``scripts/generate_data.py``) — see
``experiment.build_source``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
from deep_isochron.experiment import train
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    # Hydra sets the root logger to INFO, which lets Orbax's and JAX's absl chatter
    # through; the run's own output is the PrintLogger
    for name in ("absl", "jax", "orbax"):
        logging.getLogger(name).setLevel(logging.WARNING)
    hc = HydraConfig.get()
    run_dir = Path(hc.runtime.output_dir)
    group = Path(hc.sweep.dir).name if hc.mode == RunMode.MULTIRUN else None
    train(cfg, run_dir, group=group)


if __name__ == "__main__":
    main()
