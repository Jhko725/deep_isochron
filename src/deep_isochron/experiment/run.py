"""One training run from a composed config: ``train(cfg, run_dir)`` is what
``scripts/train.py`` calls under Hydra and what the tests call directly; ``load_run``
and ``load_model`` reopen a finished (or interrupted) run from its directory.

Order of operations matters for JAX: ``x64`` and the default device are set before the
first array exists; the device comes from ``resolve_device(cfg.device)`` (ADR-0009 §1)
and the loader is ``to_device(..., device)``. Resuming (``cfg.resume`` or a later
``train`` on the same directory) restores the latest checkpoint — the whole
``TrainerState`` — and continues the *same* data stream from ``state.step``, which is a
slice of the deterministic batched dataset (ADR-0008 Decision 3).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
from omegaconf import DictConfig, OmegaConf

from ..data import resolve_device
from ..model import AbstractPhaseAmplitudeModel
from ..training import Evaluator, OrbaxCheckpointer, TrainerState
from ..utils.provenance import git_state
from .data import (
    build_loaders,
    build_source,
    reference_normal_form,
    validation_t_split,
)
from .logging import build_logger
from .model import build_model
from .training import build_trainer, resolved


CHECKPOINT_DIR = "checkpoints"
CONFIG_FILE = "config.yaml"


def configure_jax(cfg: DictConfig) -> jax.Device:
    """``jax_enable_x64`` and the default device, before anything is built."""
    jax.config.update("jax_enable_x64", bool(cfg.x64))
    device = resolve_device(cfg.device)
    jax.config.update("jax_default_device", device)
    return device


def checkpointer(cfg: DictConfig, run_dir: Path, metadata: dict[str, Any] | None):
    return OrbaxCheckpointer(
        run_dir / CHECKPOINT_DIR,
        save_every=cfg.checkpoint.every,
        metric=cfg.checkpoint.metric,
        keep_best=cfg.checkpoint.keep_best,
        custom_metadata=metadata,
    )


def train(
    cfg: DictConfig, run_dir: str | Path, *, group: str | None = None
) -> TrainerState:
    """Run the training the config describes, writing checkpoints, the resolved config
    and logs under ``run_dir``. Returns the final state."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = configure_jax(cfg)
    config = resolved(cfg)
    source = build_source(cfg.data)
    trainer = build_trainer(cfg)

    resume_dir = Path(cfg.resume) if cfg.resume else None
    if resume_dir is not None:
        previous = OrbaxCheckpointer(resume_dir / CHECKPOINT_DIR, metric=None)
        step = previous.latest_step
        if step is None:
            raise FileNotFoundError(f"no checkpoint to resume under {resume_dir}")
        template = trainer.init(
            build_model(cfg.model, jax.random.key(cfg.seed)),
            key=jax.random.key(cfg.seed + 1),
        )
        state = previous.restore(template, step)
        previous.close()
        start_step = int(state.step)
    else:
        state = trainer.init(
            build_model(cfg.model, jax.random.key(cfg.seed)),
            key=jax.random.key(cfg.seed + 1),
        )
        start_step = 0

    loaders = build_loaders(cfg, source, device, start_step=start_step)
    assert source.metadata is not None
    metadata = {
        "config": config,
        "dataset": {
            "config_hash": source.metadata.config_hash,
            "trajectories": source.num_trajectories,
        },
        "x64": bool(cfg.x64),
        "git": dict(zip(("sha", "dirty"), git_state())),
        "batches_per_epoch": loaders.batches_per_epoch,
    }
    (run_dir / CONFIG_FILE).write_text(OmegaConf.to_yaml(cfg, resolve=True))
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))

    print(
        f"run {run_dir}: {cfg.num_steps} steps from step {start_step}; "
        f"{loaders.num_windows} windows per epoch in {loaders.batches_per_epoch} "
        f"batches of {cfg.windows.batch} × {cfg.windows.length} "
        f"(≈ {cfg.num_steps / loaders.batches_per_epoch:.2f} epochs); "
        f"{source.num_trajectories} trajectories, "
        f"{loaders.val_source.num_trajectories} held out; "
        f"device {device.platform}:{device.id}"
    )
    logger, wandb_run = build_logger(
        cfg, run_dir, config, loaders.batches_per_epoch, group=group
    )
    evaluator = Evaluator(
        loaders.val,
        reference=reference_normal_form(cfg.data),
        t_split=validation_t_split(cfg, source),
    )
    try:
        state = trainer.train(
            state,
            loaders.train,
            # the loader covers whole epochs (>= num_steps batches); run num_steps
            num_steps=cfg.num_steps - start_step,
            logger=logger,
            checkpointer=checkpointer(cfg, run_dir, metadata),
            evaluate=evaluator,
            eval_every=cfg.eval_every,
        )
    finally:
        if wandb_run is not None:
            wandb_run.finish()
    return state


def load_run(run_dir: str | Path, step: int | None = None):
    """``(cfg, state)`` of a run directory: the config it was trained with (from the
    checkpoint metadata, falling back to ``config.yaml``) and its state at ``step``
    (latest if ``None``), rebuilt with the same builders."""
    run_dir = Path(run_dir)
    ckpt = OrbaxCheckpointer(run_dir / CHECKPOINT_DIR, metric=None)
    meta = ckpt.custom_metadata
    if meta.get("config") is not None:
        cfg = OmegaConf.create(meta["config"])
    else:
        cfg = OmegaConf.load(run_dir / CONFIG_FILE)
    assert isinstance(cfg, DictConfig)
    configure_jax(cfg)
    trainer = build_trainer(cfg)
    template = trainer.init(
        build_model(cfg.model, jax.random.key(cfg.seed)),
        key=jax.random.key(cfg.seed + 1),
    )
    state = ckpt.restore(template, step)
    ckpt.close()
    return cfg, state


def load_model(
    run_dir: str | Path, step: int | None = None
) -> AbstractPhaseAmplitudeModel:
    """The trained model of a run directory at ``step`` (latest if ``None``)."""
    _, state = load_run(run_dir, step)
    return state.model
