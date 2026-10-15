"""From the ``loss``, ``schedule`` and ``optimizer`` groups to a ``Trainer``."""

from __future__ import annotations

import optax
from omegaconf import DictConfig, OmegaConf

from ..training import AbstractLoss, AbstractLossSchedule, Constant, Trainer
from .instantiate import instantiate


def build_loss(loss_cfg: DictConfig) -> AbstractLoss:
    return instantiate(loss_cfg)


def build_schedule(
    schedule_cfg: DictConfig, loss: AbstractLoss
) -> AbstractLossSchedule:
    """``Constant`` with ``values: null`` takes the loss's ``default_weights``; anything
    else is instantiated as written."""
    target = schedule_cfg.get("_target_", "")
    if target.endswith("Constant") and schedule_cfg.get("values") is None:
        return Constant(tuple(loss.default_weights))
    return instantiate(schedule_cfg)


def build_optimizer(
    optimizer_cfg: DictConfig, clip_norm: float | None = None
) -> optax.GradientTransformation:
    """The ``optimizer`` group as written (``optax.adam`` / ``adabelief`` / ``adamw``
    with its ``decay_mask``); with ``clip_norm`` the gradients are clipped to that
    global norm first (``optax.chain(clip_by_global_norm, optimizer)``)."""
    optimizer = instantiate(optimizer_cfg)
    if clip_norm is None:
        return optimizer
    return optax.chain(optax.clip_by_global_norm(float(clip_norm)), optimizer)


def build_trainer(cfg: DictConfig) -> Trainer:
    loss = build_loss(cfg.loss)
    return Trainer(
        build_optimizer(cfg.optimizer, cfg.get("clip_norm")),
        loss,
        build_schedule(cfg.schedule, loss),
    )


def resolved(cfg: DictConfig) -> dict:
    """The config as a plain, fully resolved dict (what goes into wandb and the
    checkpoint metadata)."""
    out = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    assert isinstance(out, dict)
    return out
