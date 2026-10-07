"""From the ``wandb`` group and ``log_every`` to a ``Logger``: ``PrintLogger`` always;
``WandbLogger`` when ``wandb.mode`` is not ``disabled`` — wrapping ``wandb.init`` in
whichever mode (``online`` / ``offline``) the config names, the run directory as wandb's
``dir``. The logger classes themselves live in ``training.loggers``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from ..training import DelayedLogger, EpochLogger, Logger, MultiLogger, PrintLogger


def build_logger(
    cfg: DictConfig,
    run_dir: Path,
    config: dict[str, Any],
    batches_per_epoch: int,
    *,
    group: str | None = None,
    wandb_run: Any = None,
) -> tuple[Logger, Any]:
    """``(logger, wandb_run_or_None)``. The returned logger is one-step delayed
    (``DelayedLogger``) so the host never waits on the current step. An existing
    ``wandb_run`` (``wandb.init(...)`` done by the caller, e.g. in a notebook) is logged
    to as it is, whatever ``wandb.mode`` says, and is the caller's to finish."""
    loggers: list[Logger] = [
        PrintLogger(every=cfg.log_every, keys=("loss", "val/mse", "period", "epoch"))
    ]
    run = wandb_run
    mode = cfg.wandb.mode
    if run is None and mode != "disabled":
        import wandb  # optional at runtime; only imported when asked for

        run = wandb.init(
            mode=mode,
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            dir=str(run_dir),
            config=config,
            group=group,
        )
    if run is not None:
        from ..training import WandbLogger

        loggers.append(WandbLogger(run, every=cfg.wandb.every))
    logger = EpochLogger(MultiLogger(*loggers), batches_per_epoch)
    return DelayedLogger(logger), run
