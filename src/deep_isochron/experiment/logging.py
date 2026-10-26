"""From the ``wandb`` group and ``log_every`` to a ``Logger``: a ``PrintLogger``
always, and a ``WandbLogger`` when ``wandb.mode`` is not ``disabled``, wrapping
``wandb.init`` in the mode the config names with the run directory as wandb's ``dir``.
The logger classes themselves live in ``training.loggers``."""

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
    """The run's logger and its wandb run, if any.

    **Arguments:**

    - `cfg`: The composed config; `log_every` and the `wandb` group are read.
    - `run_dir`: The run directory, wandb's ``dir``.
    - `config`: The resolved config to record on the wandb run.
    - `batches_per_epoch`: For the epoch in every record.
    - `group`: The wandb group.
    - `wandb_run`: An existing run, from ``wandb.init`` in a notebook; it is logged
        to as it is, whatever ``wandb.mode`` says, and is the caller's to finish.

    **Returns:** ``(logger, wandb_run_or_None)``; the logger is one-step delayed, so
    the host never waits on the current step.
    """
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
