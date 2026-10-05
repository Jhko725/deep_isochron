"""Loggers for a run: ``PrintLogger`` always; ``WandbLogger`` when ``wandb.mode`` is not
``disabled`` — wrapping ``wandb.init`` in whichever mode (``online`` / ``offline``) the
config names; the run directory is wandb's ``dir``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from ..training import DelayedLogger, Logger, PrintLogger


class MultiLogger(Logger):
    """Fan out to several loggers; closes all of them."""

    def __init__(self, *loggers: Logger) -> None:
        self.loggers = loggers

    def log(self, metrics, step: int) -> None:
        for logger in self.loggers:
            logger.log(metrics, step)

    def close(self) -> None:
        for logger in self.loggers:
            logger.close()


class EpochLogger(Logger):
    """Adds ``epoch = step / batches_per_epoch`` to every record before forwarding."""

    def __init__(self, inner: Logger, batches_per_epoch: int) -> None:
        self.inner, self.batches_per_epoch = inner, batches_per_epoch

    def log(self, metrics, step: int) -> None:
        self.inner.log({**metrics, "epoch": step / self.batches_per_epoch}, step)

    def close(self) -> None:
        self.inner.close()


def build_logger(
    cfg: DictConfig,
    run_dir: Path,
    config: dict[str, Any],
    batches_per_epoch: int,
    *,
    group: str | None = None,
) -> tuple[Logger, Any]:
    """``(logger, wandb_run_or_None)``. The returned logger is one-step delayed
    (``DelayedLogger``) so the host never waits on the current step."""
    loggers: list[Logger] = [
        PrintLogger(every=cfg.log_every, keys=("loss", "val/mse", "period", "epoch"))
    ]
    run = None
    mode = cfg.wandb.mode
    if mode != "disabled":
        import wandb  # optional at runtime; only imported when asked for

        from ..training import WandbLogger

        run = wandb.init(
            mode=mode,
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            dir=str(run_dir),
            config=config,
            group=group,
        )
        loggers.append(WandbLogger(run, every=cfg.wandb.every))
    logger = EpochLogger(MultiLogger(*loggers), batches_per_epoch)
    return DelayedLogger(logger), run
