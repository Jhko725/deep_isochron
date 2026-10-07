"""A training run: its directory (``Run``), its objects (``setup`` → ``Experiment``) and
reopening it (``load_run`` / ``load_model``).

- ``Run`` is the run directory as an object — ``config.yaml``, ``metadata.json`` and
  ``checkpoints/`` under one path — and the only place that layout is spelled out.
  ``Run.create`` **refuses** a directory that already holds a run; continuing one is the
  explicit ``Run.resume`` (or ``setup(..., resume=True)``). ``Run.open`` reads one back.
  ``run.checkpointer()`` is an ``OrbaxCheckpointer`` in the right place carrying the
  run's metadata, so a checkpoint written through it can always be reopened.
- ``setup(cfg, run_dir)`` builds everything the config describes *before* the loop —
  device, dataset, trainer, state (fresh, or restored), loaders, evaluator — and returns
  an ``Experiment``; ``Experiment.train(...)`` is the default loop, and a notebook can
  instead call ``trainer.train`` itself with ``exp.run.checkpointer()`` and any logger.
  ``train(cfg, run_dir)`` is ``setup(cfg, run_dir).train()``: what ``scripts/train.py``
  calls under Hydra and what the tests call directly.
- ``load_run(run_dir)`` rebuilds the state template from the stored config and restores
  a checkpoint; ``load_run(run_dir, template=...)`` restores into a template the caller
  built, for a model that never had a config (a bare ``OrbaxCheckpointer`` directory is
  accepted then).

Two ways to continue training, both explicit: ``cfg.resume = <other run dir>``
warm-starts a **new** run from that run's latest checkpoint (new directory, same data
stream from ``state.step``); ``setup(cfg, run_dir, resume=True)`` continues **this** run
in place — ``config.yaml`` is rewritten to ``cfg`` and the differences from the stored
config are recorded in ``metadata.json`` (``resumed.config_changes``), so a changed
learning rate or a raised ``num_steps`` is on record.

Order of operations matters for JAX: ``x64`` and the default device are set before the
first array exists (``configure_jax``); the loader is ``to_device(..., device)``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import jax
from omegaconf import DictConfig, OmegaConf

from ..data import resolve_device, TimeSeriesDataSource
from ..model import AbstractPhaseAmplitudeModel
from ..training import (
    Checkpointer,
    Evaluator,
    Logger,
    OrbaxCheckpointer,
    Trainer,
    TrainerState,
)
from ..utils.provenance import git_state
from .data import build_evaluator, build_loaders, build_source, Loaders
from .logging import build_logger
from .model import build_model
from .training import build_trainer, resolved


def configure_jax(cfg: DictConfig) -> jax.Device:
    """``jax_enable_x64`` and the default device, before anything is built."""
    jax.config.update("jax_enable_x64", bool(cfg.x64))
    device = resolve_device(cfg.device)
    jax.config.update("jax_default_device", device)
    return device


# ------------------------------------------------------------------ the directory --
class _Default:
    """Sentinel: ``Run.checkpointer(metric=...)`` not given → the config's metric."""


_DEFAULT = _Default()


@dataclass(frozen=True)
class Run:
    """A run directory: ``dir/config.yaml`` (the resolved config), ``dir/metadata.json``
    (dataset hash, git state, batches per epoch, resume record), ``dir/checkpoints/``
    (Orbax, with the same metadata as ``custom_metadata``)."""

    dir: Path
    cfg: DictConfig
    metadata: dict[str, Any]

    CONFIG_FILE: ClassVar[str] = "config.yaml"
    METADATA_FILE: ClassVar[str] = "metadata.json"
    CHECKPOINT_DIR: ClassVar[str] = "checkpoints"

    @property
    def checkpoint_dir(self) -> Path:
        return self.dir / self.CHECKPOINT_DIR

    @staticmethod
    def exists(run_dir: str | Path) -> bool:
        """Whether ``run_dir`` already holds a run (a config or checkpoints). A
        directory that merely exists — Hydra creates it, with ``.hydra/``, before
        ``train`` is called — is not a run."""
        d = Path(run_dir)
        return (d / Run.CONFIG_FILE).exists() or (d / Run.CHECKPOINT_DIR).exists()

    @classmethod
    def create(
        cls, cfg: DictConfig, run_dir: str | Path, metadata: dict[str, Any]
    ) -> Run:
        """Start a run in ``run_dir``: refuses if one is already there (``resume`` to
        continue it, or choose another directory). Writes the config and metadata."""
        d = Path(run_dir)
        if cls.exists(d):
            raise FileExistsError(
                f"{d} already holds a run. Continue it with Run.resume / "
                "setup(cfg, run_dir, resume=True), or choose another directory."
            )
        d.mkdir(parents=True, exist_ok=True)
        run = cls(d, cfg, dict(metadata))
        run._write()
        return run

    @classmethod
    def resume(
        cls, cfg: DictConfig, run_dir: str | Path, metadata: dict[str, Any]
    ) -> Run:
        """Continue the run in ``run_dir`` with ``cfg`` (which may differ from the
        stored config: the differences go into ``metadata["resumed"]["config_changes"]``
        and ``config.yaml`` is rewritten). Requires a run with a checkpoint."""
        previous = cls.open(run_dir)
        step = previous.latest_step()
        if step is None:
            raise FileNotFoundError(f"no checkpoint to resume under {previous.dir}")
        changes = _config_changes(resolved(previous.cfg), resolved(cfg))
        meta = {
            **dict(metadata),
            "resumed": {
                "from_step": step,
                "config_changes": changes,
                "history": [*previous.metadata.get("resumed", {}).get("history", [])]
                + [{"from_step": step, "config_changes": changes}],
            },
        }
        run = cls(previous.dir, cfg, meta)
        run._write()
        return run

    @classmethod
    def open(cls, run_dir: str | Path) -> Run:
        """Read a run back: the config from ``config.yaml`` (the latest truth — a resume
        may have changed it), falling back to the checkpoints' ``custom_metadata``."""
        d = Path(run_dir)
        if not cls.exists(d):
            raise FileNotFoundError(
                f"no run under {d} (no {cls.CONFIG_FILE} or {cls.CHECKPOINT_DIR}/). "
                "For a bare OrbaxCheckpointer directory: load_run(..., template=...)."
            )
        metadata: dict[str, Any] = {}
        if (d / cls.METADATA_FILE).exists():
            metadata = json.loads((d / cls.METADATA_FILE).read_text())
        if (d / cls.CONFIG_FILE).exists():
            cfg = OmegaConf.load(d / cls.CONFIG_FILE)
        else:
            ckpt = OrbaxCheckpointer(d / cls.CHECKPOINT_DIR, metric=None)
            try:
                stored = ckpt.custom_metadata
            finally:
                ckpt.close()
            if stored.get("config") is None:
                raise FileNotFoundError(
                    f"{d} has checkpoints but no config (neither {cls.CONFIG_FILE} nor "
                    "checkpoint metadata); use load_run(..., template=...)."
                )
            cfg = OmegaConf.create(stored["config"])
            metadata = metadata or stored
        assert isinstance(cfg, DictConfig)
        return cls(d, cfg, metadata)

    def _write(self) -> None:
        (self.dir / self.CONFIG_FILE).write_text(
            OmegaConf.to_yaml(self.cfg, resolve=True)
        )
        (self.dir / self.METADATA_FILE).write_text(
            json.dumps(self.metadata, indent=2, default=str)
        )

    def checkpointer(
        self,
        *,
        save_every: int | None = None,
        metric: str | None | _Default = _DEFAULT,
        keep_best: int | None = None,
    ) -> OrbaxCheckpointer:
        """An ``OrbaxCheckpointer`` in ``checkpoints/`` carrying this run's metadata;
        the policy defaults to ``cfg.checkpoint`` (pass ``metric=None`` explicitly to
        select no best checkpoint)."""
        c = self.cfg.checkpoint
        return OrbaxCheckpointer(
            self.checkpoint_dir,
            save_every=c.every if save_every is None else save_every,
            metric=c.metric if isinstance(metric, _Default) else metric,
            keep_best=c.keep_best if keep_best is None else keep_best,
            custom_metadata=self.metadata,
        )

    def latest_step(self) -> int | None:
        """The most recent checkpointed step, ``None`` if there is none yet."""
        if not self.checkpoint_dir.exists():
            return None
        ckpt = OrbaxCheckpointer(self.checkpoint_dir, metric=None)
        try:
            return ckpt.latest_step
        finally:
            ckpt.close()

    def restore(self, template: TrainerState, step: int | None = None) -> TrainerState:
        """The checkpointed state at ``step`` (latest if ``None``) in ``template``'s
        structure."""
        ckpt = OrbaxCheckpointer(self.checkpoint_dir, metric=None)
        try:
            return ckpt.restore(template, step)
        finally:
            ckpt.close()


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def _config_changes(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """``{dotted.key: [old, new]}`` for every leaf that differs (missing → ``None``)."""
    a, b = _flatten(old), _flatten(new)
    keys = sorted(set(a) | set(b))
    return {k: [a.get(k), b.get(k)] for k in keys if a.get(k) != b.get(k)}


def _template(cfg: DictConfig, trainer: Trainer) -> TrainerState:
    return trainer.init(
        build_model(cfg.model, jax.random.key(cfg.seed)),
        key=jax.random.key(cfg.seed + 1),
    )


# ------------------------------------------------------------------- the objects ---
@dataclass(frozen=True)
class Experiment:
    """Everything ``setup`` built from a config: the ``run`` directory, the ``device``,
    the dataset ``source``, the ``trainer``, the initial (or restored) ``state`` with
    ``start_step``, the ``loaders`` (sliced at ``start_step``) and the ``evaluator``.
    ``train()`` is the default loop; a notebook may call ``trainer.train`` itself with
    ``run.checkpointer()`` and any logger."""

    run: Run
    device: jax.Device
    source: TimeSeriesDataSource
    trainer: Trainer
    state: TrainerState
    start_step: int
    loaders: Loaders
    evaluator: Evaluator

    @property
    def cfg(self) -> DictConfig:
        return self.run.cfg

    def train(
        self,
        logger: Logger | None = None,
        *,
        checkpointer: Checkpointer | None = None,
        group: str | None = None,
        wandb_run: Any = None,
    ) -> TrainerState:
        """Run ``cfg.num_steps − start_step`` steps. ``logger`` defaults to the config's
        (``build_logger``: Print + wandb as the ``wandb`` group says, or an existing
        ``wandb_run``); ``checkpointer`` to ``run.checkpointer()``. Returns the final
        state."""
        cfg = self.cfg
        owned_wandb = None
        if logger is None:
            logger, owned_wandb = build_logger(
                cfg,
                self.run.dir,
                self.run.metadata["config"],
                self.loaders.batches_per_epoch,
                group=group,
                wandb_run=wandb_run,
            )
        if checkpointer is None:
            checkpointer = self.run.checkpointer()
        try:
            return self.trainer.train(
                self.state,
                self.loaders.train,
                # the loader covers whole epochs (>= num_steps batches); run num_steps
                num_steps=cfg.num_steps - self.start_step,
                logger=logger,
                checkpointer=checkpointer,
                evaluate=self.evaluator,
                eval_every=cfg.eval_every,
            )
        finally:
            if owned_wandb is not None and owned_wandb is not wandb_run:
                owned_wandb.finish()


def setup(cfg: DictConfig, run_dir: str | Path, *, resume: bool = False) -> Experiment:
    """Build a run's objects from its config, creating the run directory (refusing an
    existing run) or, with ``resume=True``, continuing the run already there from its
    latest checkpoint. ``cfg.resume`` (another run's directory) warm-starts a new run
    from that run's latest checkpoint instead."""
    run_dir = Path(run_dir)
    device = configure_jax(cfg)
    config = resolved(cfg)
    source = build_source(cfg.data)
    trainer = build_trainer(cfg)
    assert source.metadata is not None
    metadata: dict[str, Any] = {
        "config": config,
        "dataset": {
            "config_hash": source.metadata.config_hash,
            "trajectories": source.num_trajectories,
        },
        "x64": bool(cfg.x64),
        "git": dict(zip(("sha", "dirty"), git_state())),
    }

    if resume:
        if cfg.resume:
            raise ValueError(
                "resume=True continues run_dir; cfg.resume names another run; not both."
            )
        run = Run.resume(cfg, run_dir, metadata)
        state = run.restore(_template(cfg, trainer))
    elif cfg.resume:
        previous = Run.open(cfg.resume)
        step = previous.latest_step()
        if step is None:
            raise FileNotFoundError(f"no checkpoint to resume under {previous.dir}")
        metadata["resumed_from"] = {"run": str(previous.dir), "step": step}
        run = Run.create(cfg, run_dir, metadata)
        state = previous.restore(_template(cfg, trainer), step)
    else:
        run = Run.create(cfg, run_dir, metadata)
        state = _template(cfg, trainer)
    start_step = int(state.step)

    loaders = build_loaders(cfg, source, device, start_step=start_step)
    run.metadata["batches_per_epoch"] = loaders.batches_per_epoch
    run._write()
    evaluator = build_evaluator(cfg, source, loaders)
    print(
        f"run {run.dir}: {cfg.num_steps} steps from step {start_step}; "
        f"{loaders.num_windows} windows per epoch in {loaders.batches_per_epoch} "
        f"batches of {cfg.windows.batch} × {cfg.windows.length} "
        f"(≈ {cfg.num_steps / loaders.batches_per_epoch:.2f} epochs); "
        f"{source.num_trajectories} trajectories, "
        f"{loaders.val_source.num_trajectories} held out; "
        f"device {device.platform}:{device.id}"
    )
    return Experiment(
        run=run,
        device=device,
        source=source,
        trainer=trainer,
        state=state,
        start_step=start_step,
        loaders=loaders,
        evaluator=evaluator,
    )


def train(
    cfg: DictConfig,
    run_dir: str | Path,
    *,
    group: str | None = None,
    resume: bool = False,
) -> TrainerState:
    """``setup(cfg, run_dir, resume=resume).train(group=group)``: the whole run from a
    config, as ``scripts/train.py`` does it."""
    return setup(cfg, run_dir, resume=resume).train(group=group)


# ------------------------------------------------------------------- reopening -----
def load_run(
    run_dir: str | Path,
    step: int | None = None,
    *,
    template: TrainerState | None = None,
) -> tuple[DictConfig | None, TrainerState]:
    """``(cfg, state)`` of a run directory at ``step`` (latest if ``None``).

    Without ``template`` the state template is rebuilt from the run's config with the
    same builders (``configure_jax`` is applied). With ``template`` — a ``TrainerState``
    the caller built, for a model that never had a config — the config is only reported
    when found (else ``None``), and ``run_dir`` may be a bare ``OrbaxCheckpointer``
    directory (no ``checkpoints/`` layout)."""
    run_dir = Path(run_dir)
    if template is None:
        run = Run.open(run_dir)
        configure_jax(run.cfg)
        return run.cfg, run.restore(_template(run.cfg, build_trainer(run.cfg)), step)
    cfg = None
    ckpt_dir = run_dir / Run.CHECKPOINT_DIR
    if Run.exists(run_dir):
        try:
            cfg = Run.open(run_dir).cfg
        except FileNotFoundError:
            cfg = None
    if not ckpt_dir.exists():
        ckpt_dir = run_dir  # a bare OrbaxCheckpointer directory
    ckpt = OrbaxCheckpointer(ckpt_dir, metric=None)
    try:
        return cfg, ckpt.restore(template, step)
    finally:
        ckpt.close()


def load_model(
    run_dir: str | Path,
    step: int | None = None,
    *,
    template: TrainerState | None = None,
) -> AbstractPhaseAmplitudeModel:
    """The trained model of a run directory at ``step`` (latest if ``None``)."""
    _, state = load_run(run_dir, step, template=template)
    return state.model
