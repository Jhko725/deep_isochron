"""The experiment layer: from Hydra configs (``configs/``) to a training run (ADR-0011).

``build_source`` / ``build_loaders`` (``data``), ``build_model`` (``model``),
``build_trainer`` (``training``), ``build_logger`` (``logging``) are pure functions of
the composed config; ``run.train(cfg, run_dir)`` strings them together and
``run.load_run`` / ``run.load_model`` reopen a run directory. ``scripts/train.py`` is
the Hydra entry point; notebooks and tests use ``compose(*overrides)`` and call
``train`` or the builders directly. The package only parses configs and instantiates
objects — anything with its own logic lives in the module it belongs to
(``training.loggers``, ``provenance``, ``data``).
"""

from .compose import compose as compose, CONFIG_DIR as CONFIG_DIR
from .data import (
    build_loaders as build_loaders,
    build_source as build_source,
    build_window_source as build_window_source,
    dataset_file as dataset_file,
    generate_dataset as generate_dataset,
    Loaders as Loaders,
    reference_normal_form as reference_normal_form,
)
from .logging import build_logger as build_logger
from .model import build_inn as build_inn, build_model as build_model
from .run import (
    configure_jax as configure_jax,
    load_model as load_model,
    load_run as load_run,
    train as train,
)
from .training import (
    build_loss as build_loss,
    build_optimizer as build_optimizer,
    build_schedule as build_schedule,
    build_trainer as build_trainer,
    resolved as resolved,
)
