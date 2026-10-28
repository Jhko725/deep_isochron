"""The experiment layer: from Hydra configs in ``configs/`` to a training run
(ADR-0011).

``build_source``, ``build_model``, ``build_trainer`` and ``build_logger`` are pure
functions of the composed config; ``Experiment.from_config(cfg, run_dir)`` strings
them together into an ``Experiment`` whose ``Run`` owns the directory layout,
``train(cfg, run_dir)`` runs its loop, and ``load_run`` and ``load_model`` reopen a run
directory. ``scripts/train.py`` is the Hydra entry point; notebooks and tests use
``compose(*overrides)`` with ``Experiment.from_config``, ``train`` or the builders.

The construction style is one rule: objects the experiment layer owns (``Run``,
``Loaders``, ``Experiment``) construct themselves with a ``from_config`` classmethod,
while library objects (model, loss, schedule, optimizer, evaluator, logger, source) are
built by the free ``build_*`` functions, since ``model``, ``training`` and ``data``
must not know configs. ``compute_isochrons`` and ``load_isochrons`` do for the
reference isochrons of ``configs/isochrons/`` what ``generate_dataset`` and
``build_source`` do for datasets. The package only parses configs and instantiates
objects; anything with its own logic lives in the module it belongs to.
"""

from .compose import (
    compose as compose,
    compose_group as compose_group,
    CONFIG_DIR as CONFIG_DIR,
)
from .data import (
    build_evaluator as build_evaluator,
    build_source as build_source,
    build_window_source as build_window_source,
    dataset_file as dataset_file,
    generate_dataset as generate_dataset,
    load_or_generate_source as load_or_generate_source,
    Loaders as Loaders,
    reference_normal_form as reference_normal_form,
    validation_t_split as validation_t_split,
)
from .isochrons import (
    compute_isochrons as compute_isochrons,
    CycleConfig as CycleConfig,
    isochron_config as isochron_config,
    isochron_cycle as isochron_cycle,
    IsochronConfig as IsochronConfig,
    load_isochrons as load_isochrons,
    PhaseFieldConfig as PhaseFieldConfig,
    PhasesConfig as PhasesConfig,
)
from .logging import build_logger as build_logger
from .model import build_inn as build_inn, build_model as build_model
from .run import (
    configure_jax as configure_jax,
    Experiment as Experiment,
    load_model as load_model,
    load_run as load_run,
    Run as Run,
    train as train,
)
from .training import (
    build_loss as build_loss,
    build_optimizer as build_optimizer,
    build_schedule as build_schedule,
    build_trainer as build_trainer,
    resolved as resolved,
)


def _register_schemas() -> None:
    """Register the structured-config schemas in Hydra's `ConfigStore` (ADR-0011 §1),
    at import so that `compose` and the scripts validate against them: a config of a
    group with a schema names it first in its ``defaults``."""
    from hydra.core.config_store import ConfigStore
    from omegaconf import OmegaConf

    store = ConfigStore.instance()
    store.store(group="cycle", name="schema", node=CycleConfig)
    store.store(group="phase_field", name="schema", node=PhaseFieldConfig)
    isochrons = OmegaConf.structured(IsochronConfig)
    # A frozen dataclass makes a read-only node, which a YAML could not fill in; the
    # settings become an IsochronSettings, and frozen, when the config is converted.
    OmegaConf.set_readonly(isochrons.settings, False)
    store.store(group="isochrons", name="schema", node=isochrons, package="_global_")


_register_schemas()
