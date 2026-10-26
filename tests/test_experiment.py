"""The experiment layer, ``deep_isochron.experiment`` with ``configs/``: composing the
shipped configs builds every object, and a run trains end to end on CPU, reopens and
resumes.

In five groups. Composition: ``configs/train.yaml`` composes with every group's
shipped options. Data: `dataset_file` is where `generate_dataset` writes,
`build_source` refuses to generate, the loaders have the configured shapes, and a
resume slices the stream. Model and trainer: both model kinds and both losses build
from their configs. The run: `train` for a few steps writes config, metadata and
checkpoints, `load_model` returns the final model, and a resumed run continues the
step count and the data stream. The notebook path: `Experiment.from_config` then a
hand-rolled `trainer.train` with `run.checkpointer()` reopens with `load_run`, a
hand-built model reopens through a template, `Run.from_config` refuses an existing
run, ``resume=True`` continues in place and records config changes,
`load_or_generate_source` generates once, and `build_logger` takes an existing wandb
run. The fixture is a 12-trajectory Bautin dataset generated once per module, and
every run is 3 steps.
"""

import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from deep_isochron.experiment import (
    build_evaluator,
    build_logger,
    build_model,
    build_source,
    build_trainer,
    compose,
    compose_group,
    dataset_file,
    Experiment,
    generate_dataset,
    load_model,
    load_or_generate_source,
    load_run,
    Loaders,
    Run,
    train,
)
from deep_isochron.model import ConjugateLatentDynamics, PhaseAmplitudeAutoencoder
from deep_isochron.training import (
    ConjugacyTrajectoryLoss,
    ListLogger,
    OrbaxCheckpointer,
    PhaseAutoencoderLoss,
)
from omegaconf import DictConfig


TINY_DATA = ["data.n_trajectories=12", "data.time.n=61", "data.time.t1=12.0"]
TINY_RUN = [
    "windows.length=10",
    "windows.batch=16",
    "validation.fraction=0.25",
    "validation.batch=8",
    "num_steps=3",
    "eval_every=2",
    "log_every=1",
    "checkpoint.every=2",
]


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory) -> Path:
    """A tiny Bautin dataset generated once for the module, where the ``data`` config
    with ``TINY_DATA`` overrides says it should be."""
    root = tmp_path_factory.mktemp("data")
    cfg = compose(*TINY_DATA, f"data.out_dir={root}")
    path = generate_dataset(cfg.data)
    assert path == dataset_file(cfg.data)
    return root


def tiny(data_dir: Path, *overrides: str) -> DictConfig:
    """The tiny dataset + a short run; a two-block INN unless another model is named."""
    conjugacy = not any(o.startswith("model=") for o in overrides)
    model = ["model.inn.blocks=2"] if conjugacy else []
    return compose(
        *TINY_DATA, f"data.out_dir={data_dir}", *TINY_RUN, *model, *overrides
    )


# ---------------------------------------------------------------- 1. compose ------
@pytest.mark.parametrize(
    "overrides",
    [
        [],
        ["model=autoencoder", "loss=autoencoder", "schedule=yawata"],
        ["data=fhn"],
        ["wandb=online"],
        ["wandb=offline"],
        ["windows.sampling.kind=weighted"],
        ["windows.sampling.kind=mixed", "windows.sampling.split_idx=100"],
        ["optimizer=adabelief"],
        ["optimizer=adamw", "clip_norm=1.0"],
    ],
)
def test_shipped_configs_compose(overrides):
    """`compose` with each shipped group option returns a config with positive
    `num_steps` and batch size."""
    cfg = compose(*overrides)
    assert cfg.num_steps > 0 and cfg.windows.batch > 0


def test_compose_group_is_the_groups_node():
    """`compose_group("data", "fhn", ...)` is the `data` node of the full composition,
    with the same `dataset_file`; the same holds for the `model` group."""
    data = compose_group("data", "fhn", "n_trajectories=500")
    assert data.name == "fitzhugh_nagumo" and data.n_trajectories == 500
    full = compose("data=fhn", "data.n_trajectories=500").data
    assert dataset_file(data) == dataset_file(full)
    model = compose_group("model", "autoencoder")
    assert model.kind == "autoencoder"


# ------------------------------------------------------------------- 2. data ------
def test_build_source_requires_the_generated_file(tmp_path):
    """`build_source` raises `FileNotFoundError` on a missing dataset rather than
    generating it."""
    cfg = compose(*TINY_DATA, f"data.out_dir={tmp_path}")
    with pytest.raises(FileNotFoundError, match="never generates"):
        build_source(cfg.data)


def test_loaders_shapes_and_resume_slice(data_dir):
    """`Loaders.from_config` splits 12 trajectories 9/3 at ``fraction=0.25``, yields
    batches of shape ``(16, 10, 2)``, resumes at `start_step` on the same stream,
    rejects a start beyond the stream, and builds the weighted and mixed samplers."""
    cfg = tiny(data_dir)
    source = build_source(cfg.data)
    dev = jax.devices()[0]
    full = Loaders.from_config(cfg, source, dev)
    b = next(iter(full.train))
    assert b["u"].shape == (16, 10, 2) and b["t"].shape == (16, 10)
    assert full.train_source.num_trajectories == 9  # 12 − round(0.25 · 12)
    assert full.val_source.num_trajectories == 3
    assert len(full.train_batches) >= cfg.num_steps
    later = Loaders.from_config(cfg, source, dev, start_step=2)
    assert np.array_equal(later.train_batches[0]["u"], full.train_batches[2]["u"])
    with pytest.raises(ValueError, match="beyond"):
        Loaders.from_config(cfg, source, dev, start_step=10**6)
    for kind, extra in [("weighted", []), ("mixed", ["windows.sampling.split_idx=30"])]:
        cfg_k = tiny(data_dir, f"windows.sampling.kind={kind}", *extra)
        assert (
            next(iter(Loaders.from_config(cfg_k, source, dev).train))["u"].shape[0]
            == 16
        )


# ----------------------------------------------------------- 3. model/trainer -----
def test_models_and_trainers_build(data_dir):
    """`build_model` and `build_trainer` give the conjugacy model with ``2 * blocks``
    transforms and its loss, the autoencoder and its loss, and reject an unknown
    kind; the default optimizer leaves the model fixed on a zero gradient, `adamw`
    decays the bijection and not the latent dynamics, and `clip_norm` chains a second
    optimizer state."""
    cfg = tiny(data_dir)
    model = build_model(cfg.model, jax.random.key(0))
    assert isinstance(model, ConjugateLatentDynamics)
    assert len(model.bijection.transforms) == 2 * cfg.model.inn.blocks
    trainer = build_trainer(cfg)
    assert isinstance(trainer.loss, ConjugacyTrajectoryLoss)
    cfg_ae = tiny(data_dir, "model=autoencoder", "loss=autoencoder", "schedule=yawata")
    ae = build_model(cfg_ae.model, jax.random.key(0))
    assert isinstance(ae, PhaseAmplitudeAutoencoder)
    assert isinstance(build_trainer(cfg_ae).loss, PhaseAutoencoderLoss)
    with pytest.raises(ValueError, match="model.kind"):
        build_model(compose("model.kind=nope").model, jax.random.key(0))
    # the default optimizer is adam without decay: a zero-gradient step moves nothing;
    # the adamw group masks its decay off the latent dynamics: the normal form's
    # parameters stay exactly where they were while the bijection shrinks
    state = trainer.init(model, key=jax.random.key(1))
    zero = jax.tree.map(jnp.zeros_like, state.filter_trainable(model))
    assert eqx.tree_equal(state.take_step(zero).model, model)
    decayed = build_trainer(tiny(data_dir, "optimizer=adamw")).init(
        model, key=jax.random.key(1)
    )
    stepped = decayed.take_step(zero)
    assert eqx.tree_equal(stepped.model.latent_dynamics, model.latent_dynamics)
    assert not eqx.tree_equal(stepped.model.bijection, model.bijection)
    # clip_norm wraps the optimizer: a huge gradient moves the parameters by at most
    # about lr per leaf element (adam) whether or not it is clipped — so check the
    # chain is there by its state instead
    clipped = build_trainer(tiny(data_dir, "clip_norm=1.0"))
    assert (
        isinstance(clipped.optimizer.init(zero), tuple)
        and len(clipped.optimizer.init(zero)) == 2
    )


# -------------------------------------------------------------------- 4. run ------
def test_train_writes_a_run_and_load_model_reads_it_back(data_dir, tmp_path):
    """A 3-step `train` writes ``config.yaml`` and a ``metadata.json`` that holds the
    dataset size, the whole composed config and the git sha; `load_model` and
    `load_run` return the final model and step."""
    cfg = tiny(data_dir)
    run_dir = tmp_path / "run"
    state = train(cfg, run_dir)
    assert int(state.step) == 3
    assert (run_dir / "config.yaml").exists()
    meta = json.loads((run_dir / "metadata.json").read_text())
    assert meta["dataset"]["trajectories"] == 12 and meta["batches_per_epoch"] > 0
    # the whole composed config is in the metadata (and in wandb's config): model,
    # loss, schedule, optimizer, data and the top-level fields
    assert meta["config"]["model"]["inn"]["blocks"] == 2
    assert meta["config"]["loss"]["_target_"].endswith("ConjugacyTrajectoryLoss")
    assert set(meta["config"]) >= {"data", "model", "loss", "schedule", "optimizer"}
    assert "sha" in meta["git"]
    model = load_model(run_dir)
    assert eqx.tree_equal(model, state.model)
    loaded_cfg, loaded_state = load_run(run_dir)
    assert loaded_cfg.num_steps == 3 and int(loaded_state.step) == 3


def test_resume_continues_steps_and_stream(data_dir, tmp_path):
    """A run resumed from a 3-step run with ``num_steps=5`` ends at step 5 and
    consumes batches 3 and 4 of the same deterministic stream."""
    cfg = tiny(data_dir)
    first = tmp_path / "first"
    train(cfg, first)
    cfg_resume = tiny(data_dir, f"resume={first}", "num_steps=5")
    state = train(cfg_resume, tmp_path / "second")
    assert int(state.step) == 5
    # the resumed run consumed batches 3 and 4 of the same deterministic stream
    source = build_source(cfg.data)
    full = Loaders.from_config(cfg_resume, source, jax.devices()[0])
    resumed = Loaders.from_config(cfg_resume, source, jax.devices()[0], start_step=3)
    assert np.array_equal(resumed.train_batches[0]["u"], full.train_batches[3]["u"])


def test_validation_split_follows_the_mixed_sampler(data_dir, tmp_path):
    """``validation.t_split`` null: no split for uniform sampling, the sampler's own
    boundary for ``mixed``; an explicit value wins; the run logs the split metrics."""
    from deep_isochron.experiment import validation_t_split

    source = build_source(tiny(data_dir).data)
    assert validation_t_split(tiny(data_dir), source) is None
    mixed = tiny(
        data_dir, "windows.sampling.kind=mixed", "windows.sampling.split_idx=30"
    )
    assert validation_t_split(mixed, source) == float(source.ts[30])
    explicit = tiny(data_dir, "validation.t_split=2.5")
    assert validation_t_split(explicit, source) == 2.5
    log_run = tiny(
        data_dir,
        "validation.t_split=2.0",
        "checkpoint.metric=val/mse_early",
        "num_steps=2",
        "eval_every=2",
    )
    state = train(log_run, tmp_path / "split")
    assert int(state.step) == 2  # the checkpoint metric val/mse_early existed


def test_autoencoder_run(data_dir, tmp_path):
    """The autoencoder model, loss and Yawata schedule train for 3 steps."""
    cfg = tiny(data_dir, "model=autoencoder", "loss=autoencoder", "schedule=yawata")
    state = train(cfg, tmp_path / "ae")
    assert int(state.step) == 3


# ------------------------------------------------------------- 5. Run / setup -----
def test_from_config_then_manual_training_loop_reopens_with_load_run(
    data_dir, tmp_path
):
    """The notebook path: ``from_config`` builds the objects and the run directory; the
    loop is the caller's, with ``exp.run.checkpointer()`` and any logger; ``load_run``
    then finds the config in the directory."""
    exp = Experiment.from_config(tiny(data_dir), tmp_path / "nb")
    assert (exp.run.dir / "config.yaml").exists() and exp.start_step == 0
    assert exp.run.metadata["batches_per_epoch"] > 0
    logger = ListLogger()
    state = exp.trainer.train(
        exp.state,
        exp.loaders.train,
        num_steps=3,
        logger=logger,
        checkpointer=exp.run.checkpointer(save_every=1, metric="val/mse"),
        evaluate=exp.evaluator,
        eval_every=2,
    )
    assert int(state.step) == 3 and logger.records
    cfg, loaded = load_run(exp.run.dir)
    assert cfg.num_steps == 3 and eqx.tree_equal(loaded.model, state.model)
    assert Run.open(exp.run.dir).latest_step() == 3


def test_hand_built_model_reopens_through_a_template(data_dir, tmp_path):
    """A model built without any config, checkpointed by a bare ``OrbaxCheckpointer``:
    ``load_run(template=...)`` restores it, and reports no config."""
    cfg = tiny(data_dir)
    exp = Experiment.from_config(
        cfg, tmp_path / "run"
    )  # only for the loaders and evaluator
    trainer = build_trainer(cfg)
    state0 = trainer.init(
        build_model(cfg.model, jax.random.key(7)), key=jax.random.key(8)
    )
    ckpt_dir = tmp_path / "bare-ckpt"
    with OrbaxCheckpointer(ckpt_dir, save_every=1, metric="val/mse") as ckpt:
        state = trainer.train(
            state0,
            exp.loaders.train,
            num_steps=2,
            checkpointer=ckpt,
            evaluate=exp.evaluator,
            eval_every=2,
        )
    cfg_found, loaded = load_run(ckpt_dir, template=state0)
    assert cfg_found is None and eqx.tree_equal(loaded.model, state.model)
    assert eqx.tree_equal(load_model(ckpt_dir, template=state0), state.model)
    with pytest.raises(FileNotFoundError, match="template"):
        load_run(ckpt_dir)


def test_run_create_refuses_an_existing_run_and_resume_is_explicit(data_dir, tmp_path):
    """`Experiment.from_config` and `train` raise `FileExistsError` on an existing run
    directory; ``resume=True`` continues it in place to step 5 and records the changed
    `num_steps` and learning rate; ``resume=True`` with ``cfg.resume`` is rejected, and
    a resume without a checkpoint raises `FileNotFoundError`."""
    cfg = tiny(data_dir)
    run_dir = tmp_path / "run"
    (run_dir / ".hydra").mkdir(parents=True)  # Hydra makes the directory first: fine
    train(cfg, run_dir)
    with pytest.raises(FileExistsError, match="resume"):
        Experiment.from_config(cfg, run_dir)
    with pytest.raises(FileExistsError, match="resume"):
        train(cfg, run_dir)
    # continuing in place: more steps, a changed learning rate, both on record
    cfg_more = tiny(data_dir, "num_steps=5", "optimizer.learning_rate=2e-3")
    state = train(cfg_more, run_dir, resume=True)
    assert int(state.step) == 5
    run = Run.open(run_dir)
    assert run.cfg.num_steps == 5
    changes = run.metadata["resumed"]["config_changes"]
    assert changes["num_steps"] == [3, 5]
    assert changes["optimizer.learning_rate"][1] == 2e-3
    assert run.metadata["resumed"]["from_step"] == 3
    assert len(run.metadata["resumed"]["history"]) == 1
    # resume=True and cfg.resume together is a contradiction
    with pytest.raises(ValueError, match="not both"):
        Experiment.from_config(
            tiny(data_dir, f"resume={run_dir}"), run_dir, resume=True
        )
    # and a resume needs a checkpoint to continue from
    with pytest.raises(FileNotFoundError):
        Experiment.from_config(cfg, tmp_path / "nowhere", resume=True)


def test_warm_start_from_another_run_is_recorded(data_dir, tmp_path):
    """A run with ``resume=<other run>`` warm-starts from that run's step 3 and records
    the source run and step under ``metadata["resumed_from"]``."""
    cfg = tiny(data_dir)
    train(cfg, tmp_path / "first")
    state = train(
        tiny(data_dir, f"resume={tmp_path / 'first'}", "num_steps=5"),
        tmp_path / "second",
    )
    assert int(state.step) == 5
    meta = Run.open(tmp_path / "second").metadata
    assert meta["resumed_from"]["step"] == 3 and meta["resumed_from"]["run"].endswith(
        "first"
    )


def test_load_or_generate_source_generates_once(tmp_path):
    """`load_or_generate_source` writes the dataset on the first call and loads it,
    without rewriting, on the second."""
    cfg = compose(*TINY_DATA, f"data.out_dir={tmp_path}")
    assert not dataset_file(cfg.data).exists()
    source = load_or_generate_source(cfg.data)
    assert dataset_file(cfg.data).exists() and source.num_trajectories == 12
    mtime = dataset_file(cfg.data).stat().st_mtime_ns
    load_or_generate_source(cfg.data)
    assert dataset_file(cfg.data).stat().st_mtime_ns == mtime  # loaded, not regenerated


def test_build_logger_takes_an_existing_wandb_run(data_dir, tmp_path):
    """`build_logger` with a `wandb_run` logs to it, thinned to every 50th step with
    the epoch added, even when the config's wandb mode is off; `build_evaluator`
    carries the config's `t_split`."""

    class FakeRun:
        def __init__(self):
            self.logged = []

        def log(self, metrics, step):
            self.logged.append((step, metrics))

    cfg = tiny(data_dir)  # wandb=off: without a run no wandb logger would be made
    fake = FakeRun()
    logger, run = build_logger(cfg, tmp_path, {}, 4, wandb_run=fake)
    assert run is fake
    with logger:
        logger.log({"loss": 1.0}, 0)
        logger.log({"loss": 0.7}, 1)  # thinned by wandb.every = 50
        logger.log({"loss": 0.5}, 50)
    assert [s for s, _ in fake.logged] == [0, 50] and "epoch" in fake.logged[0][1]
    # and the evaluator builder matches what setup uses
    source = build_source(cfg.data)
    loaders = Loaders.from_config(cfg, source, jax.devices()[0])
    assert build_evaluator(cfg, source, loaders).t_split is None
