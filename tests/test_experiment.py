"""The experiment layer (``deep_isochron.experiment``, ``configs/``): composing the
shipped configs builds every object, a run trains end to end on CPU and can be reopened
and resumed.

1. compose         ``configs/train.yaml`` composes with every group's shipped options
2. data            ``dataset_file`` is where ``generate_dataset`` writes;
                   ``build_source`` refuses to generate; loaders have the configured
                   shapes; resume slices
3. model/trainer   both model kinds and both losses build from their configs
4. run             ``train`` for a few steps writes config, metadata and checkpoints;
                   ``load_model`` returns the final model; a resumed run continues the
                   step count and the data stream
"""

import json
from pathlib import Path

import equinox as eqx
import jax
import numpy as np
import pytest
from deep_isochron.experiment import (
    build_loaders,
    build_model,
    build_source,
    build_trainer,
    compose,
    dataset_file,
    generate_dataset,
    load_model,
    load_run,
    train,
)
from deep_isochron.model import ConjugateLatentDynamics, PhaseAmplitudeAutoencoder
from deep_isochron.training import ConjugacyTrajectoryLoss, PhaseAutoencoderLoss
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
    ],
)
def test_shipped_configs_compose(overrides):
    cfg = compose(*overrides)
    assert cfg.num_steps > 0 and cfg.windows.batch > 0


# ------------------------------------------------------------------- 2. data ------
def test_build_source_requires_the_generated_file(tmp_path):
    cfg = compose(*TINY_DATA, f"data.out_dir={tmp_path}")
    with pytest.raises(FileNotFoundError, match="--config-name bautin"):
        build_source(cfg.data)


def test_loaders_shapes_and_resume_slice(data_dir):
    cfg = tiny(data_dir)
    source = build_source(cfg.data)
    dev = jax.devices()[0]
    full = build_loaders(cfg, source, dev)
    b = next(iter(full.train))
    assert b["u"].shape == (16, 10, 2) and b["t"].shape == (16, 10)
    assert full.train_source.num_trajectories == 9  # 12 − round(0.25 · 12)
    assert full.val_source.num_trajectories == 3
    assert len(full.train_batches) >= cfg.num_steps
    later = build_loaders(cfg, source, dev, start_step=2)
    assert np.array_equal(later.train_batches[0]["u"], full.train_batches[2]["u"])
    with pytest.raises(ValueError, match="beyond"):
        build_loaders(cfg, source, dev, start_step=10**6)
    for kind, extra in [("weighted", []), ("mixed", ["windows.sampling.split_idx=30"])]:
        cfg_k = tiny(data_dir, f"windows.sampling.kind={kind}", *extra)
        assert next(iter(build_loaders(cfg_k, source, dev).train))["u"].shape[0] == 16


# ----------------------------------------------------------- 3. model/trainer -----
def test_models_and_trainers_build(data_dir):
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


# -------------------------------------------------------------------- 4. run ------
def test_train_writes_a_run_and_load_model_reads_it_back(data_dir, tmp_path):
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
    cfg = tiny(data_dir)
    first = tmp_path / "first"
    train(cfg, first)
    cfg_resume = tiny(data_dir, f"resume={first}", "num_steps=5")
    state = train(cfg_resume, tmp_path / "second")
    assert int(state.step) == 5
    # the resumed run consumed batches 3 and 4 of the same deterministic stream
    source = build_source(cfg.data)
    full = build_loaders(cfg_resume, source, jax.devices()[0])
    resumed = build_loaders(cfg_resume, source, jax.devices()[0], start_step=3)
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
    cfg = tiny(data_dir, "model=autoencoder", "loss=autoencoder", "schedule=yawata")
    state = train(cfg, tmp_path / "ae")
    assert int(state.step) == 3
