---
type: guide
status: current
updated: 2026-10-29
sources:
  - "ADR-0011"
  - "src/deep_isochron/experiment/"
  - "scripts/README.md (the block this guide grew from)"
---

# Running experiments from Python

The scripts in `scripts/` are thin: each composes a config and calls one function of
`deep_isochron.experiment` ([ADR-0011](../decisions/0011-experiment-layer.md)). A
notebook or an analysis calls the same functions, so what the script builds, the notebook
builds too, and a run directory is the same whichever made it.

## Composing a config

`compose` composes `configs/train.yaml` with overrides, as `scripts/train.py` does;
`compose_group` composes one group alone, for a dataset or a cycle without a training
config around it:

```python
from deep_isochron.experiment import compose, compose_group

cfg = compose(
    "data=fhn", "data.n_trajectories=500", "model=conjugacy", "num_steps=2500"
)
data_cfg = compose_group("data", "fhn", "n_trajectories=500")
```

## Datasets

`load_or_generate_source` reads a dataset's file, or generates it when a notebook asks;
`Experiment.from_config` and `train` never generate, and raise naming
`scripts/generate_data.py` when the file is missing:

```python
from deep_isochron.experiment import load_or_generate_source

source = load_or_generate_source(data_cfg)
```

## Training

`Experiment.from_config` builds the loaders, model, loss, schedule and optimizer and owns
the run directory; it refuses a directory that already holds a run. The loop is yours,
with any logger, or the default:

```python
import wandb
from deep_isochron.experiment import Experiment, train
from deep_isochron.training import DelayedLogger, WandbLogger

exp = Experiment.from_config(cfg, "runs/notebook/try-1")
state = exp.trainer.train(
    exp.state,
    exp.loaders.train,
    logger=DelayedLogger(
        WandbLogger(wandb.init(entity="jhko725", project="isochron"), every=25)
    ),
    checkpointer=exp.run.checkpointer(save_every=200),
    evaluate=exp.evaluator,
    eval_every=200,
)
state = exp.train()  # the default loop: PrintLogger, and wandb when the config says so
state = train(
    cfg, "runs/notebook/try-2"
)  # Experiment.from_config(...).train() in one call
```

Continuing a run in place is explicit, and the config change it brings is recorded in
the run directory:

```python
cfg_more = compose(..., "num_steps=5000")
state = train(cfg_more, "runs/notebook/try-1", resume=True)
```

## Reading a run back

```python
from deep_isochron.experiment import load_model, load_run

cfg, state = load_run("runs/notebook/try-1")  # any run directory, latest step
model = load_model("runs/2026-10-05/12-00-00", step=5000)
cfg, state = load_run(
    "results/bare-ckpt", template=my_state
)  # a model built without a config
```

## Isochrons

The reference isochrons of a system have a config group of their own,
`configs/isochrons/`, with `compute_isochrons` and `load_isochrons` as the pair of
functions; [`isochrons.md`](isochrons.md) covers them.
