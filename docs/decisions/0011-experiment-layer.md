---
type: decision
id: ADR-0011
status: accepted; amended 2026-10-07/13 (§6 - Run, Experiment.from_config, explicit resume, one construction style)
updated: 2026-10-28
verified_by: pending (Joon; decisions taken in discussion 2026-10-05)
---

# ADR-0011 — The experiment layer: YAML configs, builders, Hydra-owned runs

## Context

### Motivation

- The project had no layer that turns a description of an experiment into a run.
  Phases A–C produced a trainer with injected side effects
  ([ADR-0009](0009-trainer-losses-schedules-checkpoints.md)), a finite batched data
  loader ([ADR-0008](0008-dataset-format-and-sampling.md) Decision 3) and a device
  policy (`resolve_device`). Missing were configuration, construction of the objects,
  the entry point, resuming, reopening a run for analysis, and CI
  ([`2026-10-05-experiment-config.md`](../changes/2026-10-05-experiment-config.md),
  Summary).
- The notebook carried its own ad-hoc construction code: it built everything from
  scratch, with manually provided arguments and no config machinery (Joon, review of
  2026-10-27; the change document records that it persisted, Open issues, D5).
- The notebook could not produce a reopenable run. Its hand-rolled `trainer.train` used
  a bare `OrbaxCheckpointer`, which stored no config, in a directory without the run
  layout, and `load_run` expected that layout because `run.py` hard-coded it
  ([`2026-10-07-run-layout.md`](../changes/2026-10-07-run-layout.md), Summary;
  `prototype.ipynb` at `af183a6`). This point motivated Decision 6 as it now stands.

### Forces

[`scripts/generate_data.py`](../../scripts/generate_data.py) had already set a
configuration style, Hydra YAML with `_target_` entries. Generation has to be explicit
(Joon). CI and tests must not need wandb. The packages `model`, `training` and `data`
must not know configs, so the parsing of configs has to live in a layer of its own.

## Decision

### 1. YAML config groups with `_target_` leaves; no dataclass schema (for now)

[`configs/train.yaml`](../../configs/train.yaml) composes the groups `data/` (the
generation configs, unchanged), `model/`, `loss/`, `schedule/`, `optimizer/`, `wandb/`,
plus top-level fields (`seed`, `device`, `x64`, `num_steps`, `windows`, `validation`,
`eval_every`, `log_every`, `checkpoint`, `resume`). Leaf objects, such as a normal form,
a spline or an optimizer, are `_target_` mappings instantiated as written.

- *Rejected for now*: structured configs (dataclasses + `ConfigStore`), the roadmap's
  original D1. They give stronger typing and validation at the cost of a parallel schema
  for every class, and are to be revisited if compositions grow or validation becomes a
  problem (Joon's call: "re-evaluate later").
- **Lists become tuples.** Hydra's `instantiate` can only produce lists, while the
  code types fixed-size parameters as tuples and the test suite's beartype hook
  enforces that.
  [`experiment.instantiate`](../../src/deep_isochron/experiment/instantiate.py) is a
  30-line replacement: same `_target_` convention, nested targets, keyword overrides for
  what a config cannot express (per-layer PRNG keys, the alternating `flip`), and YAML
  lists as tuples. Spline constructors additionally accept any two-element sequence for
  `xy_range`.

### 2. Builders are pure functions of the config, in `deep_isochron.experiment`

**Scope rule** (review round 1): the package only parses configs and instantiates
objects. Anything with logic of its own lives in the module it belongs to:
`MultiLogger`/`EpochLogger` in `training.loggers`, `git_state` in `provenance` (shared
with `data.generate`), the window source in `data.windows`. `compose(*overrides)` is the
one convenience the package adds for notebooks and tests, the config exactly as
[`scripts/train.py`](../../scripts/train.py) composes it.

`build_source(cfg.data)`, `Loaders.from_config(cfg, source, device, start_step)`,
`build_model(cfg.model, key)`, `build_trainer(cfg)` (loss, schedule, optimizer),
`build_logger(cfg, …)`. `Experiment.from_config` calls them in turn (Decision 6), and
`scripts/train.py` is a ten-line Hydra `main` around `train(cfg, run_dir)`. Tests call
`train` directly on tiny configs; notebooks compose the same configs with
`hydra.compose` and call the same builders (D5), so there is one construction path.

- The "coupling registry" of the original D2 is Hydra's `_target_`: a model config names
  its layer classes; `build_inn` only adds keys and `flip`.

### 3. The `data` group *is* the generation config; training never generates

`build_source` recomputes the generation metadata from the config
(`generation_metadata`, factored out of `generate`) and loads
`<out_dir>/<name>-<config_hash>.nc`. A missing file is a `FileNotFoundError` naming the
`scripts/generate_data.py` command. The hash is computed under x64 regardless of the
run's `x64` setting, because datasets are float64 and the hash covers the system's
parameters as the arrays hold them.

- *Rejected*: generating on a miss. "Any generation should be explicit and intended"
  (Joon); a training job on a GPU node must not silently spend ten minutes integrating.
- `generate_dataset(cfg.data)` is the one place generation happens; the script calls it.
  `load_or_generate_source(cfg.data)` is the notebook's explicit opt-in to generation;
  `Experiment.from_config` and `train` still never generate.
- The reference isochrons of [ADR-0017](0017-isochrons-by-auto-continuation.md) follow
  the same pattern. A config in `configs/isochrons/` composes a data config, whose
  system it traces, and `compute_isochrons` saves `<out_dir>/<name>-<hash>.nc`, the
  hash over what decides the result; `load_isochrons` never computes and raises
  `FileNotFoundError` naming `scripts/compute_isochrons.py`. Unlike
  `generate_dataset`, `compute_isochrons` refuses an existing file, since a foliation
  costs hours and a rerun would compute the same one.

### 4. `num_steps` is the unit of a run; the loader covers it in whole epochs

The config field is `num_steps` (Joon: easier to grasp); the window source is sized with
`num_steps=` (`epochs = ceil(num_steps / batches_per_epoch)`,
[ADR-0008](0008-dataset-format-and-sampling.md) D3) and the trainer runs exactly
`num_steps`. The logger adds `epoch = step / batches_per_epoch` to every record and the
run header prints the epoch count.

### 5. Logging: `PrintLogger` always; wandb when asked, in its own modes

`wandb: off` is the default (`PrintLogger` only). `wandb: online` / `offline` wrap
`wandb.init(mode=…, dir=run_dir, config=resolved_config, group=…)` in a `WandbLogger`;
offline runs are synced later with `wandb sync`. The composed logger is one-step delayed
(`DelayedLogger`, [ADR-0009](0009-trainer-losses-schedules-checkpoints.md) §2). Multirun
sweeps share a `group` named after the sweep directory.
`build_logger(..., wandb_run=...)` is exposed so the notebook's logger is the script's;
an existing wandb run is logged to as it is and is the caller's to finish.

- *Rejected*: wandb as the default. CI and tests must not need it; the import happens
  only when a wandb mode is selected.

### 6. Runs: one `Run` object; checkpoints hold the config; resume is explicit

The run directory is Hydra's (`runs/<date>/<time>`; `runs/multirun/…/<job>` under `-m`).
Its layout is one object, `experiment.Run`
([`run.py`](../../src/deep_isochron/experiment/run.py)): `config.yaml` (resolved),
`metadata.json` (config, dataset hash, x64, git SHA, batches per epoch) and
`checkpoints/` (Orbax, [ADR-0009](0009-trainer-losses-schedules-checkpoints.md) §6,
with the same metadata as `custom_metadata`). The construction before the loop is
`Experiment.from_config(cfg, run_dir)` (run, device, source, trainer, state, loaders,
evaluator), and `train(cfg, run_dir)` is `Experiment.from_config(...).train()`.

- *The notebook gets the script's objects.* `exp = Experiment.from_config(cfg, run_dir)`
  gives the notebook the same objects as the script, and `exp.run.checkpointer()` writes
  checkpoints that carry the config, so `load_run` works whichever loop ran. A model
  that never had a config reopens through `load_run(dir, template=state)`, the only
  option, since Orbax stores array leaves, not Equinox structure.
  `build_evaluator(cfg, source, loaders)` is exposed so the notebook's evaluator is the
  script's.
- **The final step is always checkpointed**: `Checkpointer.save(..., force=True)`
  overrides the `save_every` policy for the last step, so `load_run` /
  `load_model(run_dir)` return the trained state without arithmetic on `save_every`.
  `OrbaxCheckpointer.save` returns whether it saved, and the trainer forces when the
  loop's last step was not saved. `load_run(run_dir, step)` rebuilds model and trainer
  from the stored config and restores.
- *Never overwrite silently, resume explicitly* (Joon). `Run.from_config` refuses a
  directory that already holds a run (a config or checkpoints; a Hydra-created
  directory with only `.hydra/` is not a run). Continuing a run is
  `Experiment.from_config(cfg, run_dir, resume=True)`: it restores the latest
  checkpoint, rewrites `config.yaml` to the given `cfg`, and records
  `resumed = {from_step, config_changes, history}` in `metadata.json`, so a raised
  `num_steps` or a changed learning rate is on record. `cfg.resume = <other run>` is a
  warm start of a *new* run from that run's latest state: it continues the *same* data
  stream from `state.step`, the slice `[step:]` of the deterministic batched dataset
  ([ADR-0008](0008-dataset-format-and-sampling.md) D3), writes into the new run
  directory, and records `resumed_from = {run, step}`. Both at once is an error.
- *`config.yaml` is the latest truth.* `Run.open` reads it first and falls back to the
  checkpoints' `custom_metadata` (fixed at directory creation, so it cannot follow a
  resume).
- *One construction style* (review round 1, 2026-10-13): objects that the experiment
  layer owns (`Run`, `Loaders`, `Experiment`) construct themselves with a `from_config`
  classmethod (`Run.from_config(cfg, run_dir, metadata, resume=...)`,
  `Loaders.from_config(cfg, source, device, start_step=...)`,
  `Experiment.from_config(cfg, run_dir, resume=...)`); library objects (model, loss,
  schedule, optimizer, evaluator, logger, source) keep the free `build_*` functions,
  because `model`/`training`/`data` must not know configs (§2). `Run.open` reads, it
  does not construct from a config, so it keeps its name.

## Consequences

- [`scripts/train.py`](../../scripts/train.py) is new, `scripts/generate_data.py` is a
  thin wrapper, and `scripts/bench_dataloader.py --config …` benchmarks a training
  config's own pipeline and model.
- `deep_isochron.experiment` holds `instantiate`, `data`, `model`, `training`, `logging`
  and `run`; [`tests/test_experiment.py`](../../tests/test_experiment.py) composes every
  shipped config and runs tiny trainings end to end, including reopening and resuming.
- `Checkpointer.save` returns `bool` and takes `force`; `Trainer.train` forces the final
  checkpoint.
- CI ([`.github/workflows/ci.yml`](../../.github/workflows/ci.yml)): lint, `ty`,
  Markdown math, the `ci` Hypothesis profile without `slow`; a `slow` job with the slow
  tests and a 20-step `train.py` run on a tiny generated dataset.
- The notebook moves onto `hydra.compose` + the builders (D5, Joon).

## Alternatives rejected

Alternatives rejected for a single decision are listed under that decision.

- *Superseded 2026-10-07*: the run layout written out in `run.py`, with
  `run.train(cfg, run_dir)` calling the builders in turn and `resume=<run_dir>`
  restoring that run's latest state into the new run directory. Replaced because the
  notebook's own loop could not produce a run that `load_run` reopens: the layout lived
  only in `run.py`.
- *Superseded 2026-10-13*: the names `Run.create` and `Experiment.setup` of the
  2026-10-07 design. Replaced by `from_config` classmethods so that every object the
  experiment layer owns constructs itself the same way (Decision 6, one construction
  style).

## History

- 2026-10-05 (`experiment-config`): accepted; the six choices were taken with Joon
  before implementation.
- 2026-10-07 (`run-layout`): Decision 6 rewritten. The layout became the `Run` object
  and the construction `Experiment.setup`; resuming became explicit; `build_evaluator`,
  `build_logger(wandb_run=...)` and `load_or_generate_source` were exposed.
- 2026-10-13 (`run-layout`, review round 1): one construction style; `Run.create` and
  `Experiment.setup` became `Run.from_config` and `Experiment.from_config`.
- 2026-10-27 (`prose`): restructured to the format of
  [ADR-0019](0019-adrs-describe-the-current-design.md).
- 2026-10-28 (`auto-isochrons`): Decision 3 covers the isochron configs
  (`configs/isochrons/`, `compute_isochrons`, `load_isochrons`); Hydra's submitit
  launcher joins as the dependency group `slurm`, with
  `configs/hydra/launcher/slurm.yaml` for training and isochron jobs alike.
