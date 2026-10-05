---
type: decision
id: ADR-0011
status: accepted
updated: 2026-10-05
verified_by: pending (Joon; decisions taken in discussion 2026-10-05)
---

# ADR-0011 — The experiment layer: YAML configs, builders, Hydra-owned runs

**Status**: accepted (2026-10-05, branch `experiment-config`; the six choices below were
taken with Joon before implementation).

## Context

Phases A–C produced a trainer with injected side effects (ADR-0009), a finite batched
data loader (ADR-0008 Decision 3) and a device policy (`resolve_device`). What was missing
was the layer that turns a description of an experiment into a run: configuration,
construction of the objects, the entry point, resuming, reopening a run for analysis, and
CI. `scripts/generate_data.py` had already set a configuration style — Hydra YAML with
`_target_` entries — and the notebook carried its own ad-hoc construction code.

## Decisions

### 1. YAML config groups with `_target_` leaves; no dataclass schema (for now)

`configs/train.yaml` composes the groups `data/` (the generation configs, unchanged),
`model/`, `loss/`, `schedule/`, `optimizer/`, `wandb/`, plus top-level fields (`seed`,
`device`, `x64`, `num_steps`, `windows`, `validation`, `eval_every`, `log_every`,
`checkpoint`, `resume`). Leaf objects — a normal form, a spline, an optimizer — are
`_target_` mappings instantiated as written.

- *Rejected for now*: structured configs (dataclasses + `ConfigStore`), the roadmap's
  original D1. Stronger typing and validation at the cost of a parallel schema for every
  class; to be revisited if compositions grow or validation becomes a problem (Joon's
  call: "re-evaluate later").
- **Lists become tuples.** Hydra's `instantiate` can only produce lists, while the code
  types fixed-size parameters as tuples and the test suite's beartype hook enforces that.
  `experiment.instantiate` is a 30-line replacement: same `_target_` convention, nested
  targets, keyword overrides for what a config cannot express (per-layer PRNG keys, the
  alternating `flip`), and YAML lists as tuples. Spline constructors additionally accept
  any two-element sequence for `xy_range`.

### 2. Builders are pure functions of the config, in `deep_isochron.experiment`

**Scope rule** (review round 1): the package only parses configs and instantiates
objects. Anything with logic of its own lives in the module it belongs to —
`MultiLogger`/`EpochLogger` in `training.loggers`, `git_state` in `provenance` (shared
with `data.generate`), the window source in `data.windows`. `compose(*overrides)` is the
one convenience the package adds for notebooks and tests: the config exactly as
`scripts/train.py` composes it.

`build_source(cfg.data)`, `build_loaders(cfg, source, device, start_step)`,
`build_model(cfg.model, key)`, `build_trainer(cfg)` (loss, schedule, optimizer),
`build_logger(cfg, …)`; `run.train(cfg, run_dir)` strings them together, and
`scripts/train.py` is a ten-line Hydra `main` around it. Tests call `train` directly on
tiny configs; notebooks compose the same configs with `hydra.compose` and call the same
builders (D5), so there is one construction path.

- The "coupling registry" of the original D2 is Hydra's `_target_`: a model config names
  its layer classes; `build_inn` only adds keys and `flip`.

### 3. The `data` group *is* the generation config; training never generates

`build_source` recomputes the generation metadata from the config (`generation_metadata`,
factored out of `generate`) and loads `<out_dir>/<name>-<config_hash>.nc`. A missing
file is a `FileNotFoundError` naming the `scripts/generate_data.py` command. The hash is
computed under x64 regardless of the run's `x64` setting, because datasets are float64
and the hash covers the system's parameters as the arrays hold them.

- *Rejected*: generating on a miss. "Any generation should be explicit and intended"
  (Joon) — a training job on a GPU node must not silently spend ten minutes integrating.
- `generate_dataset(cfg.data)` is the one place generation happens; the script calls it.

### 4. `num_steps` is the unit of a run; the loader covers it in whole epochs

The config field is `num_steps` (Joon: easier to grasp); the window source is sized with
`num_steps=` (`epochs = ceil(num_steps / batches_per_epoch)`, ADR-0008 D3) and the
trainer runs exactly `num_steps`. The logger adds `epoch = step / batches_per_epoch` to
every record and the run header prints the epoch count.

### 5. Logging: `PrintLogger` always; wandb when asked, in its own modes

`wandb: off` is the default (`PrintLogger` only). `wandb: online` / `offline` wrap
`wandb.init(mode=…, dir=run_dir, config=resolved_config, group=…)` in a `WandbLogger`;
offline runs are synced later with `wandb sync`. The composed logger is one-step delayed
(`DelayedLogger`, ADR-0009 §2). Multirun sweeps share a `group` named after the sweep
directory.

- *Rejected*: wandb as the default. CI and tests must not need it; the import happens
  only when a wandb mode is selected.

### 6. Runs: Hydra owns the directory; checkpoints hold the config; resume is a slice

The run directory is Hydra's (`runs/<date>/<time>`; `runs/multirun/…/<job>` under `-m`);
it receives `config.yaml` (resolved), `metadata.json` (config, dataset hash, x64, git
SHA, batches per epoch) and `checkpoints/` (Orbax, ADR-0009 §6, with the same metadata as
`custom_metadata`). **The final step is always checkpointed**: `Checkpointer.save(...,
force=True)` overrides the `save_every` policy for the last step, so `load_run` /
`load_model(run_dir)` return the trained state without arithmetic on `save_every` —
`OrbaxCheckpointer.save` now returns whether it saved, and the trainer forces when the
loop's last step was not saved. `load_run(run_dir, step)` rebuilds model and trainer from
the stored config and restores. `resume=<run_dir>` restores that run's latest state and
continues the *same* data stream from `state.step` — the slice `[step:]` of the
deterministic batched dataset (ADR-0008 D3) — writing into the new run directory.

## Consequences

- `scripts/train.py` (new), `scripts/generate_data.py` (thin wrapper), `scripts/bench_dataloader.py --config …` benchmarks a training config's own pipeline and model.
- `deep_isochron.experiment` (`instantiate`, `data`, `model`, `training`, `logging`,
  `run`); `tests/test_experiment.py` composes every shipped config and runs tiny trainings
  end to end, including reopening and resuming.
- `Checkpointer.save` returns `bool` and takes `force`; `Trainer.train` forces the final
  checkpoint.
- CI (`.github/workflows/ci.yml`): lint, `ty`, Markdown math, the `ci` Hypothesis profile
  without `slow`; a `slow` job with the slow tests and a 20-step `train.py` run on a tiny
  generated dataset.
- The notebook moves onto `hydra.compose` + the builders (D5, Joon).
