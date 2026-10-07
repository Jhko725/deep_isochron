---
type: change
status: ready for review
updated: 2026-10-07
branch: run-layout
---

# `run-layout` — one run layout for the script and the notebook

## Summary

Training from the notebook (`prototype.ipynb`, `af183a6`) produced checkpoints that
`load_run` could not reopen: the hand-rolled loop used a bare `OrbaxCheckpointer` (no
config in its metadata) in a directory without the run layout, and `run.py` hard-coded
that layout. The run directory is now an object, `experiment.Run`, and everything built
before the loop is `setup(cfg, run_dir) -> Experiment`, so the notebook gets the script's
objects and writes reopenable checkpoints through `exp.run.checkpointer()` while keeping
its own loop and logger. `Run.create` refuses an existing run; continuing one is the
explicit `setup(..., resume=True)`, which records the config changes. Also: a model built
without any config reopens through `load_run(dir, template=state)`; `build_evaluator` and
`build_logger(wandb_run=...)` are exposed; `load_or_generate_source` is the notebook's
explicit opt-in to generation. ADR-0011 §6 amended. (Items A, B, D of the 2026-10-07
discussion; C dropped, the plots module deferred until its contents are planned.)

## Files

- `src/deep_isochron/experiment/run.py` — rewritten: `Run` (frozen dataclass: `dir`,
  `cfg`, `metadata`; `CONFIG_FILE`/`METADATA_FILE`/`CHECKPOINT_DIR`; `exists`, `create`,
  `resume`, `open`, `checkpointer`, `latest_step`, `restore`), `Experiment` (`run`,
  `device`, `source`, `trainer`, `state`, `start_step`, `loaders`, `evaluator`; `train`),
  `setup(cfg, run_dir, resume=False)`, `train(cfg, run_dir, group, resume)`,
  `load_run(run_dir, step, template=None)`, `load_model(..., template=None)`;
  `configure_jax` unchanged.
- `src/deep_isochron/experiment/data.py` — `load_or_generate_source(data_cfg)`,
  `build_evaluator(cfg, source, loaders)` (from the inline construction in `train`).
- `src/deep_isochron/experiment/logging.py` — `build_logger(..., wandb_run=None)`: an
  existing wandb run is used as given (and left for the caller to finish).
- `src/deep_isochron/experiment/__init__.py` — exports `Run`, `Experiment`, `setup`,
  `build_evaluator`, `load_or_generate_source`; module docstring.
- `configs/wandb/{online,offline}.yaml` — `project: isochron` (the project the notebook
  logs to). `configs/wandb/off.yaml` — `every: 50` (used when a run is passed in).
- `scripts/README.md` — the Python snippet rewritten around `setup`.
- `tests/test_experiment.py` — six new tests (section 5; see Tests).
- `docs/decisions/0011-experiment-layer.md` — §6 amended (Run, setup, explicit resume).
- `docs/architecture.md` — `experiment/` tree and the "From a config to a run" diagram.
- `docs/index.md` — ADR-0011 status, change-document row. `docs/roadmap.md` — ledger row.

## Design

- **`Run.create` refuses; `Run.resume` is explicit** (Joon: "Refusing overwriting is the
  right call. Even when resuming, the decision should be made explicit"). A run exists
  when `config.yaml` or `checkpoints/` exists; Hydra's pre-created directory with only
  `.hydra/` does not count. Resume rewrites `config.yaml` to the given `cfg` and records
  `resumed = {from_step, config_changes: {dotted.key: [old, new]}, history}`; the
  alternative — forbidding any config change on resume — would have blocked the two
  things one resumes *for* (more steps, a different learning rate). `cfg.resume = <other
  run>` keeps its warm-start meaning into a *new* directory (`resumed_from`); combining
  it with `resume=True` is an error.
- **`config.yaml` first, checkpoint metadata second** in `Run.open`: Orbax's
  `custom_metadata` is fixed when the directory is created, so after a resume only the
  file is current. The checkpoint metadata remains the fallback for a `checkpoints/`
  directory moved on its own.
- **`template=` for configless models.** Orbax saves array leaves; the Equinox structure
  has to come from somewhere, and for a model assembled by hand that is the caller's own
  `TrainerState`. With a template, `load_run` also accepts a bare `OrbaxCheckpointer`
  directory (no `checkpoints/` layout) — the notebook's `results/ckpt` case.
- **The notebook's loop stays the notebook's.** `Experiment.train()` is the default loop;
  the fields are public so `trainer.train(exp.state, exp.loaders.train, ...)` with any
  logger is a first-class path, and the run's checkpointer is what makes it reopenable.
  Scope rule of ADR-0011 §2 kept: `setup` composes and instantiates, nothing more.
- **Not done, on purpose**: a `jax` memory/preallocation config (dropped), the plots
  module (planned separately).

## Bugs fixed

None in the library. In the notebook (not changed here, Joon's file): cell 27 builds the
training loader from `source` after splitting `source_train, source_val`, so the
validation trajectories are also trained on — `setup` makes this impossible, since the
split and the loaders come from one place.

## Tests

`uv run pytest tests/test_experiment.py` (20 tests; the six new ones in section 5):

- `test_setup_then_manual_training_loop_reopens_with_load_run` — the notebook path.
- `test_hand_built_model_reopens_through_a_template` — bare checkpoint dir + template;
  without a template the error names `template`.
- `test_run_create_refuses_an_existing_run_and_resume_is_explicit` — Hydra-style
  pre-created directory is fine; a second `setup`/`train` on a run refuses; `resume=True`
  continues to 5 steps and records `num_steps` and `optimizer.learning_rate` changes,
  `from_step`, `history`; `resume=True` + `cfg.resume` is an error; resuming where there
  is no checkpoint is an error.
- `test_warm_start_from_another_run_is_recorded` — `cfg.resume` → `resumed_from`.
- `test_load_or_generate_source_generates_once` — file created once, then loaded.
- `test_build_logger_takes_an_existing_wandb_run` — a fake run receives the thinned,
  epoch-annotated records; `build_evaluator` matches `setup`'s.

## Open issues

- The notebook itself still has the cell-27 leak and the bare checkpointer; moving it onto
  `setup` is Joon's (D5).
- `period` is reported signed (−11.24 for FHN, the clockwise cycle) — E6 decides the sign
  convention between `AbstractNormalForm.period()` and `Cycle.period`/`winding`.
- Plots module: planned separately.

## Review notes

| Change | Thoughts | Modifications |
|---|---|---|
| `src/deep_isochron/experiment/run.py`: `Run` (create refuses / resume explicit / open), `Experiment`, `setup`, `train`, `load_run(template=)` | | |
| `src/deep_isochron/experiment/data.py`: `load_or_generate_source`, `build_evaluator` | | |
| `src/deep_isochron/experiment/logging.py`: `build_logger(wandb_run=)`; `configs/wandb/*` (`project: isochron`, `every` in `off`) | | |
| `src/deep_isochron/experiment/__init__.py` exports; `scripts/README.md` snippet | | |
| `tests/test_experiment.py` section 5 (6 tests) | | |
| ADR-0011 §6 amendment; `docs/architecture.md`; `docs/index.md`; `docs/roadmap.md` | | |
