---
type: change
status: review round 1 applied; from_config unification and the weight-decay mask landed 2026-10-13
updated: 2026-10-13
branch: run-layout
---

# `run-layout` — one run layout for the script and the notebook

## Summary

Training from the notebook (`prototype.ipynb`, `af183a6`) produced checkpoints that
`load_run` could not reopen: the hand-rolled loop used a bare `OrbaxCheckpointer` (no
config in its metadata) in a directory without the run layout, and `run.py` hard-coded
that layout. The run directory is now an object, `experiment.Run`, and everything built
before the loop is `Experiment.setup(cfg, run_dir)`, so the notebook gets the script's
objects and writes reopenable checkpoints through `exp.run.checkpointer()` while keeping
its own loop and logger. `Run.create` refuses an existing run; continuing one is the
explicit `Experiment.setup(..., resume=True)`, which records the config changes. Also: a model built
without any config reopens through `load_run(dir, template=state)`; `build_evaluator` and
`build_logger(wandb_run=...)` are exposed; `load_or_generate_source` is the notebook's
explicit opt-in to generation. ADR-0011 §6 amended. (Items A, B, D of the 2026-10-07
discussion; C dropped, the plots module deferred until its contents are planned.)

## Files

- `src/deep_isochron/experiment/run.py` — rewritten: `Run` (frozen dataclass: `dir`,
  `cfg`, `metadata`; `CONFIG_FILE`/`METADATA_FILE`/`CHECKPOINT_DIR`; `exists`, `create`,
  `resume`, `open`, `checkpointer`, `latest_step`, `restore`), `Experiment` (`run`,
  `device`, `source`, `trainer`, `state`, `start_step`, `loaders`, `evaluator`; `train`),
  `Experiment.setup(cfg, run_dir, resume=False)` (classmethod; review round 1),
  `train(cfg, run_dir, group, resume)`,
  `load_run(run_dir, step, template=None)`, `load_model(..., template=None)`;
  `configure_jax` unchanged.
- `src/deep_isochron/experiment/data.py` — `load_or_generate_source(data_cfg)`,
  `build_evaluator(cfg, source, loaders)` (from the inline construction in `train`).
- `src/deep_isochron/experiment/logging.py` — `build_logger(..., wandb_run=None)`: an
  existing wandb run is used as given (and left for the caller to finish).
- `src/deep_isochron/experiment/compose.py` — `compose_group(group, name, *overrides)`:
  one group's config alone, overrides relative to the group, returned as that node
  (`compose_group("data", "fhn", "n_trajectories=500")` for `load_or_generate_source`).
- `src/deep_isochron/training/optimizers.py` — new: `decay_mask`; `training/__init__.py`
  export; `configs/optimizer/adamw.yaml` — `weight_decay`, `mask` (review round 1).
- `src/deep_isochron/experiment/__init__.py` — exports `Run`, `Experiment`, `Loaders`,
  `build_evaluator`, `load_or_generate_source`, `compose_group`; module docstring.
- `configs/wandb/{online,offline}.yaml` — `project: isochron` (the project the notebook
  logs to). `configs/wandb/off.yaml` — `every: 50` (used when a run is passed in).
- `scripts/README.md` — the Python snippet rewritten around `setup`.
- `tests/test_experiment.py` — seven new tests (`test_compose_group_is_the_groups_node`; section 5; see Tests).
- `docs/decisions/0011-experiment-layer.md` — §6 amended (Run, setup, explicit resume).
- `docs/architecture.md` — `experiment/` tree and the "From a config to a run" diagram.
- `docs/index.md` — ADR-0011 status, change-document row. `docs/roadmap.md` — ledger row.

## Review round 1 (2026-10-13) — what changed in response

- **`setup` → `Experiment.setup`** (Joon's edit, `53da6e1`): the tests, `scripts/README.md`,
  ADR-0011 §6, `docs/architecture.md`, the module docstrings and the `Run.create` error
  message now say `Experiment.setup`; the error string of the `resume`/`cfg.resume` clash
  is a plain string again (the triple-quoted one carried its indentation into the message).
- **`train` kept**: it is the one-call form the script and the tests use, and the only
  place `group` (the multirun → wandb group) is plumbed; three lines. Drop it if you
  prefer `Experiment.setup(...).train(group=...)` spelled out in `scripts/train.py`.
- **Unifying the construction style** (classmethods vs `build_*`): agreed and applied —
  `Experiment.setup` → `Experiment.from_config`, `build_loaders` → `Loaders.from_config`,
  `Run.create`/`Run.resume` → `Run.from_config(..., resume=)` (`Run.open` unchanged).
  The rule is recorded in ADR-0011 §6 and the package docstring; all callers, tests and
  docs updated.
- `configs/data/fhn-easy.yaml` (Joon's new config, Langfield et al. 2025 §4.2) used the
  class defaults, i.e. the *same* system as `fhn.yaml` (`c = 3`, `z = −0.4`, the 2014
  slow–fast regime); it now sets `c: 1.0`, `z: -0.8` as the paper states ("c = 1 with
  practically no time-scale separation", "z = −0.8 so that the equilibrium is shifted";
  period ≈ 10.8329, clockwise, repelling focus ≈ (0.2729, 0.5339)). *Cited* from the
  paper's §4.2; the FHN form of their Eq. (14) is ours.
- `matplotlib` moved from the `dev` group to the main dependencies: `deep_isochron.plotting`
  imports it, and a package module cannot depend on a dev-only package. **`uv.lock` is not
  updated here** — my `uv` rewrites the lock format (revision 3, 1 500 noise lines); run
  `uv lock` once after pulling.
- `tests/test_experiment.py`: `test_compose_group_is_the_groups_node` follows the dataset
  rename (`name: fitzhugh_nagumo`); `build_source`'s not-found message no longer claims the
  dataset `name` is the config file name (they differ now) and mentions
  `load_or_generate_source`.
- Answer to the `plotting` question (rcParams): see *Design*; Joon edits them himself.
- **Weight decay on the normal-form parameters** (Joon's `adam` → `adamw` switch, review
  round 1): `optax.adamw`'s default `weight_decay = 1e-4` pulled *every* leaf toward
  zero. For the bijection that is a prior toward the identity map (`raw = 0` is the
  identity, ADR-0002) — acceptable; for the latent dynamics it biases quantities under
  estimation: Bautin's `raw_a`, `raw_b` at zero are `a = 1`, `b = 0` (the Hopf form),
  `w`, `w0` at zero a frozen oscillator, the autoencoder's `omega`/`raw_decay` likewise —
  the estimated period and Floquet exponent would be systematically off. Fix at the
  optimizer level, which is where decoupled decay lives in optax:
  `training.optimizers.decay_mask(exclude=("latent_dynamics",))` returns the `mask`
  pytree (`False` under any attribute named `latent_dynamics`, which both models use),
  and `configs/optimizer/adamw.yaml` passes it (`weight_decay: 1e-4` made explicit).
  Nothing is needed at the trainer level: `is_trainable` is a different question
  (freezing), and the decay is applied inside `optimizer.update`. Tests: a zero-gradient
  `adamw` step moves the bijection and not the dynamics with the mask, both without
  (`test_training`); the composed config does the same through `TrainerState.take_step`
  (`test_experiment`). A per-group learning rate (`optax.multi_transform` with the same
  path rule) is the next step if the dynamics parameters want their own rate.

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
  module (planned separately; Joon started `deep_isochron.plotting` in review round 1).
- **One construction style (review round 1, applied).** Rule: *experiment-owned
  classes* construct themselves from a config by a classmethod named `from_config`;
  *library* objects (model, loss, schedule, optimizer, `Evaluator`) keep the free
  `build_*` functions, because `training`/`model` must not know configs (ADR-0011 §2).
  Concretely: `Experiment.setup` → `Experiment.from_config(cfg, run_dir, *, resume=False)`;
  `build_loaders` → `Loaders.from_config(cfg, source, device, *, start_step=0)`;
  `Run.create`/`Run.resume` → `Run.from_config(cfg, run_dir, metadata, *, resume=False)`
  with `Run.open` unchanged (it reads, it does not construct from a config). `build_model`,
  `build_trainer`, `build_evaluator`, `build_logger`, `build_source` stay.
- **rcParams worth setting (review round 1 question).** Beyond what `configure_rcparams`
  sets: `figure.constrained_layout.use = True` (every notebook figure currently passes
  it by hand); `figure.dpi ≈ 120` for notebooks and `savefig.dpi = 200` for the
  documentation PNGs (600 makes multi-MB files; keep 600 or vector for the paper);
  `savefig.bbox = "tight"`; `pdf.fonttype = 42` and `ps.fonttype = 42` so text in vector
  exports stays editable; `mathtext.fontset = "cm"` (or `"stix"`) so `$x$` labels match
  the documents; `legend.frameon = False`; `axes.formatter.use_mathtext = True`;
  `lines.linewidth = 1.0` (the notebook's choice); `image.cmap` a perceptual map
  (`cet_CET_L8` or `viridis`) for the phase/error grids to come; and
  `axes.prop_cycle` only if a categorical palette is the usual case — for 500
  trajectories `glasbey_light` is right, for a two-series comparison (data vs model) a
  short named cycle is clearer. Leave `font.family` alone (system-dependent).

## Bugs fixed

None in the library. In the notebook (not changed here, Joon's file): cell 27 builds the
training loader from `source` after splitting `source_train, source_val`, so the
validation trajectories are also trained on — `setup` makes this impossible, since the
split and the loaders come from one place.

## Tests

`uv run pytest tests/test_experiment.py` (21 tests; `test_compose_group_is_the_groups_node` in section 1, six new in section 5):

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
| `src/deep_isochron/experiment/run.py`: `Run` (create refuses / resume explicit / open), `Experiment`, `setup`, `train`, `load_run(template=)` | <ul><li>`setup` feels to generic of a name. Either renaming to `setup_experiment` or making it a classmethod of `Experiment` seems more adequate. <li>`train` is a super thin helper method over `setup`. Is it worth keeping? <li>Noticing that `Run` and `data.Loaders` are implemented in a different style. `Run` uses a classmethod to instantiate from config, `Loaders` uses the free `build_loaders` function. For an easy to expect codebase, the pattern used should be unified, and I prefer the `classmethod` based approach. Maybe name all these similar functionalities under the same name, for example `.from_config`? For run, this will require perhaps merging the functionalities of `create` and `resume` via a boolean flag `resume`.<ul>| `setup` changed to `Experiment.setup`. Docs will need updating accordingly. Rest deferred to discussions with Claude. |
| `src/deep_isochron/experiment/data.py`: `load_or_generate_source`, `build_evaluator` | Both look good for now. | None |
| `src/deep_isochron/experiment/logging.py`: `build_logger(wandb_run=)`; `configs/wandb/*` (`project: isochron`, `every` in `off`) | Looks good. | None |
| `src/deep_isochron/experiment/compose.py`: `compose_group`; `experiment/__init__.py` exports; `scripts/README.md` snippet | Looks good. Minor changes in `__init__.py` to comply with my edits. | None |
| `tests/test_experiment.py` section 5 (6 tests) | Had a cursory look. Looks good for now. | None |
| ADR-0011 §6 amendment; `docs/architecture.md`; `docs/index.md`; `docs/roadmap.md` | Had a readthrough. Looks good. | None |
| `prototype.ipynb`: Replaced existing code with the updated functions. | Will be migrating more stuff as I go. | None |
| `src/deep_isochron/plotting`: Started refactoring some of the plots in `prototype.ipynb` to dedicated plotting functions. | Are there other things that might be worth setting as default rcparams? | None |

### Review round 1

| Change | Thoughts | Modifications |
|---|---|---|
| `Experiment.setup` references updated everywhere (tests, README, ADR-0011, architecture, docstrings); `train` kept | Confirmed. | None |
| `configs/data/fhn-easy.yaml`: `c: 1.0`, `z: -0.8` (Langfield et al. 2025 §4.2) | Looks good. | None|
| `pyproject.toml`: matplotlib → main dependencies (`uv lock` to be run by Joon) | `uv sync --all-extras` ran to install the cuda version of JAX. | None |
| `experiment/data.py`: `build_source` not-found message; `tests/test_experiment.py` dataset rename | Good. | None |
| `from_config` classmethods: `Experiment.from_config`, `Loaders.from_config`, `Run.from_config(resume=)`; ADR-0011 §6 rule; callers, tests, docs | Confirmed. Good | None |
| `training/optimizers.py` (`decay_mask`), `configs/optimizer/adamw.yaml` (`weight_decay`, `mask`); `test_training` + `test_experiment` decay tests | Confirmed. | Need to assess later the effect of choosing the right optimizer and its parameters. Deferred for now. |
