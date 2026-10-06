---
type: change
status: review round 2 applied (utils/, validation split); D5 (notebook) done by Joon
updated: 2026-10-05
branch: experiment-config
---

# `experiment-config` — Phase D: from a config to a run

## Summary

A training run is now a composed Hydra config: `uv run python scripts/train.py data=bautin
model=conjugacy num_steps=5000` builds the dataset loader, model, loss, schedule and
optimizer from `configs/`, trains with the Phase C trainer, logs (`PrintLogger`, wandb
online/offline when asked), checkpoints into the Hydra run directory with the final step
always saved, and can be reopened (`load_run`, `load_model`) and resumed (`resume=<dir>`,
continuing the same data stream). The builders live in `deep_isochron.experiment` and are
what the tests, the benchmark (`--config`) and the notebook (D5) call. Decisions: ADR-0011.
Also: the one pre-existing red test is fixed, and CI is added.

## Files

- `configs/train.yaml` — new: the composed run (defaults `data=bautin model=conjugacy
  loss=conjugacy schedule=constant optimizer=adam wandb=off`; `seed`, `device`, `x64`,
  `num_steps`, `windows{length,batch,seed,sampling{kind,…}}`, `validation{fraction,seed,
  batch,stride}`, `eval_every`, `log_every`, `checkpoint{every,metric,keep_best}`,
  `resume`, Hydra run/sweep dirs under `runs/`).
- `configs/model/{conjugacy,autoencoder}.yaml`, `configs/loss/{conjugacy,autoencoder}.yaml`,
  `configs/schedule/{constant,yawata}.yaml`, `configs/optimizer/adam.yaml`,
  `configs/wandb/{off,online,offline}.yaml` — new groups.
- `src/deep_isochron/experiment/__init__.py` — new package, exports.
- `src/deep_isochron/experiment/instantiate.py` — `instantiate(cfg, **overrides)`:
  `_target_` convention, nested targets, **YAML lists → tuples**.
- `src/deep_isochron/experiment/data.py` — `dataset_file`, `build_source` (never
  generates; error names the command), `generate_dataset` (the one place generation
  happens; used by the script), `reference_normal_form`, `build_window_source`
  (uniform / weighted / mixed from `windows.sampling`), `build_loaders` (split, resume
  slice, `to_device`, validation dataset) → `Loaders`.
- `src/deep_isochron/experiment/model.py` — `build_inn` (per-layer keys, alternating
  `flip`), `build_model` (`kind: conjugacy | autoencoder`).
- `src/deep_isochron/experiment/training.py` — `build_loss`, `build_schedule`
  (`Constant` with `values: null` → the loss's defaults), `build_optimizer`,
  `build_trainer`, `resolved`.
- `src/deep_isochron/experiment/logging.py` — `build_logger` (PrintLogger + optional
  `WandbLogger`, then `EpochLogger`, then `DelayedLogger`).
- `src/deep_isochron/experiment/compose.py` — `compose(*overrides)` (review round 1).
- `src/deep_isochron/training/loggers.py` — `MultiLogger`, `EpochLogger` (review round 1).
- `src/deep_isochron/provenance.py` — new: `git_state`, `package_version` (from
  `data/generate.py`'s private helpers; review round 1).
- `scripts/README.md` — new (review round 1).
- Review round 2: `src/deep_isochron/utils/{__init__,numerics,provenance}.py` (from
  `misc.py`, `provenance.py`); `training/evaluation.py` (`prediction_sums`, `t_split`);
  `experiment/data.py` (`validation_t_split`), `experiment/run.py`, `configs/train.yaml`
  (`validation.t_split`); `docs/design/validation-split.md` (new); ADR-0009 §5; tests.
- `src/deep_isochron/experiment/run.py` — `configure_jax`, `train(cfg, run_dir, group)`,
  `load_run`, `load_model`.
- `scripts/train.py` — new Hydra entry point (quiets absl/jax INFO logging; multirun →
  wandb group).
- `scripts/generate_data.py` — body moved to `experiment.generate_dataset`.
- `scripts/bench_dataloader.py` — `--config <overrides>`: benchmark a training config's
  own pipeline, model and loss (D7).
- `src/deep_isochron/data/generate.py` — `generation_metadata(...)` factored out of
  `generate` (the hash without generating); `data/__init__.py` exports it.
- `src/deep_isochron/data/windows/batched.py` — `mixed_ranges(source, length, split_idx)`
  factored out of `mixed_window_batches`; `windows/__init__.py` exports it.
- `src/deep_isochron/model/invertible/splines/{base,cubic,linear,rational_quadratic}.py`
  — `xy_range` accepts any two-element sequence (`_check_range` validates length).
- `src/deep_isochron/training/checkpoint.py` — `Checkpointer.save(..., force=False) ->
  bool`; `OrbaxCheckpointer.save` passes `force` to Orbax and returns whether it saved;
  `latest_step`, `custom_metadata` properties.
- `src/deep_isochron/training/trainer.py` — the final step is always evaluated and
  checkpointed (`force=True`) unless the loop's last step already saved it.
- `tests/strategies.py` — `domain_points` keeps bijection-law points exactly `0` or at
  least `1e-9` from it (see Bugs fixed).
- `tests/test_experiment.py` — new (11 tests, see Tests). `tests/test_training.py` —
  `test_final_step_is_checkpointed_regardless_of_the_save_policy`.
- `.github/workflows/ci.yml` — new (D6). `.gitignore` — `runs/`, `data/*.nc`, CI
  scratch dirs; `wandb/` anchored to the root so `configs/wandb/` is tracked.
- `docs/decisions/0011-experiment-layer.md` — new ADR. `docs/architecture.md`
  (`experiment/`, scripts, "From a config to a run"), `docs/index.md`, `docs/roadmap.md`
  (Phase D rewritten, ledger), `docs/design/training-step-performance.md` §7.

## Review round 1 (2026-10-05) — what changed in response

- **What `resolved(cfg)` records** (question on `training.py`): the *whole* composed
  config — `data`, `model`, `loss`, `schedule`, `optimizer`, `wandb` and the top-level
  fields — as a plain resolved dict; it goes verbatim into `wandb.init(config=…)`, the
  checkpoint's `custom_metadata` and `metadata.json`. Model configs are in, so wandb can
  compare runs by `model.inn.blocks`, `model.latent_dynamics.a`, etc. (wandb flattens
  nested dicts with dots). `test_train_writes_a_run_and_load_model_reads_it_back` now
  asserts the model and loss entries are present.
- **`MultiLogger` and `EpochLogger` moved to `training/loggers.py`** (exported from
  `training`); `experiment/logging.py` keeps only `build_logger`. `EpochLogger` is
  general — any loop that knows its batches per epoch can use it — so it belongs with
  the other loggers.
- **`git_state` is shared**: new `deep_isochron/provenance.py` (`git_state`,
  `package_version`), used by `data.generate` (as before, renamed from the private
  helpers) and by `experiment.run` for `metadata.json["git"] = {sha, dirty}`; the
  duplicate `_git_sha` is gone. **Scope rule** recorded in ADR-0011 §2: `experiment` only
  parses configs and instantiates; logic lives in its own module.
- **`compose(*overrides)`** moved from the test module into
  `experiment/compose.py` (exported): `hydra.compose` over the checkout's `configs/`,
  for notebooks and tests alike; `CONFIG_DIR` is `<repo>/configs`.
- **`force` on `Checkpointer.save`** is per call, not a mode: the policy decides every
  offered step, `force=True` writes that one call regardless, and the trainer passes it
  exactly once, for the final step. The name stays (it mirrors Orbax's `force`); the
  class and method docstrings now say so explicitly.
- **`scripts/README.md`**: one table row per script (what it does, typical calls) and a
  snippet for reopening runs from Python.

## Review round 2 (2026-10-06) — `utils/` and the validation split

- **`deep_isochron/utils/`**: `misc.py` → `utils/numerics.py` (`squashed_exp`,
  `inv_softplus`, polar ↔ Cartesian — used by `model/invertible/constraints.py` *and*
  `systems/normal_forms/base.py`, so it could not move into `model/`), `provenance.py` →
  `utils/provenance.py`. Not `math.py`: shadowing a stdlib name inside a package invites a
  wrong import one day. Three import sites and one test updated.
- **Validation split by window start** (design document `validation-split.md`, ADR-0009 §5
  amended): `Evaluator(t_split=…)` reports `val/mse_early` / `val/mse_late` (windows
  starting before / at-or-after `t_split`), `val/n_early` / `val/n_late`; `val/mse`
  unchanged (the overall mean). Implemented as a jitted `prediction_sums` whose per-batch
  sums the `Evaluator` accumulates, like the phase statistics. An empty bucket is `NaN`.
  Config: `validation.t_split` (a time, or `null` = the `mixed` sampler's `ts[split_idx]`
  when that sampler is used, else no split — `experiment.validation_t_split`);
  `checkpoint.metric: val/mse_early` selects on the transient.
- Tests: `test_evaluator_splits_the_prediction_error_by_window_start` (counts,
  recombination to the overall mean, empty bucket, a model wrong only on late windows
  caught only there via `prediction_sums`); `test_validation_split_follows_the_mixed_sampler`
  (resolution rule; a run with `checkpoint.metric=val/mse_early`).
- Answers recorded for the other round-2 questions (wandb run ownership and directory,
  train-then-continue with a new loader, the `Evaluator`'s host sync, winding direction →
  Phase E `analysis`) are in the conversation summary of the Project doc; the winding
  direction and a far-from-cycle validation set are Phase E items in the roadmap.

## Design

ADR-0011 has the six decisions and their alternatives. Points specific to the branch:

- **`experiment.instantiate` instead of `hydra.utils.instantiate`.** The first end-to-end
  run failed in the beartype hook: `MonotonicRQSpline(xy_range=[-4.0, 4.0])` — Hydra
  produces lists, the code types tuples. Rather than relax every tuple annotation, the
  instantiation converts lists to tuples (and recurses into nested `_target_`s). Spline
  `xy_range` additionally accepts any sequence, since `_check_range` already normalizes.
- **The dataset hash is computed under x64.** `dataset_file` wraps metadata construction
  in `jax.enable_x64(True)`: the hash covers the system's parameters as the arrays hold
  them, and a float32 process (the test harness before the hook, a run with `x64: false`)
  would otherwise look for a file named after rounded parameters. Found when the smoke
  script, run without x64, could not find the file the test had just generated.
- **Forced final checkpoint.** With `checkpoint.every=2` and `num_steps=3`, Orbax's
  `FixedIntervalPolicy` declined the trainer's end-of-run save and `load_model` returned
  step 2. `save` now returns whether it saved, and the trainer forces the final step when
  the loop's last step did not persist — so `load_run` is always the trained state.
- **Resume writes into a new run directory** (`resume=<old>` restores from `<old>`); the
  continued stream is `WindowBatchSource[step:]`, tested against the unsliced stream.
- **`num_steps` is honored exactly**: the window source is sized to cover it in whole
  epochs (so the loader is longer than the run), and `Trainer.train` gets `num_steps`
  explicitly — no exhaustion warning.
- **`.gitignore`'s `wandb/`** silently ignored the new `configs/wandb/` group; anchored
  to the root.

## Bugs fixed

- **`test_vector_jacobian_consistent_with_inverse[circular_rq (K=8)]`** (red since Phase
  B). Hypothesis's point `x = (1e-12, 0)` sits exactly on `CircularMonotonicRQCoupling`'s
  regularization radius `eps_r = 1e-12`: outside the disc for the forward map (full polar
  Jacobian, angular eigenvalue `s'(0)`), inside after rounding for the inverse (a constant
  rotation), so `Df⁻¹·Df` has `s'(0) = 0.3635` where the test wants `1` — a boundary
  straddle, not the periodic-seam limitation. `tests/strategies.domain_points` now draws
  points that are exactly `0` or at least `1e-9` in magnitude (`magnitudes(1e-9, 5.0)`),
  with the reason in its docstring; no bijection law is meant to hold *on* a
  regularization boundary.
- Orbax declining the end-of-run save (above).

## Tests

`uv run pytest tests/test_experiment.py` (≈ 90 s; generates a 12-trajectory Bautin
dataset once per module into `tmp_path`). Eleven tests: every shipped config option
composes (defaults, autoencoder + yawata, fhn, wandb online/offline, weighted, mixed);
`build_source` refuses a missing file naming the command; loader shapes, split sizes,
the resume slice equals the unsliced stream, all three sampling kinds; both model kinds
and both losses build, unknown `kind` rejected; a 3-step run writes `config.yaml`,
`metadata.json`, checkpoints, and `load_model`/`load_run` return the final state; a
resumed run reaches `num_steps` and consumes the right batches; an autoencoder run with
the Yawata switch. `test_training.py`: final-step checkpointing with `save_every=2` and 3
steps (`[2, 3]` or `[3]`), and without `evaluate` (`[4]`).

The full suite is green for the first time since Phase B: 471 passed, 1 xfail.

## Open issues

- D5: the notebook still constructs its own objects; Joon moves it onto `hydra.compose`
  + the builders as Phase E runs begin.
- CI has not run on GitHub yet (no `.github/` existed); the first push of this branch
  will show whether `uvx ty` and the `ci` Hypothesis profile fit the 45-minute budget.
- `configs/data/*.yaml` carry `out_dir: data`; datasets are expected under the
  repository's `data/` (gitignored). A shared cluster location is a config override.
- `wandb.group` for single runs is `null`; only multirun sets it.

## Review notes

| Change | Thoughts | Modifications |
|---|---|---|
| `configs/train.yaml` and the `model/`, `loss/`, `schedule/`, `optimizer/`, `wandb/` groups | Confirmed. Whether and how these will need to be modified will clarify as experiments progress. | None |
| `src/deep_isochron/experiment/instantiate.py`: `_target_` instantiation with lists → tuples | Confirmed. | None |
| `src/deep_isochron/experiment/data.py`: `dataset_file`, `build_source`, `generate_dataset`, `reference_normal_form`, `build_window_source`, `build_loaders`, `Loaders` | Had a lookthrough. Looks good. | None |
| `src/deep_isochron/experiment/model.py`: `build_inn`, `build_model` | Looks good. | None |
| `src/deep_isochron/experiment/training.py`: `build_loss`/`build_schedule`/`build_optimizer`/`build_trainer`, `resolved` | What is the metadata that is saved to Wandb and the checkpoint via `resolved`? Looks like perhaps the model configs are not going in, which may hinder comparisons between the dmoels in Wandb. | Deferred to discussions with Claude. |
| `src/deep_isochron/experiment/logging.py`: `MultiLogger`, `EpochLogger`, `build_logger` | `MultiLogger` and perhaps `EpochLogger` look like the should belong in `training/loggers.py` | Deferred to Claude for restructuring / need to discuss to determine where is more appropriate for `EpochLogger`. |
| `src/deep_isochron/experiment/run.py`: `configure_jax`, `train`, `load_run`, `load_model` | Looks okay, but isn't there git_sha related functionality in `deep_isochron/data/generate.py` (`_git_state` function)? Overall, need to make sure `experiment` subpackage is only dealing with config parsing and object instantiation. Any lower level functionalities should be in the other relevant modules. | Deferred to discussion with Claude regarding the scope of the `experiment` module. |
| `scripts/train.py` (new), `scripts/generate_data.py` (thin), `scripts/bench_dataloader.py --config` | Looks good. But now with the increasing number of scripts, we need a Readme.md in the /scripts directory detailing what each script does and how to use them. | Deferred to Claude for readme generation. |
| `src/deep_isochron/data/generate.py`: `generation_metadata`; `data/windows/batched.py`: `mixed_ranges` | Looks good. | None |
| `src/deep_isochron/model/invertible/splines/*`: `xy_range: Sequence[float]` | Simple change to appease the type checker. | None |
| `src/deep_isochron/training/checkpoint.py` (`save -> bool`, `force`, `latest_step`, `custom_metadata`), `training/trainer.py` (final step forced) | Refine documentation of the `Checkpointer` class regarding `force`. Does it force saving for all save steps, or just the final save. If the latter, it may be better to rename it to something like `(force_)save_final`.| Deferred to discussions with Claude. |
| `tests/strategies.py`: `domain_points` off the regularization radius (fixes `circular_rq`) | Confirmed. | None |
| `tests/test_experiment.py` (new), `tests/test_training.py` (+1) | Looks good, but the `compose` function may be useful for notebook execution as well. If so, move to under `experiments`: `experiments/utils.py` for example?| None |
| `.github/workflows/ci.yml`, `.gitignore` | Confirmed | None |
| ADR-0011; `docs/architecture.md`; `docs/index.md`; `docs/roadmap.md` (Phase D, ledger); design doc §7 | Looks good. Will later need to be expanded on how to also run experiments in Jupyter notebook (for quick prototyping runs) | None |

### Review round 1

| Change | Thoughts | Modifications |
|---|---|---|
| `src/deep_isochron/training/loggers.py`: `MultiLogger`, `EpochLogger` (from `experiment/logging.py`); `training/__init__.py` exports | Confirmed. Looks good. Not related to this change, but it seems that `WandbLogger` receives a wandb run. Would it be worth adding additional initialization strategy that either runs `wandb.init` or reuse `run` depending on what is given? This would allow the trainer to manage wandb log outputs and the checkpoints in the same directory (or do we not want this?) | None |
| `src/deep_isochron/provenance.py` (new): `git_state`, `package_version`; `data/generate.py` and `experiment/run.py` use it (`metadata.json["git"]`) | Confirmed. Unrelated to this change, but maybe the top level `misc.py` belongs more in `models`? Or it is better to create a `utils` or `misc` submodule and place `provenance.py` and `misc.py` (maybe change name to something more descriptive... `math.py`? ) there? | Deferred to discussions with Claude. |
| `src/deep_isochron/experiment/compose.py` (new): `compose(*overrides)`, `CONFIG_DIR`; `tests/test_experiment.py` uses it | Confirmed. | None |
| `src/deep_isochron/training/checkpoint.py`: `force` semantics documented (per call; the trainer's final step) | Looks good. | None |
| `scripts/README.md` (new) | Like the readme. Good. | None |
| `tests/test_experiment.py`: metadata holds the whole config (model, loss) | Looks good. | None |
| ADR-0011 §2 scope rule; `docs/architecture.md` (tree); this document | Looks good. | None |
| `prototype.ipynb` | Revised to test out the developed machinery, confirmed that training runs in the notebook with `PrintLogger`| None |
| Additional comments | <ul><li>Currently, the dataloader owns the total number of training steps. Is it possible to train and save a model using n steps, then load the model from checkpoint, then train the model further with a new dataloader for m steps? (Should be possible). This is not the same as running the model for (n+m) steps, though correct? <li> Is it better to log Wandb and orbax checkpoint in the same directory, or should log dir and ckpt dir be separate? <li> Currently, `DelayedLogger` allows the accelerator to run ahead. However, the `Evaluator` doesn't have a functionality like that, so in practice, the training loop cannot run ahead, correct? Can this be remedied? <li>The validation trajectories should be sampled similar on and off the limit cycle; otherwise the saving will be biased towards getting the limit cycle correct only. Potential remedies. <li>The rotation direction of the trajectories (clock/counter-clockwise) is needed to properly initialize `w` and `w0` of the normal forms. Need algorithm to determine the winding direction from data. This algorithm would belong in the `analysis` submodule. <ul> | Deferred to further discussions with Claude |

### Review round 2

| Change | Thoughts | Modifications |
|---|---|---|
| `src/deep_isochron/utils/numerics.py` (from `misc.py`), `utils/provenance.py` (from `provenance.py`), `utils/__init__.py`; imports in `constraints.py`, `normal_forms/base.py`, `data/generate.py`, `experiment/run.py`, `tests/test_constraints.py` | Looks good. | None |
| `src/deep_isochron/training/evaluation.py`: `prediction_sums`; `Evaluator(t_split)` → `val/mse_early`, `val/mse_late`, `val/n_early`, `val/n_late` | Looks good. | None |
| `src/deep_isochron/experiment/data.py`: `validation_t_split`; `experiment/run.py` passes it; `configs/train.yaml`: `validation.t_split` | Looks good. | None |
| `docs/design/validation-split.md` (new); ADR-0009 §5 amended; `docs/index.md`; `docs/architecture.md`; roadmap Phase E items | Looks good for now. | None |
| `tests/test_training.py`: `test_evaluator_splits_the_prediction_error_by_window_start`; `tests/test_experiment.py`: `test_validation_split_follows_the_mixed_sampler` | Looks good. | None |
