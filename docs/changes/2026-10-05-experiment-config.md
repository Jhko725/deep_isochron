---
type: change
status: implemented (D1–D4, D6–D8); review pending; D5 (notebook) is Joon's
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
- `src/deep_isochron/experiment/logging.py` — `MultiLogger`, `EpochLogger` (adds
  `epoch`), `build_logger` (PrintLogger + optional `WandbLogger`, wrapped in
  `DelayedLogger`).
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
| `configs/train.yaml` and the `model/`, `loss/`, `schedule/`, `optimizer/`, `wandb/` groups | | |
| `src/deep_isochron/experiment/instantiate.py`: `_target_` instantiation with lists → tuples | | |
| `src/deep_isochron/experiment/data.py`: `dataset_file`, `build_source`, `generate_dataset`, `reference_normal_form`, `build_window_source`, `build_loaders`, `Loaders` | | |
| `src/deep_isochron/experiment/model.py`: `build_inn`, `build_model` | | |
| `src/deep_isochron/experiment/training.py`: `build_loss`/`build_schedule`/`build_optimizer`/`build_trainer`, `resolved` | | |
| `src/deep_isochron/experiment/logging.py`: `MultiLogger`, `EpochLogger`, `build_logger` | | |
| `src/deep_isochron/experiment/run.py`: `configure_jax`, `train`, `load_run`, `load_model` | | |
| `scripts/train.py` (new), `scripts/generate_data.py` (thin), `scripts/bench_dataloader.py --config` | | |
| `src/deep_isochron/data/generate.py`: `generation_metadata`; `data/windows/batched.py`: `mixed_ranges` | | |
| `src/deep_isochron/model/invertible/splines/*`: `xy_range: Sequence[float]` | | |
| `src/deep_isochron/training/checkpoint.py` (`save -> bool`, `force`, `latest_step`, `custom_metadata`), `training/trainer.py` (final step forced) | | |
| `tests/strategies.py`: `domain_points` off the regularization radius (fixes `circular_rq`) | | |
| `tests/test_experiment.py` (new), `tests/test_training.py` (+1) | | |
| `.github/workflows/ci.yml`, `.gitignore` | | |
| ADR-0011; `docs/architecture.md`; `docs/index.md`; `docs/roadmap.md` (Phase D, ledger); design doc §7 | | |
