---
type: roadmap
status: current
updated: 2026-10-06
sources: [joon, claude]
---

# Roadmap

The current plan for `deep_isochron`. This file is a statement of intent and is kept
*correct now*: finished work moves to the **Done** ledger at the bottom (one line per item,
linking the branch's change document; chronological — new rows are appended at the end), parked work is listed with its reason, and the
history is `git log -p docs/roadmap.md`. It is updated in the same commit as the work that
changes it, never on its own. Decisions live in `docs/decisions/`; branch-level detail in
`docs/changes/`.

Order of the phases was set 2026-09-29: invertible package → data generation → trainer →
Hydra/wandb/Orbax → science.

---

## Phase A — `invertible-cleanup` (merged 2026-10-02)

All items in the Done ledger; see `docs/changes/2026-10-02-invertible-cleanup.md`.

## Phase B — `data-generation` (current; B1–B12 landed; review round 3 applied 2026-10-04)

Decisions taken 2026-10-02 (ADR-0007, ADR-0008) and refined by the review of 2026-10-03.
B1–B7 landed; the review (change document, *Review notes*) set the remaining items, in
this order:

| # | Item | Why | Done when |
|---|---|---|---|
| B8 | Mechanical review fixes: single `flow -> diffrax.Solution`; `strategy` → `integration` (`normal_forms/integration.py`, `AbstractFlowIntegration`); `systems/normal_forms/{base,hopf,bautin}.py`; `AbstractODE.params()` contract; `GreaterThan(lower, at_zero=lower+1)` with `Positive` as alias; `test_normal_forms.py` split from `test_systems.py` with Langfield et al. cited | review of `6c0fdc2` | *done 2026-10-03* |
| B9 | Data layer as grain transforms: whole-trajectory frozen-dataclass source (`copy.replace`), `RandomWindow` / `WeightedWindow`, `mixed_windows`; grouped `DatasetMetadata`; `test_data.py` rewritten; ADR-0008 amended | review: idiomatic grain; organised metadata | *done 2026-10-03* |
| B10 | `docs/design/normal-forms.md` — **agreed 2026-10-03** (Joon's revision merging the B10 draft with his derivation notes): `Θ`/`Ψ` notation, `κ = ρ'(1)`, Wilson–Moehlis normalization `∂ᵣΨ(1) = 1`, Option B (`_sq` hooks), `ClosedFormIntegration` + `(Θ, Ψ)` chart in B11, log-polar member deferred. Amended 2026-10-03: §5.1 corollary sign fixed (confirmed); §5.3 rewritten from the paper and §5.4 Kvalheim & Revzen added — both *proposed*, pending Joon | review: `r`/`s` inconsistency; math laid down before implementation | *done 2026-10-03* |
| B11 | `AbstractNormalForm` reimplemented against the design document §9 (`_sq` hooks, API in `r`, `κ = ρ'(1)`, `Ψ` normalized, `eigenvalues_origin`, `(Θ, Ψ)` chart with differentiable `Ψ⁻¹`, `ClosedFormIntegration`); tests per §10; ADR-0007 amended | — | *done 2026-10-03* |
| B12 | Yawata et al. (Chaos 34, 063111, 2024) baseline per `docs/design/normal-forms.md` §5.3: `PhaseAmplitudeLatentDynamics(omega, kappa)` (replaces `LinearLatentDynamics`), `PhaseAmplitudeAutoencoder` with unit-circle normalization, `phase`/`phase_sensitivity`; `PhaseAutoencoderLoss` (Eqs. (21)–(26), `α_k` and weight schedules); `OnCycleGaussian` sampler; `tests/test_baseline.py` against the exact chart + a `slow` training test on Hopf (Θ asserted up to a constant; Ψ, κ reported) | the paper's baseline; §5.4 says what it learns | *done 2026-10-03* |

## Phase C — `trainer` (current; C1–C9 landed 2026-10-05; review round 2 applied, C9 under review)

Decisions taken with Joon on 2026-10-04 (ADR-0009): step-dependent loss weights for
curricula; the one-step-delayed logging kept as a `DelayedLogger` adapter (JAX cookbook
pattern; see the change document for the measurements); evaluation on a validation
loader with `val/mse` as the checkpoint metric and the dataset's bounding box as the
diagnostic grid; FHN reports prediction error and learned period until Phase E;
checkpoint directories owned by the caller (Hydra in D3); `scripts/training/` deleted.

| # | Item | Done when |
|---|---|---|
| C1 | `Trainer(optimizer, loss, schedule)`; `TrainerState` with schedule state; pure jitted `train_step`; `train(…, logger, checkpointer, evaluate, eval_every)`; `Logger`/`Checkpointer` base classes, `Null`/`List`/`Print`/`Wandb` loggers, `DelayedLogger` | *done 2026-10-04* |
| C2 | Losses as weighted named terms (`AbstractLoss`, building blocks); weights from the trainer's schedule (`Constant`, `StepSchedule`, `ThresholdSwitch`) | *done 2026-10-04* |
| C3 | `Evaluator(val_batches, reference)`: `val/mse`, `val/final_mse`, `period`, `kappa`, normal-form params, bijection round trip and Jacobian singular values on the bounding-box grid, phase circular std and amplitude correlation against a reference normal form | *done 2026-10-04* |
| C4 | `OrbaxCheckpointer(directory, save_every, metric, custom_metadata)`: whole `TrainerState`, `FixedInterval` + `Any([LatestN(1), BestN])`, `restore(template, step)`, resume | *done 2026-10-04* |
| C5 | `tests/test_training.py` (15 tests); the slow B12 test runs through the trainer | *done 2026-10-04* |
| C6 | `scripts/training/` deleted; notebook on the new API; ADR-0009; architecture | *done 2026-10-04* |
| C7 | Data-pipeline prefetch. `data.to_device(dataset, device)` is the training-run path; **found**: grain's default 16 reader threads (bare `iter(MapDataset)` / default `to_iter_dataset()`) contend with the loop for the GIL — 6× slower dispatch on CPU; fixed with a single reader (`single_threaded`). V100 (2026-10-05): `to_device` 94 ms/step vs 92 device-resident, dispatch floors 0.8 ms for committed/uncommitted/NumPy (no state commit needed); NumPy-in-loop 488, `mp_prefetch(4)+to_device` 464 — both rejected | *done 2026-10-05* |
| C8 | The step was host-bound on the V100 (88 ms at batch 512, dispatch ≈ total): `bench_dataloader.py --hlo-stats` found ≈ 1 700 conditionals per step from `jax.scipy.linalg.expm`'s 16-step scan of `lax.cond` in `BiLipschitzLinear` (0.649 ms per 2×2 call measured; each conditional a device-to-host round trip on GPU). **Fixed and confirmed**: Cayley transforms (ADR-0010) — 0 conditionals, step 88 → 33 ms. Remaining step: ≈ 20 ms fixed (≈ 1 900 launches) + ≈ 10.7 ms device work per 512 windows (31 ms at 512, 63 at 2 048); the root solve's 112 serialized iterations per point are the lever for both — fewer iterations / `unroll`, Phase D | *done 2026-10-05* |
| C9 | The fetch was the bound after C8 (grain per-element pipeline: 89 ms per batch of 512 on the V100 host vs a 33 ms step). **Done**: `WindowBatchSource` — a `RandomAccessDataSource` whose element is a whole batch, one vectorized gather (0.6–0.9 ms per batch on the dev CPU vs 50), epochs of every window once with a fresh permutation each, concrete `len`, resumable by slicing; `window_batches`/`mixed_window_batches`; `Trainer.train(num_steps=None)` runs the finite loader to its end (ADR-0008 Decision 3). V100: fetch 88.5 → 0.7 ms per batch, `to_device` 32.6 ms per step vs 31.0 device-resident | *done 2026-10-05* |

## Phase D — `experiment-config` (current; D1–D4, D6–D8 landed 2026-10-05; D5 Joon's)

Decisions taken with Joon on 2026-10-05 (ADR-0011): YAML + `_target_` configs (no
dataclass schema for now); builders in `deep_isochron.experiment`; `PrintLogger` default
with wandb `online`/`offline` as config groups; `num_steps` as the run's unit with epochs
logged; training never generates data (the file must exist); the pre-existing red test
fixed by keeping bijection-law points off the regularization radius instead of `xfail`.

| # | Item | Done when |
|---|---|---|
| D1 | Config layout: `configs/train.yaml` composing `data/` (the generation configs), `model/{conjugacy,autoencoder}`, `loss/`, `schedule/{constant,yawata}`, `optimizer/adam`, `wandb/{off,online,offline}`; `windows.sampling` uniform / weighted / mixed; `validation`; `checkpoint`; `resume` | *done 2026-10-05* |
| D2 | Builders: `experiment.instantiate` (lists → tuples), `build_source` (file by generation hash, no generation), `build_window_source`/`build_loaders` (resume slice), `build_model` (`_target_` layers + keys + `flip`), `build_loss`/`build_schedule`/`build_optimizer`/`build_trainer`, `build_logger` | *done 2026-10-05* |
| D3 | `scripts/train.py`: Hydra run dir, `resolve_device`, x64, config/metadata/checkpoints in the run dir, final step always checkpointed, `resume=` continues the same stream, multirun group for wandb | *done 2026-10-05* |
| D4 | `load_run(run_dir, step)` / `load_model` from the checkpoint's `custom_metadata` (config) | *done 2026-10-05* |
| D5 | Notebook on `hydra.compose` + the builders (Joon, as Phase E runs start) | pending |
| D6 | CI: `.github/workflows/ci.yml` — ruff, ty, `check_md_math`, `pytest --hypothesis-profile=ci -n 4 -m "not slow"`; `slow` job + 20-step end-to-end `train.py` run; the `circular_rq (K=8)` failure fixed (test domain off the `eps_r` disc) | *done 2026-10-05* (first green run on GitHub pending) |
| D7 | `scripts/bench_dataloader.py --config <overrides>` benchmarks a training config's own pipeline, model and loss | *done 2026-10-05* |
| D8 | ADR-0011; `docs/architecture.md` (`experiment/`, scripts, config→run data flow); change document; design doc §7 | *done 2026-10-05* |

## Phase E — science

First `analysis` items, from the Phase D review (2026-10-06): the winding direction of the
data's rotation (sign of the mean signed angular velocity about the cycle's centroid over
the late trajectory segments) to initialize the signs of `w`, `w0` in `build_model`; a
far-from-cycle validation dataset (`UniformAnnulus` initial conditions, its own `data`
config) complementing the `val/mse_early` split (design document `validation-split.md`).

`deep_isochron.analysis` (namespace reserved; algorithms chosen as the research dictates):
numerical limit cycle, monodromy/Floquet exponents, asymptotic phase by long integration,
isochrons by the BVP continuation of Langfield, Krauskopf & Osinga (2014) — the ground
truth for the learned FitzHugh–Nagumo isochrons; `t_settle` per trajectory in the dataset
once the normal-form `amplitude` or these tools provide it; `floquet_multipliers(ode,
cycle)` by the variational equation over one period — the computation
`test_hopf_floquet_multiplier_is_the_polar_monodromy` does by hand for Hopf (review round
3), and the FHN cycle points for `OnCycleGaussian`. Evaluation works through
`AbstractPhaseAmplitudeModel` (phase up to a constant, amplitude up to a fitted scale). Pushforward loss with oversampling near the repelling slow manifold; curvature-matching
(Hessian) loss; INN depth 8–12; Jacobian-anisotropy diagnostics (`diagnostics/`); endpoint
handling on `AbstractSpline` (periodic / free boundary derivatives, needed for a C¹
`CircularMonotonicRQCoupling`). As `LossConfig`/`INNConfig` entries once Phase D is in.

## Parked (with reason)

- Optimistix for `radius_from_isostable` (and other hand-rolled solves) — dropped for now
  (review round 3, 2026-10-04): the bracketed Newton with its implicit-function JVP is
  tested and small; revisit if a second root solve appears or the dependency is needed
  elsewhere.
- `TimeSeriesDataSource` as an `eqx.Module` instead of a frozen dataclass — pending
  Joon's call (review round 2); no technical gain identified (it is an I/O object never
  passed through a JAX transform), uniformity is the argument for.

- Reinstating `InvertibleLinear` — resolved 2026-10-05: `BiLipschitzLinear`'s slowness was
  `expm` (ADR-0010); with Cayley transforms it is loop-free, so there is no speed case for
  the unconstrained layer. It was markedly cheaper to evaluate, and `det W` crossing zero was
  never observed in practice; the price would be one test-exception group
  (`ORIENTATION_NOT_GUARANTEED`) coming back.
- Phase-autoencoder baseline refinements — batch normalization in the MLPs, the paper's
  once-and-for-all weight switch and per-epoch `α_k` (Phase C's trainer owns schedules),
  input standardisation, the FHN cycle for `OnCycleGaussian` (needs the numerical limit
  cycle of Phase E) — when the baseline is run for the paper.
- Grouping the scalar bijections (`affine.py`, `analytic.py`, `splines/`, `OffsetedBijection`)
  under one subpackage — cosmetic; when the vocabulary stops growing.

- Initialisation-strategy enum on `AbstractBijection` — only if training dynamics call for it
  (identity-at-init is universal; `BiLipschitzLinear.init="rotation"` is the one opt-in).
- Merging `ScalarChain` back into `SequentialINN` — only if the overlap grows.
- `_Shifted` as a mixin rather than a base — not worth the effort now.
- `at_zero` as a declared attribute on `Constraint` — not uniform (`Widths` maps 0 to a vector;
  `Free`/`Arcsinh` to 0); `is_constrained` covers the checkable half of the contract.

---

## Done

| When | Branch | What | Record |
|---|---|---|---|
| 2026-09-30 | `splines-refactor` | `AbstractSpline`; `CubicBSpline` ported; `LinearSpline`; RQ-spline bug fixes; `BijectionFactory` removed; first Hypothesis suite; change-document convention | `docs/changes/2026-09-30-splines-refactor.md` |
| 2026-10-01 | `scalar-param-refactor` | `raw`/`constrain` contract (ADR-0001/0002); declared `smoothness` (ADR-0003); `AbstractVar` as static `init=False` fields (ADR-0004); constraint primitives (ADR-0005); `ScalarChain`; `CouplingFlow` with pluggable conditioner; `PolarCouplingFlow`; circular spline as exact rotation; `InvertibleLinear` rotation init; `BiLipschitzLinear._s` vector; 303-test suite; `ty` clean on the invertible package | `docs/changes/2026-10-01-scalar-param-refactor.md`, ADRs 0001–0005 |
| 2026-10-02 | `invertible-cleanup` | A1: `Shift`/`Affine` scalar templates; `AffineCoupling`/`ResidualCoupling` as `CouplingFlow` factories (the latter now identity at init) | `docs/changes/2026-10-02-invertible-cleanup.md` |
| 2026-10-02 | `invertible-cleanup` | A2–A4: `BiLipschitzLinear` identity at init (`init="rotation"` opt-in), raw leaves + `LinearParams`; `InvertibleLinear` removed — identity-at-init and orientation are universal laws, `IDENTITY_AT_INIT`/`ORIENTATION_NOT_GUARANTEED` deleted | `docs/changes/2026-10-02-invertible-cleanup.md` |
| 2026-10-02 | `invertible-cleanup` | A5–A7: `AbstractScalarBijection` implementation checklist; ADR-0006 + `docs/design/cubic-bspline.md` (corrects the inverse error-bound claim); `docs/architecture.md` | `docs/changes/2026-10-02-invertible-cleanup.md` |
| 2026-10-02 | `invertible-cleanup` | A8–A9: `systems/` on ADR-0004; all `ty: ignore`s gone (`cast`s at the optax/orbax boundaries); `training` import bug, empty-loader crash and `HopfLatentDynamics` call fixed; `matplotlib` → dev group | `docs/changes/2026-10-02-invertible-cleanup.md` |
| 2026-10-02 | `data-generation` | B1–B7: `AbstractODE.flow` + `SolverConfig`; `AbstractNormalForm` (closed-form phase, isostable, isochrons, chart); flow strategies (cartesian / polar / `r²`); Hopf/Bautin rewritten on two rates with constrained leaves; `HopfLatentDynamics` deleted; xarray/netCDF `TimeSeriesDataSource` + `DatasetMetadata`; `generate` with loud failures and `config_hash`; `split_time`/`split_trajectories`; `weighted_windows` + `mixed_split`; `scripts/generate_data.py` + configs; `test_systems.py`, `test_data.py` | `docs/changes/2026-10-02-data-generation.md`, ADR-0007/0008 |
| 2026-10-03 | `data-generation` | B8: review round 1 — `flow -> Solution`; `integration` naming; `normal_forms/` subpackage; `params()`; `GreaterThan`; tests split, Langfield et al. cited (FHN equilibrium, eigenvalues, period) | `docs/changes/2026-10-02-data-generation.md` |
| 2026-10-03 | `data-generation` | B9: whole-trajectory `TimeSeriesDataSource` (frozen dataclass); `RandomWindow`/`WeightedWindow` grain transforms, `windows`, `mixed_windows`; grouped `DatasetMetadata` (`system/sampling/grid/solve/provenance/extra`); dict batches | `docs/changes/2026-10-02-data-generation.md`, ADR-0008 |
| 2026-10-03 | `data-generation` | B10–B11: `docs/design/normal-forms.md` agreed; normal forms reimplemented against its §9 (`log_growth_rate_sq`/`angular_rate_sq` hooks, public API in `r`, `κ = ρ'(1)`, `∂ᵣΨ(1) = 1`, `eigenvalues_origin`, `to/from_phase_amplitude` with a differentiable root solve, `ClosedFormIntegration`); `test_normal_forms.py` per §10 | `docs/changes/2026-10-02-data-generation.md`, `docs/design/normal-forms.md`, ADR-0007 |
| 2026-10-03 | `data-generation` | B12: Yawata et al. phase-autoencoder baseline — `PhaseAmplitudeLatentDynamics`, `PhaseAmplitudeAutoencoder` (`phase`, `phase_sensitivity`), `PhaseAutoencoderLoss`, `OnCycleGaussian`; `LinearLatentDynamics` removed; `tests/test_baseline.py` | `docs/changes/2026-10-02-data-generation.md`, `docs/design/normal-forms.md` §5.3 |
| 2026-10-04 | `data-generation` | Review round 3: `to_polar`/`from_polar`; `_log_growth_rate_sq`/`_angular_rate_sq`; `default_integration = "closed_form"`; `tests/helpers.SOLVERS` (Tsit5); `categorical`; `AbstractPhaseAmplitudeModel` protocol implemented by both models; design-doc inline math unwrapped; OKF-style frontmatter + `docs/index.md` | `docs/changes/2026-10-02-data-generation.md` |
| 2026-10-04 | `data-generation` | Review round 4: design-document math made GitHub-safe (standalone `$$` blocks; no backslash-punctuation escapes, `*`, `\\` or emphasis-forming `_` inside inline math) with `scripts/check_md_math.py`; Done ledger chronological; ty-unreachable asserts fixed | `docs/changes/2026-10-02-data-generation.md` |
| 2026-10-04 | `trainer` | C1–C6: `Trainer`/`TrainerState` with injected `Logger`/`Checkpointer` and `DelayedLogger`; `AbstractLoss` weighted terms + building blocks; schedules (`Constant`, `StepSchedule`, `ThresholdSwitch`) in the state; `Evaluator`; `OrbaxCheckpointer` (whole state, resume); `test_training.py`; `scripts/training/` deleted | `docs/changes/2026-10-04-trainer.md`, ADR-0009 |
| 2026-10-04 | `trainer` | Review round 1: loader-side transfer (`data.to_device`), `validation_windows` + `Evaluator` over a finite dataset with streamed statistics, logger/checkpointer context managers, loss `terms`/`aux` split, typed schedule state, `circular_std` rationale, American English | `docs/changes/2026-10-04-trainer.md`, ADR-0009 |
| 2026-10-05 | `trainer` | Review round 2: `Logger`/`Checkpointer` base classes; grain single-reader rule (`single_threaded`, `to_device(dataset, device)`); lazy `_shift` (no JAX at import); `scripts/bench_dataloader.py` redesigned; C7 closed on V100 numbers (`to_device`; `mp_prefetch` rejected; no state commit) | `docs/changes/2026-10-04-trainer.md`, ADR-0009 |
| 2026-10-05 | `trainer` | C8: `BiLipschitzLinear` rotations by Cayley transform instead of `expm` — the V100 step's ≈ 1 700 per-step conditionals (device-to-host round trips) removed; `cayley`, two tests, ADR-0010 | `docs/changes/2026-10-04-trainer.md`, ADR-0010 |
| 2026-10-05 | `trainer` | C9: `WindowBatchSource` / `window_batches` / `mixed_window_batches` — batched window source for training runs (epochs of every window once, one gather per batch, finite loader); `Trainer.train(num_steps=None)`; `resolve_device`; ADR-0008 Decision 3 | `docs/changes/2026-10-04-trainer.md`, ADR-0008 |
| 2026-10-05 | `experiment-config` | D1–D4, D6–D8: `configs/train.yaml` and groups; `deep_isochron.experiment` (`instantiate`, builders, `run.train`/`load_run`/`load_model`); `scripts/train.py`; forced final checkpoint; `generation_metadata`; CI workflow; benchmark on configs; `circular_rq` test domain fixed; ADR-0011 | `docs/changes/2026-10-05-experiment-config.md`, ADR-0011 |
