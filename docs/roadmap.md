---
type: roadmap
status: current
updated: 2026-10-04
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

## Phase C — `trainer` (current; C1–C6 landed 2026-10-04; review round 1 applied)

Decisions taken with Joon on 2026-10-04 (ADR-0009): step-dependent loss weights for
curricula; the one-step-delayed logging kept as a `DelayedLogger` adapter (JAX cookbook
pattern; see the change document for the measurements); evaluation on a validation
loader with `val/mse` as the checkpoint metric and the dataset's bounding box as the
diagnostic grid; FHN reports prediction error and learned period until Phase E;
checkpoint directories owned by the caller (Hydra in D3); `scripts/training/` deleted.

| # | Item | Done when |
|---|---|---|
| C1 | `Trainer(optimizer, loss, schedule)`; `TrainerState` with schedule state; pure jitted `train_step`; `train(…, logger, checkpointer, evaluate, eval_every)`; `Logger`/`Checkpointer` protocols, `Null`/`List`/`Print`/`Wandb` loggers, `DelayedLogger` | *done 2026-10-04* |
| C2 | Losses as weighted named terms (`AbstractLoss`, building blocks); weights from the trainer's schedule (`Constant`, `StepSchedule`, `ThresholdSwitch`) | *done 2026-10-04* |
| C3 | `Evaluator(val_batches, reference)`: `val/mse`, `val/final_mse`, `period`, `kappa`, normal-form params, bijection round trip and Jacobian singular values on the bounding-box grid, phase circular std and amplitude correlation against a reference normal form | *done 2026-10-04* |
| C4 | `OrbaxCheckpointer(directory, save_every, metric, custom_metadata)`: whole `TrainerState`, `FixedInterval` + `Any([LatestN(1), BestN])`, `restore(template, step)`, resume | *done 2026-10-04* |
| C5 | `tests/test_training.py` (15 tests); the slow B12 test runs through the trainer | *done 2026-10-04* |
| C6 | `scripts/training/` deleted; notebook on the new API; ADR-0009; architecture | *done 2026-10-04* |
| C7 | Data-pipeline prefetch: `data.to_device` (grain's two-stage prefetch, the tutorial's recommended pattern) is the training-run path — review round 1. Remaining: check that `TimeSeriesDataSource` pickles to `mp_prefetch` workers on the GPU machine (it failed in the dev container), for the case where a step is shorter than the 15 ms window slicing | `to_device` *done 2026-10-04*; `mp_prefetch` pending |

## Phase D — `experiment-config`

- D1 Structured configs (`config.py` dataclasses + `ConfigStore`); `seed: int` in config, keys
  split in `build_model(cfg, key)`.
- D2 `build_model`/`build_data`/`build_optimizer`; coupling registry replacing the notebook's
  `make_invertible_block` variants.
- D3 `scripts/train.py`: Hydra-owned run dir, `OmegaConf.to_container(resolve=True,
  throw_on_missing=True)` → `wandb.init(config=…, dir=out_dir)`, Orbax dir inside, `run.save(
  .hydra/config.yaml)`, `wandb.group` for multirun, `configs/wandb/offline.yaml`.
- D4 `load_model(run_dir, step)` from `root_metadata().custom_metadata`.
- D5 Notebook via `hydra.compose` + the same `build_*` functions.
- D6 CI: `uv sync --group dev`, `pytest --hypothesis-profile=ci -n 4 -m "not slow"`, separate
  `slow` job with a 20-step end-to-end smoke test.

## Phase E — science

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

- Reinstating `InvertibleLinear` — only if `BiLipschitzLinear` stays too slow after trying
  parametrizations of `SO(dim)` other than the matrix exponential (Cayley transform,
  Householder products). It was markedly cheaper to evaluate, and `det W` crossing zero was
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
