# Roadmap

The current plan for `deep_isochron`. This file is a statement of intent and is kept
*correct now*: finished work moves to the **Done** ledger at the bottom (one line per item,
linking the branch's change document), parked work is listed with its reason, and the
history is `git log -p docs/roadmap.md`. It is updated in the same commit as the work that
changes it, never on its own. Decisions live in `docs/decisions/`; branch-level detail in
`docs/changes/`.

Order of the phases was set 2026-09-29: invertible package → data generation → trainer →
Hydra/wandb/Orbax → science.

---

## Phase A — `invertible-cleanup` (merged 2026-10-02)

All items in the Done ledger; see `docs/changes/2026-10-02-invertible-cleanup.md`.

## Phase B — `data-generation` (current)

Decisions taken 2026-10-02 (ADR-0007, ADR-0008) and refined by the review of 2026-10-03.
B1–B7 landed; the review (change document, *Review notes*) set the remaining items, in
this order:

| # | Item | Why | Done when |
|---|---|---|---|
| B8 | Mechanical review fixes: single `flow -> diffrax.Solution`; `strategy` → `integration` (`normal_forms/integration.py`, `AbstractFlowIntegration`); `systems/normal_forms/{base,hopf,bautin}.py`; `AbstractODE.params()` contract; `GreaterThan(lower, at_zero=lower+1)` with `Positive` as alias; `test_normal_forms.py` split from `test_systems.py` with Langfield et al. cited | review of `6c0fdc2` | *done 2026-10-03* |
| B9 | Data layer as grain transforms: whole-trajectory frozen-dataclass source (`copy.replace`), `RandomWindow` / `WeightedWindow`, `mixed_windows`; grouped `DatasetMetadata`; `test_data.py` rewritten; ADR-0008 amended | review: idiomatic grain; organised metadata | *done 2026-10-03* |
| B10 | `docs/design/normal-forms.md` — **agreed 2026-10-03** (Joon's revision merging the B10 draft with his derivation notes): `Θ`/`Ψ` notation, `κ = ρ'(1)`, Wilson–Moehlis normalisation `∂ᵣΨ(1) = 1`, Option B (`_sq` hooks), `ClosedFormIntegration` + `(Θ, Ψ)` chart in B11, log-polar member deferred. Pending minor correction: §5.1 corollary sign of `A` | review: `r`/`s` inconsistency; math laid down before implementation | *done* |
| B11 | Reimplement `AbstractNormalForm`'s analytic methods against B10; ADR-0007 amended | — | tests unchanged in intent, API in `r` |
| B12 | Yawata et al. baseline, *Phase autoencoder for limit-cycle oscillators*, Chaos 34, 063111 (2024): a two-variable latent with constant phase velocity and exponentially decaying amplitude, i.e. the autoencoder learns `(φ, ψ)` directly; then retire `LinearLatentDynamics` | the paper's baseline | module + test; `LinearLatentDynamics` removed |

## Phase C — `trainer`

- C1 `Trainer(optimizer, loss_fn)` + `train(model, loader, *, logger, checkpointer, num_steps,
  eval_every, eval_loader)`; `NullLogger`/`NullCheckpointer` for tests; document the
  one-step-delayed logging overlap.
- C2 Explicit loss weights in `ConjugacyTrajectoryLoss` (the `SolverConfig` field on
  `ConjugateLatentDynamics` landed with Phase B).
- C3 Held-out evaluation; physics scalars every `eval_every`: base-system `a, b, w`, implied
  period and Floquet exponent, max round-trip error, min/max per-layer Jacobian singular values
  on a fixed grid.
- C4 `training/checkpoint.py`: Orbax wrapper — `FixedIntervalPolicy` + `AnyPreservationPolicy(
  [BestN, LatestN(1)])`, `save_async`, whole `TrainerState` saved, `custom_metadata` = resolved
  config + dataset hash + `x64` flag; `restore(template)` via a `ShapeDtypeStruct` tree.
- C5 Tests: toy linear fit (`TrainerState` determinism, trainable filter, convergence,
  `StopIteration`); Orbax save → load → `tree_equal`; `jax_enable_x64` guard.

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
once the normal-form `amplitude` or these tools provide it. Pushforward loss with oversampling near the repelling slow manifold; curvature-matching
(Hessian) loss; INN depth 8–12; Jacobian-anisotropy diagnostics (`diagnostics/`); endpoint
handling on `AbstractSpline` (periodic / free boundary derivatives, needed for a C¹
`CircularMonotonicRQCoupling`). As `LossConfig`/`INNConfig` entries once Phase D is in.

## Parked (with reason)

- Reinstating `InvertibleLinear` — only if `BiLipschitzLinear` stays too slow after trying
  parametrisations of `SO(dim)` other than the matrix exponential (Cayley transform,
  Householder products). It was markedly cheaper to evaluate, and `det W` crossing zero was
  never observed in practice; the price would be one test-exception group
  (`ORIENTATION_NOT_GUARANTEED`) coming back.
- `AbstractLatentDynamics` / `PhaseAmplitudeAutoencoder` (non-invertible baseline) stay as
  they are until the baseline is needed in the paper; they consume an `AbstractNormalForm`
  through `flow` if ever adapted.
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
| 2026-10-03 | `data-generation` | B9: whole-trajectory `TimeSeriesDataSource` (frozen dataclass); `RandomWindow`/`WeightedWindow` grain transforms, `windows`, `mixed_windows`; grouped `DatasetMetadata` (`system/sampling/grid/solve/provenance/extra`); dict batches | `docs/changes/2026-10-02-data-generation.md`, ADR-0008 |
| 2026-10-03 | `data-generation` | B8: review round 1 — `flow -> Solution`; `integration` naming; `normal_forms/` subpackage; `params()`; `GreaterThan`; tests split, Langfield et al. cited (FHN equilibrium, eigenvalues, period) | `docs/changes/2026-10-02-data-generation.md` |
| 2026-10-02 | `data-generation` | B1–B7: `AbstractODE.flow` + `SolverConfig`; `AbstractNormalForm` (closed-form phase, isostable, isochrons, chart); flow strategies (cartesian / polar / `r²`); Hopf/Bautin rewritten on two rates with constrained leaves; `HopfLatentDynamics` deleted; xarray/netCDF `TimeSeriesDataSource` + `DatasetMetadata`; `generate` with loud failures and `config_hash`; `split_time`/`split_trajectories`; `weighted_windows` + `mixed_split`; `scripts/generate_data.py` + configs; `test_systems.py`, `test_data.py` | `docs/changes/2026-10-02-data-generation.md`, ADR-0007/0008 |
| 2026-10-02 | `invertible-cleanup` | A8–A9: `systems/` on ADR-0004; all `ty: ignore`s gone (`cast`s at the optax/orbax boundaries); `training` import bug, empty-loader crash and `HopfLatentDynamics` call fixed; `matplotlib` → dev group | `docs/changes/2026-10-02-invertible-cleanup.md` |
| 2026-10-02 | `invertible-cleanup` | A5–A7: `AbstractScalarBijection` implementation checklist; ADR-0006 + `docs/design/cubic-bspline.md` (corrects the inverse error-bound claim); `docs/architecture.md` | `docs/changes/2026-10-02-invertible-cleanup.md` |
| 2026-10-02 | `invertible-cleanup` | A2–A4: `BiLipschitzLinear` identity at init (`init="rotation"` opt-in), raw leaves + `LinearParams`; `InvertibleLinear` removed — identity-at-init and orientation are universal laws, `IDENTITY_AT_INIT`/`ORIENTATION_NOT_GUARANTEED` deleted | `docs/changes/2026-10-02-invertible-cleanup.md` |
