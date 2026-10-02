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

Decisions taken 2026-10-02 (ADR-0007, ADR-0008): `AbstractODE` / `AbstractNormalForm`
two-layer hierarchy with the rule "a class carries only what is analytically available,
everything numerical is a function over `AbstractODE`"; `SolverConfig`; flow strategies as
objects; one netCDF4 file per dataset via xarray; weighted-window sampling kept alongside
`mix`. B1–B7 landed in `docs/changes/2026-10-02-data-generation.md`.

| # | Item | Why | Done when |
|---|---|---|---|
| B8 | `deep_isochron/analysis/`: numerical counterparts of `AbstractNormalForm`'s closed forms for any `AbstractODE` — `find_limit_cycle` (Poincaré/shooting), `monodromy` → Floquet exponents, `asymptotic_phase` by long integration against the located cycle, later `isochrons` by the continuation method of Langfield, Krauskopf & Osinga (2014) | ground truth for the learned FHN isochrons; `t_settle` for sampling weights | functions + tests that recover Bautin's closed forms numerically |
| B9 | `t_settle` per trajectory stored in the dataset (from `amplitude` for normal forms, from B8 for observed systems); `transient_weights` optionally keyed on it | a physical basis for oversampling rather than wall-clock start time | variable in the file; sampler option |

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

Pushforward loss with oversampling near the repelling slow manifold; curvature-matching
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
| 2026-10-02 | `data-generation` | B1–B7: `AbstractODE.flow` + `SolverConfig`; `AbstractNormalForm` (closed-form phase, isostable, isochrons, chart); flow strategies (cartesian / polar / `r²`); Hopf/Bautin rewritten on two rates with constrained leaves; `HopfLatentDynamics` deleted; xarray/netCDF `TimeSeriesDataSource` + `DatasetMetadata`; `generate` with loud failures and `config_hash`; `split_time`/`split_trajectories`; `weighted_windows` + `mixed_split`; `scripts/generate_data.py` + configs; `test_systems.py`, `test_data.py` | `docs/changes/2026-10-02-data-generation.md`, ADR-0007/0008 |
| 2026-10-02 | `invertible-cleanup` | A8–A9: `systems/` on ADR-0004; all `ty: ignore`s gone (`cast`s at the optax/orbax boundaries); `training` import bug, empty-loader crash and `HopfLatentDynamics` call fixed; `matplotlib` → dev group | `docs/changes/2026-10-02-invertible-cleanup.md` |
| 2026-10-02 | `invertible-cleanup` | A5–A7: `AbstractScalarBijection` implementation checklist; ADR-0006 + `docs/design/cubic-bspline.md` (corrects the inverse error-bound claim); `docs/architecture.md` | `docs/changes/2026-10-02-invertible-cleanup.md` |
| 2026-10-02 | `invertible-cleanup` | A2–A4: `BiLipschitzLinear` identity at init (`init="rotation"` opt-in), raw leaves + `LinearParams`; `InvertibleLinear` removed — identity-at-init and orientation are universal laws, `IDENTITY_AT_INIT`/`ORIENTATION_NOT_GUARANTEED` deleted | `docs/changes/2026-10-02-invertible-cleanup.md` |
