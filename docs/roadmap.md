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

## Phase A — `invertible-cleanup` (current)

Structural follow-ups from the two reviews of the invertible package. Each is small;
together they close the design before the data/trainer work builds on it.

| # | Item | Why | Done when |
|---|---|---|---|
| A5 | Implementation checklist in the `AbstractScalarBijection` docstring (fields → `num_params` → `constrain` → read `self.params`; nothing else is an array), cross-linked from `AbstractSpline` | the rule is enforced by tests but not visible from a class body | docstring present |
| A6 | `CubicBSpline` design: ADR-0006 (Greville pinning vs Hong & Chun Alg. 1 lines 13–17; Newton-bisection + implicit JVP vs closed-form root) and a math note `docs/design/cubic-bspline.md` (knot/coefficient indexing, exterior-width normalisation, `2^-newton_iters` bound) | the two decisions the splines branch left undocumented; required before the B-spline becomes the production architecture | both files exist; linked from the change document |
| A7 | `docs/architecture.md`: module map (systems → bijections → `ConjugateLatentDynamics` → losses → trainer), data flow, links to ADRs | the ADRs record decisions; nothing records the shape of the package | one page; `CLAUDE.md` points to it |
| A8 | `systems/` on ADR-0004 (`dim: ClassVar  # ty: ignore` → static `init=False` field, 4 classes); revisit the three `ty: ignore`s in `trainer.py` | `ty` cleanliness package-wide | `grep "ty: ignore" src` empty; `ty check src` clean |
| A9 | `matplotlib` → `dev` dependency group | runtime deps = what `src` imports | `uv sync` without `dev` has no plotting |

**Acceptance**: `uv run pytest -n 4` 0 failed with the pinned sinh-underflow xfail as the only
non-pass; `ty check src` clean; A6/A7 documents exist; change document with pre-populated review rows.

## Phase B — `data-generation`

- B1 Unify the ODE interface: `AbstractODE.flow(ts, u0, *, solver, rtol, atol, max_steps)`
  (diffrax default; `BautinNormalForm` overrides with the r² trick); delete the duplicate
  `latent_dynamics.HopfNormalForm`; `solve` stops swallowing `**kwargs`; explicit polar chart
  on the normal forms (`to_chart`/`from_chart`) so `ConjugateLatentDynamics` is chart-agnostic
  and the `r = 0` singularity lives in one place.
- B2 `generate(system, ic_sampler, ts, *, solver, rtol, atol, key) -> TimeSeriesDataSource`,
  vmapped over initial conditions (replaces the notebook cells).
- B3 `DatasetMetadata`: system class + params, solver + tolerances, time grid, IC sampler +
  seed, `n_trajectories`, dtype, created, git SHA/dirty, package version.
- B4 On-disk format `data/<name>-<cfghash8>/arrays.npz` (`ts`, `ys`, `u0`) + `metadata.json`;
  `TimeSeriesDataSource.save/load`; loud failure on dtype mismatch.
- B5 `split_time(idx)` (documented transient oversampling) + `split_trajectories(frac, seed)`
  for held-out validation.
- B6 `scripts/generate_data.py` as a Hydra entry point from `configs/data/*.yaml`.
- B7 Tests: `test_systems.py` (Bautin r² trick vs direct integration; Floquet exponent vs
  numerical monodromy; FHN fixed-point eigenvalues `0.1339 ± 0.9163i`; HH gating in `[0, 1]`),
  `test_latent_dynamics.py`, `test_data.py` (windows, splits, save/load incl. metadata, grain).

## Phase C — `trainer`

- C1 `Trainer(optimizer, loss_fn)` + `train(model, loader, *, logger, checkpointer, num_steps,
  eval_every, eval_loader)`; `NullLogger`/`NullCheckpointer` for tests; document the
  one-step-delayed logging overlap.
- C2 Solver settings out of `ConjugateLatentDynamics.__call__` into a `SolverConfig` field;
  explicit loss weights in `ConjugacyTrajectoryLoss`.
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
