---
type: roadmap
status: current
updated: 2026-10-22
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
| B9 | Data layer as grain transforms: whole-trajectory frozen-dataclass source (`copy.replace`), `RandomWindow` / `WeightedWindow`, `mixed_windows`; grouped `DatasetMetadata`; `test_data.py` rewritten; ADR-0008 amended | review: idiomatic grain; organized metadata | *done 2026-10-03* |
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
| D5 | Notebook on `hydra.compose` + the builders (Joon, as Phase E runs start). 2026-10-07: the notebook's hand-rolled loop could not be reopened → `Run`/`setup`/explicit resume (branch `run-layout`, ADR-0011 §6 amended); the notebook moves onto `setup` | in progress |
| D6 | CI: `.github/workflows/ci.yml` — ruff, ty, `check_md_math`, `pytest --hypothesis-profile=ci -n 4 -m "not slow"`; `slow` job + 20-step end-to-end `train.py` run; the `circular_rq (K=8)` failure fixed (test domain off the `eps_r` disc) | *done 2026-10-05*; red since 2026-10-06 on `uvx ty` drifting to 0.0.88 — fixed 2026-10-15 (`ci-ty`: seven ADR-0004 overrides, CI on `ty@latest` by choice) |
| D7 | `scripts/bench_dataloader.py --config <overrides>` benchmarks a training config's own pipeline, model and loss | *done 2026-10-05* |
| D8 | ADR-0011; `docs/architecture.md` (`experiment/`, scripts, config→run data flow); change document; design doc §7 | *done 2026-10-05* |

## Phase E — `analysis`, then science (current; replanned 2026-10-15; E1 landed 2026-10-06)

**Replanned 2026-10-15** (Joon; ADR-0012 amended). `deep_isochron.analysis` keeps its
two halves, but their roles and priority are now:

- **`analysis.ode`** — *given* an `AbstractODE`, compute the reference objects of a
  limit-cycle oscillator: the cycle, its Floquet exponent, the asymptotic phase, the phase
  response curve (PRC/PSF), the amplitude (isostable) coordinate and its response curve.
  These are what a trained model is **assessed** against (does its phase / isochrons /
  PRC match the ground-truth system's?). **The priority.**
- **`analysis.data`** — from sampled trajectories, the quantities that automate
  **training decisions**: winding sense → the latent `w` sign; transient detection →
  the split index of a `mixed` dataset. Later, as the runs demand.

Method, per item: **a literature survey first, delivered to Joon** (what exists, what to
credit, candidate numerical schemes, the oracles), who reads it, extends it and may
prototype; implementation follows in its own commit, with the survey recorded in
`docs/design/analysis.md`. Abstractions come *last*: functions first, shared objects
once several functions exist (the `Cycle` object of E1 is kept as it is and reassessed
then — Parked).

| # | Item | Method candidates (surveyed before implementation) | Oracle / done when |
|---|---|---|---|
| E1 | `analysis.Cycle` + `analysis.data.estimate_cycle` — the cycle from trajectory tails (data half) | Revzen & Guckenheimer; Kralemann et al. 2008 protophase-to-phase | *done 2026-10-06*: normal forms to 1e-8; FHN period 11.2279 (Langfield et al. 2014) |
| O1 | `analysis.ode`: `find_limit_cycle(ode, x0)` — points on $\gamma$ and the period $T$ | settle → first-return map on a flow-normal section (`PoincareMap`, differentiable in the point, with the minimum-period test as period-$k$ detection) → multiple shooting on the time-$T$ flow with AD Jacobians and Levenberg–Marquardt; `reverse_time` for repelling cycles; ADR-0014 | *done 2026-10-18*, review round 1 applied: normal forms to 1e-8 (T, radius; $e^{\kappa T}$ from `jacfwd` of the map), integrating the vector field numerically against the closed forms; Guckenheimer & Meloon's algebraic orbit $\max\vert g\vert < 10^{-8}$; FHN $T = 11.22789$ (Langfield et al.); Winfree's repelling inner cycle in reverse time |
| O3 | `analysis.ode`: `asymptotic_phase(ode, cycle, x)` — the phase field on a grid | direct method: integrate $x$ for $n$ periods, read the phase of the cycle point it lands beside (Guckenheimer 1975's definition made numerical; Yawata et al. 2024 Fig. 8); error controlled by $\kappa$ and $n$ | normal forms' exact `phase` (Hopf, Bautin, Winfree); the learned-phase comparison plot |
| O4 | `analysis.ode`: `phase_sensitivity(ode, cycle, θ)` — PRC/PSF $Z(\theta) = \nabla\Theta$ on $\gamma$ | adjoint equation, backward integration, normalization $Z \cdot F = \omega$ (Ermentrout & Kopell 1991; Brown, Moehlis & Holmes 2004 App. A; the "true PSF" of Yawata et al. 2024 Figs. 9/12/15); direct finite-perturbation cross-check | normal forms: $\nabla\Theta$ of the closed-form chart by autodiff; FHN: direct vs adjoint |
| O2 | `analysis.ode`: `floquet(cycle)` — $\mu$, $\kappa$, the Floquet vectors | the periodic Schur chain on the shooting-segment Jacobians, seeded with the flow direction and run in the attracting direction (the method; Bojanczyk et al. 1992, Lust 2001, Dieci et al. 1997); the explicit monodromy and Liouville's $\int\mathrm{div} f$ as checks; the trace of $DP$ rejected (leaks the trivial multiplier's error); ADR-0015 | *done 2026-10-20*: normal forms $\kappa = \rho'(1)$ to 1e-8 by all routes; Winfree's repeller forward-time; the Cassini oval's $\kappa T = -\oint\Vert\nabla g\Vert d\ell$ and $T = \oint d\ell/(s\Vert\nabla g\Vert)$ by solver-free line integrals ($\mu \approx 10^{-16}$, beyond the monodromy's reach); FHN $\mu = 1.69\cdot10^{-12}$ with the $\varepsilon/\mu$ floor of the monodromy made visible |
| O5 | `analysis.ode`: `isostable(ode, cycle, x)` — the amplitude coordinate $\Psi$ | direct: fit $d(t) \approx C e^{\kappa t}$ along the settling trajectory, normalized $\partial_r\Psi(1) = 1$; alternatively Laplace averages (Mauroy, Mezić & Moehlis 2013) | normal forms' exact `amplitude` |
| O6 | `analysis.ode`: `isostable_response(ode, cycle, θ)` — IRC/ARC $I(\theta) = \nabla\Psi$ on $\gamma$ | adjoint-type equation for the isostable gradient (Wilson & Moehlis 2016 Eqs. 9–14; Shirasaka, Kurebayashi & Nakao 2017; Wilson & Ermentrout 2018) | normal forms: $\nabla\Psi$ of the chart by autodiff |
| O7 | `analysis.ode.auto`: isochrons of a planar system by AUTO-07p continuation of the multi-segment two-point BVP (Langfield, Krauskopf & Osinga 2014; the coworker's reproduction in `src/Isochron2PBVP`), on demand as an `IsochronFoliation` dataclass; the orbit from O1, the bundle and the η-continuation in AUTO; plan `claude/isochron-auto-plan-2026-10-22` (Project doc) | step 1 *done 2026-10-22* (`symbolic-systems`): systems declared once in SymPy, JAX `rhs` derived, Fortran `RHS` + exact Jacobian generated and `INCLUDE`d into the generic BVP file (ADR-0016); step 2 (branch `auto-isochrons`): the staged pipeline, dataclasses, netCDF persistence, `scripts/install_auto.sh` and the `auto` test job, ADR-0017 | generated Fortran equals the coworker's hand-written `RHS`; `f2py`-compiled `RHS` vs `rhs`/`jacfwd`; then: AUTO orbit vs O1, bundle vs O2/O4 directions, Bautin/Winfree isochrons vs closed-form `phase` |
| M | Model-side metrics in `training/evaluation.py`, one per oracle as it lands: aligned phase-field error (offset and orientation by the circular mean, shared with the plot), PRC error in the first $k$ Fourier modes and the normalization residual $Z \cdot F - \omega$, amplitude correlation, ARC error; `amplitude_sensitivity` on `AbstractPhaseAmplitudeModel` | — | `train.py data=winfree` / `fhn` report them |
| D2 | `analysis.data`: winding sense → `build_model` sets the signs of `w`, `w0` (E1's `winding`) | — | no manual sign intervention on FHN/Winfree |
| D3 | `analysis.data`: transient detection (`settling_time` against a cycle; shared `log d(t)` fit with O5) → `t_settle` in dataset metadata, default `validation.t_split`; the far-from-cycle validation set (`UniformAnnulus`, design document `validation-split.md`) | O1 or E1 | `t_settle = ln(d_0/\epsilon)/\vert\kappa\vert` on the normal forms |
| E7 | Experiments (running on the side throughout, feeding the plots and analysis wants): Bautin identity sanity; Winfree $a < 0$ against its invariants ($T$, $\kappa$, phase, PRC); FHN; the Yawata baseline on both; Winfree $a > 0$ — what a planar diffeomorphism does with a phaseless disk (`outside_basin=resample`). Losses/diagnostics (pushforward near the slow manifold, Hessian matching, Jacobian anisotropy, INN depth, periodic spline endpoints) only as the runs demand | M | figures and tables |
| E8 | Docs: `analysis.md` one section per function; ADR-0012 amendments; architecture; `docs/figures/` from `scripts/figures.py` once the plots module exists (Joon moves plot code from notebooks as it stabilizes) | — | — |

Order: O1 → O2 (done 2026-10-20) → O3 → O4 (one survey, shared references), then O5 → O6, with M alongside
each; D2/D3 when a run asks for them; E7 continuously.

## Parked (with reason)

- `analysis.Cycle` as the shared abstraction of both halves (ADR-0012 §1, E1) — kept as
  implemented, not extended (2026-10-15): abstractions are decided after the `analysis.ode`
  functions exist, from what they actually share.
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
  input standardization, the FHN cycle for `OnCycleGaussian` (needs the numerical limit
  cycle of Phase E) — when the baseline is run for the paper.
- Grouping the scalar bijections (`affine.py`, `analytic.py`, `splines/`, `OffsetedBijection`)
  under one subpackage — cosmetic; when the vocabulary stops growing.

- Initialization-strategy enum on `AbstractBijection` — only if training dynamics call for it
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
| 2026-10-06 | `analysis` | E1: `analysis.Cycle` (Fourier closed curve in a phase uniform in time) and `analysis.data.estimate_cycle` (tails, polar protophase, Kralemann PTP, regression period, self-consistency check); literature survey in `docs/design/analysis.md` §1; ADR-0012; `tests/test_analysis.py` | `docs/changes/2026-10-06-analysis.md`, ADR-0012 |
| 2026-10-07 | `run-layout` | `experiment.Run` (the run directory; `create` refuses an existing run, `resume` explicit with recorded config changes), `setup(cfg, run_dir) -> Experiment`, `train = setup().train()`, `load_run(template=)` for configless models, `build_evaluator`, `build_logger(wandb_run=)`, `load_or_generate_source`; ADR-0011 §6 amended | `docs/changes/2026-10-07-run-layout.md`, ADR-0011 |
| 2026-10-08 | `normal-forms-basin` | Normal forms take their data in `r` on a declared basin (`basin_radii`, `in_basin`, `nan` outside); evenness is the `AbstractEvenNormalForm` subfamily (Hopf, Bautin); `WinfreeNormalForm` (Langfield et al. 2025 §4.1, closed-form isochrons, phaseless disk; `a = 0` allowed); `generate(outside_basin=error\|resample)`; design document §3.3/§9 amended; ADR-0013 | `docs/changes/2026-10-08-normal-forms-basin.md`, ADR-0013 |
| 2026-10-15 | `ci-ty` | CI green against the latest ty: five `num_params` properties → ADR-0004 fields, template `raw` keeps the base annotation; `uvx ty@latest` in CI by choice, ty in the dev group; ADR-0004 amended | `docs/changes/2026-10-15-ci-ty.md`, ADR-0004 |
| 2026-10-18 | `analysis` | O1: `analysis.ode` — `Hyperplane`, `PoincareMap` (first-return map: `__call__`, `iterate`, `fixed_point`, `jacobian`; `direction`, `reverse_time`), `multiple_shooting` (LM, AD Jacobians, coarsening), `limit_cycle` → `LimitCycle`; `NotConvergedError`; `SolverConfig.with_adjoint`; `tests/helpers.LevelSetCycle` (Guckenheimer–Meloon orbit, Cassini oval); design document §2; ADR-0014 | `docs/changes/2026-10-06-analysis.md`, ADR-0014, `docs/design/analysis.md` §2 |
| 2026-10-20 | `analysis` | O2: `analysis.ode.floquet` — `floquet(cycle, method=)` by the seeded periodic Schur chain (`segment_jacobians`, `periodic_schur`), the explicit `monodromy` and Liouville's `divergence_integral`, `return_map_multiplier`; `LimitCycle`/`FixedPoint` promoted to `eqx.Module`s carrying `ode` and `config`, `sample(n)`/`orientation()`, `f(cycle)` signatures; stage kernels jitted; design document §3; ADR-0015 | `docs/changes/2026-10-06-analysis.md`, ADR-0015, `docs/design/analysis.md` §3 |
| 2026-10-22 | `symbolic-systems` | O7 step 1: `systems.symbolic` — `SymbolicODE`, `AbstractSymbolicODE._symbolic_template` with `rhs`/`params()` derived (sympy2jax, compiled per class in `__init_subclass__`); FHN and HH as templates; normal forms' defining data as `_symbolic_rates[_sq]` with the JAX hooks derived and `symbolic()`; `analysis.ode.auto.codegen` (`write_problem`, `Idx`, `rhs.inc`/`indices.inc`/`c.<name>`) and the generic `isochron_multiseg.f90` with named slots; `tests/test_symbolic.py`, `tests/test_auto_codegen.py` (golden file; `f2py` test when gfortran is present); ADR-0016; ADR-0007 amended | `docs/changes/2026-10-22-symbolic-systems.md`, ADR-0016, ADR-0007 |
