---
type: architecture
status: current
updated: 2026-10-05
sources: [code]
---

# Architecture

The shape of the `deep_isochron` package (`docs/index.md` lists every document): what each module is for, how data flows through
a training step, and where the design decisions behind each part are recorded. Decisions
live in `docs/decisions/` (ADRs), the current plan in `docs/roadmap.md`, branch-level
detail in `docs/changes/`. This page is kept current with the code; when a phase of the
roadmap lands, the relevant section here changes in the same commit.

## Purpose

Learn a smooth conjugacy `Φ` between an oscillator with a stable limit cycle (observed
through trajectories) and a normal form whose phase/amplitude structure is known in closed
form, so that the oscillator's isochrons and phase response are read off `Φ`. Concretely:
FitzHugh–Nagumo (observed) ↔ Bautin normal form (latent), with `Φ` an invertible neural
network. The conjugacy equation `Φ ∘ φ_t = ψ_t ∘ Φ` (flows of the two systems) is enforced on
trajectories.

## Module map

```
deep_isochron
├── systems/                 ODEs (observed and normal forms)      ── AbstractODE
│   ├── base.py              rhs(t, u) · params() · SolverConfig · flow -> diffrax.Solution
│   ├── fitzhugh_nagumo.py   FitzhughNagumo (dim 2)
│   ├── hodgekin_huxley.py   HodgekinHuxley (dim 4)
│   └── normal_forms/
│       ├── base.py          AbstractNormalForm: closed-form phase Θ, isostable Ψ,
│       │                    isochrons, κ, eigenvalues at 0, polar and (Θ, Ψ) charts;
│       │                    default_integration = closed_form
│       ├── hopf.py          HopfNormalForm            (ρ, ω via `_sq` hooks in s = r²)
│       ├── bautin.py        BautinNormalForm
│       └── integration.py   CartesianIntegration / PolarIntegration /
│                            RadiusSquaredIntegration — flow(..., integration=)
├── model/
│   ├── invertible/          the INN vocabulary                     ── AbstractBijection
│   │   ├── base.py          AbstractBijection, AbstractScalarBijection, ScalarChain,
│   │   │                    SequentialINN
│   │   ├── constraints.py   Free, Arcsinh, Positive, BoundedPositive, Interval, Widths
│   │   ├── affine.py        Shift, Affine; AffineCoupling/ResidualCoupling factories
│   │   ├── analytic.py      CubicRational, SinhConjugation, CubicConjugation
│   │   ├── splines/         AbstractSpline; LinearSpline (C⁰), MonotonicRQSpline (C¹),
│   │   │                    CubicBSpline (C²)
│   │   ├── coupling.py      CouplingFlow[B] — any scalar template × any conditioner
│   │   ├── polar.py         OffsetedBijection, RadialBijection, PolarCouplingFlow,
│   │   │                    CircularMonotonicRQCoupling
│   │   └── linear.py        BiLipschitzLinear (the linear layer)
│   ├── fourier.py           TruncatedFourier (conditioner on S¹)
│   ├── base.py              AbstractPhaseAmplitudeModel: phase · amplitude · cycle_point ·
│   │                        __call__; phase_sensitivity derived      ── both models
│   ├── conjugacy.py         ConjugateLatentDynamics = bijection + normal form + SolverConfig
│   ├── latent_dynamics.py   PhaseAmplitudeLatentDynamics(omega, kappa): rotation ⊕ decay
│   ├── autoencoder.py       PhaseAmplitudeAutoencoder — Yawata et al. (2024) baseline;
│   │                        phase(x), phase_sensitivity(θ)
│   └── utils.py             zero_final_layer
├── data/                    one netCDF4 file per dataset (xarray)       ── ADR-0008
│   ├── dataset.py           DatasetMetadata (grouped); TimeSeriesDataSource = whole
│   │                        trajectories (frozen dataclass; splits; save/load)
│   ├── generate.py          IC samplers (UniformBox, UniformAnnulus, OnCycleGaussian);
│   │                        generate(system, sampler, ts, n, seed=…) with
│   │                        loud failures and config_hash
│   └── windows/             two ways to cut windows (ADR-0008 D2, D3) — see its __init__
│       ├── common.py        Element/Batch/WeightFn, range checks, transient_weight,
│       │                    categorical, start_weights (shared)
│       ├── per_element.py   RandomWindow / WeightedWindow (grain RandomMap); windows();
│       │                    mixed_windows(); validation_windows() — the reference
│       ├── batched.py       WindowBatchSource (batches by one gather, epochs of every
│       │                    window once); window_batches(); mixed_window_batches() — the
│       │                    training-run path
│       └── device.py        single_threaded(); to_device(ds, device); resolve_device()
├── analysis/                numerical limit cycle / monodromy / phase for any AbstractODE
│                            (namespace reserved; Phase E)
├── training/                                                            ── ADR-0009
│   ├── trainer.py           TrainerState (model · opt_state · step · key · schedule
│   │                        state); Trainer(optimizer, loss, schedule): jitted
│   │                        train_step, train(loader, logger, checkpointer, evaluate)
│   ├── losses.py            building blocks (trajectory_mse, step_weighted_consistency,
│   │                        alpha_schedule, batch_center_of_mass); AbstractLoss =
│   │                        weighted named terms; ConjugacyTrajectoryLoss,
│   │                        PhaseAutoencoderLoss
│   ├── schedules.py         Constant / StepSchedule (curriculum) / ThresholdSwitch
│   ├── loggers.py           Logger base class; Null / List / Print / Wandb; DelayedLogger
│   ├── checkpoint.py        Checkpointer base class; OrbaxCheckpointer (whole state)
│   └── evaluation.py        Evaluator(val_data, reference) on the model contract
└── misc.py                  inv_softplus, squashed_exp, polar ↔ cartesian

scripts/generate_data.py + configs/data/*.yaml   Hydra entry point for data generation
scripts/check_md_math.py                        GitHub-safe Markdown math check
scripts/bench_dataloader.py                     data-pipeline vs training-step benchmark (C7)
```

## The invertible package

Two abstract classes and one rule.

**`AbstractBijection`** — a diffeomorphism of `Rᵈ`: `__call__`, `inverse`, `jacobian`
(forward-mode by default), and two declared static properties, `dim` and `smoothness`
(the map is `C^k`; `None` is `C^∞`; containers take the minimum — ADR-0003). Every concrete
subclass is the identity map when freshly constructed and orientation-preserving at every
point of its parameter space; both are universal laws in the test suite.

**`AbstractScalarBijection[P]`** — `R → R` maps with exactly one array leaf, the
unconstrained vector `raw`; constrained parameters (a `NamedTuple` `P`) are computed on
read by `constrain(raw)`, with `constrain(0)` the identity (ADR-0001, ADR-0002).
`constrain` is built from the primitives in `constraints.py`, plain Python objects created
inside it and never stored (ADR-0005), each carrying its own identity-at-zero shift. An
instance with `raw = None` is a *template*: static configuration only. The class docstring
is the implementation checklist; `Affine` is the smallest example.

**The rule** that ties them together is `CouplingFlow`: it stores a scalar *template*, a
conditioner `x_const → raw` (an MLP with a zero-initialized last layer by default, a
`TruncatedFourier` for the polar variant), and writes the conditioner's output straight
into one scalar bijection per coupled coordinate through `template.from_unconstrained`.
Because `raw` is unconstrained, any conditioner output is valid; because `constrain(0)` is
the identity, a fresh layer is the identity. This is why every scalar bijection — analytic,
spline, chain, offset — is automatically a coupling layer, and why `AffineCoupling` and
`ResidualCoupling` are three-line factories rather than classes.

`BiLipschitzLinear` is the only linear layer: `U diag(s) Vᵀ` with `U, V ∈ SO(d)` (Cayley
transforms of skew generators, ADR-0010 — `expm`'s 16 `lax.cond`s per rotation were the
training step's host-side cost on GPU) and `s ∈ (1/L, L)`, so it is
orientation-preserving and well-conditioned everywhere; the
unconstrained `InvertibleLinear` was removed for failing both (change document
2026-10-02). An INN is a `SequentialINN` of alternating linear and coupling layers (the
notebook's `make_invertible_block`).

Smoothness matters here because the conjugacy must be at least `C¹` for the isochron
geometry (Jacobians) to be meaningful and `C²` for curvature-based losses (Phase E). The
cubic B-spline is the production candidate for that reason; its design is in ADR-0006 and
`docs/design/cubic-bspline.md`.

## Systems and normal forms

Two layers and one rule (ADR-0007). `AbstractODE` is any ODE in the study: `rhs(t, u, args)`
in the diffrax signature, `dim`, `params()` (constrained parameter values, for metadata),
and the numerical `flow(ts, u0, *, config: SolverConfig) -> diffrax.Solution` (`.ys` the
trajectory; `.result` for batched generation with `throw=False`).
`SolverConfig` holds solver, tolerances, `max_steps`, adjoint and `throw` as static fields;
it is leafless, so under `eqx.filter_vmap`/`filter_jit` it is static. `flow` is written for
one initial condition; batching is `eqx.filter_vmap(ode.flow, in_axes=(None, 0))`.

`AbstractNormalForm(AbstractODE)` is the subset usable as a conjugacy target: planar, with
`ṙ = rρ(r²)`, `θ̇ = ω(r²)` and a stable cycle at `r = 1`. A subclass supplies the two rates
and two closed-form integrals (`phase_shift` `h(r)`, `isostable` `ψ(r)`); the base derives
the cartesian `rhs` (smooth at the origin), `rhs_polar`, `period`, `floquet_exponent`
(`κ = ρ'(1)` by autodiff), `eigenvalues_origin`, `phase` (`Θ = θ + h(r)`), `amplitude`
(`Ψ`, `∂ᵣΨ(1) = 1`), `limit_cycle`, `isochron`, and the chart maps `to_polar`/`from_polar`
(polar) and `to_phase_amplitude`/`from_phase_amplitude` (the latter through
`radius_from_isostable`, a differentiable root solve unless the subclass overrides it with
the explicit inverse, as Hopf does). The mathematics and the implementation contract are
`docs/design/normal-forms.md` (§9). **The rule**: a class carries only what is available analytically; anything
numerical — locating a limit cycle, monodromy, asymptotic phase of FitzHugh–Nagumo — is a
function over `AbstractODE` in `analysis/` (Phase E). Everything public is cartesian.

**Integrations** (`normal_forms/integration.py`) are objects in the diffrax style — pass an
instance, or one of `"cartesian" | "polar" | "r_squared" | "closed_form"` for the
defaults (`closed_form` evolves `(Θ, Ψ)` exactly and maps back; no solver). They are
leafless modules, hence static: each traces separately and dispatch is free. All return
cartesian `.ys` in the same `Solution`, so `ConjugateLatentDynamics` never converts charts.

`base.py` is the contract both approaches implement, `AbstractPhaseAmplitudeModel`:
`phase(x)` in `(-π, π]`, `amplitude(x)` (isostable-like, no fixed normalization),
`cycle_point(θ)`, `__call__(ts, x0) -> (xt, yt)`; `phase_gradient` and
`phase_sensitivity` are derived once for all. Training and evaluation code takes the
contract and never asks which model it has. For `ConjugateLatentDynamics` the contract is
the normal form's closed forms transported by the bijection (`Θ = Θ_NF ∘ H`,
`cycle_point = H⁻¹ ∘ limit_cycle`).

`latent_dynamics.py` / `autoencoder.py` are the non-invertible baseline: the phase
autoencoder of Yawata et al. (Chaos 34, 063111, 2024), as mapped in
`docs/design/normal-forms.md` §5.3. `PhaseAmplitudeLatentDynamics(omega, kappa)` is the
closed-form flow on `R³` — `(Y₁, Y₂)` rotating at `omega`, `Y₃` decaying at `kappa < 0`
(Eqs. (12)–(14)); `PhaseAmplitudeAutoencoder` is an MLP encoder whose first two outputs
are normalized to the unit circle (Eqs. (15)–(16)), an MLP decoder, and that flow;
`phase(x) = atan2(Y₂, Y₁)` (Eq. (19)) and `phase_sensitivity(θ)` by `jax.grad` at the
decoded cycle point (Eq. (20)). Its latent space is exactly the `(Θ, Ψ)` chart of a
normal form (`(cos Θ, sin Θ, Ψ)`), which is what `tests/test_baseline.py` checks; the
loss `PhaseAutoencoderLoss` (Eqs. (21)–(26)) lives in `training/losses.py` and the
near-cycle training distribution is the `OnCycleGaussian` sampler (Eqs. (27)–(28)). It
does not use `AbstractODE`.

## Data

A dataset is **one netCDF4 file** (HDF5 underneath; ADR-0008): dims `(trajectory, time,
dim)`, the time grid as the `time` coordinate, and a grouped `DatasetMetadata` — `system`
(name, constrained `params()`), `sampling` (IC sampler, params, seed, `n_trajectories`),
`grid` (`t0`, `t1`, `n`), `solve` (solver, tolerances, `max_steps`, integration),
`provenance` (created, git SHA/dirty, package version, dtype), `extra` (the resolved Hydra
config) — one JSON-string attribute per group. `config_hash` is a property over the first
four groups, so an unchanged config regenerates to the same `<name>-<hash>.nc`.

`TimeSeriesDataSource` is a frozen dataclass (`ts`, `ys`, `metadata`) and a grain
random-access source over **whole trajectories**: `len` is the number of trajectories and
`source[i] = {"t": ts, "u": ys[i]}`. It carries no window bookkeeping. `split_time(idx)`
and `split_trajectories(frac, seed)` are `copy.replace` with sliced arrays; `save`/`load`
(with an optional dtype assertion) are the only I/O; `dataset` is the xarray view.

`generate(system, ic_sampler, ts, n, *, seed, config, integration, extra)` draws initial
conditions from an `AbstractICSampler` (`UniformBox`, `UniformAnnulus`), vmaps `flow` with
`throw=False`, and raises naming any failed indices. `scripts/generate_data.py` is the Hydra
entry point over `configs/data/*.yaml`.

Training runs read **batches of windows** from a `WindowBatchSource`
(`data/windows/batched.py`, ADR-0008 Decision 3): element `i` is the `i`-th batch of the run, one vectorized gather,
an epoch being every window of every trajectory once (fresh permutation per epoch, remainder
dropped), so the loader is finite and `len = epochs · batches_per_epoch`.
`window_batches(source, length, batch, seed=…, epochs=… | num_steps=…, weight=… |
start_range=…)` and `mixed_window_batches(…, split_idx, weights)` build it; weighted and
mixed starts are draws with replacement (`transient_weight(boost, tau)` oversamples the
transient). The per-element **grain transforms** `RandomWindow`/`WeightedWindow` behind
`windows()`/`mixed_windows()` remain the reference semantics (tests compare marginals) and
serve `validation_windows` (`data/windows/per_element.py`). Batches are dicts `{"t": (B, L), "u": (B, L, dim)}`.

## Data flow of one training step

```
loader ──► batch {t:[B,L], u:[B,L,d]}  (grain; to_device() prefetches to the accelerator)
                 │
Trainer.train_step(state, batch)                                  ── jitted, pure
                 ├─ weights = schedule.weights(state.schedule_state, state.step)
                 ├─ (loss, terms) = loss_fn(model, batch, weights)
                 │     ConjugacyTrajectoryLoss:  y0 = H(x[:, 0]); y[t] = nf.flow(t, y0).ys;
                 │                               x̂ = H⁻¹(y); data = mse(x, x̂),
                 │                               latent = mse(H(x[t]), y[t])
                 │     PhaseAutoencoderLoss:     Y = f_enc(x); Ŷ = latent flow of Y[:, 0];
                 │                               recon, pha, amp (α_k-weighted), aux
                 ├─ grads = ∇_model loss ; schedule_state' = schedule.update(…, terms)
                 ▼
state' = state.take_step(grads, schedule_state')   optax update on is_trainable leaves
                 │
                 ├─ logger.log({loss, terms, w/*}, step)   DelayedLogger: forwarded one
                 │                                          step later (host sync then)
                 └─ every eval_every: evaluate(model) → {val/mse, period, κ, …}
                                      logger.log(…); checkpointer.save(step, state, …)
```

The model is an `AbstractPhaseAmplitudeModel` — `ConjugateLatentDynamics` (bijection,
normal form, `SolverConfig`) or `PhaseAmplitudeAutoencoder`. A loss is a weighted sum of
named terms whose weights the trainer's schedule supplies each step (ADR-0009); the
conjugacy equation is what `data` and `latent` together enforce. Batches come from
`window_batches`/`mixed_window_batches` over a `TimeSeriesDataSource` — finite, the data
define the run — transferred by the loader (`to_device`); validation is `validation_windows` over the held-out trajectories — finite
and deterministic — evaluated to exhaustion by an `Evaluator`, whose `val/mse` is what the
`OrbaxCheckpointer` keeps the best checkpoint by.

## Tests

`tests/registry.py` declares *what* is tested (scalar templates, vector builders,
`UNTESTED` with reasons) and `tests/test_registry.py` fails if a concrete bijection is
missing from it. `tests/test_bijections.py` holds the laws every bijection satisfies (round
trip, identity at init, orientation, finiteness, Jacobian consistency, pytree hygiene, one
optimizer step), run through Hypothesis draws from `tests/strategies.py`. Per-family laws
are in `test_splines.py`, `test_analytic.py`, `test_constraints.py`, `test_linear.py`;
`test_normal_forms.py` pins the normal forms' closed forms by autodiff and the integrations,
`test_systems.py` the flow machinery and the observed systems' facts (Langfield et al. 2014),
`test_data.py` the data layer end to end, `test_baseline.py` the phase autoencoder
against the exact chart (plus one `slow` training test on Hopf data), `test_models.py`
the `AbstractPhaseAmplitudeModel` contract on both models, `test_training.py` the losses,
schedules, loggers, trainer loop, evaluation and Orbax round trip. Solver configurations used by
tests are named in `tests/helpers.SOLVERS` with their reasons, like `TOL`. Shape
annotations are checked at runtime by the jaxtyping/beartype import hook (`conftest.py`).

## Decision records

| ADR | Decision |
|---|---|
| [0001](decisions/0001-unconstrained-leaves.md) | one unconstrained leaf `raw`; constrained values computed on read |
| [0002](decisions/0002-identity-at-zero.md) | `constrain(0)` is the identity; the shift lives in the map |
| [0003](decisions/0003-declared-smoothness.md) | `smoothness` is declared per class; containers take the minimum |
| [0004](decisions/0004-abstractvar-fields.md) | `AbstractVar`s implemented as static `init=False` fields |
| [0005](decisions/0005-constraint-primitives.md) | constraint primitives are plain objects created inside `constrain` |
| [0006](decisions/0006-cubic-bspline-boundary-and-inverse.md) | `CubicBSpline`: Greville-pinned boundary; bracketed Newton inverse |
| [0007](decisions/0007-systems-hierarchy-and-flow-strategies.md) | `AbstractODE` / `AbstractNormalForm`; `SolverConfig`; flow integrations as objects |
| [0008](decisions/0008-dataset-format-and-sampling.md) | one netCDF4 file per dataset via xarray; weighted windows alongside `mix`; batched window source for training runs |
| [0009](decisions/0009-trainer-losses-schedules-checkpoints.md) | trainer: injected logging/checkpointing, losses as weighted terms, schedules in the state, whole-state checkpoints |
| [0010](decisions/0010-rotations-by-cayley-transform.md) | `BiLipschitzLinear` rotations by Cayley transform, not `expm` |
