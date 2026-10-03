# Architecture

The shape of the `deep_isochron` package: what each module is for, how data flows through
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
│       ├── base.py          AbstractNormalForm: closed-form phase, isostable amplitude,
│       │                    isochrons, Floquet exponent, polar chart
│       ├── hopf.py          HopfNormalForm            (two rates ρ(s), ω(s))
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
│   ├── conjugacy.py         ConjugateLatentDynamics = bijection + latent ODE + SolverConfig
│   ├── latent_dynamics.py   LinearLatentDynamics (autoencoder baseline's latent flow)
│   ├── autoencoder.py       PhaseAmplitudeAutoencoder (non-invertible baseline)
│   └── utils.py             zero_final_layer
├── data/                    one netCDF4 file per dataset (xarray)       ── ADR-0008
│   ├── dataset.py           DatasetMetadata; TimeSeriesDataSource (windows, splits, save/load)
│   ├── generate.py          IC samplers; generate(system, sampler, ts, n, seed=…) with
│   │                        loud failures and config_hash
│   └── sampling.py          weighted_windows + transient_weights; mixed_split (grain)
├── analysis/                numerical limit cycle / monodromy / phase for any AbstractODE
│                            (namespace reserved; Phase E)
├── training/                TrainerState / Trainer (optax + orbax + wandb);
│                            ConjugacyTrajectoryLoss                        ── Phase C
└── misc.py                  inv_softplus, squashed_exp, polar ↔ cartesian

scripts/generate_data.py + configs/data/*.yaml   Hydra entry point for data generation
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
conditioner `x_const → raw` (an MLP with a zero-initialised last layer by default, a
`TruncatedFourier` for the polar variant), and writes the conditioner's output straight
into one scalar bijection per coupled coordinate through `template.from_unconstrained`.
Because `raw` is unconstrained, any conditioner output is valid; because `constrain(0)` is
the identity, a fresh layer is the identity. This is why every scalar bijection — analytic,
spline, chain, offset — is automatically a coupling layer, and why `AffineCoupling` and
`ResidualCoupling` are three-line factories rather than classes.

`BiLipschitzLinear` is the only linear layer: `U diag(s) Vᵀ` with `U, V ∈ SO(d)` and
`s ∈ (1/L, L)`, so it is orientation-preserving and well-conditioned everywhere; the
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
(`2ρ'(1)` by autodiff), `phase` (`θ + h(r)`), `amplitude`, `limit_cycle`, `isochron`, and the
chart maps. **The rule**: a class carries only what is available analytically; anything
numerical — locating a limit cycle, monodromy, asymptotic phase of FitzHugh–Nagumo — is a
function over `AbstractODE` in `analysis/` (Phase E). Everything public is cartesian.

**Integrations** (`normal_forms/integration.py`) are objects in the diffrax style — pass an
instance, or one of `"cartesian" | "polar" | "r_squared"` for the defaults. They are
leafless modules, hence static: each traces separately and dispatch is free. All return
cartesian `.ys` in the same `Solution`, so `ConjugateLatentDynamics` never converts charts.

`latent_dynamics.py` / `autoencoder.py` are the non-invertible baseline
(`PhaseAmplitudeAutoencoder` with `LinearLatentDynamics`), kept for comparison; they do not
use `AbstractODE`.

## Data

A dataset is **one netCDF4 file** (HDF5 underneath; ADR-0008): dims `(trajectory, time,
dim)`, the time grid as the `time` coordinate, and `DatasetMetadata` — system and
parameters, IC sampler and seed, grid, solver and tolerances, integration, dtype, `config_hash`,
timestamp, git SHA/dirty, package version, free-form `extra` (the resolved Hydra config) —
flattened into the attributes. `TimeSeriesDataSource` wraps it and serves fixed-length
windows through grain's random-access protocol; `split_time(idx)` and
`split_trajectories(frac, seed)` return sources over views; `save`/`load` (with an optional
dtype assertion) are the only I/O.

`generate(system, ic_sampler, ts, n, *, seed, config, integration, window_size, extra)` draws
initial conditions from an `AbstractICSampler` (`UniformBox`, `UniformAnnulus`), vmaps
`flow_result` with `throw=False`, and raises naming any failed indices. Files are named
`<name>-<config_hash>.nc` (`dataset_path`); the hash covers the generation-defining fields
only, so an unchanged config regenerates to the same file. `scripts/generate_data.py` is the
Hydra entry point over `configs/data/*.yaml`.

Training draws windows either by **weighted sampling** — `weighted_windows(source,
transient_weights(boost, tau), seed)`, one grain dataset drawing window indices with
probability ∝ weight, oversampling the transient — or by the **two-loader mixture**
`mixed_split(source, split_idx, weights, seed)` (`grain.MapDataset.mix` of the shuffled
halves). Both are kept so they can be compared.

## Data flow of one training step

```
batch (t[B,T], x[B,T,d])  ──►  ConjugacyTrajectoryLoss(model, batch)
                                 │
                                 ├─ y0 = Φ(x[:, 0])                       bijection forward
                                 ├─ y[t]  = latent.flow(t, y0, config).ys  systems (integration)
                                 ├─ x̂[t]  = Φ⁻¹(y[t])                      bijection inverse
                                 ├─ mse(x, x̂)  +  w · mse(Φ(x[t]), y[t])
                                 ▼
TrainerState.take_step(grads)  ──►  optax update on eqx.filter(model, is_trainable)
                                 └─ wandb log (one step delayed), orbax checkpoint
```

`ConjugateLatentDynamics` is the model: a bijection, a latent ODE and the `SolverConfig`
used to integrate it. The loss is the trajectory reconstruction error plus a weighted
latent-consistency term; the conjugacy equation is what the two terms together enforce.

Batches come from one of the two samplers above over a `TimeSeriesDataSource`.

## Tests

`tests/registry.py` declares *what* is tested (scalar templates, vector builders,
`UNTESTED` with reasons) and `tests/test_registry.py` fails if a concrete bijection is
missing from it. `tests/test_bijections.py` holds the laws every bijection satisfies (round
trip, identity at init, orientation, finiteness, Jacobian consistency, pytree hygiene, one
optimiser step), run through Hypothesis draws from `tests/strategies.py`. Per-family laws
are in `test_splines.py`, `test_analytic.py`, `test_constraints.py`, `test_linear.py`;
`test_normal_forms.py` pins the normal forms' closed forms by autodiff and the integrations,
`test_systems.py` the flow machinery and the observed systems' facts (Langfield et al. 2014),
`test_data.py` the data layer end to end. Shape
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
| [0008](decisions/0008-dataset-format-and-sampling.md) | one netCDF4 file per dataset via xarray; weighted windows alongside `mix` |
