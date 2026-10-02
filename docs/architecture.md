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
├── systems/                 ODEs (observed and latent)            ── AbstractODE
│   ├── base.py              rhs(t, u) + solve(ts, u0, …) via diffrax
│   ├── fitzhugh_nagumo.py   FitzhughNagumo (dim 2)
│   ├── hodgekin_huxley.py   HodgekinHuxley (dim 4)
│   └── normal_forms.py      HopfNormalForm, BautinNormalForm (polar chart; r² trick)
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
│   ├── conjugacy.py         ConjugateLatentDynamics = bijection + latent ODE
│   ├── latent_dynamics.py   LinearLatentDynamics, HopfLatentDynamics (autoencoder path)
│   ├── autoencoder.py       PhaseAmplitudeAutoencoder (non-invertible baseline)
│   └── utils.py             zero_final_layer
├── data/                    TimeSeriesDataSource (windows over trajectories); generate.py
│                            placeholder                                   ── Phase B
├── training/                TrainerState / Trainer (optax + orbax + wandb);
│                            ConjugacyTrajectoryLoss                        ── Phase C
└── misc.py                  inv_softplus, squashed_exp, polar ↔ cartesian
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

## Systems and the latent chart

`AbstractODE` subclasses provide `rhs` (diffrax signature) and `solve` (`diffeqsolve` with
`SaveAt(ts)` and a PID controller). The normal forms are written in **polar coordinates**
`(r, θ)`: `ṙ = a r (1 − r²)(1 + b r²)`, `θ̇ = w₀ + (w − w₀) r²` for Bautin; `BautinNormalForm.solve`
integrates `r²` instead of `r` (removes the `r = 0` singularity of the chart and the
square root) and recovers `θ` by quadrature. Phase B unifies this interface (`flow(ts, u0,
*, solver, …)`, explicit `to_chart`/`from_chart`). `latent_dynamics.py` and
`autoencoder.py` are the earlier, non-invertible autoencoder path (`HopfLatentDynamics`
calls the normal form as if it were callable, which it no longer is); they are legacy
until Phase B decides what of them survives.

## Data flow of one training step

```
batch (t[B,T], x[B,T,d])  ──►  ConjugacyTrajectoryLoss(model, batch)
                                 │
                                 ├─ y0 = Φ(x[:, 0])                       bijection forward
                                 ├─ polar = cartesian_to_polar(y0)         misc
                                 ├─ y[t]  = latent.solve(t, polar)         systems (diffrax)
                                 ├─ x̂[t]  = Φ⁻¹(polar_to_cartesian(y[t]))  bijection inverse
                                 ├─ mse(x, x̂)  +  w · mse(Φ(x[t]), y[t])
                                 ▼
TrainerState.take_step(grads)  ──►  optax update on eqx.filter(model, is_trainable)
                                 └─ wandb log (one step delayed), orbax checkpoint
```

`ConjugateLatentDynamics` is the model: a bijection and a latent ODE. The loss is the
trajectory reconstruction error plus a weighted latent-consistency term; the conjugacy
equation is what the two terms together enforce. Solver settings currently live in
`ConjugateLatentDynamics.__call__` (Phase C moves them to a `SolverConfig`).

`TimeSeriesDataSource` holds `ts[T]`, `ys[N, T, d]` and serves fixed-length windows;
`split(idx)` cuts in time. Generation, metadata and on-disk format are Phase B.

## Tests

`tests/registry.py` declares *what* is tested (scalar templates, vector builders,
`UNTESTED` with reasons) and `tests/test_registry.py` fails if a concrete bijection is
missing from it. `tests/test_bijections.py` holds the laws every bijection satisfies (round
trip, identity at init, orientation, finiteness, Jacobian consistency, pytree hygiene, one
optimiser step), run through Hypothesis draws from `tests/strategies.py`. Per-family laws
are in `test_splines.py`, `test_analytic.py`, `test_constraints.py`, `test_linear.py`. Shape
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
