---
type: decision
id: ADR-0007
status: accepted; amended (ADR-0013 - basins, even subfamily; ADR-0016 - symbolic defining data)
updated: 2026-10-27
verified_by: joon
---

# ADR-0007 — `AbstractODE` / `AbstractNormalForm`, `SolverConfig`, and flow integrations as objects

The mathematics lives in [`normal-forms.md`](../design/normal-forms.md), which is
authoritative for every symbol and normalization below: where this ADR and the design
document disagree, the design document wins.

## Context

### Motivation

- `AbstractODE` was a vector field plus a `solve` that swallowed `**kwargs`, so a
  misspelled `rtol` was ignored without an error
  ([`2026-10-02-data-generation.md`](../changes/2026-10-02-data-generation.md):386-387).
  `BautinNormalForm` overrode it with an `r²` trick in the polar chart.
- `AbstractLatentDynamics` was a closed-form flow for the autoencoder baseline, with a
  `HopfLatentDynamics` that called a normal form as if it were callable and would have
  raised a `TypeError` (`2026-10-02-data-generation.md`:384).
- `ConjugateLatentDynamics` took the union of the two, converted to polar coordinates
  itself, and hard-coded solver settings (`2026-10-02-data-generation.md`:48).
- The normal forms had to expose their phase–amplitude structure, since isochrons are
  the point of the project.
- The flow had to be vmappable with adjustable solver and tolerances, and the Bautin
  form integrable in several ways to compare them.

### Forces

Before this decision the ODE layer had grown in two directions at different times, and
the conjugacy model depended on both. The observed systems have no closed forms, while
the normal forms are chosen precisely because they have them; one interface has to hold
both without pretending the observed systems carry what they do not.

## Decision

### 1. Unified interface for general ODEs and normal forms

`AbstractODE` is *any* ODE in the study: `rhs(t, u, args)`, `dim`, `params()` and the
numerical `flow`. `AbstractNormalForm(AbstractODE)` is the subset usable as a conjugacy
target: planar, with a stable cycle at `r = 1` inside a declared basin
`basin_radii() -> (r_in, r_out)` (default $(0, \infty)$) with `in_basin(u)`, written
through two rates `ρ(r)` (log growth rate, `ṙ = rρ(r)`) and `ω(r)` (`θ̇ = ω(r)`), and
carrying in **closed form** what the observed systems cannot: `period`,
`floquet_exponent` (`κ = ρ'(1)`, by autodiff of `log_growth_rate`), `phase` (asymptotic
phase `Θ = θ + h(r)`), `amplitude` (isostable coordinate `Ψ(r)`, normalized
`∂ᵣΨ(1) = 1`, `Ψ < 0` inside), `limit_cycle`, `isochron`, the polar chart
`to_polar`/`from_polar` and the phase–amplitude chart
`to_phase_amplitude`/`from_phase_amplitude`. The charts are named after their targets,
because `to_chart` was ambiguous once there were two charts. The basin and its `nan`
convention outside it are [ADR-0013](0013-normal-form-basins.md)'s.

The defining data `ρ(r)` and `ω(r)` are declared as SymPy expressions,
`_symbolic_rates()` returning `(r, ρ(r), ω(r), params)`, and the JAX hooks
`log_growth_rate` and `angular_rate` are *derived* from them, compiled once per class
through `sympy2jax` ([ADR-0016](0016-symbolic-systems.md)); the same expressions
generate the AUTO-07p Fortran. `h(r)` and `Ψ(r)` stay hand-derived JAX, verified
symbolically in `tests/test_symbolic.py`. Observed systems have the same structure one
level up (`AbstractSymbolicODE`, `_symbolic_template`).

The even subfamily `AbstractEvenNormalForm(AbstractNormalForm)` declares its data in
`s = r²` instead, `_symbolic_rates_sq()`, with the derived hooks `_log_growth_rate_sq`
and `_angular_rate_sq` (design document §9, "Option B"), so that its cartesian `rhs` is
smooth at the origin; it alone has `eigenvalues_origin`. The public API is in `r` for
every form. The `_sq` hooks carry a leading underscore like the spline contract's
`_forward_in_range` / `_inverse_in_range`: they are the subclass's implementation hooks,
not the user API.

The rule is that **a class carries only what is analytically available; everything
numerical is a function over `AbstractODE`** (`deep_isochron.analysis`: limit-cycle
location, monodromy, numerical asymptotic phase). The rule is unchanged by the move of
the defining data to SymPy, which changed only their language. The name
`AbstractLatentDynamics` was not reused because "latent" names the role in a model, not
the property that justifies the subclass; it stays as the interface of the
non-invertible autoencoder baseline only.

### 2. `SolverConfig`: every solver setting in one leafless module

`flow(ts, u0, args=None, *, config: SolverConfig) -> diffrax.Solution` (``.ys`` the
trajectory, ``.result`` the outcome, ``.stats`` the step counts). `SolverConfig` holds
`solver`, `rtol`, `atol`, `max_steps`, `adjoint`, `throw` as static fields. It has no
array leaves, so under `eqx.filter_vmap`/`filter_jit` it is a static argument and never
traced. The same object is a static field of `ConjugateLatentDynamics` (what Phase C
planned as C2). `flow` is written for one initial condition; batching is
`eqx.filter_vmap(ode.flow, in_axes=(None, 0))`. `Solution.result` carries diffrax's
`RESULTS`, which is needed because under `vmap` `throw=True` raises if *any* element
fails [diffrax docs]; data generation uses `throw=False` and reports the failing
indices. Integrations that change chart return the same `Solution` with `.ys` replaced
(`eqx.tree_at`), so callers always read cartesian `.ys`.

### 3. Flow integrations are objects, resolved from names

`AbstractNormalForm.flow(..., integration=...)` takes an `AbstractFlowIntegration`
instance from
[`integration.py`](../../src/deep_isochron/systems/normal_forms/integration.py):
`CartesianIntegration`, `PolarIntegration`, `RadiusSquaredIntegration` (the even family
only; it raises otherwise), or `ClosedFormIntegration` (`Θ(t) = Θ₀ + ω₁t`,
`Ψ(t) = Ψ₀e^{κt}` in the phase–amplitude chart, then `from_phase_amplitude`; no ODE
solve, the `SolverConfig` is ignored).

This design corresponds to the Strategy pattern, but we deliberately avoid that word
here since `tests/strategies.py` already reserves the word for Hypothesis strategies. A
caller passes an integration instance, or one of the names
`"cartesian" | "polar" | "r_squared" | "closed_form"`, which maps to a default instance;
`None` selects the class's `default_integration`.

For every normal form, `default_integration` is `"closed_form"`: it is exact and cheaper
than any ODE solve, so a numerical integration is an explicit choice made for
comparison.

Note that diffrax also adopts the same pattern to pass solver objects to `diffeqsolve`.
Since an integration is an `eqx.Module` with no array leaves, it is static under the
filter transforms: each traces separately and dispatch costs nothing at run time
(`test_integrations_are_static_under_filter_jit`). Every integration takes and returns
**cartesian** coordinates, so the caller never knows which chart was used internally;
`ConjugateLatentDynamics` lost its polar conversion as a result.

### 4. Normal-form parameters as unconstrained leaves

`a > 0` is kept by `GreaterThan(0.0)` and Bautin's `b > −1` by `GreaterThan(−1.0)`. With
the default `at_zero = lower + 1` these map `raw = 0` to `a = 1` and `b = 0`, the Hopf
form, without any explicit shift at the call site (`A_CONSTRAINT`, `B_CONSTRAINT`).
`b > −1` is exactly `ρ'(1) < 0`, the cycle's stability. For `−1 < b < 0` a second,
unstable cycle at `r² = −1/b` bounds the basin, the Bautin bifurcation's two-cycle
regime; as a conjugacy target `b ≥ 0` is used and the tests restrict the flow laws to
it, while a dedicated test pins the outer cycle. `w0` defaults to `w`; the former `None`
made `θ̇ = w` but reported zero imaginary part at the origin.

## Consequences

- `BautinNormalForm.solve` (the `r²` trick) is now `RadiusSquaredIntegration`, usable
  for any even normal form; the former code was Bautin-specific.
- Closed forms are tested exactly, by autodiff of the defining identities
  (`ω(r) + h'(r)·rρ(r) = ω₁`, `Ψ'(r)·rρ(r) = κΨ`, `Ψ'(1) = 1`), and again by integration
  (design document §10).
- `from_phase_amplitude` needs `Ψ⁻¹`; Hopf has it explicitly, Bautin by a bracketed,
  bisection-safeguarded Newton solve in `ℓ = ln r` with an implicit-function JVP
  (`eqx.filter_custom_jvp`), so `radius_from_isostable` is differentiable in `Ψ` and in
  the parameters, returns `nan` for `Ψ` outside the range (`Ψ ≥ Ψ(∞)` for `b > 0`), and
  handles the basin edge for `−1 < b < 0`.
- Both models implement `AbstractPhaseAmplitudeModel` of
  [`model/base.py`](../../src/deep_isochron/model/base.py): `phase`, `amplitude`
  (isostable-like, with *no fixed normalization*: the conjugacy model's is `Ψ`, the
  autoencoder's `Y₃` is only up to scale), `cycle_point`, `__call__`;
  `phase_gradient`/`phase_sensitivity` derived. The contract is minimal on purpose and
  grows with the science. `ConjugateLatentDynamics.latent_dynamics` is typed
  `AbstractNormalForm` accordingly (phase and amplitude are transported closed forms).
- `HopfLatentDynamics` deleted. The non-invertible baseline is the phase autoencoder of
  Yawata et al. 2024 (B12): `PhaseAmplitudeLatentDynamics(omega, kappa)` replaced
  `LinearLatentDynamics`; its latent space is the normal forms' `(Θ, Ψ)` chart (design
  document §5.3).
- The numerical counterparts live in `analysis/` (Phase E).
- `AbstractODE.params()` returns the constrained parameter values keyed by their
  mathematical names; dataset metadata records it, nothing inspects attribute names.

## Alternatives rejected

- Making `AbstractLatentDynamics` the subclass. It has no `rhs`; it is a closed-form
  flow, which is a different contract.
- Numerical isochron methods on `AbstractODE`. They need solver choices, grids and
  continuation parameters that do not belong on a vector field.
- Four keyword arguments instead of `SolverConfig`. They were already being swallowed by
  `**kwargs`, and the trainer will want to carry the same settings around.
- A `Literal` string keyword with an `if`/dict dispatch. Works (strings are static under
  `filter_jit` too) but cannot carry options and is not extensible without editing the
  class; kept only as the shorthand.
- `lax.switch` on a traced mode. Compiles all branches into one program and forbids
  branch-specific state shapes; nothing here needs a runtime choice.
- *Superseded 2026-10-08*: every normal form's defining data as smooth functions of
  `s = r²` (`_log_growth_rate_sq`, `_angular_rate_sq`) on the punctured plane, with the
  basin implicit. Replaced because Winfree's model with a hole is not even in `r` and
  has a phaseless disk, which needs the base class in `r` with a declared basin; the
  `s = r²` contract became the even subfamily (ADR-0013).
- *Superseded 2026-10-22*: the defining data written as JAX hooks by hand. Replaced
  because the same expressions must also generate the AUTO-07p Fortran (ADR-0016).
- *Superseded 2026-10-03*: a separate `flow_result` beside `flow`, and the first
  implementation that mixed `r` and `s` in the public methods (review round 1). Replaced
  by `flow` returning the `diffrax.Solution` and a public API in `r` (B11).

## History

- 2026-10-02 (`data-generation`): accepted.
- 2026-10-03 (`data-generation`, review round 1): the objects named *integration*, not
  *strategy*; `flow` returns the `diffrax.Solution`; `params()` added to the
  `AbstractODE` contract; `GreaterThan`.
- 2026-10-03 (`data-generation`, B11): the normal forms reimplemented against the design
  document, `_sq` hooks with the public API in `r`; `ClosedFormIntegration`.
- 2026-10-04 (`data-generation`, review round 3): `to_polar`; the `_sq` hooks
  underscored; `default_integration`; `AbstractPhaseAmplitudeModel`.
- 2026-10-08 (`normal-forms-basin`, ADR-0013): the base class in `r` with a declared
  basin; the `s = r²` contract moved to `AbstractEvenNormalForm`.
- 2026-10-22 (`symbolic-systems`, ADR-0016): the defining data declared in SymPy and the
  JAX hooks derived.
