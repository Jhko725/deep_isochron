---
type: decision
id: ADR-0007
status: accepted; amended
updated: 2026-10-04
verified_by: joon
---

# ADR-0007 — `AbstractODE` / `AbstractNormalForm`, `SolverConfig`, and flow integrations as objects

**Status**: accepted (2026-10-02); amended 2026-10-03 after review (names: *integration* not *strategy*; `flow` returns the `diffrax.Solution`; `params()` added to the `AbstractODE` contract; `GreaterThan`); amended 2026-10-03 (B11) and 2026-10-04 (review round 3: `to_polar`, `_sq` hooks
underscored, `default_integration`, `AbstractPhaseAmplitudeModel`): the mathematics lives in
`docs/design/normal-forms.md`, which is authoritative for every symbol and normalisation
below — where this ADR and the design document disagree, the design document wins.

## Context

Before this branch the ODE layer had grown in two directions at different times:
`AbstractODE` (vector field + a `solve` that swallowed `**kwargs`, with `BautinNormalForm`
overriding it with an `r²` trick in the polar chart) and `AbstractLatentDynamics` (a
closed-form flow for the autoencoder baseline, with a `HopfLatentDynamics` that called a
normal form as if it were callable). `ConjugateLatentDynamics` took the union of the two,
converted to polar coordinates itself, and hard-coded solver settings. Three needs drove the
redesign: the normal forms must expose their phase–amplitude structure (isochrons are the
point of the project); the flow must be vmappable with adjustable solver and tolerances;
and the Bautin form must be integrable by several strategies to compare them.

## Decisions

### 1. Two layers, one rule

`AbstractODE` is *any* ODE in the study: `rhs(t, u, args)`, `dim`, and the numerical
`flow`/`flow_result`. `AbstractNormalForm(AbstractODE)` is the subset usable as a conjugacy
target: planar, with a stable cycle at `r = 1`, written through two rates `ρ(r)` (log
growth rate, `ṙ = rρ(r)`) and `ω(r)` (`θ̇ = ω(r)`), and carrying in **closed form** what
the observed systems cannot: `period`, `floquet_exponent` (`κ = ρ'(1)`, by autodiff of
`log_growth_rate`), `eigenvalues_origin`, `phase` (asymptotic phase `Θ = θ + h(r)`),
`amplitude` (isostable coordinate `Ψ(r)`, normalised `∂ᵣΨ(1) = 1`, `Ψ < 0` inside),
`limit_cycle`, `isochron`, the polar chart `to_polar`/`from_polar` and the phase–amplitude
chart `to_phase_amplitude`/`from_phase_amplitude` (named after their targets since review
round 3; `to_chart` was ambiguous once there were two charts). Subclasses implement the defining data
as smooth functions of `s = r²` (`_log_growth_rate_sq`, `_angular_rate_sq`, design document
§9, "Option B"), so the cartesian `rhs` is smooth at the origin; the public API is in `r`
(B11; the first implementation mixed `r` and `s` in the public methods — review round 1).
The hooks carry a leading underscore like the spline contract's `_forward_in_range` /
`_inverse_in_range`: they are the subclass's implementation surface, not the user API
(review round 3). The rule: **a class
carries only what is analytically available; everything numerical is a function over
`AbstractODE`** (the future `deep_isochron.analysis`: limit-cycle location, monodromy,
numerical asymptotic phase). The name `AbstractLatentDynamics` was not reused because
"latent" names the role in a model, not the property that justifies the subclass; it stays
as the interface of the non-invertible autoencoder baseline only.

- *Rejected*: making `AbstractLatentDynamics` the subclass. It has no `rhs`; it is a
  closed-form flow, which is a different contract.
- *Rejected*: numerical isochron methods on `AbstractODE`. They need solver choices,
  grids and continuation parameters that do not belong on a vector field.

### 2. `SolverConfig`: every solver setting in one leafless module

`flow(ts, u0, args=None, *, config: SolverConfig) -> diffrax.Solution` (``.ys`` the trajectory, ``.result`` the outcome, ``.stats`` the step counts; there is no separate `flow_result`). `SolverConfig` holds `solver`, `rtol`,
`atol`, `max_steps`, `adjoint`, `throw` as static fields; it has no array leaves, so under
`eqx.filter_vmap`/`filter_jit` it is a static argument and never traced. The same object is
a static field of `ConjugateLatentDynamics` (what Phase C planned as C2). `flow` is written
for one initial condition; batching is `eqx.filter_vmap(ode.flow, in_axes=(None, 0))`.
`Solution.result` carries diffrax's `RESULTS`, needed because under `vmap` `throw=True`
raises if *any* element fails [diffrax docs]; data generation uses `throw=False` and reports
the failing indices. Integrations that change chart return the same `Solution` with `.ys`
replaced (`eqx.tree_at`), so callers always read cartesian `.ys`.

- *Rejected*: four keyword arguments. They were already being swallowed by `**kwargs`,
  and the trainer will want to carry the same settings around.

### 3. Flow integrations are objects, resolved from names

`AbstractNormalForm.flow(..., integration=...)` takes an `AbstractFlowIntegration` instance
(`systems/normal_forms/integration.py`; the word *strategy* was dropped because
`tests/strategies.py` already means Hypothesis strategies) —
`CartesianIntegration`, `PolarIntegration`, `RadiusSquaredIntegration`,
`ClosedFormIntegration` (B11: `Θ(t) = Θ₀ + ω₁t`, `Ψ(t) = Ψ₀e^{κt}` in the phase–amplitude
chart, then `from_phase_amplitude`; no ODE solve, the `SolverConfig` is ignored) — or one
of the names `"cartesian" | "polar" | "r_squared" | "closed_form"` mapped to default
instances, or `None` for the class's `default_integration` — `"closed_form"` for every
normal form (review round 3): it is exact and cheaper than any ODE solve, so a numerical
integration is an explicit choice made for comparison. This is the
diffrax pattern (pass a solver object). A strategy is an `eqx.Module` with no array leaves,
hence static under the filter transforms: each traces separately and dispatch costs nothing
at run time (`test_strategies_are_static_under_filter_jit`). Every strategy takes and
returns **cartesian** coordinates, so the caller never knows which chart was used
internally; `ConjugateLatentDynamics` lost its polar conversion as a result.

- *Rejected*: a `Literal` string keyword with an `if`/dict dispatch. Works (strings are
  static under `filter_jit` too) but cannot carry options and is not extensible without
  editing the class; kept only as the shorthand.
- *Rejected*: `lax.switch` on a traced mode. Compiles all branches into one program and
  forbids branch-specific state shapes; nothing here needs a runtime choice.

### 4. Normal-form parameters as unconstrained leaves

`a > 0` via `GreaterThan(0.0)` and Bautin's `b > −1` via `GreaterThan(−1.0)`; with the
default `at_zero = lower + 1` these map `raw = 0` to `a = 1` and `b = 0` (the Hopf form)
without any explicit shift at the call site (`A_CONSTRAINT`, `B_CONSTRAINT`). `b > −1` is exactly
`ρ'(1) < 0`, the cycle's stability. For `−1 < b < 0` a second, unstable cycle at `r² = −1/b`
bounds the basin (the Bautin bifurcation's two-cycle regime); as a conjugacy target
`b ≥ 0` is used and the tests restrict the flow laws to it, while a dedicated test pins the
outer cycle. `w0` defaults to `w` (the former `None` made `θ̇ = w` but reported zero
imaginary part at the origin).

## Consequences

- `BautinNormalForm.solve` (the `r²` trick) is now `RadiusSquaredIntegration`, usable for
  any normal form; the former code was Bautin-specific.
- Closed forms are tested exactly, by autodiff of the defining identities
  (`ω(r) + h'(r)·rρ(r) = ω₁`, `Ψ'(r)·rρ(r) = κΨ`, `Ψ'(1) = 1`), and again by integration
  (design document §10).
- `from_phase_amplitude` needs `Ψ⁻¹`; Hopf has it explicitly, Bautin by a bracketed,
  bisection-safeguarded Newton solve in `ℓ = ln r` with an implicit-function JVP
  (`eqx.filter_custom_jvp`), so `radius_from_isostable` is differentiable in `Ψ` and in the
  parameters, returns `nan` for `Ψ` outside the range (`Ψ ≥ Ψ(∞)` for `b > 0`), and handles
  the basin edge for `−1 < b < 0`.
- Both models implement `model/base.py: AbstractPhaseAmplitudeModel` (review round 3):
  `phase`, `amplitude` (isostable-like, *no fixed normalisation* — the conjugacy model's
  is `Ψ`, the autoencoder's `Y₃` is only up to scale), `cycle_point`, `__call__`;
  `phase_gradient`/`phase_sensitivity` derived. Minimal on purpose; it grows with the
  science. `ConjugateLatentDynamics.latent_dynamics` is typed `AbstractNormalForm`
  accordingly (phase and amplitude are transported closed forms).
- `HopfLatentDynamics` deleted. The non-invertible baseline is the phase autoencoder of
  Yawata et al. (2024) (B12): `PhaseAmplitudeLatentDynamics(omega, kappa)` replaced
  `LinearLatentDynamics`; its latent space is the normal forms' `(Θ, Ψ)` chart (design
  document §5.3).
- The numerical counterparts (`analysis/`) are Phase E; the namespace is reserved.
- `AbstractODE.params()` returns the constrained parameter values keyed by their
  mathematical names; dataset metadata records it, nothing inspects attribute names.
