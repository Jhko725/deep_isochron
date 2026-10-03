# ADR-0007 — `AbstractODE` / `AbstractNormalForm`, `SolverConfig`, and flow integrations as objects

**Status**: accepted (2026-10-02); amended 2026-10-03 after review (names: *integration* not *strategy*; `flow` returns the `diffrax.Solution`; `params()` added to the `AbstractODE` contract; `GreaterThan`). The mathematics will move to `docs/design/normal-forms.md` (roadmap B10).

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
target: planar, with a stable cycle at `r = 1`, written through two rates `ρ(s)`, `ω(s)`
(`s = r²`) so that `ṙ = rρ(r²)`, `θ̇ = ω(r²)`, and carrying in **closed form** what the
observed systems cannot: `period`, `floquet_exponent` (`2ρ'(1)`, by autodiff of `ρ`),
`phase` (asymptotic phase `θ + h(r)`), `amplitude` (isostable coordinate `ψ(r)`),
`limit_cycle`, `isochron`, and the polar chart `to_chart`/`from_chart`. The rule: **a class
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
`CartesianIntegration`, `PolarIntegration`, `RadiusSquaredIntegration` — or one of the
names `"cartesian" | "polar" | "r_squared"` mapped to default instances. This is the
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
  (`ω(r²) + h'(r)·rρ(r²) = ω(1)`, `ψ'(r)·rρ(r²) = κψ`), and again by integration.
- `HopfLatentDynamics` deleted; `LinearLatentDynamics` and `PhaseAmplitudeAutoencoder` kept
  as the non-invertible baseline.
- The numerical counterparts (`analysis/`) are Phase E; the namespace is reserved.
- `AbstractODE.params()` returns the constrained parameter values keyed by their
  mathematical names; dataset metadata records it, nothing inspects attribute names.
