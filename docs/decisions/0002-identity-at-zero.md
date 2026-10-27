---
type: decision
id: ADR-0002
status: accepted
updated: 2026-10-27
verified_by: joon
---

# ADR-0002 — Identity at zero is a property of the constraining map, not of initialization

## Context

### Motivation

- `CouplingFlow` had no identity at init: it wrote MLP output straight into the
  constrained leaves
  ([`2026-09-30-splines-refactor.md`](../changes/2026-09-30-splines-refactor.md),
  lines 104–105).
- The identity shift of `MonotonicRQSpline` had a sign error, `1 − inv_softplus(1 − ε)`
  for `+ inv_softplus(1 − ε)`, which was the source of the 1e-3 `IDENTITY_TOL` that the
  tests carried (same document, lines 92–93).
- The tests skipped the identity-at-init check for `residual_coupling`,
  `invertible_linear` and `bilipschitz_linear` through `IDENTITY_AT_INIT`, since those
  layers were not the identity at init
  ([the scalar-param change document](../changes/2026-10-01-scalar-param-refactor.md),
  line 248).

### Forces

`CouplingFlow` zero-initializes its conditioner's final layer so that a fresh layer is
the identity. The zero output is the identity only if the scalar bijection built from
`raw = 0` is the identity map. flowjax and bijx reach identity at init differently: they
*initialize* the raw value to the preimage of the identity
(`inv_softplus(1 - min_derivative)`, `softplus⁻¹(0.9) - 1`) and leave the constraining
map unshifted. That works for standalone modules but not for a zero-initialized
conditioner, whose output is zero, not the preimage.

## Decision

Every constraint primitive with a neutral value takes an `at_zero` argument and maps
`raw = 0` to it (`Positive(eps, at_zero)`, `Interval(lo, hi, at_zero)`,
`BoundedPositive(eps, at_zero, a)`, `Widths` → equal widths). `constrain(0)` is the
identity map's parameters for every scalar bijection, and `identity_like()` is
`from_unconstrained(0)`. The shift is the constant `_inverse(at_zero)`, computed on
first use under `jax.ensure_compile_time_eval`
([ADR-0005](0005-constraint-primitives.md), rule 3).

## Consequences

- Zero conditioner output gives the identity layer, for MLP and Fourier conditioners
  alike.
- `cls(config)` is the identity instance without any per-class initialization code.
- Pinned by `test_scalar_identity_at_zero` (atol 1e-12) and `test_elementwise_at_zero`.

## Alternatives rejected

- Preimage initialization (flowjax, bijx): incompatible with zero-initialized
  conditioners.
- *Superseded 2026-10-01*: shift computation using the `math` module. Avoids triggering
  recompilation without `jax.ensure_compile_time_eval`, but requires one to duplicate
  the `.inverse` logic with `math` functions; replaced because of that duplication
  ([ADR-0005](0005-constraint-primitives.md), rule 3).

## History

- 2026-10-01 (`scalar-param-refactor`): accepted. Within the same branch the shift moved
  from `math` to the primitive's own `_inverse` under `jax.ensure_compile_time_eval`
  ([ADR-0005](0005-constraint-primitives.md), rule 3).
- 2026-10-04 (`trainer`): the shift is computed on first use rather than at construction
  (ADR-0005's history).
- 2026-10-27 (`prose`): the Decision's shift sentence had described the `math` route
  until Joon's review corrected it; it now links to ADR-0005 instead of restating the
  mechanism.
