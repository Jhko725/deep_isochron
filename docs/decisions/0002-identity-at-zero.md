---
type: decision
id: ADR-0002
status: accepted
updated: 2026-10-01
verified_by: joon
---

# ADR-0002 — Identity at zero is a property of the constraining map, not of initialization

**Status**: accepted (2026-10-01)

## Context

`CouplingFlow` zero-initializes its conditioner's final layer so that a fresh layer is the
identity. This only works if the scalar bijection built from `raw = 0` is the identity map.
flowjax and bijx achieve identity-at-init differently: they *initialize* the raw value to the
preimage of the identity (`inv_softplus(1 - min_derivative)`, `softplus⁻¹(0.9) - 1`), with the
constraining map itself unshifted. That works for standalone modules but not for a
zero-initialized conditioner, whose output is zero, not the preimage.

## Decision

Every constraint primitive with a neutral value takes an `at_zero` argument and maps
`raw = 0` to it (`Positive(eps, at_zero)`, `Interval(lo, hi, at_zero)`,
`BoundedPositive(eps, at_zero, a)`, `Widths` → equal widths). `constrain(0)` is the identity
map's parameters for every scalar bijection; `identity_like()` is `from_unconstrained(0)`.
The shift is a constant computed once with `math` in the primitive's constructor.

## Consequences

- Zero conditioner output ⇒ identity layer, for MLP and Fourier conditioners alike.
- `cls(config)` is the identity instance without any per-class initialization code.
- Pinned by `test_scalar_identity_at_zero` (atol 1e-12) and `test_elementwise_at_zero`.

## Alternatives rejected

- Preimage initialization (flowjax, bijx): incompatible with zero-initialized conditioners.
