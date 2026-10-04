---
type: decision
id: ADR-0003
status: accepted
updated: 2026-10-01
verified_by: joon
---

# ADR-0003 — Regularity is declared, not inferred

**Status**: accepted (2026-10-01)

## Context

The project needs the conjugating map to be C² (iPRC/iARC and curvature computations take
second derivatives). Different bijection families have different regularity — linear spline
C⁰, RQ spline C¹, cubic B-spline C², analytic maps C^∞ — and containers inherit the minimum.
The spline tests originally kept this as a `JOIN_ORDER` table in the test file.

## Decision

`smoothness: eqx.AbstractVar[int | None]` on `AbstractBijection` (C^k; `None` = C^∞). Leaf
classes fix it (`CubicBSpline`: 2); containers derive it (`CouplingFlow` = template's;
`ScalarChain`/`SequentialINN` = minimum over members, `None` as +∞; wrappers = wrapped).
The conditioner's activation is *not* tracked: the docstring states it must be at least as
smooth as the template, and `CouplingFlow.__init__` rejects the non-smooth `jax.nn`
activations (`relu`, `relu6`, `leaky_relu`, `hard_tanh`, `hard_sigmoid`, `hard_swish`,
`hard_silu`) when the template is C¹ or better. A custom conditioner is assumed smooth.

## Consequences

- Whole-INN regularity is a one-line query on the model (`inn.smoothness`).
- `test_splines.JOIN_ORDER` is derived from the classes, not declared in the test.
- `CircularMonotonicRQCoupling` declares `0` (C⁰ across θ = ±π, continuous only at the
  origin) — the attribute makes a limitation visible that a docstring alone would not.

## Alternatives rejected

- `activation_smoothness` field or a per-activation table — manual bookkeeping that protects
  against nothing a user supplying it isn't already aware of; the guard covers the one
  realistic mistake at zero maintenance.
