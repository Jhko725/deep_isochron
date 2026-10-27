---
type: decision
id: ADR-0003
status: accepted
updated: 2026-10-27
verified_by: joon
---

# ADR-0003 — Regularity is declared, not inferred

**Status**: accepted (2026-10-01)

## Context

The project needs the conjugating map to be C², because the iPRC, the iARC and the
curvature computations take second derivatives. The bijection families differ in
regularity (linear spline C⁰, RQ spline C¹, cubic B-spline C², analytic maps C^∞), and a
container inherits the minimum of its members. The spline tests originally kept these
facts as a `JOIN_ORDER` table in the test file.

## Decision

`AbstractBijection` declares `smoothness: eqx.AbstractVar[int | None]`, corresponding to the `k` of $C^k$. `None` corresponds to C^∞. Leaf classes fix `smoothness` (`CubicBSpline`: 2). Containers derive it:
`CouplingFlow` reports its template's, `ScalarChain` and `SequentialINN` the minimum over
their members with `None` as +∞, and a wrapper the wrapped bijection's.

Note that the conditioner's
activation is *not* tracked. The docstring states that it must be at least as smooth as
the template, and `CouplingFlow.__init__` performs only a minimal check, rejecting the non-smooth `jax.nn` activations
(`relu`, `relu6`, `leaky_relu`, `hard_tanh`, `hard_sigmoid`, `hard_swish`, `hard_silu`)
when the template is C¹ or better. A custom conditioner is assumed smooth.

## Consequences

- The regularity of a whole INN is a one-line query on the model (`inn.smoothness`).
- `test_splines.JOIN_ORDER` is derived from the classes, not declared in the test.
- `CircularMonotonicRQCoupling` declares `0`, since it is C⁰ across θ = ±π and continuous
  only at the origin. The attribute makes a limitation visible that a docstring alone
  would not.

## Alternatives rejected

- An `activation_smoothness` field or a per-activation table: manual bookkeeping that
  protects against nothing a user supplying the activation is not already aware of. The
  guard covers the one realistic mistake at zero maintenance.
