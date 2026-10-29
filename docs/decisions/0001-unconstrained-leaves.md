---
type: decision
id: ADR-0001
status: accepted
updated: 2026-10-27
verified_by: joon
---

# ADR-0001 — Unconstrained parameters are the trainable leaves of scalar bijections

## Context

### Motivation

- An optimizer step on a standalone instance could leave the constrained set when the
  constrained values were the leaves, because `__check_init__` runs at construction only
  and `eqx.apply_updates` bypasses `__init__`.
- That design forced an `_identity` helper and a second kind of object in the tests
  (Joon's review of [the splines branch](../changes/2026-09-30-splines-refactor.md),
  line 160).
- The design before it computed the constrained values as properties of raw fields, and
  duplicated the softplus and sigmoid formulas in a `from_unnormalized_params`
  classmethod.
- `CouplingFlow` wrote MLP output straight into the constrained leaves through
  `BijectionFactory`, with no positivity and no identity at init
  ([`2026-09-30-splines-refactor.md`](../changes/2026-09-30-splines-refactor.md),
  lines 104–105).

### Forces

Scalar bijections (analytic maps, monotone splines) have parameters that must lie in a
constrained set: positive widths and derivatives, `alpha` in an interval, a positive
scale. They are used in two ways. Standalone, a bijection is trainable state held
directly by a module (`RadialBijection`, `CircularMonotonicRQCoupling`); inside
`CouplingFlow`, a conditioner regenerates its parameters from a flat unconstrained
vector on every call. A design has to serve both uses.

## Decision

Every `AbstractScalarBijection` has exactly one trainable leaf, `raw`, which is an
unconstrained vector of length `num_params`. One method, `constrain(raw) -> Params`,
computes the constrained parameters on read from the primitives in
[`invertible/constraints.py`](../../src/deep_isochron/model/invertible/constraints.py);
`Params` is a NamedTuple that the class is generic over. `__call__` and `inverse` read
`self.params`. `from_unconstrained(raw)` is a `tree_at` on `raw`, defined once on the base
class. An instance with `raw=None` is a *template*: static configuration only, hashable,
of zero trainable size, and what `CouplingFlow` stores. `cls(config)` with `raw` omitted
is the identity instance.

## Consequences

- Standalone training is safe by construction, because the constrained set is the image
  of the map. Pinned by `test_scalar_standalone_training_keeps_validity`.
- The conversion formulas exist in one place per class; `check_positive`,
  `__check_init__` and `constrain_widths` are gone.
- Inspecting a trained spline means `spline.params.x_widths`, not a leaf.
- `constrain` is a *method*, not a declarative table, because the B-spline's coefficient
  constraint depends on another constrained quantity, the Greville span from the knots.

## Alternatives rejected

- *Superseded 2026-10-01*: **constrained leaves** (the splines branch, 2026-09-30). It
  gave `num_trainable_params == num_params` and a simpler coupling path; replaced
  because it is unsafe for standalone training, as above.
- *Superseded 2026-09-30*: constrained values as properties of raw fields (the first
  version). It trained safely; replaced because it duplicated the conversion formulas in
  `from_unnormalized_params`.
- **Per-parameter wrappers** (flowjax/paramax `Parameterize` + `unwrap`, bijx
  `TransformedParameter.get_value()`). The same invariant, but per field: they cannot
  express the B-spline's dependent constraint, and they need a reconstructor to bridge the
  conditioner's flat vector to per-field leaves (bijx `ModuleReconstructor`, the old
  `BijectionFactory`).
- **quax `ArrayValue`.** A dispatching array type for uniform semantics (units, sparsity).
  A sliced vector with heterogeneous constraints is not that, every consumer would need
  `quaxify`, and the values are not `jax.Array`s for jaxtyping.

## History

- 2026-10-01 (`scalar-param-refactor`): accepted.
