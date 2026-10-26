---
type: decision
id: ADR-0001
status: accepted
updated: 2026-10-27
verified_by: joon
---

# ADR-0001 — Unconstrained parameters are the trainable leaves of scalar bijections

**Status**: accepted (2026-10-01, branch `scalar-param-refactor`)

## Context

Scalar bijections (analytic maps, monotone splines) have parameters that must lie in a
constrained set: positive widths and derivatives, `alpha` in an interval, a positive scale.
They are used in two ways. Standalone, a bijection is trainable state held directly by a
module (`RadialBijection`, `CircularMonotonicRQCoupling`); inside `CouplingFlow`, a
conditioner regenerates its parameters from a flat unconstrained vector on every call.

Two earlier designs each satisfied one use only. The first version computed the
constrained values as properties of raw fields. It trained safely but duplicated the
softplus and sigmoid formulas in a `from_unnormalized_params` classmethod. The splines
branch made the constrained values the leaves, which gave
`num_trainable_params == num_params` and a simpler coupling path, but an optimizer step
on a standalone instance could leave the constrained set, because `__check_init__` runs
at construction only and `eqx.apply_updates` bypasses `__init__`. That design forced an
`_identity` helper and a second kind of object in the tests.

## Decision

Every `AbstractScalarBijection` has exactly one trainable leaf, `raw`, an unconstrained
vector of length `num_params`. One method, `constrain(raw) -> Params`, computes the
constrained parameters on read from the primitives in `invertible/constraints.py`;
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

- **Constrained leaves.** Unsafe for standalone training, as above.
- **Per-parameter wrappers** (flowjax/paramax `Parameterize` + `unwrap`, bijx
  `TransformedParameter.get_value()`). The same invariant, but per field: they cannot
  express the B-spline's dependent constraint, and they need a reconstructor to bridge the
  conditioner's flat vector to per-field leaves (bijx `ModuleReconstructor`, the old
  `BijectionFactory`).
- **quax `ArrayValue`.** A dispatching array type for uniform semantics (units, sparsity).
  A sliced vector with heterogeneous constraints is not that, every consumer would need
  `quaxify`, and the values are not `jax.Array`s for jaxtyping.
