---
type: decision
id: ADR-0001
status: accepted
updated: 2026-10-01
verified_by: joon
---

# ADR-0001 — Unconstrained parameters are the trainable leaves of scalar bijections

**Status**: accepted (2026-10-01, branch `scalar-param-refactor`)

## Context

Scalar bijections (analytic maps, monotone splines) have parameters that must lie in a
constrained set — positive widths and derivatives, `alpha` in an interval, a positive scale.
They are used in two ways: standalone, as trainable state held directly by a module
(`RadialBijection`, `CircularMonotonicRQCoupling`), and inside `CouplingFlow`, where a
conditioner regenerates them from a flat unconstrained vector on every call.

Two earlier designs each satisfied one use only. Constrained values as properties computed
from raw fields (first version) trained safely but duplicated the softplus/sigmoid formulas
in a `from_unnormalized_params` classmethod. Constrained values as leaves (splines branch)
made `num_trainable_params == num_params` and simplified the coupling path, but an optimizer
step on a standalone instance could leave the constrained set — `__check_init__` runs at
construction only, and `eqx.apply_updates` bypasses `__init__` — which forced an
`_identity` helper and a second kind of object in the tests.

## Decision

Every `AbstractScalarBijection` has exactly one trainable leaf, `raw`, an unconstrained
vector of length `num_params`. Constrained parameters are computed on read by one method,
`constrain(raw) -> Params` (a NamedTuple the class is generic over), built from the
primitives in `invertible/constraints.py`. `__call__`/`inverse` read `self.params`;
`from_unconstrained(raw)` is a `tree_at` on `raw` defined once on the base class; an
instance with `raw=None` is a *template* (static configuration only, hashable, zero
trainable size) and is what `CouplingFlow` stores. `cls(config)` with `raw` omitted is the
identity instance.

## Consequences

- Standalone training is safe by construction: the constrained set is the image of the map.
  Pinned by `test_scalar_standalone_training_keeps_validity`.
- The conversion formulas exist in one place per class; `check_positive`,
  `__check_init__` and `constrain_widths` are gone.
- Inspecting a trained spline means `spline.params.x_widths`, not a leaf.
- `constrain` is a *method*, not a declarative table, because the B-spline's coefficient
  constraint depends on another constrained quantity (the Greville span from the knots).

## Alternatives rejected

- **Constrained leaves** — unsafe for standalone training (above).
- **Per-parameter wrappers** (flowjax/paramax `Parameterize` + `unwrap`, bijx
  `TransformedParameter.get_value()`) — the same invariant, but per field: cannot express the
  B-spline's dependent constraint, and needs a reconstructor to bridge the conditioner's flat
  vector to per-field leaves (bijx `ModuleReconstructor`, the old `BijectionFactory`).
- **quax `ArrayValue`** — a dispatching array type for uniform semantics (units, sparsity);
  a sliced vector with heterogeneous constraints is not that, every consumer would need
  `quaxify`, and the values are not `jax.Array`s for jaxtyping.
