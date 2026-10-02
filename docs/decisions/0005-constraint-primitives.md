# ADR-0005 — Constraint primitives are plain Python objects created inside `constrain`

**Status**: accepted (2026-10-01)

## Context

ADR-0001 makes `constrain(raw)` the single place where a scalar bijection maps its
unconstrained leaf onto the constrained set. The maps themselves — shifted softplus,
shifted sigmoid, floored softmax, `arcsinh`, the bounded `squashed_exp` — are shared by
several classes and each needs an inverse (for `from_constrained`) and the identity-at-zero
shift (ADR-0002). They need a home, and the question is what kind of object that home is,
given that it is used inside `jit`/`vmap`/`grad`-transformed code.

Optax implements its transformations' *states* as `NamedTuple`s because a state is an
argument and a return value of `update`, which runs under `jit`: anything crossing such a
boundary must be a registered pytree so JAX can flatten it to arrays and rebuild it. The same
reasoning is why bijections are Equinox modules. It is tempting to conclude that every
object touching JAX code must be a pytree; that is not the rule.

## Decision

Constraint primitives (`Free`, `Positive`, `Interval`, `BoundedPositive`, `Arcsinh`,
`Widths` in `invertible/constraints.py`) are plain Python classes (`abc.ABC`, not
`eqx.Module`) with `__call__` (raw → constrained) and `inverse` (constrained → raw). They
hold only static configuration — `eps`, `at_zero`, bounds, `min_rel`, and the shift constant
computed once in `__init__` under `jax.ensure_compile_time_eval()`. They are **instantiated inside `constrain`**,
called once, and dropped:

```python
def constrain(self, raw):  # traced
    widths = Widths(self.range_width, self.min_rel_width)  # lives inside the trace
    return RQSplineParams(widths(x_raw), widths(y_raw), ...)  # only arrays leave
```

### Why this is correct under `jit`/`vmap`/`grad`

JAX needs to understand an object's structure only when it is an **argument or return value**
of a transformed function. A primitive created and consumed inside the trace is no different
from calling a helper function with closed-over Python constants: there is nothing to flatten,
nothing to trace, and nothing to retrace on later calls. The test suite exercises exactly
this path — `roundtrip`/`jacobian_dets` are `eqx.filter_jit` over `jax.vmap`, the
orientation laws go through `jacfwd`, the extreme-regime tests through `grad³`, and
`CouplingFlow` calls `constrain` under `filter_vmap` of `from_unconstrained` on every forward
pass. Equinox's own layers call plain Python callables (activation functions) inside the
trace the same way.

### Rules that keep it correct

1. **Never store a primitive on a module.** As a field it would have to be an array leaf
   (it isn't) or a hashable static field (its `total` may be an array). Constructing them in
   `constrain` makes the question moot. This is also why they are not `eqx.Module`s: making
   them modules would invite storing them.
2. **A tracer may live in a primitive only within the trace that created it.**
   `CubicBSpline.constrain` builds `Widths(total=span)` with a traced `span` (the Greville
   span, which depends on `raw`); fine, because the object is consumed in the same trace.
   Returning or caching such an object would leak a tracer — the standard JAX rule.
3. **Shift constants are computed under `jax.ensure_compile_time_eval()`.** A shifted
   primitive implements the *unshifted* map pair `_forward`/`_inverse` once; the shift is
   `float(_inverse(at_zero))`, evaluated in that context so it is a concrete Python float even
   when the primitive is constructed inside a `jit`/`vmap` trace (verified eagerly, under
   `jit`, `vmap`, `jit(vmap)` and `filter_vmap`). Without it, `float()` of a staged `jnp` op
   raised `ConcretizationTypeError` under `vmap` during the migration. The earlier workaround
   — re-deriving each shift with `math` — duplicated the inverse formula and is gone.
4. **Validation happens at construction**: `Positive` requires `at_zero > eps`, `Interval`
   requires `lo < at_zero < hi`, `BoundedPositive` requires `at_zero - eps` in
   `(exp(-a), exp(a))`, `Widths` requires `n * min_rel < 1` (checked per call, since `n` is
   the length of `raw`). These are Python `ValueError`s raised at trace time, which is when a
   misconfigured bijection should fail.

### What the primitives are *not*

- They are not the parameter *values*. Those are the `*Params` NamedTuples returned by
  `constrain` — pytrees, because `params` can be returned from or passed into jitted code.
- They are not pytrees and must not be passed through `jax.tree.*`, `vmap`'d over, or used
  as `jit` static arguments (they define no `__hash__`/`__eq__` beyond identity).
- They have no bearing on a bijection's leaf structure. That is decided by the bijection's
  own fields alone: `raw` is the array leaf, and `eps_*`, `min_rel_*`, `xy_range` are static
  fields on the bijection (not on the primitive) and would be static whether the map were a
  class, a function or inlined code. "The class keeps `eps` opaque to JAX" is therefore not a
  reason for the design; rule 1 is what keeps the primitives out of `jax.tree.leaves`.

## Consequences

- One implementation per map, with its inverse next to it; `from_constrained` is a few
  `inverse` calls; the identity-at-zero shift exists in exactly one place per primitive.
- `Widths.inverse` is defined up to a constant (softmax is shift-invariant); the chosen gauge
  is mean-zero raw, which is also the gauge in which equal widths map back to `raw = 0`.
  Tests state the law as `inverse(c(r_centred)) == r_centred`.
- `is_constrained(value)` is part of the contract: the primitive that claims a set also
  decides membership, so tests can check the image generically over every primitive without
  a hand-maintained predicate table.
- Laws pinned in `tests/test_constraints.py`, generic over `PRIMITIVES`: `inverse∘forward`,
  `c(0) == at_zero` (equal widths for `Widths`), `is_constrained(c(raw))` and finite
  Jacobian for `|raw| <= 30`, non-vacuity of `is_constrained`, constructor validation,
  construction inside `jit(vmap)`.
- Cost per call: one softplus/sigmoid per scalar parameter and one softmax over `K` widths per
  spline — the same work the coupling path already did, now also on the standalone path.

## Alternatives rejected

- **`eqx.Module` primitives stored as fields** — static leaves on every bijection for no
  benefit, and a `total` that may be traced cannot be a static field.
- **`NamedTuple` primitives** — a pytree solves a boundary-crossing problem the primitives do
  not have; `__call__`/`inverse` methods on a NamedTuple are awkward and the tuple fields
  would become (meaningless) leaves if one ever crossed a boundary.
- **Pure functions** (`positive(raw, eps, at_zero)`) — the same semantics, but the inverse
  and the forward then live in two functions that must agree on the shift; the class keeps
  them together with their validated configuration.
- **Per-parameter wrapper leaves** (paramax `Parameterize`, bijx `TransformedParameter`) —
  covered by ADR-0001; they move the map into the leaf rather than into `constrain`.
