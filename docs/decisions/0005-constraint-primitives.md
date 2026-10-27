---
type: decision
id: ADR-0005
status: accepted; amended
updated: 2026-10-27
verified_by: joon
---

# ADR-0005 — Constraint primitives are plain Python objects created inside `constrain`

## Context

### Motivation

- The constraining maps were spread over modules: `constrain_widths` and
  `check_positive` lived in
  [`splines/base.py`](../../src/deep_isochron/model/invertible/splines/base.py)
  ([`2026-09-30-splines-refactor.md`](../changes/2026-09-30-splines-refactor.md),
  line 21), and positivity was checked in `__check_init__` for concrete arrays only,
  since `check_positive` skipped tracers (same document, lines 60–61).
- A duplicated shift formula was wrong: the identity shift of `MonotonicRQSpline` had a
  sign error (same document, lines 92–93).
- The first primitives re-derived each shift with `math`
  ([the scalar-param change document](../changes/2026-10-01-scalar-param-refactor.md),
  line 28), which duplicated each `_inverse`.
- Joon's review of that branch asked whether plain classes are correct inside
  JAX-traced code (same document, line 256); this ADR records the answer.
- Before 2026-10-04, every grain worker process that imported the package took a CUDA
  context and preallocated GPU memory, which ran a V100 out of memory: the module-level
  primitives (`A_CONSTRAINT`, `B_CONSTRAINT`, `DECAY_CONSTRAINT`) computed their shift
  at construction and so initialized a JAX backend at import time.

### Forces

[ADR-0001](0001-unconstrained-leaves.md) makes `constrain(raw)` the single place where a
scalar bijection maps its unconstrained leaf onto the constrained set. The maps
themselves (shifted softplus, shifted sigmoid, floored softmax, `arcsinh`, the bounded
`squashed_exp`) are shared by several classes, and each needs an inverse, for
`from_constrained`, and the identity-at-zero shift of
[ADR-0002](0002-identity-at-zero.md). They need a home, and the question is what kind of
object that home is, given that it is used inside `jit`-, `vmap`- and `grad`-transformed
code.

Optax implements its transformations' *states* as `NamedTuple`s because a state is an
argument and a return value of `update`, which runs under `jit`: anything crossing such
a boundary must be a registered pytree so that JAX can flatten it to arrays and rebuild
it. The same reasoning is why bijections are Equinox modules. It is tempting to conclude
that every object touching JAX code must be a pytree; that is not the rule.

## Decision

Constraint primitives (`Free`, `GreaterThan` and its case `Positive`, `Interval`,
`BoundedPositive`, `Arcsinh`, `Widths` in
[`invertible/constraints.py`](../../src/deep_isochron/model/invertible/constraints.py))
are plain Python classes (`abc.ABC`, not `eqx.Module`) with `__call__` (raw →
constrained) and `inverse` (constrained → raw). They hold only static configuration:
`eps`, `at_zero`, bounds, `min_rel`, and the shift constant, computed on first use (rule
3). They are **instantiated inside `constrain`**, called once, and dropped:

```python
def constrain(self, raw):  # traced
    widths = Widths(self.range_width, self.min_rel_width)  # lives inside the trace
    return RQSplineParams(widths(x_raw), widths(y_raw), ...)  # only arrays leave
```

### Why this is correct under `jit`/`vmap`/`grad`

JAX needs to understand an object's structure only when it is an **argument or return
value** of a transformed function. A primitive created and consumed inside the trace is
no different from calling a helper function with closed-over Python constants: there is
nothing to flatten, nothing to trace, and nothing to retrace on later calls. The test
suite exercises exactly this path. `roundtrip` and `jacobian_dets` are `eqx.filter_jit`
over `jax.vmap`, the orientation laws go through `jacfwd`, the extreme-regime tests
through `grad³`, and `CouplingFlow` calls `constrain` under `filter_vmap` of
`from_unconstrained` on every forward pass. Equinox's own layers call plain Python
callables (activation functions) inside the trace the same way.

### Rules that keep it correct

1. **Never store a primitive on a module.** As a field it would have to be an array
   leaf, which it is not, or a hashable static field, which its `total` cannot be when
   it is an array. Constructing them in `constrain` makes the question moot. The same
   reason is why they are not `eqx.Module`s: making them modules would invite storing
   them.
2. **A tracer may live in a primitive only within the trace that created it.**
   `CubicBSpline.constrain` builds `Widths(total=span)` with a traced `span` (the
   Greville span, which depends on `raw`); fine, because the object is consumed in the
   same trace. Returning or caching such an object would leak a tracer, which is the
   standard JAX rule.
3. **Shift constants are computed on first use, under
   `jax.ensure_compile_time_eval()`.** A shifted primitive implements the *unshifted*
   map pair `_forward`/`_inverse` once; the shift is `float(_inverse(at_zero))`, a
   `functools.cached_property` evaluated in that context so that it is a concrete Python
   float even when the primitive is constructed inside a `jit`/`vmap` trace (verified
   eagerly, under `jit`, `vmap`, `jit(vmap)` and `filter_vmap`). Without the context,
   `float()` of a staged `jnp` op raised `ConcretizationTypeError` under `vmap` during
   the migration. Computing it on first use rather than at construction keeps the
   module-level primitives (`A_CONSTRAINT`, `B_CONSTRAINT`, `DECAY_CONSTRAINT`) from
   initializing a JAX backend at import time. `_inverse` is plain `jnp` with no
   jit-decorated helper, so that the evaluation is not staged (the `_shift` docstring in
   `constraints.py`, lines 67–79).
4. **Validation happens at construction**: `GreaterThan` (and so `Positive`) requires
   `at_zero > lower`, `Interval` requires `lo < at_zero < hi`, `BoundedPositive`
   requires `at_zero - lower` in `(exp(-a), exp(a))`, `Widths` requires
   `n * min_rel < 1` (checked per call, since `n` is the length of `raw`). These are
   Python `ValueError`s raised at trace time, which is when a misconfigured bijection
   should fail.

### What the primitives are *not*

- They are not the parameter *values*. Those are the `*Params` NamedTuples returned by
  `constrain`, pytrees because `params` can be returned from or passed into jitted code.
- They are not pytrees and must not be passed through `jax.tree.*`, `vmap`'d over, or
  used as `jit` static arguments (they define no `__hash__`/`__eq__` beyond identity).
- They have no bearing on a bijection's leaf structure. The bijection's own fields alone
  decide that: `raw` is the array leaf, and `eps_*`, `min_rel_*`, `xy_range` are static
  fields on the bijection (not on the primitive) and would be static whether the map
  were a class, a function or inlined code. "The class keeps `eps` opaque to JAX" is
  therefore not a reason for the design; rule 1 is what keeps the primitives out of
  `jax.tree.leaves`.

## Consequences

- One implementation per map, with its inverse next to it; `from_constrained` is a few
  `inverse` calls; the identity-at-zero shift exists in exactly one place per primitive.
- `Widths.inverse` is defined up to a constant, since softmax is shift-invariant. The
  chosen gauge is mean-zero raw, which is also the gauge in which equal widths map back
  to `raw = 0`. Tests state the law as `inverse(c(r_centered)) == r_centered`.
- `is_constrained(value)` is part of the contract: the primitive that claims a set also
  decides membership, so tests can check the image generically over every primitive
  without a hand-maintained predicate table.
- Laws pinned in [`tests/test_constraints.py`](../../tests/test_constraints.py), generic
  over `PRIMITIVES`: `inverse∘forward`, `c(0) == at_zero` (equal widths for `Widths`),
  `is_constrained(c(raw))` and finite Jacobian for `|raw| <= 30`, non-vacuity of
  `is_constrained`, constructor validation, construction inside `jit(vmap)`.
- Cost per call: one softplus or sigmoid per scalar parameter and one softmax over `K`
  widths per spline. The coupling path already did the same work, and the standalone
  path now does it too.

## Alternatives rejected

- **`eqx.Module` primitives stored as fields.** Static leaves on every bijection for no
  benefit, and a `total` that may be traced cannot be a static field.
- **`NamedTuple` primitives.** A pytree solves a boundary-crossing problem that the
  primitives do not have; `__call__`/`inverse` methods on a NamedTuple are awkward, and
  the tuple fields would become meaningless leaves if one ever crossed a boundary.
- **Pure functions** (`positive(raw, eps, at_zero)`). The same semantics, but the
  inverse and the forward then live in two functions that must agree on the shift; the
  class keeps them together with their validated configuration.
- **Per-parameter wrapper leaves** (paramax `Parameterize`, bijx
  `TransformedParameter`). Covered by [ADR-0001](0001-unconstrained-leaves.md); they
  move the map into the leaf rather than into `constrain`.
- *Superseded 2026-10-01*: re-deriving each shift with `math`. The earlier workaround
  duplicated the inverse formula; replaced by `float(_inverse(at_zero))` under
  `jax.ensure_compile_time_eval()`.
- *Superseded 2026-10-04*: the shift constant computed once in `__init__` under
  `jax.ensure_compile_time_eval()`; replaced because the module-level primitives then
  initialized a JAX backend at import time, which ran a V100 out of memory through the
  grain workers.

## History

- 2026-10-01 (`scalar-param-refactor`): accepted.
- 2026-10-04 (`trainer`): the identity-at-zero shift of `_Shifted` primitives is
  computed lazily (`functools.cached_property`) rather than at construction. Nothing
  else about the contract changed; the shift is still concrete under a trace
  (`ensure_compile_time_eval`).
