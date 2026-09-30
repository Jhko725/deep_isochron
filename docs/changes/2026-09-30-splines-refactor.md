# 2026-09-30 · `splines-refactor`

Base commit: `18d6908`. Branch commit(s): see `git log main..splines-refactor`.

## Summary

Extracts the logic shared by the monotone spline bijections into
`AbstractSpline`, ports `CubicBSpline` to the `AbstractScalarBijection`
interface (constrained parameters as leaves, `from_unconstrained`, identity at
zero), fixes a set of bugs in `MonotonicRQSpline`, adds a `LinearSpline` as
the C⁰ instance of the base class, and replaces the ad-hoc `test_bspline.py`
with property tests that fit the registry/strategies infrastructure. Making
the splines work inside `CouplingFlow` required moving the coupling layer to
`template.from_unconstrained` as well, which the interface change had
anticipated but not yet done.

## Files

| File | Change |
|---|---|
| `src/.../invertible/splines/base.py` | **new** — `AbstractSpline`, `constrain_widths`, `bin_index`, `check_positive` |
| `src/.../invertible/splines/rational_quadratic.py` | rewritten on the base class; bug fixes below |
| `src/.../invertible/splines/cubic.py` | rewritten on the base class; constrained parameters in real units, `identity`, `from_unconstrained`, `xs`/`ys`/`knot_derivs` |
| `src/.../invertible/splines/linear.py` | **new** — `LinearSpline` (C⁰) |
| `src/.../invertible/splines/__init__.py`, `invertible/__init__.py` | export `AbstractSpline`, `LinearSpline` |
| `src/.../invertible/coupling.py` | `BijectionFactory` removed; bijections built with `template.from_unconstrained` under `vmap`; template stored static without array leaves; `split_idx` check un-inverted |
| `src/.../invertible/base.py` | `AbstractScalarBijection.jacobian` = `jax.grad`; `SequentialINN.num_params`/`from_unconstrained`; `SequentialINN` shape annotations loosened to `"..."` |
| `src/.../invertible/polar.py` | import `.splines`; `CircularMonotonicRQCoupling.make_spline` passes widths and interior derivatives; `OffsetedBijection` annotations loosened |
| `src/.../invertible/linear.py` | `BiLipschitzLinear._s`/`s` annotated as scalars (they are) |
| `tests/test_splines.py` | **new** — spline-specific laws (see Tests) |
| `tests/test_bspline.py` | **deleted** (standalone script, superseded) |
| `tests/registry.py` | templates via `cls.identity(num_bins, xy_range)`; B-spline and linear templates; B-spline coupling; `_identity` helper for `RadialBijection`; `IDENTITY_TOL` hack removed |
| `tests/strategies.py` | `spline_templates(cls)` replaces `rq_spline_templates`; `with_unconstrained` → `from_unconstrained`; type strategies for all splines |
| `tests/test_bijections.py` | imports from `tests.helpers`; `KNOTTED` by `isinstance(AbstractSpline)`, `KNOTTED_C1` for `knot_derivs`; scalar tests pass 0-d batches; generic `test_spline_any_bin_count_round_trip` |
| `CLAUDE.md`, `docs/changes/…` | this convention |

## Design

**`AbstractSpline` contract.** A spline is a strictly increasing map that is
piecewise on `xy_range` and the identity outside. Subclasses provide `xs`,
`ys` (knot positions/values, endpoints included), `_forward_in_range`,
`_inverse_in_range`, `num_params`, `from_unconstrained`. The base provides
the tail switch, `bin_value`, `_knots_from_widths` (last knot snapped to the
range end, as nflows does), and `identity`-style templates are per-class
classmethods. `knot_derivs` is optional: defined for C¹⁺ splines, used by
the shared knot-derivative test, absent on `LinearSpline`.
*Rejected*: a `linear_tails` flag. All three splines need identity tails
for this project, and the boundary conditions that make the joins C¹/C² are
baked into the parametrisation, so a flag would have no consistent meaning.

**Interpolant evaluated at the clipped input.** `__call__` is
`where(in_range, f_in(clip(x)), x)`. With `jnp.where` (unlike `jnp.piecewise`)
a NaN in the unselected branch reaches the gradient; for tail inputs the RQ
discriminant is negative about half the time, so this was a real NaN
source. Clipping keeps the unselected branch finite.

**Constrained parameters as leaves.** Both splines store constrained arrays
(positive widths, increments, derivatives) so that `num_trainable_params ==
num_params` for a standalone spline and `from_unconstrained` is the single
place where the softmax/softplus maps live. Positivity is checked in
`__check_init__` only for concrete arrays (`check_positive` skips tracers),
since inside a coupling layer the arrays are traced and positivity is
guaranteed by `from_unconstrained`.

**B-spline boundary.** Instead of lines 13–17 of Algorithm 1 in Hong & Chun
(which fix only f(a)=a, f(b)=b), the three outermost coefficients on each
side are pinned to their Greville abscissae, so f=id, f′=1, f″=0 at both
range endpoints (Marsden's identity) and the identity tails join C². Cost:
the two edge bins lose flexibility. Coefficients live in real units (no
normalise/denormalise); knots `t_{-2}…t_{K+2}` with `t_0=a`, `t_K=b`.

**B-spline inverse.** Bisection-safeguarded Newton (20 fixed iterations) on
the local cubic, with a `custom_jvp` implicit rule
`du = (dy − ∂_c p·dc)/p′(u)` that is itself differentiable (second
derivatives of the inverse are tested). This replaces the closed-form cubic
root of the reference implementation, which needs 1e-7 quadratic/linear
fallbacks and which the authors suspect behind their 1-in-10 outliers.

**Coupling template without leaves.** `CouplingFlow.template` is
`eqx.partition(bijection, eqx.is_array)[1]`, i.e. the module with `None`
array leaves, stored as a static field. It is hashable, contributes nothing
to `num_trainable_params`, and `from_unconstrained` only needs the static
configuration. `SequentialINN` got `num_params`/`from_unconstrained` so a
stack of scalar bijections remains a valid template (the registry uses one).

## Bugs fixed

`MonotonicRQSpline` (all broke construction or use before this branch):
- `__init__` assigned `self._x_widths`, `self.min_rel_x_bin_width`, … — not
  fields; `TypeError` on construction.
- `num_params` was a method, not a property.
- Identity-at-init shift was `1 − inv_softplus(1 − ε)`; correct is
  `+ inv_softplus(1 − ε)`. This was the source of the 1e-3 `IDENTITY_TOL`;
  identity now holds to 1e-12.
- `from_unconstrained` used `copy.replace` with a kwarg (`derivatives`) that
  did not match the field name.
- Bin index clipped to `num_knots − 1` (one past the last bin) → silent
  out-of-bounds gather for `y ≥ hi`.
- `jnp.any` in `__check_init__` raises under tracing (i.e. in every coupling).
- NaN gradients for tail inputs (see Design).
- Dead `searchsorted` line in `inverse`.

Elsewhere:
- `CouplingFlow` wrote MLP output straight into the constrained leaves through
  `BijectionFactory` (no positivity, no identity at init); `split_idx` check
  rejected exactly the valid values.
- `polar.py` imported `.spline` (module renamed) and passed knot positions
  where widths are expected.
- `BiLipschitzLinear._s` annotated as a vector; it is a scalar.

## Tests

`uv run pytest tests/test_splines.py tests/test_bijections.py tests/test_registry.py`
(`--hypothesis-profile=dev -n 4` for a quick run).

`tests/test_splines.py` pins:
1. identity tails (value and inverse), 2. finite gradients w.r.t. x and
parameters for tail inputs with tail slope 1, 3. join regularity per class
(`JOIN_ORDER = {Linear: 0, RQ: 1, Cubic: 2}`, one-sided derivatives at both
endpoints) plus C² across interior knots, 4. B-spline forward map vs
`scipy.interpolate.BSpline` to 1e-12, strictly increasing knots/coefficients
(premise of Theorem 1), 5. round trip to 1e-11 at raw scale 20, and
`(f⁻¹)″ = −f″/f′³` through the implicit JVP.

Status at the branch commit (default profile): all scalar, spline, registry
and hygiene tests pass. Remaining failures are pre-existing and not spline
related (next section).

## Open issues

- `CircularMonotonicRQCoupling`: the RQ spline now fixes boundary
  derivatives to 1, so the previous `pad(mode="wrap")` derivative matching
  at θ=±π is lost (map is C⁰ there). Needs free boundary derivatives in
  `MonotonicRQSpline`; TODO left in `make_spline`.
- Pre-existing test failures left alone: `PolarConditionalBijection` calls
  `SinhConjugation.from_unnormalized_params` (renamed); `InvertibleLinear`
  reflects at init (its own regression test); `CircularMonotonicRQCoupling`
  has NaN Jacobians at the origin (Hypothesis draws 0 deliberately);
  `test_sequential_inn_composes_inverse` fails through the first of these.
- `CouplingFlow` uses `jax.nn.gelu` by default; a C² flow needs a C²
  conditioner activation, so keep it away from ReLU.
- The two notebooks reference `MonotonicRQCoupling` and old constructors.
- `min_rel_*` defaults are 1e-3 (matching the RQ class); the paper uses 1e-6.

## Review notes

<!-- Joon: fill in while vetting. Claude reads this at the start of the next
session on this branch. -->
