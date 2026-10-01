# 2026-10-01 · `scalar-param-refactor`

Base commit: `eed3731`. Plan: `claude/refactor-plan-2026-10-01.md` (project doc). Branch
commit(s): see `git log master..scalar-param-refactor`.

## Summary

Scalar bijections store their *unconstrained* parameter vector as the single trainable leaf
and compute constrained values on read through one method, `constrain(raw)`, built from a
small set of constraint primitives. This makes standalone bijections safe to train (an
optimiser step cannot leave the constrained set), makes `from_unconstrained` a one-line
`tree_at` shared by every class, and puts the softplus/sigmoid/softmax maps in exactly one
place per class. Alongside: `smoothness` declared on every bijection, `ScalarChain` for
scalar composition, `CouplingFlow` with a pluggable conditioner (the polar flow becomes an
instance of it), and the remaining known defects (`InvertibleLinear` reflection,
`BiLipschitzLinear._s`, `eps_*` config loss).

Steps land as separate commits; this document grows with them.

## Files

| File | Change |
|---|---|
| `src/.../invertible/constraints.py` | **new** — `Constraint`, `Free`, `Positive`, `Interval`, `Arcsinh`, `Widths`; `at_zero` convention |
| `src/.../invertible/splines/base.py` | `constrain_widths` now delegates to `Widths` (temporary alias, removed in step 3) |
| `src/deep_isochron/misc.py` | `squashed_exp(x, a)` honours `a` (was hard-coded to 2) |
| `tests/test_constraints.py` | **new** — laws of the primitives; `squashed_exp`/`inv_squashed_exp` inverse law |

## Design

**Constraint primitives are plain Python objects, not modules.** They carry only static
configuration (`eps`, `at_zero`, bounds, `min_rel`) and are created inside `constrain`, so
there is nothing to register as a pytree and nothing the optimiser can see.
*Rejected*: `eqx.Module` primitives stored as fields — would add static leaves to every
bijection for no benefit.

**`Widths.inverse` gauge.** The floored softmax is shift-invariant, so its inverse is defined
up to a constant. The inverse returns mean-zero raw values, which is also the gauge in which
equal widths map back to `raw = 0`. Tests state the law as `inverse(c(r_centred)) == r_centred`.

**`at_zero` validated at construction**: `Positive(eps, at_zero)` requires `at_zero > eps`,
`Interval(lo, hi, at_zero)` requires `lo < at_zero < hi`; the shifts are computed once as
Python floats.

## Bugs fixed

- `misc.squashed_exp(x, a)` ignored `a` (body hard-coded `2.0`), so `inv_squashed_exp(..., a)`
  was not its inverse for `a != 2`. Symptom: silent mismatch for any non-default `a`.

## Tests

`uv run pytest tests/test_constraints.py --hypothesis-profile=dev` — 26 passed (step 1).
Spline tests unchanged and green with `constrain_widths` delegating to `Widths`
(`tests/test_splines.py`: 16 passed).

Pinned: inverse∘forward for every primitive; `c(0) == at_zero` (equal widths for `Widths`);
image in the constrained set and finite gradients for `|raw| <= 30`; `Widths` sum and floor;
constructor validation of impossible floors / `at_zero` outside the set.

## Open issues

- Steps 2–6 of the plan (base contract, analytic and spline migration, conditioner,
  polar classes, housekeeping) follow in later commits on this branch.

## Review notes

| Change | Thoughts | Modifications |
|---|---|---|
