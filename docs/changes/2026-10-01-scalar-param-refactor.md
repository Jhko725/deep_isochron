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
| `src/.../invertible/constraints.py` | (step 2) `BoundedPositive` added (the `squashed_exp` family, for `CubicConjugation`); shifts computed with `math`, not `jax` |
| `src/.../invertible/base.py` | **rewritten** — `smoothness` on `AbstractBijection`; `AbstractScalarBijection[P]` with `raw` leaf, `constrain(raw) -> P`, `params`, final `from_unconstrained`/`identity_like`, `_init_raw`; `ScalarChain`; `SequentialINN` vector-only with `dim`/`smoothness` fields; `min_smoothness` |
| `src/.../invertible/analytic.py` | `CubicRational`, `SinhConjugation`, `CubicConjugation` on the new contract: `*Params` NamedTuples, `eps_*` static fields, `constrain` from primitives, `from_constrained` classmethods |
| `src/.../invertible/coupling.py` | template via `eqx.partition(...)[1]` (documented as exactly "all `raw` leaves dropped"); `isinstance(AbstractScalarBijection)` check; non-smooth-activation guard; `smoothness` |
| `src/.../invertible/linear.py` | `InvertibleLinear` init is a rotation (sign-corrected QR + det fix); `smoothness` fields |
| `src/.../invertible/affine.py`, `polar.py` | `smoothness` fields; `dim` as static `init=False` fields instead of `ClassVar`/property; `OffsetedBijection`/`RadialBijection` derive `smoothness` from the wrapped bijection |
| `src/.../invertible/__init__.py` | export `AbstractScalarBijection`, `ScalarChain`, the constraint primitives |
| `tests/registry.py` | templates are `cls(config)`; config variants real again; `ScalarChain` template; `split_idx=1 of 3` coupling; `ORIENTATION_NOT_GUARANTEED`; spline classes, circular and polar-conditional parked in `UNTESTED` with the step that unparks them |
| `tests/strategies.py` | `analytic_templates` via `st.builds` over static config; `spline_templates` uses `cls(K, xy_range)` |
| `tests/test_bijections.py` | knot tests moved out; template-has-no-trainable-state law; standalone-training laws (`optax.adam` step then round trip / monotonicity) |
| `tests/test_registry.py` | `test_smoothness_declared`, `test_abstractvars_are_not_init_args` |
| `tests/test_analytic.py` | **new** — Sinh inverse symmetry, asymptotics, extreme-regime finiteness, `from_constrained` round trips, pinned underflow xfail |
| `tests/test_splines.py` | (step 2) parked; (step 3) unparked, `JOIN_ORDER` derived from `smoothness`, constructors `cls(K, xy_range)`, knot/boundary tests with `@example` pins |
| `src/.../invertible/splines/base.py` | (step 3) `constrain_widths`/`check_positive` deleted; `_check_range`; docstring on the `raw`/`constrain` contract; generic `AbstractSpline[P]` |
| `src/.../invertible/splines/linear.py`, `rational_quadratic.py`, `cubic.py` | (step 3) on the contract: `*Params` NamedTuples, `raw` leaf, `constrain`, `smoothness` 0/1/2, floors as static fields, `identity` classmethods removed; B-spline docstring notes exterior-width normalisation and the Newton accuracy bound |
| `src/.../invertible/polar.py` | (step 3) `CircularMonotonicRQCoupling` rewritten: holds a standalone `MonotonicRQSpline` on `(-π, π)`; exact rotation form with a `where`-safe angle (origin fixed, Jacobians there mutually inverse rotations); duplicated softmax code removed |
| `tests/registry.py` | (step 3) splines, spline couplings, `ScalarChain([rq, cubic])` and the circular class re-registered; their `UNTESTED` entries removed |
| `tests/strategies.py` | (step 3) `vector_bijections(..., scale=)` |
| `tests/test_bijections.py` | (step 3) composition test draws 3 mildly perturbed layers and checks the reversed-chain identity (see Design) |
| `src/.../invertible/coupling.py` | (step 4) pluggable `conditioner: x_const -> raw` (default: zero-final-layer MLP; `key` optional when a conditioner is given); field renamed `mlp` → `conditioner` |
| `src/.../invertible/polar.py` | (step 4) `OffsetedBijection` is an `AbstractScalarBijection` (delegates `num_params`/`constrain`/`smoothness`; usable as template/chain member); `RadialBijection` typed and documented; **`PolarConditionalBijection` deleted, replaced by `PolarCouplingFlow`** (coupling in the `(r, θ)` chart with a zero-initialised `TruncatedFourier` conditioner); shared `_scaled_polar` |
| `src/.../invertible/linear.py` | (step 4) `BiLipschitzLinear._s` is a `(dim,)` vector of unconstrained singular values (was a scalar: the layer was a similarity transform); unused key dropped |
| `src/.../invertible/__init__.py` | (step 4) export `CircularMonotonicRQCoupling`, `PolarCouplingFlow` |
| `tests/registry.py` | (step 4) `OffsetedBijection` templates (standalone and over a chain), two `PolarCouplingFlow` builders; `UNTESTED` down to `SequentialINN` |
| `pyproject.toml` | (step 4) ignore Hypothesis's `random`-in-strategy deprecation (tripped by JAX's compiler during a first compile inside a draw, not by any strategy); (step 5) `quax` removed from runtime dependencies (unused) |
| `docs/decisions/0001..0004` | **new** — ADRs for the four decisions above |
| `CLAUDE.md` | (step 5) rules: `AbstractVar` as static `init=False` field, `raw`/`constrain` contract, ADRs in `docs/decisions/` |
| `tests/strategies.py` | (step 5) type annotations on every strategy function |
| `src/deep_isochron/data/generate.py` | (step 5) placeholder docstring with the intended `generate(...)` signature (was an empty file) |
| `prototype.ipynb` | (step 5) imports and constructors updated to the new API (`MonotonicRQSpline(num_bins=9, ...)` template, `PolarCouplingFlow`, `conjugacy` module name) |

## Design

Decisions are recorded as ADRs: [0001 unconstrained leaves](../decisions/0001-unconstrained-leaves.md),
[0002 identity at zero](../decisions/0002-identity-at-zero.md), [0003 declared smoothness](../decisions/0003-declared-smoothness.md),
[0004 AbstractVar fields](../decisions/0004-abstractvar-fields.md). The paragraphs below give the branch-level detail.

**Constraint primitives are plain Python objects, not modules.** They carry only static
configuration (`eps`, `at_zero`, bounds, `min_rel`) and are created inside `constrain`, so
there is nothing to register as a pytree and nothing the optimiser can see.
*Rejected*: `eqx.Module` primitives stored as fields — would add static leaves to every
bijection for no benefit.

**`Widths.inverse` gauge.** The floored softmax is shift-invariant, so its inverse is defined
up to a constant. The inverse returns mean-zero raw values, which is also the gauge in which
equal widths map back to `raw = 0`. Tests state the law as `inverse(c(r_centred)) == r_centred`.

**Unconstrained leaves (ADR-0001).** `raw` is the only trainable leaf of a scalar bijection;
`constrain(raw)` is the single conversion site, used by `params` (read path) and, through the
constructor, by `from_unconstrained` (`tree_at` on `raw`). A standalone bijection can be optimised
directly: no step can leave the constrained set, because the set is the image of the map.
*Rejected*: constrained leaves (previous branch) — unsafe under training, needed `_identity`
helpers and `check_positive`. *Rejected*: per-parameter wrappers (paramax `Parameterize`, bijx
`TransformedParameter`) — cannot express the B-spline's knot-dependent coefficient constraint.

**Templates are instances with `raw=None`.** `cls(config)` is the identity instance; dropping
`raw` yields a hashable template with zero trainable size, which is what `CouplingFlow` stores.
`eqx.partition(b, eqx.is_array)[1]` is exactly this because `raw` is by contract the only array
leaf (for a `ScalarChain`, the members' `raw` leaves). *Rejected*: a separate `Template` type.

**`AbstractVar`s as static `init=False` fields (ADR-0004).** `dim`, `smoothness`, `num_params`
are implemented as `eqx.field(static=True, init=False)` with a default (fixed by the class) or
assigned in `__init__` (derived). Type checkers reject a bare attribute, a `ClassVar` or a
property as overrides of an `AbstractVar`; the field passes `ty` and pyright and is runtime
equivalent. All `# ty: ignore` on these names are gone; `ty check` is clean on `base.py`,
`analytic.py`, `coupling.py`, `constraints.py`.

**`AbstractScalarBijection[P]` is generic over its params NamedTuple**, so `self.params.alpha`
type-checks. `ScalarChain` is `AbstractScalarBijection[tuple]` (its params are the members').

**Regularity is declared (ADR-0003).** `smoothness: int | None` (C^k; `None` = C^∞); leaves fix
it, containers take the minimum. `CouplingFlow` reports its template's value and rejects the
non-smooth `jax.nn` activations (`relu`, `relu6`, `leaky_relu`, `hard_*`) when the template is
C^1 or better. *Rejected*: tracking the activation's smoothness in a field or table.

**`ScalarChain` vs `SequentialINN`.** Scalar composition is its own class so that the
`num_params`/`from_unconstrained` contract is checkable by `isinstance`; `SequentialINN` is
vector-only again. May be merged later if the overlap proves too large.

**Angular spline as a rotation (step 3).** `CircularMonotonicRQCoupling` applies
`x -> R(s(θ) - θ) x`: `|x|` is preserved exactly and the inverse is `R(s⁻¹(θ') - θ')`. The angle
comes from a double-`where` `arctan2` that returns a *fill angle* inside `|x| < eps_r` — `0` for
the forward map, `s(0)` for the inverse — so the origin is a fixed point and the two Jacobians
there, `R(s(0))` and `R(-s(0))`, compose to the identity. *Rejected*: `sqrt(r² + eps²)` as the
radius (changes `|x|`, so `f(0) ≠ 0` and the map is no longer exactly invertible — the
`RadialBijection` form only works because it applies the regularised radius as a ratio).
*Rejected*: a smooth window on the angular warp near the origin (makes the inverse implicit).
The map is a diffeomorphism of the punctured plane; at the origin it is continuous only, and
`smoothness = 0` also reflects the C⁰ join at `θ = ±π`. Periodic endpoint handling deferred.

**Composition test scope (step 3).** `test_sequential_inn_composes_inverse` composes three
mildly perturbed layers and additionally checks that `f⁻¹` equals the reversed chain of member
inverses exactly. Every registered layer round-trips to ≤ 4e-10 on its own, but a chain of ten
aggressively perturbed layers reaches intermediate magnitudes ~1e2 and amplifies a layer's
inverse error by its Lipschitz constant to ~1e-6 — conditioning, not composition.

**`PolarCouplingFlow` (step 4).** The old `PolarConditionalBijection` rebuilt
`SinhConjugation` instances from Fourier output on every call through a removed classmethod. It
is structurally `CouplingFlow` in the polar chart — `(r, θ) -> (g_θ(r), θ)` with
`g_θ = template.from_unconstrained(conditioner(θ))` — so it is now written that way: an
`OffsetedBijection(template)` (so `g_θ(0) = 0`), a `TruncatedFourier` conditioner with zero
coefficients at init (identity at init, like the MLP's zero final layer), and the
`RadialBijection` radius/angle handling. *Rejected*: generalising `CouplingFlow` itself over
charts — the polar chart's regularised radius and safe angle are specific enough that a
separate class is clearer; the shared machinery is the template/`from_unconstrained` contract.

**`OffsetedBijection` is a scalar bijection.** `g(r) = f(r) - f(0)` is itself increasing on
`R`, identity at zero raw, with `f`'s parameters, so it satisfies the scalar contract by
delegation and can be a chain member or a template. Its `raw` is `None` (lives in `f`), like
`ScalarChain`.

**`at_zero` validated at construction**: `Positive(eps, at_zero)` requires `at_zero > eps`,
`Interval(lo, hi, at_zero)` requires `lo < at_zero < hi`; the shifts are computed once as
Python floats.

## Bugs fixed

- `misc.squashed_exp(x, a)` ignored `a` (body hard-coded `2.0`), so `inv_squashed_exp(..., a)`
  was not its inverse for `a != 2`. Symptom: silent mismatch for any non-default `a`.
- `InvertibleLinear` initialised as a *reflection*: Householder QR gives `det Q = -1` for
  essentially every key, so every linear layer (and any INN with an odd number of them) started
  orientation-reversing. Now a Haar rotation (`Q * sign(diag R)`, last column flipped if needed).
- `eps_*` of the analytic classes had become `from_unconstrained` kwargs, so a template could not
  carry its own epsilons and the registry's "config variants" were identical. Back as static
  fields.
- `BiLipschitzLinear._s` was a scalar, making Σ a multiple of the identity (a similarity
  transform, not a general bi-Lipschitz map). Now a `(dim,)` vector, initialised so `s = 1`.
- Analytic classes stored *constrained* values as trainable leaves; one optimiser step could make
  `scale <= 0`. Not reachable from the coupling path, but `RadialBijection` holds a standalone
  instance. Fixed by construction; pinned by `test_scalar_standalone_training_keeps_validity`.

## Findings recorded, not fixed

- `InvertibleLinear` is an unconstrained matrix: `det W` can change sign under training, so the
  perturbed-weights orientation law is skipped for it (`registry.ORIENTATION_NOT_GUARANTEED`,
  with reason). `BiLipschitzLinear` is the constrained alternative. Worth a decision: an INN that
  must stay orientation-preserving should not use `InvertibleLinear`.
- `sinh_conj_nonlinearity`'s second derivative underflows to NaN for `0 < |x - loc| < ~1e-20`
  (`x = 0` exactly is fine). Pinned as a strict xfail in `test_analytic.py`; the extreme-regime
  strategies keep magnitudes `>= 1e-12` / `>= 1e-6` or exactly zero.

## Tests

Step 1: `uv run pytest tests/test_constraints.py --hypothesis-profile=dev` — 26 passed.

Step 2: `uv run pytest -n 4` (default profile, 50 examples) —
155 passed, 7 skipped, 18 xfailed (the parked `test_splines.py` plus the pinned sinh underflow).

Step 3: `uv run pytest -n 4` (default profile) — **244 passed, 4 skipped, 1 xfailed** (the
pinned sinh underflow). `ty check src/deep_isochron/model/invertible` reports only the two
pre-existing `from_unnormalized_params` references in `PolarConditionalBijection` (step 4).
Spline laws (tails, tail gradients, join regularity by class, scipy oracle, Newton inverse,
knots) all green on the new contract; `test_spline_any_bin_count_round_trip` covers drawn bin
counts and ranges for all three classes.

Step 4: `uv run pytest -n 4` (default profile) — 274 passed, 4 skipped, 1 xfailed.

Final (step 5): `uv run pytest -n 4` (default profile, 50 examples) — **274 passed, 4 skipped,
1 xfailed** in 8 min; `--hypothesis-profile=dev` in 3.7 min. `ty check
src/deep_isochron/model/invertible` — all checks passed. Acceptance checks from the plan:
`CouplingFlow(..., CubicBSpline).smoothness == 2`, `(..., LinearSpline) == 0`,
`SequentialINN([bspline coupling, linear]).smoothness == 2`; `relu` with a C² template raises
`ValueError` (accepted with a C⁰ one); `dim=`/`smoothness=` constructor arguments raise
`TypeError`; the greps for `check_positive`, `constrain_widths`, `_identity(`,
`from_unnormalized`, `step 3/4`, and `ty: ignore` (invertible package) are empty.
`ty check src/deep_isochron/model/invertible` — all checks passed. The nine failures present at
the base commit are gone (`PolarConditionalBijection`'s missing classmethod, the circular
class's origin NaNs and spline-signature mismatch, `InvertibleLinear`'s reflection, and the
composition test that failed through them).

New laws: template has no trainable state and is hashable; standalone training keeps validity
(one `optax.adam(1.0)` step on a random gradient, then round trip + monotonicity); every
registered class declares `smoothness`; `dim`/`smoothness`/`num_params` are not constructor
arguments; Sinh inverse = parameter swap; CubicRational asymptotic bound `|alpha|/(beta|x_|)`;
Sinh asymptotically linear; `grad..grad^3` finite on the extreme regime; `from_constrained`
round trips.

Pinned: inverse∘forward for every primitive; `c(0) == at_zero` (equal widths for `Widths`);
image in the constrained set and finite gradients for `|raw| <= 30`; `Widths` sum and floor;
constructor validation of impossible floors / `at_zero` outside the set.

## Open issues

- `CircularMonotonicRQCoupling` is C⁰ across `θ = ±π` and not differentiable at the origin
  (documented in the class). Periodic / free-boundary endpoint handling on `AbstractSpline` is
  the intended fix; not in this branch.
- `# ty: ignore` remains in `systems/` and `training/trainer.py` (outside the invertible package).
- `PolarCouplingFlow.__init__` accepts and ignores `key` so registry builders stay uniform; the
  layer is deterministic (zero-initialised). Remove the argument if that uniformity is not wanted.
- CI job (`pytest --hypothesis-profile=ci -n 4`) not added: no workflow file exists yet in the
  repository; the `ci` profile is registered and runs locally.
- The 4 skips are the `IDENTITY_AT_INIT` skips for layers that are not identity at init by
  design (`residual_coupling`, `invertible_linear`, `bilipschitz_linear`) and the
  perturbed-orientation skip for `invertible_linear`.

## Review notes

| Change | Thoughts | Modifications |
|---|---|---|
