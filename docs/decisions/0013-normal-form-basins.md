---
type: decision
id: ADR-0013
status: accepted
updated: 2026-10-30
verified_by: pending (Joon; the four choices below were taken in discussion 2026-10-08)
---

# ADR-0013 — Normal forms take their data in r on a declared basin; evenness is a subfamily

## Context

### Motivation

- To compare learned conjugacies against systems with *analytically known*
  isochrons, the next target is Winfree's planar model with a hole, as used by
  Langfield et al. 2025 (§4.1): $\dot r = (1-r)(r-a)\thinspace r$,
  $\dot\psi = -(1 + \omega(1-r))$. Its isochrons are closed form (their Eq. 12), the
  unit circle attracts, the circle $r = a$ is a repelling cycle bounding the basin, and
  the closed disk $r \le a$ is a phaseless set.
- Two of our conventions ([ADR-0007](0007-systems-hierarchy-and-flow-strategies.md),
  design document [`normal-forms.md`](../design/normal-forms.md) §1/§9 "Option B") rule
  it out. The defining data $\rho$, $\omega$ had to be *even* in $r$, supplied in
  $s = r^2$ so that the cartesian field is smooth at the origin. The basin was
  implicitly the plane minus the origin: $\Psi \to -\infty$ at $r \to 0$, eigenvalues
  at the origin as conjugacy invariants, samplers avoiding only the origin. Winfree's
  $\rho = -a + (1+a)r - r^2$ is not even, and its origin is a stable focus *outside*
  the basin.
- The first runs against $a = 0$ went to `nan`, because a hyperbolic latent focus
  cannot be conjugate to a non-hyperbolic point (design document §3.3): the map sent
  near-origin data to extreme latent radii where the chart degenerates (commit
  `432bba6`;
  [`2026-10-08-normal-forms-basin.md`](../changes/2026-10-08-normal-forms-basin.md),
  review round 1). This point motivated Decision 5's default.

### Forces

Data generation fails loudly rather than silently. A Winfree system is an
observed/target system, not a latent one, so the rule for learnable parameters
([ADR-0001](0001-unconstrained-leaves.md)) does not reach it. An INN is a
diffeomorphism of the plane and cannot map a disk to a point.

## Decision

### 1. Base class in $r$ with a declared basin; the even forms become a subclass

`AbstractNormalForm` takes its rates $\rho(r)$ and $\omega(r)$ from the polar template
that a subclass writes ([ADR-0016](0016-symbolic-systems.md)), declares `phase_shift(r)`
and `isostable(r)` abstract in $r$, and has `basin_radii() -> (r_in, r_out)` (default
$(0, \infty)$; Python floats) and `in_basin(u)`. Its `rhs` uses $|u|$ through a safe
square root. `AbstractEvenNormalForm(AbstractNormalForm)` is the family whose rates are
functions of $r^2$; it has the `_sq` hooks, the square-root-free `rhs` and
`eigenvalues_origin`, and Hopf and Bautin belong to it. A form joins the family by
declaring it as its base class, and ADR-0016 checks the membership when the class is
created. The `r_squared` integration requires the even family and raises otherwise.

*Rejected*: a sibling class with the chart methods duplicated, or a Winfree
`AbstractODE` with hand-written phase and amplitude. One chart API means one `Evaluator`
reference type, one closed-form integration, one set of law tests: the Winfree closed
forms are checked by the same autodiff identities as Hopf's and Bautin's.

### 2. Outside the basin: `nan`, never a number

`to_phase_amplitude` (hence `phase`, `amplitude`, the closed-form flow) and `isochron`
return `nan` for $r \notin (r_{\rm in}, r_{\rm out})$, and a concrete form also makes
its closed forms `nan` there. The root solve relies on the `nan` to detect the basin's
edge: for integer $1/a$ a negative base raised to $1/a$ would otherwise be a finite
number with no meaning, which we hit. The cartesian field stays defined everywhere;
inside the hole it flows to the origin.

### 3. Initial conditions outside the basin: error or resample, both explicit

`generate(..., outside_basin="error" | "resample")` and the matching `data` config
field. `"error"`, the default, consistent with "loud failures", raises naming the
indices. `"resample"` redraws with fresh keys until `n_trajectories` lie inside
(bounded rounds), warns with the count and records it in
`metadata.provenance.rejected_ics`. The mode is part of `SamplingSpec`, hence of
`config_hash`, since the accepted set depends on it. `AbstractODE.in_basin` defaults to
`True` (FitzHugh–Nagumo's phaseless set is a point). **Consequence**: adding the field
changes every dataset's hash, so existing files must be regenerated once
([`scripts/generate_data.py`](../../scripts/generate_data.py)).

*Rejected*: silently dropping (fewer trajectories than asked) and keeping the mode out
of the hash (two different datasets under one name).

### 4. `eigenvalues_origin` is declared where the origin is an invariant

`AbstractNormalForm` does not declare `eigenvalues_origin`; the even family does. For
Winfree with $a > 0$ the origin's eigenvalues $-a \pm i(1+w)$ exist but are not
invariants of the conjugacy on the basin; the inner boundary's invariant is the
repelling cycle's exponent $\kappa_u = a\thinspace\rho'(a)$
(`WinfreeNormalForm.inner_floquet_exponent`). For $a < 0$ the origin is the basin's
enclosed hyperbolic fixed point, and `WinfreeNormalForm` defines `eigenvalues_origin`
for that regime
([`winfree.py`](../../src/deep_isochron/systems/normal_forms/winfree.py)).
Moving the method rather than generalizing it keeps its meaning exact. Revisit if a
second holed form appears.

### 5. Winfree's parameters are unconstrained scalar arrays; $a < 0$ is the training member

It is an observed/target system, not a latent one, so it has no constraints and no
`raw` leaves (ADR-0001 is about learnable bijection parameters); `a` and `w` are plain
scalar array leaves, as the other normal forms' parameters are, not Python floats,
which would be static under `filter_jit` and make every parameter value a new program
for every jitted function. Any $a < 1$ is allowed: $a > 0$ is the paper's hole, $a = 0$
the non-hyperbolic point, $a < 0$ a hyperbolic repelling focus. The closed forms
evaluate both the general formula and its $a \to 0$ limit and select by `jnp.where`,
with $a$ replaced by 1 inside the formula where it is 0 so that no `nan` reaches a
derivative; `basin_radii` returns $\max(a, 0)$ as an array. The default and the data config are $a = -0.25$, the member with closed-form
isochrons *and* an origin that a Bautin latent can match.

## Consequences

- `systems/normal_forms/{base,winfree,integration}.py`, `systems/base.py`
  (`in_basin`), `data/{generate,dataset}.py`; exports `AbstractEvenNormalForm`,
  `WinfreeNormalForm`.
- Tests: the shared normal-form laws run over even forms and Winfree; §4 "basins" in
  [`test_normal_forms.py`](../../tests/test_normal_forms.py); `test_data.py` for
  `outside_basin`.
- The research question that the system raises is Phase E work, with Winfree as an
  *observed* system against the Bautin latent: an INN is a diffeomorphism of the plane
  and cannot map a disk to a point, so what does a learned conjugacy do with the hole? A
  latent family with a hole is not in scope (decision 2026-10-08).

## Alternatives rejected

Alternatives rejected for a single decision are listed under that decision.

- *Superseded 2026-10-14*: $a = 0$ in the data config and $a = 0.25$ as the class
  default, with `basin_radii = (a, ∞)` and no `eigenvalues_origin` on Winfree. Replaced
  because the runs against $a = 0$ went to `nan` (Motivation).

## History

- 2026-10-08 (`normal-forms-basin`): accepted. Amends ADR-0007 and the design document
  `normal-forms.md` §1, §3, §7, §9, §11.
- 2026-10-14 (`normal-forms-basin`, review round 1): Decision 5's default became
  $a = -0.25$; Decision 4: `WinfreeNormalForm` defines `eigenvalues_origin` for
  $a < 0$; `basin_radii` became $(\max(a, 0), \infty)$.
- 2026-10-27 (`prose`): restructured to the format of
  [ADR-0019](0019-adrs-describe-the-current-design.md).
- 2026-10-29 (`auto-isochrons`): Decision 5's parameters became array leaves; as static
  floats they compiled every jitted function again for each Hypothesis draw, which kept
  `tests/test_normal_forms.py` at 43 minutes and a killed worker at the `ci` profile.
- 2026-10-27 (`symbolic-systems`, ADR-0016 review round 1): Decision 1 restated for the
  polar template; even-family membership is declared by the base class and checked, no
  longer written as a second hook.
- 2026-10-30 (`phase-turns`, ADR-0020): the chart's phase is a fraction of the period in `[0, 1)`;
  `to_phase_amplitude` negates the polar angle for a clockwise form so that the phase
  increases with time, and `isochron(Θ, r)` takes that phase.
