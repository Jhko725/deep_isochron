---
type: decision
id: ADR-0013
status: accepted
updated: 2026-10-08
verified_by: pending (Joon; the four choices below were taken in discussion 2026-10-08)
---

# ADR-0013 — Normal forms take their data in r on a declared basin; evenness is a subfamily

**Status**: accepted (2026-10-08, branch `normal-forms-basin`). Amends ADR-0007 and the
design document `normal-forms.md` §1, §3, §7, §9, §11.

## Context

To compare learned conjugacies against systems with *analytically known* isochrons, the
next target is Winfree's planar model with a hole as used by Langfield, Krauskopf, Lee &
Osinga (2025, §4.1): $\dot r = (1-r)(r-a)\thinspace r$, $\dot\psi = -(1 + \omega(1-r))$. Its
isochrons are closed form (their Eq. (12)), the unit circle attracts, the circle $r = a$
is a repelling cycle bounding the basin, and the closed disk $r \le a$ is a phaseless set.
Two of our conventions (ADR-0007, design document §1/§9 "Option B") rule it out: the
defining data $\rho$, $\omega$ had to be *even* in $r$ (supplied in $s = r^2$, so that the
cartesian field is smooth at the origin), and the basin was implicitly the plane minus the
origin ($\Psi \to -\infty$ at $r \to 0$, eigenvalues at the origin as conjugacy
invariants, samplers avoiding only the origin). Winfree's $\rho = -a + (1+a)r - r^2$ is not
even, and its origin is a stable focus *outside* the basin.

## Decisions

### 1. Base class in $r$ with a declared basin; the even forms become a subclass

`AbstractNormalForm` now declares `log_growth_rate(r)`, `angular_rate(r)`, `phase_shift(r)`,
`isostable(r)` abstract in $r$, plus `basin_radii() -> (r_in, r_out)` (default
$(0, \infty)$; Python floats) and `in_basin(u)`. Its `rhs` uses $|u|$ through a safe
square root. `AbstractEvenNormalForm(AbstractNormalForm)` keeps the former contract — the
`_sq` hooks, the derived $r$-views, the square-root-free `rhs`, `eigenvalues_origin` — and
Hopf and Bautin move under it unchanged. The `r_squared` integration requires the even
family and raises otherwise.

*Rejected*: a sibling class with the chart methods duplicated, or a Winfree
`AbstractODE` with hand-written phase and amplitude. One chart API means one `Evaluator`
reference type, one closed-form integration, one set of law tests — the Winfree closed
forms are checked by the same autodiff identities as Hopf's and Bautin's.

### 2. Outside the basin: `nan`, never a number

`to_phase_amplitude` (hence `phase`, `amplitude`, the closed-form flow) and `isochron`
return `nan` for $r \notin (r_{\rm in}, r_{\rm out})$; a concrete form also makes its
closed forms `nan` there (the root solve relies on it to detect the basin's edge — for
integer $1/a$ a negative base raised to $1/a$ would otherwise be finite garbage, which
we hit). The cartesian field stays defined everywhere (inside the hole it flows to the
origin).

### 3. Initial conditions outside the basin: error or resample, both explicit

`generate(..., outside_basin="error" | "resample")` and the matching `data` config field.
`"error"` (default, consistent with "loud failures") raises naming the indices;
`"resample"` redraws with fresh keys until `n_trajectories` lie inside (bounded rounds),
warns with the count, records it in `metadata.provenance.rejected_ics`, and the mode is
part of `SamplingSpec`, hence of `config_hash` (the accepted set depends on it).
`AbstractODE.in_basin` defaults to `True` (FitzHugh–Nagumo's phaseless set is a point).
**Consequence**: adding the field changes every dataset's hash; existing files must be
regenerated (`scripts/generate_data.py`), one-off.

*Rejected*: silently dropping (fewer trajectories than asked) and keeping the mode out of
the hash (two different datasets under one name).

### 4. `eigenvalues_origin` belongs to the even family

For Winfree the origin's eigenvalues $-a \pm i(1+w)$ exist but are not invariants of the
conjugacy on the basin; the inner boundary's invariant is the repelling cycle's exponent
$\kappa_u = a\thinspace\rho'(a)$ (`WinfreeNormalForm.inner_floquet_exponent`). Moving rather
than generalizing keeps the method's meaning exact. Revisit if a second holed form appears.

### 5. Winfree's parameters are plain floats

It is an observed/target system, not a latent one: no constraints, no `raw` leaves
(ADR-0001 is about learnable bijection parameters). $a = 0$ is allowed and is the member
the experiments will mostly use (phaseless set a point, non-hyperbolic origin); the
$a \to 0$ limits of the closed forms are implemented explicitly.

## Consequences

- `systems/normal_forms/{base,winfree,integration}.py`, `systems/base.py` (`in_basin`),
  `data/{generate,dataset}.py`; exports `AbstractEvenNormalForm`, `WinfreeNormalForm`.
- Tests: the shared normal-form laws run over even forms and Winfree; §4 "basins" in
  `test_normal_forms.py`; `test_data.py` for `outside_basin`.
- The research question the system raises — an INN is a diffeomorphism of the plane and
  cannot map a disk to a point, so what does a learned conjugacy do with the hole — is
  Phase E work with Winfree as an *observed* system against the Bautin latent; a latent
  family with a hole is not in scope (decision 2026-10-08).
