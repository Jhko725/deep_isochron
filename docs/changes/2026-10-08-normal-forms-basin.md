---
type: change
status: ready for review
updated: 2026-10-08
branch: normal-forms-basin
---

# `normal-forms-basin` — normal forms with a basin; Winfree's model with a hole

## Summary

Winfree's planar model as used by Langfield, Krauskopf, Lee & Osinga (2025, §4.1,
Eqs. (10)–(13)) has closed-form isochrons and a phaseless *disk* — exactly the reference
wanted for comparing learned conjugacies — but its $\rho(r) = (1-r)(r-a)$ and
$\omega(r) = -(1 + w(1-r))$ are not even in $r$ and its basin is $r > a$, which our normal
forms could not express (data in $s = r^2$, basin assumed to be the plane minus the
origin). The base class now takes its data in $r$ on a declared basin
`basin_radii() = (r_in, r_out)`, returns `nan` outside it, and the former contract lives
on as the subfamily `AbstractEvenNormalForm` (Hopf, Bautin — unchanged in substance).
`WinfreeNormalForm(a, w)` is the first non-even form, with $h$ and $\Psi$ derived from
the design document's §5.1–5.2 and checked against the paper's Eq. (12); $a = 0$ (the
member the experiments will mostly use) is handled by its explicit limits.
`generate(outside_basin="error" | "resample")` decides what happens to initial conditions
drawn in a phaseless set. Decisions: ADR-0013; mathematics: design document §3.3, §9.

## Files

- `src/deep_isochron/systems/normal_forms/base.py` — `AbstractNormalForm`: abstract
  `log_growth_rate(r)`, `angular_rate(r)` (were derived from `_sq` hooks); `basin_radii`,
  `in_basin`; `rhs` with a safe square root; `to_phase_amplitude` and `isochron` `nan`
  outside the basin; `eigenvalues_origin` moved. New `AbstractEvenNormalForm`: the `_sq`
  hooks, derived $r$-views, square-root-free `rhs`, `eigenvalues_origin`. Root-solve
  docstring: the inner edge is where $\Psi \to -\infty$.
- `src/deep_isochron/systems/normal_forms/winfree.py` — new: `WinfreeNormalForm(a=0.25,
  w=-0.5)` (plain float fields; `params`; closed forms with explicit `nan` outside
  $r > a$ and the $a = 0$ limits; `basin_radii = (a, ∞)`; `inner_floquet_exponent`).
- `src/deep_isochron/systems/normal_forms/{hopf,bautin}.py` — subclass
  `AbstractEvenNormalForm` (one line each).
- `src/deep_isochron/systems/normal_forms/integration.py` — `RadiusSquaredIntegration`
  refuses non-even forms (`TypeError`).
- `src/deep_isochron/systems/normal_forms/__init__.py`, `systems/__init__.py` — exports.
- `src/deep_isochron/systems/base.py` — `AbstractODE.in_basin(u)` (default `True`).
- `src/deep_isochron/data/generate.py` — `generate(..., outside_basin=)`;
  `_sample_in_basin` (reject / resample with a warning; `RESAMPLE_ROUNDS`);
  `generation_metadata(..., outside_basin=)`.
- `src/deep_isochron/data/dataset.py` — `SamplingSpec.outside_basin` (hashed),
  `Provenance.rejected_ics`.
- `src/deep_isochron/experiment/data.py` — passes `data_cfg.outside_basin` (default
  `"error"` when absent).
- `configs/data/{bautin,fhn}.yaml` — `outside_basin: error`. `configs/data/winfree.yaml`
  — new.
- `tests/test_normal_forms.py` — the shared laws run over `all_normal_forms` (even forms
  + Winfree, radii clamped into the basin); `test_even_forms_r_views_and_origin` split
  out; section 4 "basins" (4 tests). `tests/test_data.py` —
  `test_generate_handles_initial_conditions_outside_the_basin`.
- `docs/design/normal-forms.md` — amended: frontmatter/status, §1 table and the evenness
  paragraph, §3 split into §3.1–3.2 and new §3.3 (Winfree, with the cited/deduced
  split, the $a = 0$ member, why the system), §7 chart domain, §9 API, §10 tests, §11
  decisions, reference. `docs/decisions/0013-normal-form-basins.md` — new.
  `docs/architecture.md`, `docs/index.md`, `docs/roadmap.md` (ledger).

## Design

ADR-0013 has the five decisions and their alternatives. Points found while implementing:

- **`nan` must be explicit in the closed forms.** The root solve detects the basin's edge
  by a non-finite $\Psi$; Winfree's $\bigl((1-a)r/(r-a)\bigr)^{1/a}$ with $1/a = 4$ is
  *finite* for $r < a$ (negative base, integer-valued exponent), and the bracket expansion
  happily walked into the hole — `radius_from_isostable` returned `nan` for every
  $\Psi < -24$. Masking with `jnp.where(r > a, ·, nan)` (and evaluating the formula at a
  safe radius so no spurious `inf` leaks through autodiff) fixed it; recorded in §3.3.
- **Radii in shape annotations.** `isochron` calls `phase_shift` with a vector of radii,
  as Hopf and Bautin allow by leaving those methods unannotated; Winfree does the same
  (the jaxtyping hook rejected the scalar annotations under test).
- **Hash change.** `SamplingSpec.outside_basin` enters `config_hash`, so every existing
  dataset file gets a new name and must be regenerated once (ADR-0013 §3); the
  alternative — a mode that changes the data but not the hash — was rejected.
- **Dependency on the `analysis` branch**: none; this branch is from `master`. The
  signed `period()` (−2π for Winfree) is the same convention E1 noted; E6 decides.

## Bugs fixed

None pre-existing in the library (the root-solve issue above was introduced and fixed
within the branch).

## Tests

`uv run pytest tests/test_normal_forms.py tests/test_data.py`.

- Shared laws now over even forms *and* Winfree ($a \in \lbrace 0 \rbrace \cup [0.05, 0.6]$,
  $w \in [-2, 2]$): $\dot\Theta = \omega_1$, $\dot\Psi = \kappa\Psi$ by autodiff and along
  the flow; $h(1) = \Psi(1) = 0$, $\partial_r\Psi(1) = 1$, $\Psi < 0$ inside; cartesian =
  polar `rhs`; isochron / limit-cycle consistency; chart round trip; root solve
  differentiable; cartesian / polar / closed-form flows agree (`r_squared` for even forms).
- `test_even_forms_r_views_and_origin`: `_sq` views, `eigenvalues_origin`, basin from 0.
- §4: `test_winfree_structure` ($T = -2\pi$, $\kappa = a-1$, $\kappa_u = a(1-a)$, $\Psi$
  limits, `nan` outside while `rhs` is finite); `test_winfree_isochrons_are_langfield_eq_12`
  (the paper's formula, $(a, w) = (0.25, -0.5)$, direct); `test_winfree_a_zero_is_the_limit_of_small_a`
  (+ constructor checks); `test_r_squared_integration_refuses_non_even_forms`.
- `test_generate_handles_initial_conditions_outside_the_basin`: error names the basin;
  resample warns, yields $n$ trajectories in the basin, records `rejected_ics`, changes the
  hash; no rejection → no count; FHN's `in_basin` is `True`.

Full suite: 479 passed, 1 xfail (`-m "not slow"`, `--hypothesis-profile=dev`).

## Open issues

- Existing dataset files need regenerating (hash change).
- The INN-vs-hole question (ADR-0013 Consequences) is Phase E science, with Winfree as an
  observed system against the Bautin latent; $a = 0$ first.
- `Evaluator`'s reference is any `AbstractNormalForm`, so Winfree data get exact phase and
  amplitude metrics for free; the `period` metric is signed (clockwise → negative).

## Review notes

| Change | Thoughts | Modifications |
|---|---|---|
| `systems/normal_forms/base.py`: data in `r`, `basin_radii`/`in_basin`, `nan` outside, `AbstractEvenNormalForm` | | |
| `systems/normal_forms/winfree.py` (new): `WinfreeNormalForm` | | |
| `systems/normal_forms/{hopf,bautin,integration,__init__}.py`, `systems/{base,__init__}.py` | | |
| `data/generate.py` (`outside_basin`, `_sample_in_basin`), `data/dataset.py` (`SamplingSpec.outside_basin`, `Provenance.rejected_ics`), `experiment/data.py`, `configs/data/*` | | |
| `tests/test_normal_forms.py` (`all_normal_forms`, §4), `tests/test_data.py` | | |
| `docs/design/normal-forms.md` amendment (§1, §3.3, §7, §9–11); ADR-0013; architecture; index; roadmap | | |
