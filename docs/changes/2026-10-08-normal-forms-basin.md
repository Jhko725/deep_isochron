---
type: change
status: review round 3 applied (trainer stops before checkpointing a non-finite state; find_nan.py walks back to a finite checkpoint); nan hunt pending Joon's rerun
updated: 2026-10-14
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

## Review round 1 (2026-10-14) — the `nan` runs

Joon's runs against Winfree with `a = 0` (his edit of the default; initial conditions
from `r_min = 0.01`) hit `nan` after ≈ 1 400 steps (loss spike, then nothing). Assessment
and changes:

- **Cause, structural.** With `a = 0` the origin is non-hyperbolic ($\dot r = r^2(1-r)$,
  algebraic escape: from $r_0 = 0.01$ it takes ≈ 100 time units, i.e. longer than the
  trajectory). The eigenvalues at the enclosed fixed point are a conjugacy invariant, and
  $0 \pm i(1+w)$ cannot be matched by Bautin's $a \pm i\omega_0$; the map can only imitate
  algebraic escape by sending near-origin data to extreme latent radii. Checked what the
  Bautin chart does there: the closed-form flow and its gradient are finite down to
  $r \approx 10^{-100}$ (then $\Psi$ overflows to $-\infty$ and everything is `nan`), and
  outside the cycle $\Psi$ saturates so that any latent point beyond $r \approx 3$ is
  pulled to $r = 1.73$ within $0.05$ time units with gradients growing like $r$. Which of
  the two the run hit is what the new diagnostic will tell; both are the same disease.
- **`WinfreeNormalForm` accepts `a < 0`** (hyperbolic repelling focus, eigenvalues
  $-a \pm i(1+w)$, same closed forms — verified by autodiff), default `a = -0.25`,
  `basin_radii = (max(a, 0), ∞)`, `eigenvalues_origin` on the class; `winfree.yaml` uses
  `a: -0.25`, `r_min: 0.2`. `a = 0` stays available for the isochron pictures, documented
  as not a conjugacy target (design document §3.3 "three regimes"; ADR-0013 §5).
- **Latent-extent diagnostic** in the `Evaluator` (`training/evaluation.py`,
  `latent_extent`): `latent/r_min|r_max` (conjugacy models), `latent/amp_min|amp_max`,
  `latent/nonfinite`, over the validation window starts, combined across batches by
  min/max/sum. Logged with the other validation metrics, so wandb shows the data being
  flung before the loss does (ADR-0009 §5 amended). Test:
  `test_latent_extent_reports_where_the_data_are_sent`.
- Recommended for the rerun: `clip_norm=10` (turns a bad step into a bad step, not a
  `nan`), and `jax.config.update("jax_debug_nans", True)` from the last good checkpoint
  if it still happens — the diagnostic will then say *where* the points were.
- Tests: the Winfree strategies and `test_winfree_structure` cover $a < 0$ (including the
  origin's Jacobian against the eigenvalues).

## Review round 2 (2026-10-14) — `nan` persists with `a < 0`

Joon's two runs with `a = -0.25` (without / with `clip_norm = 10`): both reach `nan`
after a small loss jump; `latent/nonfinite` goes from 0 to one point first; without
clipping, `phase/circ_std` is intermittently `nan` *before* that (that metric runs over
every sample of the validation windows while the extent ran over window starts only —
so the first non-finite point was an interior sample). Clipping delays the event (≈ 2.6 k
→ ≈ 4.4 k steps) and removes the intermittent `nan`s, which says the gradient spikes are a
contributing path but not the cause. What this round changes:

- **Bautin's closed forms were the first thing to overflow.** `isostable` as written
  (design document §3) pairs $r^{2(1+b)}$ in the denominator with $(1 + br^2)^b$ in the
  numerator: from $r \approx 10^{150}$ it is $\infty/\infty = $ `nan`, and `phase_shift`
  is $\infty - \infty$ from $r \approx 10^{200}$ — while a learned bijection can reach such
  radii long before its own layers go non-finite. Rewritten in $r^{-2}$ (algebraically
  identical; the design-document table now shows both forms), finite up to the float
  range, limits exact; `test_bautin_closed_forms_are_finite_up_to_the_float_range`.
  With this, a non-finite *amplitude* at a finite image can no longer happen for Bautin.
- **The extent runs over every validation sample** and reports
  `latent/nonfinite_image` (the bijection's output itself non-finite) next to
  `latent/nonfinite` (the chart's): the two cases the next run needs to tell apart. If
  `nonfinite_image` is the one that rises, the problem is inside the INN (a spline
  pushed out of its range, a conditioner output overflowing a `softplus`/`exp`), not the
  chart.
- **`scripts/find_nan.py`**: reopens a run at a checkpoint, replays the *same* batches,
  prints the batch extent every 25 steps and, at the first non-finite gradient, the loss
  terms, the offending parameter leaves, the windows with non-finite predictions, the
  first bad window's start, its image $H(x_0)$ and its latent trajectory; `--debug-nans`
  re-runs the batch so JAX names the primitive. This is the tool for the next run —
  one command from the last good checkpoint — and replaces guessing. Smoke-tested on a
  tiny run.
- Not changed: the INN layers. Until `find_nan.py` points at one, hardening splines or
  conditioners would be speculation.

## Review round 3 (2026-10-14) — the first `find_nan.py` output

Joon's run of `find_nan.py runs/c2_no_weight_decay --step 3400 --debug-nans`: the
checkpoint at step 3400 *already* holds `nan` in every parameter — `jax_debug_nans`
fires on the very first array touched (`raw_U - raw_U.T` in `BiLipschitzLinear.constrain`,
a `convert_element_type` of a `nan` leaf), all 2048 windows are non-finite, $H(x_0)$ is
`nan` for a perfectly ordinary $x_0 = (0.93, 0.37)$. The clue is about the *checkpoint*,
not the model: the run kept training after the event and the checkpointer saved the
poisoned state as "latest" (with `save_every = 50` and the preservation policy
`LatestN(1) + BestN(1)`, the earlier finite checkpoints had been pruned). So this output
locates nothing yet; it says where to look from. Two changes so that the next run can be
hunted:

- **`Trainer.train` stops at the first non-finite validation metric** (`stop_on_nonfinite`,
  default on): `FloatingPointError` naming the step and the last checkpoint, raised
  *before* `checkpointer.save`, so the last checkpoint on disk is finite. `val/mse_early`
  / `late` being `nan` for an empty bucket (design document `validation-split.md`) does
  not stop a run. The loop still reads no device values between evaluations; the check
  lives at the evaluations, which already sync.
- **`find_nan.py` walks back** from a checkpoint with non-finite parameters to the last
  finite one on disk (listing the steps it found), or tells you there is none.

Next: rerun from `Experiment.from_config` with `eval_every` small enough (50) that the
stop lands within a few dozen steps of the event; then
`find_nan.py <run> --debug-nans` from the stopped run — the first non-finite gradient
will be at most `eval_every` steps away, and the traceback will point into the bijection
or the chart rather than at a poisoned leaf.

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
| `systems/normal_forms/base.py`: data in `r`, `basin_radii`/`in_basin`, `nan` outside, `AbstractEvenNormalForm` | Confirmed. Looks good for now. | None |
| `systems/normal_forms/winfree.py` (new): `WinfreeNormalForm` | Will mostly be experimenting with `a=0`, so default case needs to be changed. Also confirmed that the analytical isochrons visually match those of Langfield et al. (2025). | Changed the default case and the constructor. |
| `systems/normal_forms/{hopf,bautin,integration,__init__}.py`, `systems/{base,__init__}.py` | Trivial changes. | None |
| `data/generate.py` (`outside_basin`, `_sample_in_basin`), `data/dataset.py` (`SamplingSpec.outside_basin`, `Provenance.rejected_ics`), `experiment/data.py`, `configs/data/*` | Confirmed. Looks good. | None |
| `tests/test_normal_forms.py` (`all_normal_forms`, §4), `tests/test_data.py` |Looks good for now. | None |
| `docs/design/normal-forms.md` amendment (§1, §3.3, §7, §9–11); ADR-0013; architecture; index; roadmap | Had a readthrough. Looks good. | None |
| `prototype.ipynb` | Performing training on the WinfreeNormalForm as in the current notebook cells result in nan appearing after certain number of epochs. Possible reasons? Confirmed that there is no nan in source.ys. | Deferred to discussions with Claude. |

### Review round 1

| Change | Thoughts | Modifications |
|---|---|---|
| `WinfreeNormalForm`: `a < 0` allowed and default (`-0.25`), `basin_radii = (max(a,0), ∞)`, `eigenvalues_origin`; `configs/data/winfree.yaml`; tests | Looks good. Trajectories and isochrons looks okay (in a qualitative sense) too. | None |
| `training/evaluation.py`: `latent_extent` → `latent/r_min\|r_max\|amp_min\|amp_max\|nonfinite`; ADR-0009 §5; test | Confirmed. Looks good. | None |
| Design document §3.3 "three regimes"; ADR-0013 §5 amendment | Confirmed. Looks good. | None |
| `prototype.ipynb`: Performed data generation and training for the Winfree's model with `a=-0.25`. | Without gradient clipping - training progresses much better, but NaN does occur after a while, after a small loss jump. From the eval metrics, the nonfinite count increases from 0 to 1. Weirdly, circular std intermittantly shows NaN before the actual losses go bad. With `grad_clip=10`, the spurious NaNs in circular std disappears and the appearence of NaNs in the loss is quite delayed, but still occurs nonetheless. | Deferred to discussions with Claude. |

### Review round 2

| Change | Thoughts | Modifications |
|---|---|---|
| `systems/normal_forms/bautin.py`: overflow-safe `phase_shift`/`isostable` (in $r^{-2}$); design document §3 table; test | Confirmed. Training on the Winfree system does not explode now. | None |
| `training/evaluation.py`: extent over all samples, `latent/nonfinite_image`; ADR-0009 §5; test | Confirmed. | None |
| `scripts/find_nan.py` (new); `scripts/README.md`; architecture | Had a readthrough. | None |

### Review round 3

| Change | Thoughts | Modifications |
|---|---|---|
| `training/trainer.py`: `stop_on_nonfinite` (stop before checkpointing; empty-bucket `nan`s exempt); test | Looks good. | None |
| `scripts/find_nan.py`: walk back to the last finite checkpoint | Confirmed. Looks good. NaN problem was solved with the Bautin change, so have not had a change to use yet, though. | None |
