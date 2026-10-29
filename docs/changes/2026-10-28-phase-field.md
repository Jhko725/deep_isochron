---
type: change
status: implemented 2026-10-28; review pending
updated: 2026-10-28
---

# `phase-field` — the asymptotic phase fields of fhn and fhn-easy (O3, route (a))

## Summary

This branch computes the asymptotic phase on a grid for the two FitzHugh–Nagumo systems
that the models are trained on, so that the trained models' phases can be compared with
it in the presentation of 2026-10-29. It implements route (a) of
[`design/analysis.md`](../design/analysis.md) §4.1, which was quicker to make reliable
than the method that §4.2 proposes for O3. `asymptotic_phase` integrates a point for
`t_end`, reads its phase off the cycle at `t_end` and again one period later, and
reports the gap between the two readings and the distance from the cycle.
`phase_grid` evaluates it on a grid in chunks and masks the points where either
diagnostic is $10^{-6}$ or more.

The branch starts from `auto-isochrons`, whose `isochron_cycle` supplies the cycle and
the phase origin, so the field, the AUTO isochrons and the model share one phase 0. It
carries the O3/O4 survey of branch `phase` (four commits, cherry-picked), which this
document replaces as that branch's record; `phase` is retired. The method of §4.2 and
the ledger of §4.7 remain the plan for O3 proper.

## Files

- `src/deep_isochron/analysis/ode/phase.py` — new: `AsymptoticPhase`,
  `asymptotic_phase`, `phase_grid`.
- `src/deep_isochron/analysis/ode/limit_cycle.py` — `nearest_cycle_time`, the Newton
  iteration in time that `reanchor` had inline, now jit- and vmap-compatible and
  shared; `reanchor` checks its last step.
- `src/deep_isochron/analysis/ode/__init__.py` — exports and module docstring (O3).
- `src/deep_isochron/utils/numerics.py` — `wrap_phase`.
- `src/deep_isochron/systems/normal_forms/base.py` — `_wrap` replaced by `wrap_phase`.
- `src/deep_isochron/analysis/ode/auto/isochron.py` — `_wrap_phase` replaced by
  `wrap_phase`.
- `configs/isochrons/fhn.yaml`, `configs/isochrons/fhn-easy.yaml` — the `phase_field`
  block with `t_end` and its reason.
- `prototype_analysis.ipynb` — cells for the two fields, the check against AUTO and the
  comparison with a trained model.
- `tests/test_phase.py` — new: the O3 oracles and `wrap_phase`.
- `tests/helpers.py` — `TOL["phase"]`, `TOL["golden_phase"]`, `PHASE_T_END`,
  `GOLDEN_NEAR_CYCLE`; the golden-file helpers moved here from `test_isochron.py`.
- `tests/test_isochron.py` — uses the moved golden-file helpers.
- `tests/test_experiment_isochrons.py` — the `phase_field` block leaves the hash alone.
- `docs/design/analysis.md` — §4–5 the O3/O4 survey, §4.6 the second sweep, §4.7 the
  prototyping ledger (from branch `phase`); the §4 phase convention corrected to
  $(-\pi, \pi]$; §4.2 a note on route (a); citations of three or more authors
  rewritten as "First et al." for the bibliography check.
- `docs/references.md` — 17 entries for the works that the survey cites, which was
  written before the bibliography existed.
- `scripts/check_references.py` — a surname may carry the particle "de la", for
  Huguet & de la Llave 2013.
- `docs/changes/2026-10-20-phase.md` — removed; replaced by this document.
- `docs/changes/2026-10-28-phase-field.md` — this document.
- `docs/architecture.md`, `docs/index.md`, `docs/roadmap.md`, `CLAUDE.md` — the module
  map, the document list (with three change documents it lacked), the O3 row and the
  ledger, the float64 rule.

## Design

The decisions were settled in discussion on 2026-10-28 (questions Q1–Q22). None meets
the test for an ADR: the method is chosen for a deadline and may be replaced once O3
proper is prototyped, and replacing it is a revert of `phase.py`.

- **Two readings, one period apart, instead of one.** A single reading cannot tell a
  converged orbit from one that is still far from the cycle, because the nearest-point
  read-off errs at first order in that distance. The gap between two readings and the
  distance at the second are the diagnostics, returned in `AsymptoticPhase` beside the
  phase. *Rejected*: a fixed number of periods chosen from the multiplier, which says
  nothing about the escape from the source.
- **`t_end` is a required argument, set per system in the isochron config.** fhn needs
  80 and fhn-easy 300 (the reasons are in the configs). *Rejected*: a default derived
  from the multiplier, for the same reason.
- **The solve uses the cycle's configuration with `throw=False`.** A failed solve gives
  `nan` for that point instead of an error for the whole chunk; non-finite end points
  are replaced by `x0s[0]` before the read-off and the result set to `nan`, so no `nan`
  reaches the Newton steps. *Rejected*: `throw=True`, which loses a grid to one point.
- **`nearest_cycle_time` lives in `limit_cycle.py` and serves `reanchor` too.** It
  returns the last Newton step, which `reanchor` checks against $10^{-12}T$; the steps
  are clipped to half a period so that a far point cannot send the flow around the
  cycle many times. It lives beside `LimitCycle` because `phase.py` imports
  `limit_cycle.py`. *Rejected*: a second copy of the iteration in `phase.py`.
- **`phase_grid` is the host-side driver.** It refuses float32 with `RuntimeError`,
  jits the chunked `jax.lax.map` over `asymptotic_phase`, and returns
  `(field, mask, result)` with `field[i, j]` at `(xs[j], ys[i])`. *Rejected*: a
  pre-mask by `in_basin`, which FitzHugh–Nagumo does not define; the diagnostics mask
  the source.
- **One `wrap_phase`.** `jax.lax.rem` by $2\pi$ and one correction by $\pm 2\pi$, both
  exact, so it equals `math.remainder` bit for bit away from $\pm\pi$ and the isochron
  file hashes do not change. It replaces the normal forms' `arctan2` wrap and the AUTO
  pipeline's `math.remainder` wrap. *Rejected*: `remainder(θ + π, 2π) − π`, whose
  shift by $\pi$ rounds.
- **The comparison with a model is in the notebook, not the library.** The alignment
  fits a circular-mean offset for each direction of rotation and keeps the better one,
  since a model's phase has its own origin and may run the other way. The cells load
  the model last, because `load_model` applies the run's own `x64` setting.
- **Reverse time is refused.** A cycle found in reverse time is repelling, and its
  phase field belongs to the reversed flow; `asymptotic_phase` raises `ValueError`.

## Bugs fixed

None.

## Tests

`uv run just test dev tests/test_phase.py` runs the quick tier of the new tests;
`uv run pytest tests/test_phase.py -m slow` the FitzHugh–Nagumo oracle.

- `wrap_phase` equals `math.remainder(θ, 2π)` bit for bit on 2031 angles, and its
  derivative is 1.
- `nearest_cycle_time` under `vmap` returns the radial projection on Hopf's circle and
  its time, from points on both sides of the circle.
- On Bautin (counterclockwise, with a radius-dependent rate) and Winfree (clockwise),
  the phase equals the closed-form phase, signed by the sense of rotation, to
  `TOL["phase"]` on an annulus from 0.3 to 2.5; the phase along the cycle is
  $2\pi t/T$; on Hopf the phase advances by $2\pi s/T$ along the flow. The measured
  error is 1.5e-9 at most, the floor of the solve at rtol 1e-10.
- `phase_grid` orients the field, masks exactly the equilibrium and refuses float32;
  `asymptotic_phase` refuses a repelling cycle.
- Slow: at the points of the coworker's FitzHugh–Nagumo isochrons 26, 51 and 100 within
  0.01 of the cycle, the phase equals the file's phase to `TOL["golden_phase"]`
  (measured 4.5e-9). Farther out the gap grows to 1.8 rad near the source; direct
  integration over five periods confirms the phase computed here, not the stored
  one, so the stored points' position error is the cause.
- `test_hash_follows_what_decides_the_result` also checks that `phase_field` leaves the
  hash unchanged.

Results, 2026-10-28, on the staged code: `tests/test_phase.py` gives 13 passed, the
slow FitzHugh–Nagumo oracle included. The touched test files
(`test_phase`, `test_experiment_isochrons`, `test_normal_forms`, `test_isochron`,
`test_analysis_ode`) give 101 passed and 9 skipped without an AUTO tree. The full quick
tier gives 643 passed, 10 skipped, 1 xfailed and 1 failed: the worker running
`test_floquet_is_invariant_under_the_base_point_and_the_mesh` was killed for memory
(two workers in 7 GB, as in the base branch's record), and the test passes on its own.

## Open issues

- **Reverse-time cycles.** The phase of a repelling cycle under the reversed flow is
  refused, not computed; to be decided with O3 proper.
- **float64 in the other drivers.** `phase_grid` raises without float64; the existing
  drivers (`find_limit_cycle`, `reanchor`, `isochron_foliation`) do not check yet, as
  the new `CLAUDE.md` rule asks.
- **Deferred to O3 and O4.** The return-time cocycle (P1), the correction at the
  section (P2), the parametrization method (P4), the sensitivity flag (P9) and the
  local expansion (P10) of the §4.7 ledger.
- **The accuracy is set by the solver**, about $10^{-9}$ at rtol $10^{-10}$, not by
  `t_end`; a tighter field needs a tighter `cycle.config`.

## Review notes

| Change | Thoughts | Modifications |
|---|---|---|
| `docs/design/analysis.md` §4–5: the O3/O4 survey, the second sweep (§4.6), the prototyping ledger (§4.7) | | |
| `src/deep_isochron/analysis/ode/phase.py`: `AsymptoticPhase`, `asymptotic_phase`, `phase_grid` | | |
| `src/deep_isochron/analysis/ode/limit_cycle.py`: `nearest_cycle_time`, shared by `reanchor` | | |
| `src/deep_isochron/analysis/ode/__init__.py`: O3 exports and docstring | | |
| `src/deep_isochron/utils/numerics.py`: `wrap_phase` | | |
| `src/deep_isochron/systems/normal_forms/base.py`: `wrap_phase` replaces `_wrap` | | |
| `src/deep_isochron/analysis/ode/auto/isochron.py`: `wrap_phase` replaces `_wrap_phase` | | |
| `configs/isochrons/fhn.yaml`, `fhn-easy.yaml`: `phase_field.t_end` | | |
| `prototype_analysis.ipynb`: the phase fields, the AUTO check, the model comparison | | |
| `tests/test_phase.py`: the O3 oracles and `wrap_phase` | | |
| `tests/helpers.py`: O3 tolerances, `PHASE_T_END`, `GOLDEN_NEAR_CYCLE`, golden-file helpers | | |
| `tests/test_isochron.py`: golden-file helpers from `helpers.py` | | |
| `tests/test_experiment_isochrons.py`: `phase_field` outside the hash | | |
| `docs/changes/2026-10-20-phase.md`: removed, replaced by this document | | |
| `docs/architecture.md`, `docs/index.md`, `docs/roadmap.md`, `CLAUDE.md`: module map, document list, O3 row and ledger, float64 rule | | |
| `docs/references.md`: 17 entries for the survey's citations | | |
| `scripts/check_references.py`: the particle "de la" in a surname | | |
