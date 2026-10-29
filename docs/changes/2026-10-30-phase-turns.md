---
type: change
status: ADR written 2026-10-30; the change pending
updated: 2026-10-30
---

# `phase-turns` — the phase as a fraction of the period in $[0, 1)$

## Summary

[ADR-0020](../decisions/0020-phase-in-turns.md) moves the phase from an angle in
$(-\pi, \pi]$ to a fraction of the period in $[0, 1)$, increasing with time on every
system, with angles kept only where the quantity is one. The ADR was written first and
is the only change so far; the code follows in dependency order, each layer in its own
commit, so that no commit leaves master with two conventions.

Irreversible in effect: the isochron hashes are repinned and every saved foliation is
recomputed (ADR-0020, Consequences).

## Files

- `docs/decisions/0020-phase-in-turns.md` (**new**) — the decision, with Joon's
  rationale (the angle reading misleads on a cycle that is not star-shaped), the
  coworker's existing fractions, and the orientation rule.
- `docs/index.md` — the ADR's row; this document's row.
- `.claude/agents/implementer.md` (**new**) — the implementer subagent that
  `orchestrate-implementation` starts once per commit: its boundary (code, tests,
  docstrings, and this document's *Files* bullets and review rows), stop rule and
  report format, versioned here so that a brief points instead of restating them.
- `src/deep_isochron/utils/numerics.py` — `wrap_phase` wraps to $[0, 1)$; the radian
  wrap is `wrap_angle`; `phase_difference` is the signed shortest arc in
  $(-\tfrac12, \tfrac12]$, against the array API as the wraps are.
- `tests/test_phase.py` — the wrap test split in two: `wrap_angle` against
  `math.remainder`, and `wrap_phase` and `phase_difference` against floor-based
  oracles across the five input forms, the tie and the gradient.
- `src/deep_isochron/systems/normal_forms/base.py` — `to_phase_amplitude` returns
  $\Theta = \mathrm{wrap}(\sigma(\theta + h(r))/2\pi) \in [0, 1)$, with $\sigma$
  the sign of $\omega_1$; `from_phase_amplitude`, `limit_cycle` and `isochron` invert
  the same sign and $2\pi$; the docstrings of `phase`, `phase_shift` and the module
  state the phase in turns.
- `src/deep_isochron/systems/normal_forms/integration.py` — `ClosedFormIntegration`
  advances the phase by $t/T$, with the period positive.
- `tests/test_normal_forms.py` — phases compared through `phase_difference`; a
  `phases` strategy in $[0, 1)$; the flow law states that the phase advances by the
  elapsed fraction of the period on every form, clockwise ones included; Langfield's
  Eq. 12 identity reads $\Theta = \vartheta$.
- `src/deep_isochron/model/base.py` — `AbstractPhaseAmplitudeModel.phase` promises a
  fraction of the period in $[0, 1)$, up to the model's origin (and orientation, for
  the autoencoder); `cycle_point`, `isochron` and `phase_sensitivity` take that phase,
  the last normalized by $Z \cdot F = 1/T$. Docstrings only.
- `src/deep_isochron/model/conjugacy.py` — docstrings say turns; the phase is the
  normal form's, so no body changes.
- `src/deep_isochron/model/autoencoder.py` — `phase` returns
  $\mathrm{wrap}(\mathrm{atan2}(Y_2, Y_1)/2\pi)$; `cycle_point` and `isochron`
  encode the angle $2\pi\Theta$; the latent plane's angle stays an angle.
- `tests/test_models.py` — the range law is $0 \le \Theta < 1$; the conjugacy laws
  take phases in turns and compare them through `phase_difference`.
- `tests/test_baseline.py` — the range law is $0 \le \Theta < 1$; the exact chart
  encodes $(\cos 2\pi\Theta, \sin 2\pi\Theta, \Psi)$ and decodes through
  $\mathrm{atan2}/2\pi$; `_exact_dynamics` rotates at $|\omega_1|$, the rate of the
  encoded angle.
- `tests/test_training.py` — `_exact_autoencoder` encodes and decodes the same chart,
  with the latent rotation at $|\omega_1|$.
- `justfile` — the `test` recipe takes positional arguments, so that a quoted `-k`
  expression reaches pytest as one argument (`just test dev tests/test_x.py -k "a or b"`
  used to split it into three).
- `src/deep_isochron/systems/normal_forms/base.py` — `period` is $2\pi/|\omega_1| > 0$,
  with the orientation in the sign of `omega` alone; `floquet_multiplier`
  $e^{\kappa T}$ is therefore below 1 for a stable clockwise cycle.
- `src/deep_isochron/systems/normal_forms/integration.py` — `ClosedFormIntegration`
  advances the phase by $t/T$, without the absolute value.
- `tests/test_normal_forms.py` — the identity reads $T = 2\pi/|\omega_1|$; the
  clockwise Winfree has $T = 2\pi$ and $\omega_1 < 0$; the flow law divides by $T$.
- `tests/test_analysis_ode.py` — every `abs` around the normal form's period removed.
- `tests/test_training.py`, `tests/test_baseline.py` — the docstrings of the exact
  latent rotation read $2\pi/T = |\omega_1|$.
- `tests/test_analysis.py` — the trajectory helper takes the normal form's period
  without an absolute value.
- `src/deep_isochron/analysis/cycle.py` — `Cycle`'s phase is a fraction of the period
  in $[0, 1)$: `phases` is uniform on $[0, 1)$, `__call__`, `derivative` and
  `from_points` evaluate or fit the series at $2\pi\varphi$, `derivative` and the
  second derivative are taken in turns, and `nearest_phase` takes its Newton step in
  turns and wraps by `wrap_phase`; `omega` stays the angular frequency of the series.
- `src/deep_isochron/analysis/data/cycle.py` — `estimate_cycle` keeps the protophase,
  the density step and the period in radians and hands $\varphi/2\pi$ to
  `Cycle.from_points` and to the self-consistency check.
- `tests/test_analysis.py` — `circular_offset` removed: the `Cycle`'s phases are
  compared through `phase_difference` and the protophase angles through `wrap_angle`;
  the circle test and the Fourier fit take phases in turns.
- `tests/test_analysis_ode.py` — the phases handed to `Cycle.from_points` are
  `times / period`.
- `src/deep_isochron/analysis/ode/phase.py` — `asymptotic_phase` reads the phase as
  $\mathrm{wrap}((t - t_{\rm read})/T) \in [0, 1)$, and its discrepancy is the
  unsigned `phase_difference` of the two readings; the module and `AsymptoticPhase`
  docstrings state turns.
- `src/deep_isochron/analysis/model.py` — `phase_sense` is the sign of the
  `phase_difference` of the model's phases; `phase_offset` samples the learned cycle at
  phases in $[0, 1)$ and returns the offset in turns.
- `tests/test_phase.py` — `_gap` is the unsigned `phase_difference`; the radial
  projection, offset, closed-form, on-the-cycle ($t/T$) and flow laws are stated in
  turns; the sense of a conjugacy model is $+1$ for both orientations, and the
  closed-form comparison drops the orientation factor.
- `src/deep_isochron/analysis/ode/auto/stages.py` — `rotate_phase` places the user
  point of a requested phase $\Theta$ at $\tau_2 = (1 - \Theta)T$, a fraction of the
  period converted to a time, so no angle reaches AUTO.
- `src/deep_isochron/analysis/ode/auto/foliation.py` — `wrap_phases` and
  `isochron_foliation` take phases in turns and wrap them to $[0, 1)$.
- `src/deep_isochron/analysis/ode/auto/results.py` — `Isochron.phase` is in turns; the
  netCDF `phase` variable carries `units = "turns"`, and `IsochronFoliation.load`
  refuses a file without it (ADR-0020, decision 5); "windings" for the spiral's turns.
- `tests/helpers.py` — `golden_phase(k)` is the coworker's fraction $1 - k/100$;
  `hand_foliation`'s phases are its polar angles over $2\pi$; the tolerances that bound
  a phase are restated in turns, and `isochron_phase` is 1e-8 turns, the user-point
  acceptance of stage 3.
- `tests/test_auto_stages.py` — the requested phases are fractions; the gaps are taken
  by `phase_difference`, without the sense of rotation.
- `tests/test_isochron.py` — the `wrap_phases` test in turns; a new test that `load`
  refuses a file without a phase unit.
- `src/deep_isochron/utils/numerics.py` — `wrap_angle` is deleted, since nothing
  calls it; `wrap_phase` and `phase_difference` share one private prologue,
  `_array_and_namespace`, and overloads type a JAX array in as a JAX array out and a
  NumPy array or a float in as a NumPy array out.
- `src/deep_isochron/model/autoencoder.py`, `src/deep_isochron/analysis/model.py`,
  `src/deep_isochron/training/evaluation.py`, `src/deep_isochron/analysis/cycle.py`,
  `src/deep_isochron/analysis/ode/auto/foliation.py`,
  `src/deep_isochron/plotting/isochrons.py` — the eight calls of `wrap_phase` drop
  their `asarray` wrapper; the `wrap_phases` docstring says why it rounds to 12
  decimals before looking for repeats.
- `tests/test_phase.py` — the `wrap_angle` test is deleted; the wrap test adds the type
  law for a Python float and for a call that mixes a JAX array and a float; `_gap`
  drops its `asarray`.
- `tests/test_analysis.py` — the warped-protophase law takes the angle offset by the
  complex exponential, as before `wrap_angle`.
- `docs/decisions/0020-phase-in-turns.md` — decision 3 amended: `wrap_angle` deleted.

## Design

The ledger of the change, in the order the commits will follow. Each item becomes a
*Files* bullet and a review row when its commit lands.

1. `utils/numerics.py`: `wrap_phase` to $[0, 1)$, `wrap_angle` for radians,
   `phase_difference` in $(-\tfrac12, \tfrac12]$, all against the array API. — done.
   Review of 2026-10-30: `wrap_angle` deleted (no caller), one prologue helper,
   overloads in place of `asarray` at the call sites.
2. `systems/normal_forms/base.py`: `phase`, `to_phase_amplitude`, `limit_cycle`,
   `isochron` at the seam; the polar angle negated for $\omega_1 < 0$;
   `from_phase_amplitude` and the closed-form integration follow; `period` is
   $2\pi/|\omega_1|$ (Joon, 2026-10-30). — done.
3. `model/base.py`, `conjugacy.py`, `autoencoder.py`: the contract in turns;
   `cycle_point` and `isochron` take $2\pi\Theta$; `phase_sensitivity` in turns.
   — done.
4. `analysis/cycle.py`, `analysis/data/cycle.py`: `Cycle`'s phase in turns, the
   Fourier series in $2\pi\varphi$ inside. — done.
5. `analysis/ode/phase.py`: `phase = wrap(t/T)`, the discrepancy by
   `phase_difference`. — done.
6. `analysis/model.py`: `phase_offset` in turns; `phase_sense` by `phase_difference`.
   — done.
7. `analysis/ode/auto/`: `rotate_phase`'s $\tau_2 = (1 - \Theta)T$; `wrap_phases`;
   `Isochron.phase`; the netCDF `units` attribute and the refusal in `load`. — done.
8. `experiment/isochrons.py`, `configs/isochrons/`: `phases.count` → $k/$`count`;
   the hashes repinned.
9. `training/evaluation.py`: `circular_std` and the sums with $e^{2\pi i\Delta}$;
   `align_phases` in turns.
10. `plotting/isochrons.py`: `PHASE_NORM = Normalize(0, 1)`, the sample grids, the
    gaps by `phase_difference`.
11. `tests/`: every phase assertion and the golden isochrons' phase labels
    ($k/100$); the three `_gap` helpers replaced by `phase_difference`; the golden
    labels done with unit 6.
12. `prototype_analysis.ipynb`; the design documents and ADR History lines named in
    ADR-0020; `architecture.md`; the roadmap's item M.

## Bugs fixed

None yet.

## Tests

- `uv run just check`: every hook passed on the ADR commit.
- The code's tests: pending.

## Open issues

- The foliation files on the cluster are recomputed after the merge.

## Review notes

| Change | Thoughts | Modifications |
|---|---|---|
| `docs/decisions/0020-phase-in-turns.md`, `docs/index.md` — ADR-0020 and its rows | Looks good. | Changed the date of my presentation. |
| `.claude/agents/implementer.md` — the implementer subagent's rules | Looks god. | None |
| `src/deep_isochron/utils/numerics.py` — `wrap_phase` to $[0, 1)$, `wrap_angle`, `phase_difference` | `wrap_angle` and `wrap_phase` are essentially the same function. Seems like different implementation of the same logic, upto scaling and offsets. Also we will be implementing more array standards api-based function. To might be useful to create a common helper function that either - passes numpy array and JAX arrays through and coerces floats to numpy / if the object has array_namespace use that, else (such as floats) use a default (which would be numpy). | To be discussed. |
| `tests/test_phase.py` — the wrap tests of both wraps and the difference | Tests like `test_wrap_angle_equals_the_ieee_remainder` feel like they should be tested with hypothesis strategies, but have numpy rngs baked into them. Is this our intended rule? Are the subagents properly getting instructions on how we setup tests (hypothesis for specific properties; unit tests ; golden tests). Also, if we make all property tests hypothesis tests, will this slow down testing? | To be discussed. |
| `src/deep_isochron/systems/normal_forms/base.py` — the phase in turns at the chart, cycle and isochron seams | Confirmed. | None |
| `src/deep_isochron/systems/normal_forms/integration.py` — the closed form advances the phase by $t/T$, with the period positive | Confirmed. | None |
| `tests/test_normal_forms.py` — phase comparisons in turns; the orientation law | Confirmed. | None |
| `src/deep_isochron/model/base.py`, `src/deep_isochron/model/conjugacy.py` — the model contract in turns | Confirmed. | None |
| `src/deep_isochron/model/autoencoder.py` — the phase seams $\mathrm{atan2}/2\pi$ and $2\pi\Theta$ | Confirmed. | None |
| `tests/test_models.py`, `tests/test_baseline.py`, `tests/test_training.py` — the range law, phase comparisons and the exact chart in turns | Confirmed. | None |
| `justfile` — `test` recipe with positional arguments for `-k` | Confirmed. | None |
| `src/deep_isochron/systems/normal_forms/base.py`, `src/deep_isochron/systems/normal_forms/integration.py` — `period` is $2\pi/\lvert\omega_1\rvert$; the closed form advances by $t/T$ | Confirmed. | None |
| `tests/test_normal_forms.py`, `tests/test_analysis_ode.py` — the period positive; the Winfree orientation through `omega` | Confirmed. | None |
| `tests/test_training.py`, `tests/test_baseline.py` — docstrings read $2\pi/T$ | Confirmed. | None |
| `tests/test_analysis.py` — the trajectory helper's period without `abs` | `circular_offset` looks like a reimplementation of a logic we already have (e.g., `wrap_phase`). | To be discussed. |
| `src/deep_isochron/analysis/cycle.py`, `src/deep_isochron/analysis/data/cycle.py` — `Cycle`'s phase in turns, the series in $2\pi\varphi$; $\varphi/2\pi$ handed to the fit | Confirmed. | None |
| `tests/test_analysis.py`, `tests/test_analysis_ode.py` — `Cycle` phases in turns; `phase_difference` and `wrap_angle` replace `circular_offset` | Confirmed. | None |
| `src/deep_isochron/analysis/ode/phase.py` — `asymptotic_phase` and its discrepancy in turns | Confirmed. | None |
| `src/deep_isochron/analysis/model.py` — `phase_sense` and `phase_offset` in turns | Confirmed. | None |
| `tests/test_phase.py` — the asymptotic-phase and offset laws in turns; `_gap` by `phase_difference` | Do we need an `np.asrray` around `phase_difference`? If it's properly implemented in terms of array api, we don't need it, do we? | To be discussed. |
| `src/deep_isochron/analysis/ode/auto/stages.py` — stage 3's $\tau_2 = (1 - \Theta)T$ | Confirmed. | None |
| `src/deep_isochron/analysis/ode/auto/foliation.py` — `wrap_phases` to $[0, 1)$ | This is `wrap_phase` with extra housekeeping.. but I guess we keep it like this because t1) the housekeeping is important and 2) its not used elsewhere? Why do we do the rounding? | To be discussed. |
| `src/deep_isochron/analysis/ode/auto/results.py` — the `units` attribute and the refusal in `load` | Confirmed. | None |
| `tests/helpers.py` — `golden_phase`, `hand_foliation` and the phase tolerances in turns | Confirmed. | None |
| `tests/test_auto_stages.py`, `tests/test_isochron.py` — the AUTO tests in turns; the refusal test | Confirmed. | None |
| `src/deep_isochron/utils/numerics.py` — one prologue helper, overloads, `wrap_angle` deleted | | |
| `src/deep_isochron/model/autoencoder.py`, `src/deep_isochron/analysis/model.py`, `src/deep_isochron/training/evaluation.py`, `src/deep_isochron/analysis/cycle.py`, `src/deep_isochron/analysis/ode/auto/foliation.py`, `src/deep_isochron/plotting/isochrons.py` — the `asarray` wrappers dropped; `wrap_phases` documents its rounding | | |
| `tests/test_phase.py` — the `wrap_angle` test deleted; the type law for a float and a mixed call | | |
| `tests/test_analysis.py` — the angle offset by the complex exponential | | |
| `docs/decisions/0020-phase-in-turns.md` — the `wrap_angle` amendment | | |
