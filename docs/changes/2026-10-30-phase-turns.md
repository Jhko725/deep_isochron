---
type: change
status: landed 2026-10-30 (nine units, the period unit, the review's cleanup, the CI change, the documents); review of the documents commit pending
updated: 2026-10-30
---

# `phase-turns` — the phase as a fraction of the period in $[0, 1)$

## Summary

[ADR-0020](../decisions/0020-phase-in-turns.md) moves the phase from an angle in
$(-\pi, \pi]$ to a fraction of the period in $[0, 1)$, increasing with time on every
system, with angles kept only where the quantity is one. The ADR was written first;
the code followed in dependency order, each layer in its own commit, so that no commit
left master with two conventions: the numerics helpers, the normal forms, the model
contract, `Cycle`, the ODE-side phase, the AUTO pipeline, the isochron configs, the
evaluation, the figures and the notebook, then the documents. Two decisions were taken
on the way: a normal form's `period` is a positive time, and `wrap_angle` is deleted.
The CI `checks` job, which had reached its time limit, keeps JAX's compilation cache
across runs and runs 100 hypothesis examples.

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
- `src/deep_isochron/experiment/isochrons.py` — `phases.count` gives the fractions
  $k/$`count` and `phases.values` is read in turns, wrapped to $[0, 1)$; the
  `PhasesConfig` and `wrapped_phases` docstrings say turns, and the unused `math`
  import goes.
- `configs/isochrons/fhn.yaml`, `configs/isochrons/fhn-easy.yaml` — the comments on
  `phases` read turns.
- `tests/test_experiment_isochrons.py` — the config test expects the fractions
  $(0, \tfrac14, \tfrac12, \tfrac34)$ and `[0.5, 7.75]` wrapped to $(0.5, 0.75)$;
  `SHIPPED_HASHES` is repinned to `11305875` (`fhn`, was `d9ded7af`) and `fb823a3f`
  (`fhn-easy`, was `b688cbfe`). The repin is irreversible: every foliation saved under
  the old hashes is orphaned and recomputed on the cluster after the merge.
- `src/deep_isochron/training/evaluation.py` — `circular_std` and the phase-amplitude
  sums take phase differences in turns and form their resultants from
  $e^{2\pi i\Delta}$; the spread is the circular standard deviation of $2\pi\Delta$
  divided by $2\pi$, so `phase/circ_std` reads in turns under its old name;
  `align_phases` finds the offset as the circular mean in turns and the residual by
  `phase_difference`, and `PhaseAlignment.offset` is stored in $[0, 1)$. Both
  orientations are still tried (ADR-0020).
- `src/deep_isochron/utils/numerics.py` — `phase_difference` annotates its arguments
  `*#shape`, so `a` and `b` broadcast, as a phase against an offset.
- `tests/test_training.py` — the `circular_std` and `align_phases` laws in turns.
- `tests/test_phase.py` — the wrap test adds a scalar offset against a phase array.
- `src/deep_isochron/plotting/isochrons.py` — `PHASE_NORM` is `Normalize(0, 1)` and the
  learned curve is sampled on turns; `plot_phase_field` takes a matplotlib `norm` in
  place of the symmetric `limit`, since a phase in $[0, 1)$ has no symmetric range; the
  residual panel passes `Normalize(-r, r)` from the 99.5th percentile of the residual;
  the isochron gaps go through `phase_difference`; the colorbars say turns.
- `prototype_analysis.ipynb` — cells 7, 9, 13, 15 and 16 follow the API (the learned
  isochron at 0.875 turns, the gap through `phase_difference`, labels in turns) and
  cell 8 reads "a sixth of a turn"; the outputs of the edited code cells are cleared.
- `.github/workflows/ci.yml`, `tests/conftest.py` — JAX's persistent compilation cache
  kept across CI runs by `actions/cache` in each job, and the `ci` hypothesis profile at
  100 examples: the `checks` job had reached its 45-minute limit at 300.
- `tests/conftest.py` — an autouse fixture that calls `jax.clear_caches()` after every
  test, so a worker's memory no longer grows with each compiled program.
- `tests/helpers.py` — `CompileCounter` counts staged programs only; a program named
  after a `jax.numpy` or `jax.lax` function is an eager operation and is skipped.

- `docs/design/normal-forms.md` — §1's symbol table ($\Theta \in [0, 1)$, $T > 0$, $h$
  an angle), the notation paragraph ("we use turns, $\Theta = \vartheta$"), §5.1's
  derivation and §7's chart API, each as a dated amendment at the sentence.
- `docs/design/phase-field.md` — §1's conventions paragraph (the rescaling by $1/T$,
  every $\omega t$ read as $t/T$), §1.2's normalization $Z \cdot F = 1/T$, the
  `phase_sensitivity` route, the O4 identities and item M's residual, as amendments.
- `docs/design/isochrons.md` — §1's convention ($\Theta = \vartheta$), §3.4's stage 3
  ($\tau_2 = (1 - \Theta)T$), the adjoint normalization and the bundle's scale, as
  amendments.
- `docs/design/cycle-estimation.md` — §1.1's convention (`Cycle`'s phase in turns, the
  series in $2\pi\varphi$, the protophase and period estimate in radians) and the
  affine-image oracle row, as amendments.
- `docs/decisions/0007-systems-hierarchy-and-flow-strategies.md`,
  `docs/decisions/0009-trainer-losses-schedules-checkpoints.md`,
  `docs/decisions/0013-normal-form-basins.md`,
  `docs/decisions/0017-isochrons-by-auto-continuation.md` — the sentences that stated
  the radian convention amended in place; a History line each. ADR-0020's History says
  the change landed.
- `docs/architecture.md` — the phase contract lines of the normal forms and the models
  in $[0, 1)$.
- `docs/roadmap.md` — O4's and M's normalization $Z \cdot F = 1/T$; the Done ledger row.
- `docs/changes/2026-10-30-phase-turns.md` — this document: *Summary*, *Design*'s
  decisions, *Bugs fixed*, *Tests*, *Open issues*.
- `tests/conftest.py`, `.github/workflows/ci.yml`, `CLAUDE.md`, `docs/coding-standards.md`
  — retro: the compilation cache is turned on by `conftest.py` so every launcher gets it
  (CI only caches the directory); the `ci` profile's count corrected in the standards;
  the cache and the memory-cgroup fact in `CLAUDE.md`.
- `docs/design/normal-forms.md` — converted to the turn convention: the chart's angle
  named $\phi$, $\Theta = \phi/(\omega_1 T)$. The derivations of §5.1, the isochron
  corollary and the closed forms are about $\phi$; §1, §2, §5.3–5.4, §6.1, §7 and §10
  state the phase; the Winfree period is $2\pi$; the amendment lines are gone.
- `docs/design/phase-field.md` — converted to the turn convention: $t/T$, $\bmod 1$,
  $Z\cdot F = 1/T$ in §1, the routes, the ledger and the oracle tables; the cited
  normalizations keep their angle form with ours beside it.
- `docs/design/isochrons.md` — converted to the turn convention: $t/T$, $\bmod 1$,
  $Z\cdot F = 1/T$. Langfield's $\vartheta$ is the phase; §1,
  §3.1, §3.4, §4 and §5 in turns, with the bundle's $Z\cdot F = 1/T = 0.0891$.
- `docs/design/cycle-estimation.md` — converted to the turn convention: $t/T$,
  $\bmod 1$, $Z\cdot F = 1/T$. `Cycle`'s series in $2\pi\varphi$;
  the radian quantity of §3 named the angle $\varphi_{\rm angle}$, handed to the fit as
  $\varphi_{\rm angle}/2\pi$; the oracle rows in turns and the period positive.
- `docs/decisions/0020-phase-in-turns.md` — its History line says the design documents
  were rewritten in the convention rather than amended.

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
   the hashes repinned. — done.
9. `training/evaluation.py`: `circular_std` and the sums with $e^{2\pi i\Delta}$;
   `align_phases` in turns. — done.
10. `plotting/isochrons.py`: `PHASE_NORM = Normalize(0, 1)`, the sample grids, the
    gaps by `phase_difference`. — done.
11. `tests/`: every phase assertion and the golden isochrons' phase labels
    ($k/100$); the three `_gap` helpers replaced by `phase_difference`; the golden
    labels done with unit 6. — done, each file with its unit.
12. `prototype_analysis.ipynb`; the design documents and ADR History lines named in
    ADR-0020; `architecture.md`; the roadmap's item M. — `prototype_analysis.ipynb`
    done with unit 9; the documents done in the last commit.

The decisions, each with its reason and where it is recorded. The structural ones are
ADR-0020's; the rest are this branch's.

- $\Theta \in [0, 1)$, a fraction of the period, 0 at `x0s[0]`, increasing with time
  on every system. An angle reads as geometry and misleads on a cycle that is not
  star-shaped; a fraction is defined for every cycle; the coworker's code already uses
  fractions. Rejected: $[0, T)$ (the range would differ between systems) and
  $(-\tfrac12, \tfrac12]$ (the seam half a turn from the base point, negative phases
  for half of every period). ADR-0020 decision 1 and Alternatives.
- Angles stay where the quantity is an angle, and the $2\pi$ is written at named seams:
  `to_phase_amplitude`/`from_phase_amplitude`, `limit_cycle`, `isochron`; the
  autoencoder's `phase`, `cycle_point`, `isochron`; `Cycle`'s series (`__call__`, the
  two derivatives, `from_points`); `phase_offset`'s and `_learned_curve`'s sample grids
  are phases and live in $[0, 1)$; `isochron_coordinate` is radial and untouched. One
  conversion site per seam, no inline constants. ADR-0020 decision 2.
- The sign of $\omega_1$ is applied inside `to_phase_amplitude` and inverted in
  `from_phase_amplitude`, `limit_cycle` and `isochron`, so that a clockwise form's cycle
  point at $\Theta = \tfrac14$ is the point reached a quarter period after the origin.
  The orientation rule has to hold for the closed forms too. The handoff's assumption,
  confirmed by the flow law of `tests/test_normal_forms.py`.
- `phase_difference(a, b)` in $(-\tfrac12, \tfrac12]$, one array-API helper with
  overloads (JAX in, JAX out; NumPy or float in, NumPy out) and broadcastable arguments
  (a phase array against a scalar offset). The signed shortest arc is what an RMS error
  needs; the overloads remove the `asarray` wrapper the union return type forced at
  eight call sites. ADR-0020 decision 3; the overloads and the `*#shape` annotation from
  the review of units 1–6 and unit 8.
- `wrap_angle` deleted. The ADR kept the radian wrap for the geometry of decision 2, but
  once the normal forms wrapped on the turn no site in `src/` wrapped an angle; the one
  test that compared angles uses the inline complex-exponential offset. Joon, review of
  units 1–6; ADR-0020 decision 3 amended.
- `wrap_phase` sends the one rounding case `x - floor(x) == 1.0` (a negative phase
  within half an ulp of 0) to 0, so that the range holds bit for bit; the derivative is
  0 at that float and 1 elsewhere, and the function is discontinuous there in any case.
- `period()` is $2\pi/|\omega_1|$, a positive time; the orientation is the sign of
  `omega()` alone. A period is a time, the data-side periods were positive by
  construction, and every consumer that used the normal form's period as a time already
  took its absolute value. Joon, 2026-10-30, after unit 2. Reversible, though
  `floquet_multiplier` would be wrong again for a clockwise form.
- $Z \cdot F = 1/T$: the normalization in turns. ADR-0020 decision 4;
  `phase-field.md` §1.2; roadmap M.
- The netCDF `phase` variable carries `units = "turns"`, and `IsochronFoliation.load`
  refuses a file without it, so an old file fails loudly rather than reading as turns.
  The hashes are repinned (`fhn` `d9ded7af → 11305875`, `fhn-easy`
  `b688cbfe → fb823a3f`) and the foliations are recomputed on the cluster. ADR-0020
  decision 5. Irreversible in effect.
- `Isochron.phase` records the requested phase, not the one found by stage 3. The two
  differ by at most `1e-8 * period`, stage 3's acceptance of a user point
  (`stages.py:175`); the requested value is what the config, the hash and a user's
  lookup carry. Joon, 2026-10-30, gate 2. The test tolerance `isochron_phase` is 1e-8
  turns for that reason; the other phase tolerances were divided by $2\pi$ so that
  they bound what they bounded before.
- Metric names unchanged, units change: `phase/circ_std` and the aligned residual read
  in turns under their names; a renamed metric would be a second name for one quantity.
  ADR-0020 decision 6. wandb history across this branch is not comparable.
- `plot_phase_field` takes a matplotlib `norm` in place of a symmetric `limit`: a phase
  in $[0, 1)$ has no symmetric range; the residual panel passes its own symmetric
  range. `limit` is replaced, not kept beside `norm`.
- The CI `checks` job keeps JAX's persistent compilation cache across runs and the `ci`
  hypothesis profile runs 100 examples (300 took the job to its 45-minute limit on
  2026-10-29). Joon, 2026-10-30. Reversible.

## Bugs fixed

- `AbstractNormalForm.floquet_multiplier` returned $e^{\kappa T}$ with the signed
  period, which exceeds 1 for a stable clockwise cycle ($\kappa < 0$, $T < 0$). The
  positive period fixes it; `tests/test_normal_forms.py` asserts the Winfree form's
  orientation through `omega` and its multiplier below 1.
- `TOL["isochron_phase"]` at 1e-8 rad was tighter than stage 3's acceptance of a user
  point (1e-8 of a period, up to 6e-8 rad) and passed by luck; it is 1e-8 turns with
  that reason (`tests/helpers.py`).

## Tests

- `uv run just check`: clean on every commit of the branch.
- The full tier, `uv run just test default` at `c6ac9f7` (two cores, 7 GB, JAX's
  persistent compilation cache on): `673 passed, 1 xfailed in 914 s`. The `xfail` is
  the pinned `sinh` second-derivative case of `tests/test_analytic.py`, unrelated. At
  `cca3c27`, before the memory fix, the same run had both workers killed by the
  memory cgroup at 3.4 and 4.1 GB RSS (`2 failed, 671 passed`, the two "failures" the
  tests running at the kill); serially it was `673 passed, 1 xfailed` at a peak of
  4.96 GB. With the fix the workers stay near 1.2 GB.
- `uv run just test-auto quick` at `cca3c27`, with the AUTO tree: `10 passed in 280 s`.
- The golden FitzHugh–Nagumo phases (`slow`, three cases): `uv run pytest
  tests/test_phase.py -k test_fitzhugh_nagumo_phase_on_the_coworkers_isochrons -m slow
  -n 2 --hypothesis-profile=dev`: `3 passed`.
- The `slow` tier and the `ci` profile run in CI or on Joon's machine.
- Per unit, the fast tier of the touched files passed before its commit; the counts
  are in the commit messages' evidence as the orchestrating session recorded them:
  `test_normal_forms.py` 29, `test_models.py` 7, `test_baseline.py` 9,
  `test_analysis.py` 12, `test_analysis_ode.py` 44, `test_phase.py` 14 (+3 slow),
  `test_auto_stages.py` and `test_isochron.py` 23 together, `test_experiment_isochrons.py`
  11, `test_training.py` 22, `test_plotting.py` 5.

## Open issues

- The foliation files on the cluster are recomputed after the merge, under the hashes
  `11305875` (`fhn`) and `fb823a3f` (`fhn-easy`).
- `test_fitzhugh_nagumo_isochrons_lie_on_the_coworkers` failed once under `-n 2` at
  `00b5cdb` and passed alone (the handoff's baseline); it passed in every run of this
  branch. Not reproduced; watch it.
- Stage 3 accepts a user point within `1e-8 * period` of its target, so a traced
  isochron's base point sits up to 1e-8 turns off its recorded phase; a tighter base
  point would come from that acceptance, in the AUTO settings round (`EPSL`/`EPSU`/
  `EPSS`).
- The hypothesis-versus-`default_rng` question of the wrap tests (review of units 1–6)
  has no rule in the repository; a rule for `coding-standards.md` and the implementer
  agent file is a retro item.
- `docs/index.md` keeps ADR-0020 at `accepted (review pending)` until Joon reviews the
  documents commit.

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
| `src/deep_isochron/utils/numerics.py` — one prologue helper, overloads, `wrap_angle` deleted | Looks good. | None |
| `src/deep_isochron/model/autoencoder.py`, `src/deep_isochron/analysis/model.py`, `src/deep_isochron/training/evaluation.py`, `src/deep_isochron/analysis/cycle.py`, `src/deep_isochron/analysis/ode/auto/foliation.py`, `src/deep_isochron/plotting/isochrons.py` — the `asarray` wrappers dropped; `wrap_phases` documents its rounding | Looks good. | None |
| `tests/test_phase.py` — the `wrap_angle` test deleted; the type law for a float and a mixed call | Confirmed. | None |
| `tests/test_analysis.py` — the angle offset by the complex exponential | Confirmed. | None |
| `docs/decisions/0020-phase-in-turns.md` — the `wrap_angle` amendment | Looks good. | None |
| `src/deep_isochron/experiment/isochrons.py`, `configs/isochrons/fhn.yaml`, `configs/isochrons/fhn-easy.yaml` — `phases.count` and `phases.values` in turns | Confirmed. | None |
| `tests/test_experiment_isochrons.py` — the config test in turns; the two shipped hashes repinned | Confirmed. | None |
| `src/deep_isochron/training/evaluation.py` — `circular_std`, the sums and `align_phases` in turns | Confirmed. | None |
| `src/deep_isochron/utils/numerics.py` — `phase_difference` broadcasts its arguments | Confirmed. | None |
| `tests/test_training.py` — the spread and alignment laws in turns | Confirmed. | None |
| `tests/test_phase.py` — the scalar-offset case of the difference | Confirmed. | None |
| `src/deep_isochron/plotting/isochrons.py` — `PHASE_NORM` over $[0, 1)$, `norm` in place of `limit`, the gaps by `phase_difference`, colorbars in turns | Confirmed. | None |
| `prototype_analysis.ipynb` — cells 7, 8, 9, 13, 15, 16 follow the API; outputs of the edited code cells cleared | Confirmed. | None |
| `.github/workflows/ci.yml`, `tests/conftest.py` — the JAX cache in CI; `ci` at 100 examples | `conftest.py`'s docs said that the ci run is still 300. | Module docstring fixed. |
| `tests/conftest.py` — the autouse fixture that clears JAX's caches after every test | Confirmed. | None |
| `tests/helpers.py` — `CompileCounter` skips eager `jax.numpy` and `jax.lax` programs | Confirmed. | None |
| `docs/design/normal-forms.md`, `docs/design/phase-field.md`, `docs/design/isochrons.md`, `docs/design/cycle-estimation.md` — the dated amendments to turns | Writing in the old convention and adding amendents below (`isochrons.py` does this for eaxmple) make the reader constantly have to remember the rule and perform substitutions in the head. Better to convert to the new convention, unless there is a very good reason. | To be doscissed. |
| `docs/decisions/0007-systems-hierarchy-and-flow-strategies.md`, `docs/decisions/0009-trainer-losses-schedules-checkpoints.md`, `docs/decisions/0013-normal-form-basins.md`, `docs/decisions/0017-isochrons-by-auto-continuation.md` — amended sentences and History lines | Confirmed. | None |
| `docs/architecture.md`, `docs/roadmap.md` — the contract lines, the normalizations, the Done row | Confirmed. | None |
| `docs/changes/2026-10-30-phase-turns.md` — the documents commit's sections | Confirmed. | None |
| `tests/conftest.py`, `.github/workflows/ci.yml`, `CLAUDE.md`, `docs/coding-standards.md` — retro: the cache in `conftest.py`, the facts written down | Confirmed. | None |
| `docs/design/normal-forms.md`, `docs/design/phase-field.md`, `docs/design/isochrons.md`, `docs/design/cycle-estimation.md` — converted to the turn convention, the amendment lines removed | Looks good. | None |
| `docs/decisions/0020-phase-in-turns.md` — History: the design documents rewritten, not amended | Looks good. | None |
