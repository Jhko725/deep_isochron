---
type: decision
id: ADR-0020
status: accepted (review pending)
updated: 2026-10-30
verified_by: pending (Joon; the decisions were taken in discussion 2026-10-29/30 with Andrus Giraldo)
---

# ADR-0020 — The phase is a fraction of the period in $[0, 1)$, increasing with time

## Context

### Motivation

- A phase written as an angle in $(-\pi, \pi]$ invites reading it as a geometric angle
  about an interior point. That reading is right for a star-shaped cycle (the angle
  between a reference axis and the ray from an interior point to the cycle point) and
  wrong in general, and it led a listener at Joon's presentation of 2026-10-28 to ask
  whether the phase is defined for a cycle that is not star-shaped. It is: the
  asymptotic phase is a rescaled time along the orbit (Winfree 1974; Guckenheimer 1975,
  Theorem A), defined for every hyperbolic attracting cycle. A phase that is visibly a
  fraction of the period carries that meaning in its range.
- The coworker's AUTO-07p reproduction already works in fractions of the period: its
  `phaseVec` and the `UZR` targets of its stage 3 are fractions multiplied by the
  period (`isochron2pbvp/fncMultiSeg/mainIsochronMultiSegFNC.py`), as is this pipeline's
  stage 3, which converts $\theta/2\pi$ back to a fraction before AUTO sees it
  (`analysis/ode/auto/stages.py`, `rotate_phase`). The radian form was a conversion in
  both directions around a quantity that was never an angle.
- The orientation of the reference phase was tied to the polar angle of the normal
  form's chart, so for a clockwise form (Winfree with $w < 0$, Bautin with
  $\omega_1 < 0$) the reference phase *decreased* in time, and the evaluation had to
  keep "the better of the two orientations" (`training/evaluation.py`,
  `phase_amplitude_agreement`). A phase that is a fraction of elapsed time has one
  orientation.

### Forces

The phase is the quantity every layer shares: the normal forms' closed forms, the model
contract, the data-side `Cycle`, the ODE-side `asymptotic_phase`, the AUTO pipeline's
base points, the evaluation metrics and the plots. A convention has to be one thing in
all of them, with the conversion to and from the geometry that is an angle (the normal
forms' polar chart, the invertible polar layer, the latent rotation) written at named
seams and nowhere else. The foliation files on disk and the hashes that name them carry
the phases, so a change of convention renames every saved foliation. Models learn a
phase up to its origin and, for the autoencoder, up to its orientation, so the metrics
and plots still align a model's phase to the reference; the reference itself has no
such freedom.

## Decision

1. **The phase $\Theta$ is a fraction of the period**, $\Theta \in [0, 1)$, with
   $\Theta = 0$ at the cycle's phase origin `x0s[0]` and
   $\Theta(\Phi_t x) = \Theta(x) + t/T \pmod 1$: it increases with time on every
   system, clockwise or counterclockwise. The unit is the turn; a docstring says
   "phase" and means this.
2. **Angles stay where the quantity is an angle**: the normal forms' polar coordinate
   $\theta$ and its shift $h(r)$, the angular rates $\omega$ and $\omega_1$, the
   invertible polar layer's angle, the latent rotation of the autoencoder, the
   initial-condition sampler. The conversion $2\pi\Theta$ or $\theta/2\pi$ is written at
   the seams that produce or consume a phase — the normal forms' `phase`, `limit_cycle`
   and `isochron`, the models' `phase`, `cycle_point` and `isochron`, `Cycle`'s Fourier
   series, `phase_offset`'s sample grid — and the normal forms negate the polar angle
   when $\omega_1 < 0$, so that decision 1 holds for them too.
3. **Phase differences are taken by one helper**, `phase_difference(a, b)` in
   `utils/numerics.py`, the signed shortest arc in $(-\tfrac12, \tfrac12]$ (positive
   when `a` is ahead of `b`), written against the array API as `wrap_phase` is, so it
   serves NumPy and JAX arrays under `jit`, `vmap` and `grad`. `wrap_phase` wraps to
   $[0, 1)$; the radian wrap survives as `wrap_angle` for the geometry of decision 2.
   *Amended 2026-10-30 (`phase-turns`, review of units 1–6): `wrap_angle` was deleted;
   once the normal forms wrapped on the turn, no site in `src/` wrapped an angle.*
4. **The phase sensitivity is in turns**: $Z(\Theta) = \nabla\Theta$ on the cycle,
   normalized by $Z \cdot F = 1/T$ (Brown et al. 2004, Eq. 2.4, and Nakao 2016, Eqs.
   12–13, carry $\omega = 2\pi/T$ and $1$ respectively; both become $1/T$ under decision
   1).
5. **Saved foliations say their unit**: the netCDF `phase` variable carries
   `units = "turns"`, and `IsochronFoliation.load` refuses a file without the
   attribute, so a file written before this decision is never read as turns. The
   isochron config's hash covers the wrapped phases, so every shipped hash is repinned
   and every saved foliation is recomputed.
6. **Metric names stay**: `phase/circ_std` and the aligned residual are reported in
   turns under their current names, since a renamed metric would be a second name for
   one quantity; wandb history across this decision is not comparable.

## Consequences

- Every public phase signature changes meaning in one branch, `phase-turns`, in
  dependency order (`numerics` and the normal forms and the model contract; the
  analyses; the AUTO pipeline and the experiment layer; plots, notebook, documents), so
  that master never carries two conventions. The change document of that branch is the
  ledger.
- Irreversible in effect: the foliation files on Joon's cluster are recomputed under
  the new hashes (about twenty minutes per system on a 24-core node, ADR-0017), and
  the `SHIPPED_HASHES` of `tests/test_experiment_isochrons.py` are repinned.
  Checkpoints and datasets carry no phase and are untouched.
- The reference phase has one orientation, so `phase_amplitude_agreement`'s choice of
  orientation and `phase_sense` describe only the model, which may still learn either
  way round.
- The design documents that fix the convention — `normal-forms.md` §1, §5 and §7,
  `phase-field.md` §1, `isochrons.md` §1 and §3.4, `cycle-estimation.md` §1.1 — are
  amended in the same branch, with ADR-0007, ADR-0013 and ADR-0017 given History lines.

## Alternatives rejected

- **$(-\tfrac12, \tfrac12]$**, the symmetric range. It keeps today's symmetry about the
  origin, but puts the wrap's seam at half a turn from the base point, where nothing
  useful happens, and makes $\Theta$ negative for the first half of every period, which
  the "rescaled time" reading does not.
- **$[0, T)$, the time itself.** The most literal rescaled time, but the range would
  differ between systems, so a tolerance, a colormap and a config value would all have
  to be per system.
- **Radians with a documented orientation.** Keeps every signature, but keeps the
  angle reading that motivates this decision, and the orientation rule would be a
  sentence rather than a range.
- **Two helpers for the difference**, one per array library. `wrap_phase` already
  showed that one array-API function serves both.

## History

- 2026-10-30 (`phase-turns`): accepted; written before the change it decides, as agreed
  on 2026-10-29.
- 2026-10-30 (`phase-turns`, review of units 1–6): `wrap_angle` deleted for lack of a
  caller (decision 3 amended).
