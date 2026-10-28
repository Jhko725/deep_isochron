---
type: decision
id: ADR-0012
status: accepted; amended 2026-10-15; amended 2026-10-20 (ADR-0015)
updated: 2026-10-29
verified_by: pending (Joon; the split and the ordering were decided in discussion 2026-10-06)
---

# ADR-0012 — `analysis`: a data-driven half and an ODE half

## Context

### Motivation

- Phase E needed numerical answers that the normal forms give in closed form but
  FitzHugh–Nagumo (and real data) do not: where the limit cycle is, its period and
  orientation, when a trajectory has settled onto it, which side of it a point lies on,
  and the nontrivial Floquet multiplier ([`roadmap.md`](../roadmap.md) Phase E). The
  project had none of them.
- Some of these are needed from *sampled trajectories*, for data generation metadata,
  validation splits, initializing a model's rotation sense, and eventually real
  recordings with noise.
- Others are needed from the *vector field*, as ground truth to compare learned
  isochrons against. After the first Winfree runs these became the more urgent: the
  experiments needed to assess trained models against the ground-truth system, and
  Joon asked to "substantially replan this phase"
  ([`2026-10-06-analysis.md`](../changes/2026-10-06-analysis.md), review row on
  `analysis.md`; the 2026-10-15 replan, `:124`).

### Forces

[`architecture.md`](../architecture.md) had reserved `analysis/` for "anything
numerical — locating a limit cycle, monodromy, asymptotic phase of FitzHugh–Nagumo — as
a function over `AbstractODE`", the rule that a normal-form class carries only what is
analytic. The data half must work when no ODE exists. The two kinds of input fail
differently (noise against solver tolerance) and are checked against different oracles.

## Decision

The ODE half is the priority. It assesses trained models (phase, isochrons, PRC,
amplitude, ARC of the learned conjugacy against the ground-truth system) and is what
the experiments need now; the data half automates training decisions (winding → latent
`w` sign, transient → `mixed` split index) and follows as the runs ask. Roadmap Phase E,
items O1–O6, M, D2–D3.

### 1. Two subpackages; shared objects chosen from what the functions share

`deep_isochron.analysis.data` works from trajectories (arrays and a time grid; nothing
about the vector field); `deep_isochron.analysis.ode` works from an `AbstractODE`.

**`analysis.Cycle`** is a planar closed curve as a Fourier series in a phase that
advances uniformly in time, with its period (design document
[`cycle-estimation.md`](../design/cycle-estimation.md) §1). E1's `Cycle` stays as implemented (tested,
used by `estimate_cycle`). The shared objects of the module are chosen *after* several
`analysis.ode` functions exist, from what they actually share (Parked in the roadmap).

The first `analysis.ode` objects are `CrossSection`, `PoincareMap` and `LimitCycle`
([ADR-0014](0014-limit-cycle-by-shooting.md)), chosen from the O1 survey and Joon's
prototype: functions and one map abstraction, with `Cycle` consumed (not produced) by
`limit_cycle`'s output.

*Rejected*: one flat module of functions keyed on what they accept. The two halves have
different inputs, different failure modes (noise vs. solver tolerance) and different
oracles, and the data half must also work when no ODE exists.

### 2. The data-driven cycle estimate is self-contained; settling time depends on a cycle

`estimate_cycle` (E1) works on trajectory **tails** and checks its own assumption
(`converged`), so it needs no settling time. `settling_time` (E2) takes *a* `Cycle`,
from E1 or from E5, and the distance to it along a trajectory. The two can be iterated
(estimate, settle, re-estimate on the settled part), but neither requires the other to
run. Joon's question "which one takes precedence?" is answered: E1.

*Rejected*: a settling-time detector based on the trajectory alone (e.g. a change point
in the speed or in the radius about the centroid), followed by a cycle fit on the part
after it. It has to know what "settled" looks like without a reference, which is exactly
the quantity in question, and it fails on cycles whose shape makes radius-about-centroid
non-monotone (FitzHugh–Nagumo).

### 3. The data half is host-side NumPy; the ODE half's results are `eqx.Module`s

The data half's functions are post-processing: they run once per dataset or per run,
are not differentiated, and involve data-dependent control flow (tail selection,
crossing times, convergence checks). `Cycle` is a frozen dataclass of NumPy arrays;
`estimate_cycle` accepts JAX arrays and converts.

The ODE half's results are `eqx.Module`s: `LimitCycle` and `FixedPoint`
([`limit_cycle.py`](../../src/deep_isochron/analysis/ode/limit_cycle.py),
[`poincare.py`](../../src/deep_isochron/analysis/ode/poincare.py)). The cycle carries
its ODE and solver configuration, drivers take the cycle (`f(cycle, ...)`) and jitted
kernels take arrays ([ADR-0015](0015-floquet-by-seeded-periodic-schur.md) §5).

*Rejected* for the data half: `eqx.Module`s and jitted functions, for uniformity with
the model code. No caller needs gradients or batching over cycles, and every function
here would need `jax.lax` control flow for no gain.

### 4. One commit per analysis function, after a literature survey delivered first

Each of E1–E5 lands in its own commit, preceded by a survey recorded in its own design
document under `docs/design/` (`cycle-estimation.md`, `limit-cycle.md`, `floquet.md`,
`phase-field.md`, `isochrons.md`): what exists, what we credit, what we do differently and why,
with claims marked *cited* or *deduced* (the Project's standing instruction). Each
survey (what exists, what to credit, candidate schemes, oracles) is *delivered to Joon
first*, for reading, further search and possibly his own prototype; the implementation
commit follows, with the survey recorded in that document. The survey is
not optional: E1 took its phase convention and Fourier representation from Revzen &
Guckenheimer and its protophase-to-phase step from Kralemann et al. rather than
reinventing either.

## Consequences

- `src/deep_isochron/analysis/{__init__,cycle}.py`, `analysis/data/`, later
  `analysis/ode/`; [`tests/test_analysis.py`](../../tests/test_analysis.py);
  one design document per item under `docs/design/`.
- `Cycle`'s phase is the asymptotic phase on the cycle up to a constant; comparisons
  against a learned phase or a normal form's `phase` are up to that constant (circular
  offset), never absolute.
- The normal forms' `period()` is signed (the sign is the sense of rotation, design
  document [`normal-forms.md`](../design/normal-forms.md) §2); `Cycle.period` is
  positive and the orientation is `winding`. Consumers that compare the two take the
  absolute value.

## Alternatives rejected

Alternatives rejected for a single decision are listed under that decision.

- *Superseded 2026-10-15*: the data half first, and `Cycle` as the object that every
  function produces and consumes. Both subpackages were to produce and consume a
  `Cycle`, and every downstream consumer would take a `Cycle` without regard to which
  half made it: settling time, inside/outside, Floquet estimates, the `Evaluator`'s
  reference, `OnCycleGaussian` for FitzHugh–Nagumo. Replaced because the ODE half is
  what the experiments need now, and because `Cycle`'s role was decided before the
  functions existed.
- *Superseded 2026-10-20*: host-side NumPy for the ODE half too. The ODE half (E5) was
  to call the systems' `flow` (JAX, diffrax) and return NumPy, its results host-side
  frozen dataclasses. Replaced because a cycle is coupled to its ODE and to the solver
  configuration it was found with, so every later analysis must integrate the same
  system the same way ([ADR-0015](0015-floquet-by-seeded-periodic-schur.md) §5).

## History

- 2026-10-06 (`analysis`): accepted; decisions taken with Joon before implementation,
  roadmap Phase E.
- 2026-10-15 (`analysis`, Joon, after the first Winfree runs): the ODE half became the
  priority; Decision 1 narrowed (`Cycle` no longer the organizing object); Decision 4
  extended (each survey delivered first). Decisions 2 and 3 unchanged.
- 2026-10-18 (`analysis`): the first `analysis.ode` objects recorded in Decision 1
  (ADR-0014).
- 2026-10-20 (`analysis`, ADR-0015): Decision 3's ODE-half results became
  `eqx.Module`s.
- 2026-10-27 (`prose`): restructured to the format of
  [ADR-0019](0019-adrs-describe-the-current-design.md).
- 2026-10-29 (`docs-layout`): `design/analysis.md` split into one document per item.
