---
type: decision
id: ADR-0012
status: accepted; amended 2026-10-15; amended 2026-10-20 (ADR-0015)
updated: 2026-10-20
verified_by: pending (Joon; the split and the ordering were decided in discussion 2026-10-06)
---

# ADR-0012 — `analysis`: a data-driven half and an ODE half sharing one `Cycle`

**Status**: accepted (2026-10-06, branch `analysis`; decisions taken with Joon before
implementation, roadmap Phase E). **Amended 2026-10-15** — see the end: the ODE half is
the priority, `Cycle` is no longer the organizing object, and the survey is delivered
before anything is implemented.

## Context

Phase E needs numerical answers the normal forms give in closed form but FitzHugh–Nagumo
(and real data) do not: where the limit cycle is, its period and orientation, when a
trajectory has settled onto it, which side of it a point lies on, and the nontrivial
Floquet multiplier. Some of these are needed from *sampled trajectories* (data generation
metadata, validation splits, initializing a model's rotation sense, and eventually real
recordings with noise); some from the *vector field* (ground truth to compare learned
isochrons against). `architecture.md` had reserved `analysis/` for "anything numerical —
locating a limit cycle, monodromy, asymptotic phase of FitzHugh–Nagumo — as a function
over `AbstractODE`" (the rule that a normal-form class carries only what is analytic).

## Decisions

### 1. Two subpackages, one shared object

`deep_isochron.analysis.data` works from trajectories (arrays and a time grid; nothing
about the vector field); `deep_isochron.analysis.ode` works from an `AbstractODE`. Both
produce and consume **`analysis.Cycle`**: a planar closed curve as a Fourier series in a
phase that advances uniformly in time, with its period (design document `analysis.md`
§0). Every downstream consumer — settling time, inside/outside, Floquet estimates, the
`Evaluator`'s reference, `OnCycleGaussian` for FitzHugh–Nagumo — takes a `Cycle` and does
not care which half made it.

*Rejected*: one flat module of functions keyed on what they accept. The two halves have
different inputs, different failure modes (noise vs. solver tolerance) and different
oracles, and the data half must also work when no ODE exists.

### 2. The data-driven cycle estimate is self-contained; settling time depends on a cycle

`estimate_cycle` (E1) works on trajectory **tails** and checks its own assumption
(`converged`), so it needs no settling time. `settling_time` (E2) takes *a* `Cycle` — from
E1 or from E5 — and the distance to it along a trajectory. The two can be iterated
(estimate → settle → re-estimate on the settled part) but neither requires the other to
run. Joon's question "which one takes precedence?" is answered: E1.

*Rejected*: a settling-time detector based on the trajectory alone (e.g. a change point
in the speed or in the radius about the centroid), followed by a cycle fit on the part
after it. It has to know what "settled" looks like without a reference, which is exactly
the quantity in question, and it fails on cycles whose shape makes radius-about-centroid
non-monotone (FitzHugh–Nagumo).

### 3. Host-side NumPy, not JAX

The analysis functions are post-processing: they run once per dataset or per run, are
not differentiated, and involve data-dependent control flow (tail selection, crossing
times, convergence checks). `Cycle` is a frozen dataclass of NumPy arrays; `estimate_cycle`
accepts JAX arrays and converts. The ODE half (E5) will call the systems' `flow` (JAX,
diffrax) and return NumPy.

*Rejected*: `eqx.Module`s and jitted functions, for uniformity with the model code. No
caller needs gradients or batching over cycles, and every function here would need
`jax.lax` control flow for no gain.

### 4. One commit per analysis function, after a literature survey

Each of E1–E5 lands in its own commit, preceded by a survey recorded in
`docs/design/analysis.md` — what exists, what we credit, what we do differently and why,
with claims marked *cited* or *deduced* (the Project's standing instruction). The survey
is not optional: E1 took its phase convention and Fourier representation from Revzen &
Guckenheimer and its protophase-to-phase step from Kralemann et al. rather than
reinventing either.

## Consequences

- `src/deep_isochron/analysis/{__init__,cycle}.py`, `analysis/data/`, later
  `analysis/ode/`; `tests/test_analysis.py`; `docs/design/analysis.md` grows one section
  per item.
- `Cycle`'s phase is the asymptotic phase on the cycle up to a constant; comparisons
  against a learned phase or a normal form's `phase` are up to that constant (circular
  offset), never absolute.
- The normal forms' `period()` is signed (the sign is the sense of rotation, design
  document `normal-forms.md` §2); `Cycle.period` is positive and the orientation is
  `winding`. Consumers that compare the two take the absolute value.

## Amendment 2026-10-15 (Joon, after the first Winfree runs)

- **Priority inverted.** The ODE half assesses trained models (phase, isochrons, PRC,
  amplitude, ARC of the learned conjugacy against the ground-truth system) and is what the
  experiments need now; the data half automates training decisions (winding → latent `w`
  sign, transient → `mixed` split index) and follows as the runs ask. Roadmap Phase E,
  items O1–O6, M, D2–D3.
- **Decision 1 narrowed.** The two subpackages stand; `Cycle` as the object every
  function produces and consumes does not — it was decided before the functions existed.
  E1's `Cycle` stays as implemented (tested, used by `estimate_cycle`), and the shared
  objects of the module are chosen *after* several `analysis.ode` functions exist, from
  what they actually share (Parked in the roadmap).
- **Decision 4 extended.** Each survey is *delivered to Joon first* — what exists, what
  to credit, candidate schemes, oracles — for reading, further search and possibly his own
  prototype; the implementation commit follows, with the survey recorded in
  `docs/design/analysis.md`. Decisions 2 and 3 are unchanged.
- **2026-10-18**: the first `analysis.ode` objects are `CrossSection`, `PoincareMap` and
  `LimitCycle` (ADR-0014), chosen from the O1 survey and Joon's prototype — functions
  and one map abstraction, with `Cycle` consumed (not produced) by `limit_cycle`'s output.

## Amendment 2026-10-20 — ADR-0015

§3's host-side frozen dataclasses for the ODE half's results are superseded: `LimitCycle` and `FixedPoint` are `eqx.Module`s, the cycle carries its ODE and solver configuration, drivers take the cycle (`f(cycle, ...)`) and jitted kernels take arrays (ADR-0015 §5).
