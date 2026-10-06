---
type: decision
id: ADR-0012
status: accepted
updated: 2026-10-06
verified_by: pending (Joon; the split and the ordering were decided in discussion 2026-10-06)
---

# ADR-0012 — `analysis`: a data-driven half and an ODE half sharing one `Cycle`

**Status**: accepted (2026-10-06, branch `analysis`; decisions taken with Joon before
implementation, roadmap Phase E).

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
