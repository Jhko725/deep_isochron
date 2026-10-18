---
type: decision
id: ADR-0014
status: accepted; amended 2026-10-18 (review rounds 1–2)
updated: 2026-10-19
verified_by: pending (Joon; the decisions were taken in discussion 2026-10-17/18 after the O1 survey and his notebook prototype)
---

# ADR-0014 — The limit cycle from the vector field: first-return map on a flow-normal section, multiple shooting with AD Jacobians

**Status**: accepted (2026-10-18, branch `analysis`, roadmap O1). Design document
`analysis.md` §2 has the survey and the method; this records the choices.

## Context

`analysis.ode` must give the reference objects a trained model is assessed against,
starting with the periodic orbit $\gamma$ and its period $T$ of any planar `AbstractODE`
(roadmap Phase E, replanned 2026-10-15). The literature offers brute force, single and
multiple shooting (Newton on the time-$T$ flow), orthogonal collocation (AUTO), Taylor-series
shooting with automatic differentiation (Guckenheimer & Meloon 2000) and least-squares
shooting (Dednam & Botha 2015); the survey is in the design document. Joon prototyped the
pipeline in `prototype_analysis.ipynb` (settle, section, return times by diffrax events,
multiple shooting by Levenberg–Marquardt) before this refactoring.

## Decisions

### 1. Shooting on the time-$T$ flow, not collocation

The orbit is found by multiple shooting: unknowns $(x_1, \dots, x_m, T)$, residual
$\Phi_{T/m}(x_i) - x_{i+1}$ (cyclic) plus one phase condition, solved by Levenberg–Marquardt
(optimistix) with the Jacobian by forward-mode AD through the diffrax solve, $m$ fixed. *Why*: the segments are independent IVP solves (`vmap`), the Jacobian is
the exact derivative of the numerical flow (the property Guckenheimer & Meloon identify
as the source of accuracy), the adaptive solver's error control comes for free, and our
orbits are attracting. *Rejected*: collocation (a BVP solver to write, no error control
without mesh adaptation; its strengths — continuation, unstable orbits, Langfield-style
isochron BVPs — are not O1's problem); a Taylor integrator (one order of magnitude at the
roundoff floor, irrelevant at the 1e-10 we need); harmonic balance (ill-suited to
relaxation oscillations).

### 2. The phase condition is a Poincaré section normal to the flow

$\Sigma$ through the settled point $p$ with unit normal $f(p)/\Vert f(p)\Vert$ pins the first
shooting point: $\langle n, x_1 - p\rangle = 0$. It is transversal by construction and is
what the shooting degeneracy needs (the Jacobian's null direction is the flow direction;
any admissible condition must have a component along it). Parker & Chua's "Choosing Σ",
BifurcationKit's `SectionSS` default. *Rejected*: Mees's iteration-dependent constraint
$\langle f(x^{(i)}), \Delta x\rangle = 0$ (not a Newton on a fixed residual); Doedel's integral
phase condition (the right one for *continuation*, needs a reference orbit we do not have
in a one-off solve; evaluable by shooting through a quadrature state if a continuation is
ever wanted); BifurcationKit's factor $T$ on the phase row (a row scaling with no stated
rationale; we keep the row in state units).

### 3. `PoincareMap` is the one abstraction; results are lightweight containers

A `PoincareMap(ode, section, config, t_max, direction, reverse_time)` is the discrete map
the flow induces on the section. It unifies three operations the plan had as separate
functions: the return time and point ($P(x)$), the settle stage (iterate $P$ to its fixed
point, with the minimum-period test as period-$k$ detection), and the brute-force
multiplier (the contraction of the iterates). It is a differentiable function of the
point through a custom JVP (Parker & Chua Eq. 5.21 for the point, implicit
differentiation of the crossing for the time; no differentiation through the event), so
`jax.jacfwd(P)` is $DP$ — the O2 hook.
`direction` (positive / negative / both) and `reverse_time` are static constructor
arguments — the map's identity, not per-call options; `fixed_point` is defined for the
one-sided maps only. `LimitCycle` and `FixedPoint` (host-side) are frozen dataclasses of NumPy arrays
(ADR-0012 §3); `ShootingResult` (the output of a jittable function) is a `NamedTuple` of
arrays — kept lightweight until the package has enough functions for the shared
abstractions to be chosen from what they actually share (ADR-0012 amendment).

### 4. Reverse time finds repelling cycles

With `reverse_time=True` the map and the shooting integrate the time-reversed field; a
repelling cycle is attracting there (Parker & Chua §5.1.1, footnote 1). The reported
monodromy is always the forward-time one (the reversed monodromy inverted). First use:
Winfree's inner cycle $r = a$ ($a > 0$), the basin boundary, with its exact exponent
$\kappa_u = a(1 - a)$ as oracle.

### 5. Guards, and what is deferred

`find_limit_cycle` reports the shooting residual and rejects the two extraneous solutions
of shooting (an equilibrium; $T \to 0$; Parker & Chua §5.5). Properties *of* the orbit —
the monodromy with its trivial-multiplier error (Guckenheimer & Meloon §5.4), the
nontrivial multiplier, Liouville's cross-check $\exp\int\mathrm{div}f\thinspace dt$ — are
functions of a `LimitCycle` and belong to O2 (review round 1, Joon 2026-10-18); symmetric forward/backward shooting for canard-type segments
(Guckenheimer & Meloon §5.3) and per-point sections with variable segment times are
documented, not built.

### 6. Implementation findings

- diffrax 0.7.2 reads a zero of the event condition at $t_0$ as a crossing, and in a
  reverse-time solve passes the condition its *internal* increasing time rather than the
  signed one. The departure guard is therefore a separate short solve, not a time test
  inside the condition.
- The shooting solver's tolerances are the residual floor: Joon's prototype at
  rtol 1e-6 had a residual of 1e-6 *at the initial guess*, so the polish could not act.
  `SHOOTING_SOLVER_CONFIG` (Dopri8, rtol 1e-10, atol 1e-12, `ForwardMode`) is the default.

## Consequences

- `analysis/errors.py`, `analysis/ode/{sections,poincare,shooting,limit_cycle}.py`;
  `SolverConfig.with_adjoint`; `tests/helpers.LevelSetCycle` (a cycle designed as a level
  set, $f = s(R\nabla g - g\nabla g)$, Guckenheimer & Meloon's §5.1 orbit and a Cassini oval);
  `tests/test_analysis_ode.py`.
- Oracles: normal forms (exact $T$, radius, $e^{\kappa T}$), Guckenheimer & Meloon's algebraic
  orbit ($\max\vert g\vert$), FitzHugh–Nagumo (Langfield et al. 2014), Winfree's repelling
  cycle in reverse time.
- Open: the sign convention between the normal forms' signed `period()`, `Cycle.winding`
  and `LimitCycle.orientation` (all three exist; a test keeps them consistent).

## Amendment 2026-10-18 — review round 1

- **Jit scope** (decision 7): the map (`__call__`, `iterate`), the shooting solve and the
  flow are jit- and vmap-compatible, with failures as runtime errors (`throw`, as in
  diffrax); `fixed_point` and `find_limit_cycle` are host-side drivers over them. A jitted
  settle would need a bounded history buffer and a masked minimal-period search for a
  stage that runs once per system; revisit if a use case asks for it.
- **Custom JVP** instead of a `jacobian` method; **no grid coarsening** (the $m$-point
  solution is as valid as a single-shooting one, and the monodromy can be taken from any
  point); the stages of `find_limit_cycle` are functions, not a class; the monodromy and
  the multipliers leave `LimitCycle` for O2.
- **Names**: `CrossSection` (the dynamical-systems term) with `signed_distance` and
  `normal_to_flow`; `find_limit_cycle`; `AbstractODE.flow_to` replaces the three local
  flow helpers; `LevelSetCycle` moves to `systems/level_set.py`.
- **`analysis` works through the `flow` contract alone** (`flow`, `flow_to`); which
  integration a system's `flow` performs is the *system's* choice. The normal forms'
  `flow` defaults to their *closed-form* chart, which as the input of an analysis (i)
  makes the tests against them circular and (ii) loses precision at a basin edge — found
  when the reverse-time Winfree run returned garbage at $r = a$ and a $10^{-7}$ period just
  inside. So the tests construct them with `integration="cartesian"` (a constructor
  argument, review round 2) and the closed forms stay the oracles. *Rejected* (round 1's
  first fix, reverted in round 2): an `AbstractODE.integrate` that always solves `rhs`
  numerically — a near-duplicate of `flow` on the base contract for a need the normal
  forms already meet with their integration switch.
