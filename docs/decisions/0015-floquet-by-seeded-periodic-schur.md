---
type: decision
id: ADR-0015
status: accepted; amended 2026-10-20 (review round 5 - jit rule, light FixedPoint)
updated: 2026-10-29
verified_by: pending (Joon; the decisions were taken in discussion 2026-10-19/20 after the O2 survey and his notebook prototype)
sources:
  - "Bojanczyk et al. 1992"
  - "Lust 2001"
  - "Dieci et al. 1997"
  - "Guckenheimer & Meloon 2000"
  - "Fairgrieve & Jepson 1991"
  - "Mauroy & Mezić 2018 (Remark 4)"
  - "Wilson & Moehlis 2016"
  - "Kvalheim & Revzen 2021 (Prop. 5)"
  - "BifurcationKit source (`PeriodicSchurBifurcationKit.jl`)"
---

# ADR-0015 — Floquet multipliers by a seeded periodic Schur chain on the shooting mesh; `LimitCycle` as a module

## Context

### Motivation

- The project needs the nontrivial Floquet multiplier $\mu$ of a planar cycle and its
  exponent $\kappa = \log\mu/T$: $\kappa$ is the rate of the amplitude coordinate, and
  with $T$ it is the one number that a learned conjugacy must reproduce.
- The textbook route, eigenvalues of the explicit monodromy $M = M_m\cdots M_1$, is
  limited by rounding to a relative error $\varepsilon\Vert M\Vert/\mu$ (Lust 2001,
  Eq. 10): $10^{-4}$ for FitzHugh–Nagumo ($\mu \approx 1.7\cdot10^{-12}$), nothing at
  all for the Cassini oval ($\mu \approx 10^{-16}$).
- Joon prototyped the periodic Schur decomposition of Bojanczyk et al. 1992 on the
  shooting-segment Jacobians
  ([`prototype_analysis.ipynb`](../../prototype_analysis.ipynb), 2026-10-20). The chain
  as first written (backward, by RQ) did not close, which fixed the direction rule of
  Decision 1.

### Forces

The method has to serve the cycles that
[ADR-0014](0014-limit-cycle-by-shooting.md) finds, attracting ones and repellers found
in reverse time, and has to report Floquet vectors and an accuracy diagnostic as well
as the multiplier. The cycles in the project are planar, so the transverse block is
$1\times1$; a method for `dim > 2` is not needed yet.

## Decision

[`floquet.md`](../design/floquet.md) has the survey and the method; this ADR
records the choices.

### 1. The method: the periodic Schur chain, seeded with the flow direction, run forward

The multipliers are the diagonal products of the triangular factors
$R_i = Q_{i+1}^{\top}M_iQ_i$ of the segment Jacobians, accumulated as
$\sum_i \log R_{i,jj}$. Bojanczyk et al.'s general algorithm finds the Schur basis by a
periodic QR *iteration*. Here one Schur vector is known analytically: the flow
direction $f(x_1)$, the eigenvector of the trivial multiplier and the *dominant* one
for an attracting cycle. So $Q_1$ is seeded with it and the decomposition is
completed by the forward QR chain $M_i \thinspace Q_i = Q_{i+1} \thinspace R_i$ with
positive diagonals, the discrete QR method for Lyapunov exponents over one period
(Dieci et al. 1997, §3, Claim 2), with the seed in the spirit of Guckenheimer &
Meloon's transformation to cross-section coordinates (2000, §2 algorithm). In the
plane the transverse block is $1\times1$, so no iteration remains.

*Why forward*: the chain propagates the first Schur vector. Propagated forward it is
the dominant direction and self-correcting; propagated backward through the transposes
(the RQ chain of the prototype) it is the subdominant one and the chain does not close.
That is a *deduced* rule, confirmed numerically, with no literature statement found.

The planar cross-check is `method="liouville"`, and the explicit monodromy is kept as
a diagnostic, `method="monodromy"` (Alternatives rejected). For `dim > 2`,
`periodic_schur` raises `NotImplementedError` naming a general periodic QR library.

### 2. Direction: the chain runs in the time direction in which the cycle attracts

For a cycle found with `reverse_time` (a repeller), $f$ is the *subdominant* direction
of the forward monodromy and the forward chain would be unstable; the segment
Jacobians are therefore those of the reversed field ($M_i^{-1}$ of the forward
segments), the chain gives $\lbrace 1, 1/\mu\rbrace$, and the result is inverted:
reported multipliers and exponents are always forward-time
([ADR-0014](0014-limit-cycle-by-shooting.md) §4).

### 3. Signs and diagnostics

The QR factors are normalized to a positive diagonal (rows of $R$ and columns of $Q$
flipped together, which keeps the factorization exact; the prototype flipped columns
of $R$). The chain closes on $Q_{m+1} = Q_1D$ with $D = \mathrm{diag}(\pm1)$: $D$
carries the multipliers' signs (a negative multiplier appears in the closure, not in
the factors; in the plane $D = I$ always, $\det M > 0$), and the defect
$\Vert Q_{m+1} - Q_1D\Vert$ with $\vert\mu_0 - 1\vert$ is the trivial-multiplier
diagnostic (Guckenheimer & Meloon 2000, §5.4). The trace of the return-map Jacobian is
*not* an estimator: $\mathrm{tr}\thinspace DP = (1 + \mu) - 1$ leaks the trivial
multiplier's error ($\sim10^{-9}$) onto $\mu$ (found on the prototype's FHN run:
$3.5\cdot10^{-9}$ for $\mu = 1.7\cdot10^{-12}$); `return_map_multiplier` takes the
eigenvalue and documents its absolute floor.

### 4. Interface: three routes behind one `floquet(cycle, *, method=Literal)`

Functions first (`segment_jacobians`, `periodic_schur`, `monodromy`,
`multipliers_from_monodromy`, `divergence_integral`, `return_map_multiplier`), one
dispatching driver returning a `Floquet` module (multipliers, exponents,
trivial error, closure, Floquet vectors at every mesh point, the last two `None` for
the other routes). Each Floquet route returns a `Floquet` (`floquet_periodic_schur`,
`floquet_monodromy`, `floquet_liouville`); `floquet(cycle, method=)` dispatches. The
`Literal` turns into a strategy object without changing call sites.

### 5. `LimitCycle` and `FixedPoint` are `eqx.Module`s; the cycle carries its ODE and configuration; analysis functions take the cycle

A cycle is tightly coupled to its ODE and to the solver configuration it was found
with, and every later analysis must integrate the same system the same way, so
`LimitCycle` holds `ode`, `x0s` (the converged shooting mesh, which the periodic Schur
chain needs), `period`, `residual`, `settle`, `section`, `config` (static),
`reverse_time` (static). It samples itself (`sample(n_points)`, re-integrating; the
sampled orbit is a derived quantity, no longer stored) and reports its `orientation()`;
the signed-area test lives in `analysis.data.cycle`, since it acts on a discretized
curve. This replaces the host-side frozen dataclasses of
[ADR-0012](0012-analysis-layout.md) §3 and ADR-0014 §3.

The cycle is a *fixed-shape* pytree: `FixedPoint` holds only `x`, `tau`, `k`,
`contraction`, `converged`, and the search record (`taus`, `iterates`, per-cycle
length) is returned by `fixed_point` separately as a `FixedPointIteration` and not
stored on the cycle.

The signature rule for `analysis.ode` is that functions of a cycle take the cycle,
`f(cycle, ...)`, and are jit- and vmap-compatible as they stand; a batch of cycles of
one mesh size (the ODE's parameters being leaves) is one `vmap`.

### 6. Jit scope

A function that is naturally jit- and vmap-compatible is left unjitted, so the caller
chooses the level (`eqx.filter_jit(floquet)`, `eqx.filter_vmap(floquet)` over a batch of
cycles, or neither); a function that is not, a host-side driver with Python control
flow and informative errors, jits the subcomponents that it calls. So `settle`,
`polish`, `multiple_shooting`, `LimitCycle.sample`, `PoincareMap.__call__`/`iterate`,
`segment_jacobians`, `periodic_schur`, `monodromy`, `divergence_integral` and the
`floquet_*` routes carry no decorator; `find_limit_cycle` runs `eqx.filter_jit(settle)`
and `eqx.filter_jit(polish)`, `fixed_point` the jitted return solve. This is Joon's
rule from review round 5.

## Consequences

- [`analysis/ode/floquet.py`](../../src/deep_isochron/analysis/ode/floquet.py);
  `LimitCycle`/`FixedPoint` as modules; tests on the normal forms (all routes,
  $\kappa = \rho'(1)$), Winfree's repeller, the Cassini oval by solver-free line
  integrals ($\kappa T = -\oint\Vert\nabla g\Vert d\ell$,
  $T = \oint d\ell/(s\Vert\nabla g\Vert)$, also an oracle for O1's period) and
  FitzHugh–Nagumo (the floors made visible).
- Oracles are extended by a *conditioning* regime: Cassini's $\mu \approx 10^{-16}$ is
  beyond the monodromy's reach and inside the chain's.
- Open: `dim \geq 3` (deflate $f$, periodic QR on the transverse block through
  `slycot`, or BifurcationKit-style section-reduced products); the forward-direction
  stability argument without a citation; strategy objects for `method`.

## Alternatives rejected

- *Liouville as the method*: $\kappa = T^{-1}\int_0^T\mathrm{div}f$ is exact and cheap
  in the plane, but it is planar only, gives no Floquet vectors and no accuracy
  diagnostic; it is kept as the planar cross-check (`method="liouville"`).
- *The explicit monodromy*: the floor in Motivation; kept as a diagnostic
  (`method="monodromy"`).
- *A general periodic QR library* (SLICOT `MB03WD` via `slycot`, as BifurcationKit's
  `FloquetPQZ` uses `PeriodicSchurDecompositions.jl`): only needed for `dim > 2`, where
  `periodic_schur` raises `NotImplementedError` naming it.
- *Strategy objects per method*
  ([ADR-0007](0007-systems-hierarchy-and-flow-strategies.md) style), rejected for now:
  to be adopted when a method grows its own options, such as a quadrature choice or a
  `slycot` callback.
- *Storing a diffrax dense interpolant* (`SaveAt(dense=True)`, 8th order for `Dopri8`)
  for resampling. Its size is the step count, a static-shape and consistency burden,
  for a resampling that costs a fraction of a second; revisit if resampling becomes
  hot.
- *Superseded 2026-10-20*: jitted kernels that take arrays and the static
  configuration, never the cycle, because the cycle's `settle` record had a per-cycle
  shape and would recompile every kernel per cycle. The first version of this branch
  decorated the kernels and kept kernel/driver pairs
  (`_divergence_integral`/`divergence_integral`). Replaced in review round 5 by the
  jit rule of Decision 6; the pairs are gone with the rule.

## History

- 2026-10-20 (`analysis`, roadmap O2): accepted.
- 2026-10-20 (`analysis`, review round 5): the jit rule of Decision 6; `FixedPoint`
  without the search record, which `fixed_point` returns as a `FixedPointIteration`;
  each Floquet route returns a `Floquet`; the signed-area test of `orientation()` moved
  to `analysis.data.cycle`.
- 2026-10-27 (`prose`): restructured to the format of
  [ADR-0019](0019-adrs-describe-the-current-design.md); the Sources section folded into
  the frontmatter `sources:` line.
- 2026-10-29 (`docs-layout`): the design document is now the item's own, split out of
  `design/analysis.md`.
