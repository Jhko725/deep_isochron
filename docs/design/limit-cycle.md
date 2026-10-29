---
type: design
status: tentative
updated: 2026-10-29
verified_by: pending (Joon); the O1 tests in tests/test_analysis_ode.py pass
sources:
  - "ADR-0012 (analysis layout)"
  - "ADR-0014 (limit cycle by shooting)"
  - "roadmap Phase E"
  - "Guckenheimer & Meloon 2000"
  - "Parker & Chua 1989"
  - "Aprille & Trick 1972"
  - "Morrison et al. 1962"
  - "Urabe 1965"
  - "Hénon 1982"
  - "Tucker 2002"
  - "Shampine & Thompson 2000"
  - "BifurcationKit and diffrax documentation"
---

# `analysis.ode.limit_cycle` — the periodic orbit and its period from the vector field (O1)

A method document in the form of [`document-conventions.md`](../document-conventions.md)
§ Design documents: the problem, what exists (the literature survey, with what we take
from it and what we do differently and why), the method as implemented, what is exact
and what approximated, the oracles (what the tests pin down and against which exact
answers) and the limits (what the function does not do, and which later item does).
Claims are labeled [cited] (read in the source), [cited via X] (read in X, which cites
it) or [deduced] (our inference from the source or from the mathematics). The analysis package as a whole is laid out by
[ADR-0012](../decisions/0012-analysis-layout.md); the other analyses are
[`cycle-estimation.md`](cycle-estimation.md) (E1), this document
(O1), [`floquet.md`](floquet.md) (O2), [`phase-field.md`](phase-field.md) (O3–O4) and
[`isochrons.md`](isochrons.md) (O7).

## 1. The problem

Given a planar `AbstractODE` with a hyperbolic attracting periodic orbit $\gamma$, find a point $x^\ast \in \gamma$, the period $T$, and the orbit sampled uniformly in time — the reference every later item (asymptotic phase, PRC, isostables) is built on, to an accuracy far beyond what a learned model's comparison can use (target $10^{-10}$ on $T$). Claims are [cited] (read in the source), [cited via X] (read in X, which cites it) or [deduced].

## 2. What exists

Every method solves the same boundary value problem: $\dot x = T f(x)$ on $[0, 1]$, $x(0) = x(1)$, plus one phase condition. The methods differ in how the trajectory between mesh points is represented.

| Family | Trajectory between mesh points | Jacobian | References |
|---|---|---|---|
| Brute force: settle, then measure returns to a section | adaptive IVP | none | Parker & Chua 1989, §5.1.1, and the minimum-period test of §5.5 read forwards [cited] |
| Single shooting: Newton on $\Phi^T(x) - x$ with $T$ unknown | one adaptive IVP | variational equation or AD | Aprille & Trick 1972 and Mees 1981 [cited via Parker & Chua 1989, §5.5]; the Poincaré-map variant with $DP_+ = [I - f(y)h^{\top}/\langle h, f(y)\rangle]\thinspace\Phi_T(x)$, their Eq. (5.21) |
| Multiple shooting: $m$ segments with matching | $m$ adaptive IVPs | block-cyclic, blocks $\partial_x\Phi^{T/m}(x_i)$ | Morrison et al. 1962 [cited via Wikipedia: Shooting method]; Keller 1968; Osborne 1969, J. Math. Anal. Appl. 27:417–433 ([cited], author's list); Stoer & Bulirsch §7.3.5 [cited via Wikipedia: Shooting method]; Deuflhard 1980 survey [cited via Guckenheimer & Meloon 2000 and Dednam & Botha 2015]; Ascher et al. 1988 [cited via citing papers] |
| Symmetric multiple shooting: forward from $x_i$, backward from $x_{i+1}$ | half-segments both ways | same structure | Guckenheimer & Meloon 2000, §2, §5.3 [cited] |
| Finite differences | one step of a one-step formula per interval | sparse cyclic | Parker & Chua §5.5 [cited] — multiple shooting with one solver step per segment |
| Orthogonal collocation | piecewise polynomial zeroed at Gauss nodes | sparse, explicit | Doedel 1981; Doedel et al. 1991; Doedel 2007 [cited via the Osinga/Krauskopf lecture notes and BifurcationKit's documentation]; AUTO, MatCont, CoCo |
| Taylor-series shooting with AD | degree-$d$ Taylor polynomial of the trajectory | AD of the Taylor map | Guckenheimer & Meloon 2000 [cited] |
| Least-squares shooting | fixed-step RK on rescaled time, stacked residual | finite differences | Dednam & Botha 2015 [cited] |
| Harmonic balance | truncated Fourier series, Galerkin | dense | Parker & Chua §5.5 [cited]; Urabe 1965 [cited via citing papers] |

What the two modern papers add, and do not [cited]. Guckenheimer & Meloon's accuracy comes from exact derivatives of the discrete flow map ("no finite difference calculations of derivatives implicit in our methods", §6) and coarse meshes with increasing polynomial degree; on their algebraic test orbit (§5.1, Fig. 5.1) AUTO reaches $\approx 5\cdot10^{-15}$, RK4 at its optimal step $\approx 8\cdot10^{-15}$, their method $\approx 6\cdot10^{-16}$: one order of magnitude at the roundoff floor. Their stronger case is a canard cycle (§5.3), which forward integration cannot follow at all and symmetric shooting can. Dednam & Botha's extra residual components $x(1 + \Delta s) - x(\Delta s)$ are redundant at a zero (uniqueness of the flow), so their Levenberg–Marquardt contributes *globalization*, not accuracy [deduced]; their finite-difference Jacobian is their own stated weakness (§2). Neither has grounds to beat AD-Jacobian multiple shooting on an adaptive high-order solver: JAX gives the exact-Jacobian property for free, and $10^{-10}$ is reachable with `Dopri8` at tight tolerances.

**The degeneracy and the phase condition.** At a solution $DG = [\thinspace M - I \mid f(x^\ast)\thinspace]$ with $M$ the monodromy and $Mf = f$, so the null space is the flow direction at fixed $T$; an admissible phase condition has a nonzero component along $f(x^\ast)$, that is, a transversal section [deduced]. Two are in use: the Poincaré condition $\langle n, x_1 - p\rangle = 0$ (local, fixes the starting point) and Doedel's integral condition $\int_0^1 \langle x, \dot x_{\rm ref}\rangle = 0$, the stationarity condition of the time-shift distance to a reference orbit [deduced], the natural one for continuation. Shooting *can* evaluate the integral through a quadrature state; for a one-off solve with no reference orbit the Poincaré condition is the choice (ADR-0014 §2). BifurcationKit's default $s = T\langle x - x_\pi, \phi\rangle$ is this condition with `center` $x_\pi$ and unit `normal` $\phi = F(x_1)/\Vert F(x_1)\Vert$ (its `SectionSS`, [cited] from the source); the factor $T$ is a row scaling with no stated rationale, which we drop.

**The initial guess** is not packaged anywhere (BifurcationKit, BoundaryValueDiffEq, MatCont, CoCo, AUTO take an orbit guess from the user or start from a Hopf point; [cited] from their documentation, or [cited via citing papers]). The recipe is Parker & Chua's brute force on a section: settle, place $\Sigma$ normal to the flow at the settled point ("let the program automatically set $h = f(x^{(0)})$ and $x_\Sigma = x^{(0)}$", §5.5), count positive crossings (footnote 4), measure the return time. Crossings are located by event detection with a root finder (Shampine & Thompson 2000, Comput. Math. Appl. 39:43–54 — *verify*; the classical remedy for the final-step error is Hénon 1982, via Tucker 2002, [cited]). Peak-to-peak timing is inferior: a smooth extremum localizes time to $\sqrt{2\varepsilon/\vert\ddot x\vert}$, a transversal crossing to $\varepsilon/\vert\dot x\vert$ [deduced].

## 3. The method (`analysis/ode/`)

`CrossSection` (`sections.py`): $\Sigma = (p, n)$ with unit normal, `signed_distance(x)`, `normal_to_flow(ode, p)` the flow-normal section.

`PoincareMap` (`poincare.py`) is the first-return map $P$ of the flow on $\Sigma$. `__call__(x) -> (P(x), τ(x))` locates the crossing by a diffrax `Event` after a short departure solve (a zero of the condition at $t_0$ is read as a crossing; in reverse-time solves diffrax 0.7.2 hands the condition its internal increasing time, so the guard is a separate solve, not a time test). `direction` (positive, negative or both, relative to the field being integrated, so that in reverse time a point of a cycle returns to itself after one period) and `reverse_time` are static constructor arguments. `__call__` and `iterate` (a `lax.scan`) are jit- and vmap-compatible, a missing crossing being a runtime error (`throw=True`, as in diffrax) or `τ = ∞`. `fixed_point(x, tol, max_iter)` is a host-side driver over the jitted map that iterates the one-sided map until it returns to within `tol` of an earlier iterate, a period-$k$ orbit of $P$ with $k$ minimal (Parker & Chua §5.4; a cycle cutting $\Sigma$ in $2k$ points), and reports the contraction of successive differences as the brute-force multiplier estimate. The map is differentiable in $x$ through a custom JVP that never differentiates through the event: differentiating the crossing condition $\langle n, \Phi_\tau(x) - p\rangle = 0$ gives $d\tau = -\langle n, J\thinspace dx\rangle / \langle n, v\rangle$ and $dy = [I - v n^{\top}/\langle n, v\rangle]\thinspace J\thinspace dx$ with $J = \partial_x\Phi_\tau(x)$ by forward-mode AD over the known $\tau$ and $v$ the velocity at the crossing, which is Parker & Chua Eq. (5.21) for the point and implicit differentiation for the time [deduced]. So `jax.jacfwd(P)` is $DP$ with eigenvalues $\lbrace 0, e^{\kappa T}\rbrace$.

`multiple_shooting` (`shooting.py`): unknowns $(x_1, \dots, x_m, T)$, residual $(\Phi_{T/m}(x_i) - x_{i+1}\ \text{cyclic},\ \langle n, x_1 - p\rangle)$, segments `vmap`ped, Jacobian by `jax.jacfwd` with `diffrax.ForwardMode`, solved by `optimistix.least_squares(LevenbergMarquardt)`; jit-compatible, $m$ fixed. `SHOOTING_SOLVER_CONFIG`: `Dopri8` (an eighth-order method needs far fewer steps than a fifth-order one at $10^{-10}$–$10^{-12}$; `Tsit5` gives the same answers more slowly), rtol $10^{-10}$, atol $10^{-12}$ — the solver's tolerance is the residual floor (Joon's prototype at rtol $10^{-6}$ had residual $10^{-6}$ at the *initial guess*).

`find_limit_cycle` (`limit_cycle.py`), a host-side composition of jitted stages, each exported: `settle` (flow for `t_settle`) → `initial_guess` (`CrossSection.normal_to_flow` and `PoincareMap.fixed_point`: $\hat x$, $\hat T$ over $k$ returns, $\hat\lambda$) → `polish` (`multiple_shooting` from $m$ points of the orbit through $\hat x$) → the guards `is_equilibrium` and `period_consistent` (the two extraneous solutions of shooting, Parker & Chua §5.5, "Extraneous solutions") → `sample_orbit` (the orbit at `n_points` uniform times in $[0, T)$, forward order even in reverse time) and `orientation` (sign of the signed area) → `LimitCycle` (orbit, period, orientation, section, residual, settle record). Properties *of* the orbit, such as the monodromy, the multipliers and their cross-checks, are O2's functions of a `LimitCycle`, not products of the search.

`nearest_cycle_time(cycle, y)` and `reanchor(cycle, point)` (`limit_cycle.py`, added for O7, which anchors its isochrons at phase 0, and used by O3's route (a) to read a phase off the cycle). The first finds the time $t^\ast$ from `x0s[0]` of the cycle point nearest $y$: from the sample before the nearest of $n$ samples, `optimistix.Newton` solves for the time offset at which the stationarity condition $g = (\gamma - y)\cdot f(\gamma)$ vanishes, differentiating through the flow and seeded at one sample spacing; it starts one sample back because diffrax's derivative of a flow in its end time is identically zero for a zero-length solve (and $f(\gamma)$ elsewhere), so a solve seeded at the nearest sample itself would never move. It returns the solver's convergence flag, stays jit- and vmap-compatible and never raises. `reanchor` moves the phase origin to that point without solving the orbit again: the new mesh is the flow from the new origin at times $iT/m$, the residual is the shooting residual evaluated once there, and the section is the flow-normal one through the new origin, as `find_limit_cycle` builds through its own. (Review of 2026-10-28: the first version hand-rolled the Newton steps and re-ran the shooting solve from the moved mesh; both were replaced.)

`AbstractODE.flow_to(t, u0)` (the state at a signed time, `flow(ts=[0, t]).ys[-1]`, differentiable in both) and `LevelSetCycle` (`systems/level_set.py`, the designed-cycle family of §5) were added to `systems` for this item. `analysis` works through the `flow` contract alone; which integration a system's `flow` performs is the system's choice. For the normal forms it defaults to the closed-form chart, which as an *input* would make the oracle tests circular and, at a basin edge ($r \to a$ for Winfree), loses precision — so the tests construct them with `integration="cartesian"` (a constructor argument) and the closed forms stay the oracles, the solver the input (ADR-0014, round 2).

## 4. Exact and approximated

The section, the phase condition and the shooting residual are exact in form; what is
approximated is each segment's flow, so the residual floor and the error of every
output are the solver's: `SHOOTING_SOLVER_CONFIG` (`Dopri8`, rtol $10^{-10}$, atol
$10^{-12}$) gives a global error of about $10^{-9}$ over a period (§3), which is the
bound §5 pins at $10^{-8}$ on the period, the radius and the sampled orbit. The
brute-force estimate $\hat T$ of the initial guess is a measurement over $k$ returns and
is replaced by the polished $T$; the two agree to $10^{-6}$ on FitzHugh–Nagumo (§5).
`nearest_cycle_time` is a Newton solve on the stationarity condition and reports whether
it converged, so a point it returns is exact to that solve's tolerance.

## 5. Oracles (`tests/test_analysis_ode.py`)

| System | Exact | Pinned |
|---|---|---|
| Hopf, Bautin, Winfree ($a < 0$) | $r = 1$, $T = 2\pi/\vert\omega_1\vert$, $\kappa = \rho'(1)$ | $T$ and radius to $10^{-8}$ (the solver's global error over a period at rtol $10^{-10}$ is $\sim 10^{-9}$); return time and point of $P$, under jit and vmap; `jacfwd(P)`'s eigenvalues $\lbrace 0, e^{\kappa T}\rbrace$ and $d\tau$ against finite differences; contraction of the iterates $\to e^{\kappa T}$ in the linear regime; orientation = sign of $\omega$ |
| Designed level-set cycles, `systems.LevelSetCycle`: $f = s\thinspace(R\nabla g - g\nabla g)$, so $\dot g = -s\thinspace g\thinspace\Vert\nabla g\Vert^2$ and $\lbrace g = 0\rbrace$ is an attracting cycle traversed counterclockwise ([deduced]; Guckenheimer & Meloon's §5.1 field is this with $s = 1/2$) | the curve | **Guckenheimer & Meloon's algebraic orbit** $g = x^2 - y^2 + \tfrac23 y^3 + c$, $c = 0.07$, start $(0, 0.2952161257895192)$ ([cited], p. 970): $\max\vert g\vert$ along the output $< 10^{-8}$ (theirs $6\cdot10^{-16}$ at roundoff; ours is bounded by the solver's global error, $\sim 10^{-9}$), $T \approx 7.7$; **a Cassini oval** (non-convex): a horizontal section above the waist is cut four times — the two-sided map visits the four intersections in order, the one-sided map has a period-2 orbit and `fixed_point` reports $k = 2$ |
| FitzHugh–Nagumo | — | $T = 11.2279$ (Langfield et al. 2014, §III, [cited]) to $2\cdot10^{-3}$; brute-force $\hat T$ agrees with the polished $T$ to $10^{-6}$; clockwise (`slow`) |
| Winfree $a = 0.25$, reverse time | inner cycle $r = a$, $T = 2\pi/\vert\omega(a)\vert$, $\kappa_u = a(1 - a)$ | radius and period to $10^{-8}$; the reversed map contracts by $e^{-\kappa_u T}$ per return and `jacfwd` of it has that eigenvalue |
| Failure modes | — | outside the basin (`ValueError`), an equilibrium start (`ValueError`), no crossing within `t_max`, `max_iter` exhausted, the $T \to 0$ extraneous solution — each a `NotConvergedError` with the stage named |

The suite also covers a near-Hopf Bautin cycle ($\kappa \to 0$) where the fixed-point stage alone cannot reach a tight tolerance and the polish finishes from a loose settle; tolerance monotonicity (tighter solver, smaller period error); the output consumed by `Cycle.from_points` (phase-uniform points) with negligible residual; Hypothesis over Hopf parameters and starting points.

## 6. Limits

- **Attracting orbits only** (repelling ones through `reverse_time`); a saddle cycle in higher dimension would need shooting proper (error growth $e^{\kappa_u T/m}$ per segment is the lever) — not planar, not planned.
- **No canard safety.** Forward segments; a cycle hugging a repelling slow manifold needs symmetric shooting (Guckenheimer & Meloon §5.3). FitzHugh–Nagumo at $\varepsilon = 0.08$ follows the attracting branches; Langfield et al.'s sensitivity near the repelling slow manifold concerns the *isochrons*, not the orbit.
- **No continuation.** A parameter sweep would want the integral phase condition and pseudo-arclength; the quadrature-state formulation makes that a small addition. Collocation only if Langfield-style isochron BVPs are ever wanted.
- **Floquet multipliers** are available (`jax.jacfwd` of the map, the contraction of its iterates, the monodromy by `jacfwd` of `flow_to`) but their interpretation and the Liouville cross-check $\exp\int_0^T \mathrm{div} f\thinspace dt$ belong to O2.
- **Jit scope** (Joon, 2026-10-18): the map, its iteration, the shooting solve and the flow are jitted and vmappable; `fixed_point` and `find_limit_cycle` are host-side drivers (they run once per system and raise informative errors). Revisit if a use case needs a jitted settle. First call per `(ODE class, config, m)` pays XLA compilation (≈ 15 s on CPU); later instances of the same class reuse it, and so do later calls with another `t_settle` or period estimate, which reach the jitted stages as arrays (a Python float is a static argument of `filter_jit`, and every new value compiled three programs; found 2026-10-28 when the `ci` test profile exhausted a process's memory mappings).
