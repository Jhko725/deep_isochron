---
type: design
status: tentative
updated: 2026-10-27
verified_by: pending (Joon); the E1 oracle tests in tests/test_analysis.py and the O1/O2 tests in tests/test_analysis_ode.py pass
sources: [ADR-0012 (analysis layout), ADR-0014 (limit cycle by shooting), ADR-0015 (Floquet by seeded periodic Schur), roadmap Phase E, docs/design/normal-forms.md (phase conventions); §3 survey: Guckenheimer & Meloon 2000, Lust 2001, Fairgrieve & Jepson 1991, Mauroy & Mezić 2018, Wilson & Moehlis 2016, Kvalheim & Revzen 2021, Giacomini & Grau 2005, Bojanczyk et al. 1992, Dieci et al. 1997, Geist et al. 1990, BifurcationKit and diffrax documentation; §4–5 survey: Guckenheimer 1975, Brown et al. 2004, Langfield et al. 2014, Osinga & Moehlis 2010, Mauroy & Mezić 2012, Monga et al. 2019, Pérez-Cervera et al. 2020, Yawata et al. 2024, Kvalheim & Revzen 2021]
---

# `deep_isochron.analysis` — the limit cycle and its neighborhood, from data and from the vector field

One section per analysis function, in the order they land (roadmap Phase E). Each
section has the same parts: **what exists** (the literature survey, with what we take
from it and what we do differently and why), **the method** as implemented, **oracles**
(what the tests pin down and against which exact answers) and **limits** (what the
function does not do, and which later item does). Claims are marked *cited* (read in the
source) or *deduced* (our inference from the source or from the mathematics).

## 0. The shared object: `Cycle`

A planar limit cycle is represented by a closed curve as a truncated Fourier series in a
phase $\varphi \in [0, 2\pi)$,

$$
\gamma(\varphi) = c + \sum_{k=1}^{K} a_k \cos k\varphi + b_k \sin k\varphi ,
$$

with its period $T$, and the **convention that $\varphi$ advances uniformly in time**
along the cycle, $\varphi(t) = \varphi_0 + 2\pi t / T$. On the cycle this makes $\varphi$
the asymptotic phase up to an additive constant (the asymptotic phase is defined by
uniform advance along the orbit; design document `normal-forms.md` §2, §4), so a point
of the cycle and its phase are interchangeable and a learned phase can be compared
against `Cycle.nearest_phase`. The same representation is Revzen & Guckenheimer's
"order 11 Fourier series model of the orbit" (2012; *cited*, see §1.1). `Cycle` is a
frozen dataclass holding NumPy arrays, because the analysis functions are host-side,
non-differentiable post-processing and JAX adds nothing here (ADR-0012).

`Cycle` provides the curve ($\gamma$, $\gamma'$), sampling (`phases`, `points`), point
queries (`nearest_phase`, `nearest_point`, `distance`: dense argmin over `resolution`
samples followed by one Newton step on the stationarity condition
$(\gamma(\varphi) - x) \cdot \gamma'(\varphi) = 0$) and two constructors: `from_points(phases, points, period)` (least
squares in the $1 + 2K$ coefficients — any order, any spacing, repeated phases, which is
what pooled trajectories produce) and `circle` (the normal forms' cycle in their own
coordinates). E3 adds `winding` and `contains`.

## 1. E1 — `analysis.data.estimate_cycle`: the cycle from trajectory tails

**The problem.** From one or several trajectories of a planar oscillator sampled on a
common time grid, converging to the cycle from initial conditions anywhere and possibly
with measurement noise, find the limit cycle as a `Cycle` (curve, period, phase
parameterization), **without a prior notion of when the transient ends** (E2 defines the
settling time *from* a cycle, so E1 cannot assume one).

### 1.1 What exists

- **Revzen & Guckenheimer 2008**, *Estimating the phase of synchronized oscillators*,
  Phys. Rev. E **78**, 051907 ("Phaser"). From the survey (the abstract page could not
  be re-fetched when this document was written, so the method description is from the
  earlier reading, *cited* at that level): a protophase is obtained from each scalar
  measurement by the Hilbert transform; a correction expressed as a Fourier series makes
  the phase advance at a constant rate; several measurements are combined; the method is
  designed for data that already lie near the limit cycle (rhythmic locomotion data).
  Companion code: `github.com/sheim/phaser`.
- **Revzen & Guckenheimer 2012**, *Finding the dimension of slow dynamics in a rhythmic
  system*, J. R. Soc. Interface **9**, 957–971. *Cited*: the orbit is modeled as an
  order-11 Fourier series in the phase; return maps are taken at a set of phases and the
  Floquet structure is estimated by least squares, with bootstrap for confidence. This is
  the source of our `harmonics=11` default and of the Fourier-in-phase representation.
- **Kralemann et al. 2008**, *Phase dynamics of
  coupled oscillators reconstructed from data*, Phys. Rev. E **77**, 066205. *Cited* (as
  restated by Gengel & Pikovsky, arXiv:2111.10300, §7.1, Eqs. 23–24, which we could read):
  a protophase $\psi$ is any $2\pi$-periodic observable angle; the phase is the one whose
  density in time is uniform, $\rho(\varphi) = (2\pi)^{-1}$; with $F_k$ the Fourier
  coefficients of the protophase density, the **protophase-to-phase (PTP)
  transformation** is

$$
\varphi(\psi) = \int_0^{\psi} \rho(\tilde\psi)\thinspace d\tilde\psi
= \psi + \sum_{k \ne 0} \frac{F_k}{ik}\left(e^{ik\psi} - 1\right),
\qquad F_k = \langle e^{-ik\psi(t)} \rangle_t ,
$$

  that is, the cumulative density rescaled to $2\pi$. Our `phase_from_protophase` is this
  formula with the $k \ne 0$ sum written as $2\thinspace\mathrm{Re}\sum_{k \ge 1}$.
- **Gengel & Pikovsky 2022**, *Phase reconstruction from oscillatory data with
  iterated Hilbert transform embeddings* (arXiv:2111.10300; the Springer chapter
  Gengel & Pikovsky 2021 is the earlier paper). *Cited*: the state of the art for protophases from a *scalar* signal;
  the PTP step is unchanged. Not needed here (§1.2).
- **Zielinski et al. 2014**, *Strengths and limitations of
  period estimation methods for circadian data*, PLoS ONE **9**(5), e96462. *Cited* from
  the survey: among period estimators (Fourier/periodogram, autocorrelation, peak-to-peak
  timing, curve fitting), the curve-fitting family that uses every sample is the most
  robust to noise, and timing-based estimators (two samples per cycle) degrade fastest.
  *Deduced*: this is the reason our period comes from a regression over all tail samples
  rather than from crossing intervals (§1.2, step 5).
- **Namura et al. 2022**, *Estimating asymptotic phase and
  amplitude functions of limit-cycle oscillators from time series data*, Phys. Rev. E
  **106**, 014204. *Cited*: phase and amplitude functions are fitted *off* the cycle by
  polynomial regression on time-series data, with the cycle and Floquet exponent as
  inputs. Relevant to E4/E7 (comparison with the learned phase), not to E1.
- **Mauroy & Mezić 2012**, Chaos **22**, 033112; **Mauroy et al. 2013**,
  Physica D **261**, 19–30: isochrons and isostables by Fourier–Laplace averages of
  observables along trajectories — E5's method when the vector field is known.
- **Langfield et al. 2014**, Chaos **24**, 013131: the published FitzHugh–
  Nagumo isochrons; their period $T_\Gamma \approx 11.2279$ and clockwise orientation are
  E1's FitzHugh–Nagumo oracle (`tests/test_systems.py` already pins the period).

**What we do differently, and why.** The phase-reconstruction literature starts from a
*scalar* observable, so its first and hardest step is the protophase (Hilbert transform,
embedding, iterated transforms). Here the **full planar state is observed**, so the polar
angle about any interior point of the loop is a protophase at no cost (*deduced*: the
angle about an interior point winds once per revolution, which is all a protophase
needs), and the PTP transformation of Kralemann et al., a density correction rather than
a signal-processing step, carries over unchanged. Second, the literature assumes the data lie on or near the
cycle; our trajectories start anywhere and converge, so the estimate must pick its own
window (the **tail**) and *check* that the window is on the cycle. Third, the period
estimate exploits the phase: once $\varphi$ is uniform in time, the period is a slope.

### 1.2 The method

The input is trajectories `ys (N, T, 2)` (or one, `(T, 2)`) on a common grid `ts (T,)`;
`tail_periods` (default 3), `harmonics` $K$ (11), `density_harmonics` ($2K + 2$),
`tolerance` (2).

1. **Center.** The coordinate-wise median of the second half of every trajectory: a
   robust interior point of the loop (the transient of a converging trajectory is shorter
   than half of it in every dataset we generate; a center need only be *inside*).
2. **Protophase.** $\psi(t) = \mathrm{unwrap}\thinspace\mathrm{atan2}(y - c_y,\thinspace x - c_x)$,
   oriented so that it increases with time; the sign is the **winding** ($+1$
   counterclockwise, $-1$ clockwise) and is reported.
3. **Tail.** The samples of the last `tail_periods` revolutions of each trajectory,
   $\psi > \psi_{\rm end} - 2\pi\cdot$`tail_periods`. The times at which $\psi$ crosses
   successive multiples of $2\pi$ (linear interpolation between bracketing samples) give
   one period sample per completed revolution per trajectory, pooled into
   `period_samples` — a per-revolution diagnostic (their spread is the noise level of a
   two-sample estimator; their drift, a transient).
4. **Density and phase.** $F_k$, $k = 1..K_\rho$, is the **time average** of $e^{-ik\psi}$
   over exactly `tail_periods` revolutions per trajectory, by the trapezoid rule with the
   window's start interpolated in time, pooled over trajectories weighted by duration;
   then $\varphi = \psi + 2\thinspace\mathrm{Re}\sum_k F_k (e^{ik\psi} - 1)/(ik)$ on the
   unwrapped $\psi$ (the correction is $2\pi$-periodic in $\psi$, so this is the unwrapped
   $\varphi$). *Deduced, and the reason the average is an integral and not a sample mean*:
   over an integer number of revolutions the density is exact; a window whose sample count
   is off by one at the edge (which happens whenever the grid is commensurate with the
   period) biases every $F_k$ by $1/n$, which with $n \approx 300$ is a phase error of
   $\approx 3\cdot10^{-3}$ — we measured exactly this before switching to the integral
   (the Bautin phase error went from $5\cdot10^{-3}$ to $2\cdot10^{-7}$).
5. **Period.** $T = 2\pi / \hat\omega$ with $\hat\omega$ the least-squares slope of the
   unwrapped $\varphi$ against $t$ over the tail, one intercept per trajectory (pooled
   numerator and denominator). Every sample contributes, so measurement noise averages
   out as $\sigma / (\sqrt{n}\thinspace\mathrm{std}(t))$ — on the Bautin test with
   $\sigma = 0.05$ and 16 trajectories the period error is $2\cdot10^{-5}$ where the
   crossing-interval median gave $3\cdot10^{-3}$.
6. **Curve.** `Cycle.from_points(φ, tail points, T, harmonics)` — the least-squares
   Fourier series.
7. **Check.** The curve fitted to the **last revolution alone** is the reference. Its RMS
   misfit on those samples is `residual_last` (the noise floor); the RMS misfit of the
   *earlier* tail revolutions to it is `residual_earlier`. `converged` is
   `residual_earlier ≤ tolerance · residual_last`: a tail on the cycle traces the same
   curve every revolution, a tail still in the transient fits worse the further back it
   goes. (A whole-tail fit compared with itself does not work: the fitted curve is then
   the *average* of the revolutions and the transient hides in it — ratio 1.1 on a tail
   whose radius changes by 0.3; the last-revolution reference gives 2.9.)

The returned `cycle` is always the whole-tail fit (the most averaging). When
`converged` is `False` it is biased toward the transient and the caller shortens
`tail_periods` or supplies longer trajectories; E2 turns this into an iteration.

### 1.3 Oracles (`tests/test_analysis.py`)

| Test | Pins down |
|---|---|
| Bautin $(a, b, \omega_1, \omega_0) = (1, 0.5, 2, 1)$ and Hopf $(1, 1.5, 0.7)$, 6 trajectories from an annulus $r \in [0.3, 2.5]$, 8 periods at 100 samples per period | radius 1 and center 0 to $2\cdot10^{-3}$ (achieved $10^{-8}$); period `nf.period()` to $10^{-6}$ relative (achieved $10^{-8}$); the recovered phase equals the polar angle up to one constant (spread $< 2\cdot10^{-2}$; achieved $2\cdot10^{-7}$); `winding = +1`; `converged` |
| the same trajectories through a fixed affine map | the image curve, with the phase *still* the original polar angle up to a constant — the PTP step is what makes this true, since the polar angle about the image center is not uniform in time |
| one trajectory vs the pooled four | the same period and curve; `period_samples` has one entry per completed tail revolution |
| Gaussian measurement noise $\sigma = 0.05$ on 4 and on 16 trajectories | both converge; the curve residual is the noise level ($\sigma\sqrt 2$ within a factor 2); the curve's radial error is below $10\thinspace\sigma\sqrt{(2K+1)/n}$ and *shrinks* by at least 0.8 when the sample is quadrupled (achieved 0.018 → 0.0077) |
| Bautin with $a = 0.1$ ($\kappa = -0.2$, the deviation halves per period), 5 periods from $r \in [0.2, 0.5]$ | `converged` is `False`; the same system over 40 periods converges |
| Bautin with $\omega_1, \omega_0 < 0$ | `winding = -1`, the period is $\vert 2\pi/\omega_1 \vert$ (the normal forms' `period()` is signed, design document §2), and the curve runs clockwise (negative signed area) |
| FitzHugh–Nagumo (slow), 8 trajectories from the box $[-3, 3]^2$, 120 time units at the `fhn` config's step | period $11.2279 \pm 2\cdot10^{-3}$ (Langfield et al.; achieved $11.22792$), clockwise, converged; residual $< 10^{-2}$ with `harmonics=24` (11 harmonics leave $3\cdot10^{-2}$ on the relaxation jumps) |
| `phase_from_protophase`, `density_coefficients` | the identity on a uniform protophase; recovers $\varphi$ from $\psi = \varphi + 0.05\sin(\varphi - 0.4)$ up to a constant; the integral estimator equals the sample mean over a closed period and ignores samples outside its window |
| input checks | fewer than `tail_periods` revolutions, `tail_periods < 1`, too few points for the harmonics |

Tolerances are entries of `tests/helpers.TOL` (`cycle_query`, `cycle_period`,
`cycle_curve`, `cycle_phase`) with their reasons.

### 1.4 Limits

- **Noise model.** Measurement noise, i.i.d. per sample, is what the averaging handles.
  Dynamical noise (a stochastic oscillator) spreads the tail around the cycle; the fit
  still returns the mean loop but `converged` may misreport — the residual floor then
  carries the dynamical spread and the check only compares revolutions with each other.
- **Self-consistency, not settling.** `converged` says the *tail* is on the cycle; it does
  not say when the trajectory got there. That is E2 (`settling_time`), which takes this
  `Cycle` and the distance to it along each trajectory.
- **Planar only.** The polar-angle protophase needs a plane; in higher dimension the
  protophase would come from a 2-D projection (PCA, as Revzen & Guckenheimer do) and the
  curve from the full state.
- **Not differentiable, not jitted.** Host-side NumPy by design (ADR-0012).
- **Harmonics.** 11 (Revzen & Guckenheimer) suits a normal form; FitzHugh–Nagumo's
  relaxation jumps want about 24. The density order defaults to $2K + 2$ because it is
  cheap and its truncation is the phase's error (a 2.5 : 1 affine image of the circle
  needs 24 density terms for a $2\cdot10^{-6}$ phase; 11 leave $3\cdot10^{-3}$).

## 2. O1 — `analysis.ode.limit_cycle`: the periodic orbit and its period from the vector field

**The problem.** Given a planar `AbstractODE` with a hyperbolic attracting periodic orbit $\gamma$, find a point $x^\ast \in \gamma$, the period $T$, and the orbit sampled uniformly in time — the reference every later item (asymptotic phase, PRC, isostables) is built on, to an accuracy far beyond what a learned model's comparison can use (target $10^{-10}$ on $T$). Claims are *cited* (read in the source), *secondary* (via a source that cites it) or *deduced*.

### 2.1 What exists

Every method solves the same boundary value problem: $\dot x = T f(x)$ on $[0, 1]$, $x(0) = x(1)$, plus one phase condition. The methods differ in how the trajectory between mesh points is represented.

| Family | Trajectory between mesh points | Jacobian | References |
|---|---|---|---|
| Brute force: settle, then measure returns to a section | adaptive IVP | none | Parker & Chua 1989, §5.1.1, and the minimum-period test of §5.5 read forwards (*cited*) |
| Single shooting: Newton on $\Phi^T(x) - x$ with $T$ unknown | one adaptive IVP | variational equation or AD | Aprille & Trick 1972 and Mees 1981 (*secondary* via Parker & Chua §5.5, *cited*); the Poincaré-map variant with $DP_+ = [I - f(y)h^{\top}/\langle h, f(y)\rangle]\thinspace\Phi_T(x)$, their Eq. (5.21) |
| Multiple shooting: $m$ segments with matching | $m$ adaptive IVPs | block-cyclic, blocks $\partial_x\Phi^{T/m}(x_i)$ | Morrison et al. 1962 (*cited* via Wikipedia); Keller 1968; Osborne 1969, J. Math. Anal. Appl. 27:417–433 (*cited*, author's list); Stoer & Bulirsch §7.3.5 (*cited* via Wikipedia); Deuflhard 1980 survey (*secondary*: Guckenheimer & Meloon and Dednam & Botha both cite it); Ascher et al. 1988 (*secondary*) |
| Symmetric multiple shooting: forward from $x_i$, backward from $x_{i+1}$ | half-segments both ways | same structure | Guckenheimer & Meloon 2000, §2, §5.3 (*cited*) |
| Finite differences | one step of a one-step formula per interval | sparse cyclic | Parker & Chua §5.5 (*cited*) — multiple shooting with one solver step per segment |
| Orthogonal collocation | piecewise polynomial zeroed at Gauss nodes | sparse, explicit | Doedel 1981; Doedel et al. 1991; Doedel 2007 (*cited* via the Osinga/Krauskopf lecture notes and BifurcationKit's documentation); AUTO, MatCont, CoCo |
| Taylor-series shooting with AD | degree-$d$ Taylor polynomial of the trajectory | AD of the Taylor map | Guckenheimer & Meloon 2000 (*cited*) |
| Least-squares shooting | fixed-step RK on rescaled time, stacked residual | finite differences | Dednam & Botha 2015 (*cited*) |
| Harmonic balance | truncated Fourier series, Galerkin | dense | Parker & Chua §5.5 (*cited*); Urabe 1965 (*secondary*) |

What the two modern papers add, and do not (*cited*). Guckenheimer & Meloon's accuracy comes from exact derivatives of the discrete flow map ("no finite difference calculations of derivatives implicit in our methods", §6) and coarse meshes with increasing polynomial degree; on their algebraic test orbit (§5.1, Fig. 5.1) AUTO reaches $\approx 5\cdot10^{-15}$, RK4 at its optimal step $\approx 8\cdot10^{-15}$, their method $\approx 6\cdot10^{-16}$: one order of magnitude at the roundoff floor. Their stronger case is a canard cycle (§5.3), which forward integration cannot follow at all and symmetric shooting can. Dednam & Botha's extra residual components $x(1 + \Delta s) - x(\Delta s)$ are redundant at a zero (uniqueness of the flow), so their Levenberg–Marquardt contributes *globalization*, not accuracy (*deduced*); their finite-difference Jacobian is their own stated weakness (§2). Neither has grounds to beat AD-Jacobian multiple shooting on an adaptive high-order solver: JAX gives the exact-Jacobian property for free, and $10^{-10}$ is reachable with `Dopri8` at tight tolerances.

**The degeneracy and the phase condition.** At a solution $DG = [\thinspace M - I \mid f(x^\ast)\thinspace]$ with $M$ the monodromy and $Mf = f$, so the null space is the flow direction at fixed $T$; an admissible phase condition has a nonzero component along $f(x^\ast)$, that is, a transversal section (*deduced*). Two are in use: the Poincaré condition $\langle n, x_1 - p\rangle = 0$ (local, fixes the starting point) and Doedel's integral condition $\int_0^1 \langle x, \dot x_{\rm ref}\rangle = 0$, the stationarity condition of the time-shift distance to a reference orbit (*deduced*), the natural one for continuation. Shooting *can* evaluate the integral through a quadrature state; for a one-off solve with no reference orbit the Poincaré condition is the choice (ADR-0014 §2). BifurcationKit's default $s = T\langle x - x_\pi, \phi\rangle$ is this condition with `center` $x_\pi$ and unit `normal` $\phi = F(x_1)/\Vert F(x_1)\Vert$ (its `SectionSS`, *cited* from the source); the factor $T$ is a row scaling with no stated rationale, which we drop.

**The initial guess** is not packaged anywhere (BifurcationKit, BoundaryValueDiffEq, MatCont, CoCo, AUTO take an orbit guess from the user or start from a Hopf point; *cited*/*secondary*). The recipe is Parker & Chua's brute force on a section: settle, place $\Sigma$ normal to the flow at the settled point ("let the program automatically set $h = f(x^{(0)})$ and $x_\Sigma = x^{(0)}$", §5.5), count positive crossings (footnote 4), measure the return time. Crossings are located by event detection with a root finder (Shampine & Thompson 2000, Comput. Math. Appl. 39:43–54 — *verify*; the classical remedy for the final-step error is Hénon 1982, via Tucker 2002, *cited*). Peak-to-peak timing is inferior: a smooth extremum localizes time to $\sqrt{2\varepsilon/\vert\ddot x\vert}$, a transversal crossing to $\varepsilon/\vert\dot x\vert$ (*deduced*).

### 2.2 The method (`analysis/ode/`)

`CrossSection` (`sections.py`): $\Sigma = (p, n)$ with unit normal, `signed_distance(x)`, `normal_to_flow(ode, p)` the flow-normal section.

`PoincareMap` (`poincare.py`) is the first-return map $P$ of the flow on $\Sigma$. `__call__(x) -> (P(x), τ(x))` locates the crossing by a diffrax `Event` after a short departure solve (a zero of the condition at $t_0$ is read as a crossing; in reverse-time solves diffrax 0.7.2 hands the condition its internal increasing time, so the guard is a separate solve, not a time test). `direction` (positive, negative or both, relative to the field being integrated, so that in reverse time a point of a cycle returns to itself after one period) and `reverse_time` are static constructor arguments. `__call__` and `iterate` (a `lax.scan`) are jit- and vmap-compatible, a missing crossing being a runtime error (`throw=True`, as in diffrax) or `τ = ∞`. `fixed_point(x, tol, max_iter)` is a host-side driver over the jitted map that iterates the one-sided map until it returns to within `tol` of an earlier iterate, a period-$k$ orbit of $P$ with $k$ minimal (Parker & Chua §5.4; a cycle cutting $\Sigma$ in $2k$ points), and reports the contraction of successive differences as the brute-force multiplier estimate. The map is differentiable in $x$ through a custom JVP that never differentiates through the event: differentiating the crossing condition $\langle n, \Phi_\tau(x) - p\rangle = 0$ gives $d\tau = -\langle n, J\thinspace dx\rangle / \langle n, v\rangle$ and $dy = [I - v n^{\top}/\langle n, v\rangle]\thinspace J\thinspace dx$ with $J = \partial_x\Phi_\tau(x)$ by forward-mode AD over the known $\tau$ and $v$ the velocity at the crossing, which is Parker & Chua Eq. (5.21) for the point and implicit differentiation for the time (*deduced*). So `jax.jacfwd(P)` is $DP$ with eigenvalues $\lbrace 0, e^{\kappa T}\rbrace$.

`multiple_shooting` (`shooting.py`): unknowns $(x_1, \dots, x_m, T)$, residual $(\Phi_{T/m}(x_i) - x_{i+1}\ \text{cyclic},\ \langle n, x_1 - p\rangle)$, segments `vmap`ped, Jacobian by `jax.jacfwd` with `diffrax.ForwardMode`, solved by `optimistix.least_squares(LevenbergMarquardt)`; jit-compatible, $m$ fixed. `SHOOTING_SOLVER_CONFIG`: `Dopri8` (an eighth-order method needs far fewer steps than a fifth-order one at $10^{-10}$–$10^{-12}$; `Tsit5` gives the same answers more slowly), rtol $10^{-10}$, atol $10^{-12}$ — the solver's tolerance is the residual floor (Joon's prototype at rtol $10^{-6}$ had residual $10^{-6}$ at the *initial guess*).

`find_limit_cycle` (`limit_cycle.py`), a host-side composition of jitted stages, each exported: `settle` (flow for `t_settle`) → `initial_guess` (`CrossSection.normal_to_flow` and `PoincareMap.fixed_point`: $\hat x$, $\hat T$ over $k$ returns, $\hat\lambda$) → `polish` (`multiple_shooting` from $m$ points of the orbit through $\hat x$) → the guards `is_equilibrium` and `period_consistent` (the two extraneous solutions of shooting, Parker & Chua §5.5, "Extraneous solutions") → `sample_orbit` (the orbit at `n_points` uniform times in $[0, T)$, forward order even in reverse time) and `orientation` (sign of the signed area) → `LimitCycle` (orbit, period, orientation, section, residual, settle record). Properties *of* the orbit, such as the monodromy, the multipliers and their cross-checks, are O2's functions of a `LimitCycle`, not products of the search.

`AbstractODE.flow_to(t, u0)` (the state at a signed time, `flow(ts=[0, t]).ys[-1]`, differentiable in both) and `LevelSetCycle` (`systems/level_set.py`, the designed-cycle family of §2.3) were added to `systems` for this item. `analysis` works through the `flow` contract alone; which integration a system's `flow` performs is the system's choice. For the normal forms it defaults to the closed-form chart, which as an *input* would make the oracle tests circular and, at a basin edge ($r \to a$ for Winfree), loses precision — so the tests construct them with `integration="cartesian"` (a constructor argument) and the closed forms stay the oracles, the solver the input (ADR-0014, round 2).

### 2.3 Oracles (`tests/test_analysis_ode.py`)

| System | Exact | Pinned |
|---|---|---|
| Hopf, Bautin, Winfree ($a < 0$) | $r = 1$, $T = 2\pi/\vert\omega_1\vert$, $\kappa = \rho'(1)$ | $T$ and radius to $10^{-8}$ (the solver's global error over a period at rtol $10^{-10}$ is $\sim 10^{-9}$); return time and point of $P$, under jit and vmap; `jacfwd(P)`'s eigenvalues $\lbrace 0, e^{\kappa T}\rbrace$ and $d\tau$ against finite differences; contraction of the iterates $\to e^{\kappa T}$ in the linear regime; orientation = sign of $\omega$ |
| Designed level-set cycles, `systems.LevelSetCycle`: $f = s\thinspace(R\nabla g - g\nabla g)$, so $\dot g = -s\thinspace g\thinspace\Vert\nabla g\Vert^2$ and $\lbrace g = 0\rbrace$ is an attracting cycle traversed counterclockwise (*deduced*; Guckenheimer & Meloon's §5.1 field is this with $s = 1/2$) | the curve | **Guckenheimer & Meloon's algebraic orbit** $g = x^2 - y^2 + \tfrac23 y^3 + c$, $c = 0.07$, start $(0, 0.2952161257895192)$ (*cited*, p. 970): $\max\vert g\vert$ along the output $< 10^{-8}$ (theirs $6\cdot10^{-16}$ at roundoff; ours is bounded by the solver's global error, $\sim 10^{-9}$), $T \approx 7.7$; **a Cassini oval** (non-convex): a horizontal section above the waist is cut four times — the two-sided map visits the four intersections in order, the one-sided map has a period-2 orbit and `fixed_point` reports $k = 2$ |
| FitzHugh–Nagumo | — | $T = 11.2279$ (Langfield et al. 2014, §III, *cited*) to $2\cdot10^{-3}$; brute-force $\hat T$ agrees with the polished $T$ to $10^{-6}$; clockwise (`slow`) |
| Winfree $a = 0.25$, reverse time | inner cycle $r = a$, $T = 2\pi/\vert\omega(a)\vert$, $\kappa_u = a(1 - a)$ | radius and period to $10^{-8}$; the reversed map contracts by $e^{-\kappa_u T}$ per return and `jacfwd` of it has that eigenvalue |
| Failure modes | — | outside the basin (`ValueError`), an equilibrium start (`ValueError`), no crossing within `t_max`, `max_iter` exhausted, the $T \to 0$ extraneous solution — each a `NotConvergedError` with the stage named |

The suite also covers a near-Hopf Bautin cycle ($\kappa \to 0$) where the fixed-point stage alone cannot reach a tight tolerance and the polish finishes from a loose settle; tolerance monotonicity (tighter solver, smaller period error); the output consumed by `Cycle.from_points` (phase-uniform points) with negligible residual; Hypothesis over Hopf parameters and starting points.

### 2.4 Limits

- **Attracting orbits only** (repelling ones through `reverse_time`); a saddle cycle in higher dimension would need shooting proper (error growth $e^{\kappa_u T/m}$ per segment is the lever) — not planar, not planned.
- **No canard safety.** Forward segments; a cycle hugging a repelling slow manifold needs symmetric shooting (Guckenheimer & Meloon §5.3). FitzHugh–Nagumo at $\varepsilon = 0.08$ follows the attracting branches; Langfield et al.'s sensitivity near the repelling slow manifold concerns the *isochrons*, not the orbit.
- **No continuation.** A parameter sweep would want the integral phase condition and pseudo-arclength; the quadrature-state formulation makes that a small addition. Collocation only if Langfield-style isochron BVPs are ever wanted.
- **Floquet multipliers** are available (`jax.jacfwd` of the map, the contraction of its iterates, the monodromy by `jacfwd` of `flow_to`) but their interpretation and the Liouville cross-check $\exp\int_0^T \mathrm{div} f\thinspace dt$ belong to O2.
- **Jit scope** (Joon, 2026-10-18): the map, its iteration, the shooting solve and the flow are jitted and vmappable; `fixed_point` and `find_limit_cycle` are host-side drivers (they run once per system and raise informative errors). Revisit if a use case needs a jitted settle. First call per `(ODE class, config, m)` pays XLA compilation (≈ 15 s on CPU); later instances of the same class reuse it.

## 3. O2 — `analysis.ode.floquet`: the Floquet exponent of the orbit

**Status: survey delivered 2026-10-19; implemented 2026-10-20 after Joon's prototype (ADR-0015). §3.1 keeps the survey, with the two corrections the implementation forced marked ⟂.**

**The problem.** Given a `LimitCycle` of a planar `AbstractODE` (O1: the shooting mesh $x_1, \dots, x_m \in \gamma$, $T$), compute the nontrivial Floquet multiplier $\mu$ and exponent $\kappa = T^{-1}\ln\mu$ of $\gamma$ to the accuracy of the orbit itself, with its own error estimate; $\kappa$ is the rate at which the amplitude (isostable) coordinate decays, $\dot\psi = \kappa\psi$, and the quantity a learned conjugacy must reproduce: the normal forms carry it in closed form ($\kappa = \rho'(1)$, `floquet_exponent`), so $\kappa$ of FitzHugh–Nagumo pins the normal form's amplitude parameters (§3.4). Claims are *cited*, *secondary* or *deduced* as in §2.

### 3.1 What exists

**The object.** The monodromy $M = \partial_x\Phi_T(x^\ast)$, the fundamental solution of the variational equation $\dot Y = Df(\gamma(t))\thinspace Y$, $Y(0) = I$, at $t = T$. Its eigenvalues are the Floquet multipliers; they do not depend on the point $x^\ast \in \gamma$ or on the coordinates; one of them is $1$ with eigenvector $f(x^\ast)$ (the trivial multiplier), and the orbit is elementary (hyperbolic) when $1$ is simple (Guckenheimer & Meloon 2000, §1, *cited*; classical Floquet theory, Hale 1980, §III.7 as cited by Kvalheim & Revzen 2021, §3.1, *secondary*). The Jacobian of a return map $P$ on a section $\Sigma$ at its fixed point is the quotient of $M$ by the flow direction ("the Jacobian of the return map can be obtained as a quotient of the monodromy matrix by projecting out the flow direction", Guckenheimer & Meloon §1, *cited*), so $DP$ has the eigenvalues of $M$ except the trivial one, which $DP$ replaces by $0$ when written on the ambient space (O1's `jacfwd(P)`, eigenvalues $\lbrace 0, \mu\rbrace$, *deduced* in §2.2 and tested). Wilson & Moehlis 2016 define the isostable coordinates from exactly this $DP$ and its eigenvectors (their Eqs. 3–5, *cited*) and set $\kappa_i = \log(\lambda_i)/T$ (Eq. 7).

**Why it matters to this project.** The Floquet exponent is the Koopman eigenvalue of the principal eigenfunction transverse to the cycle: Kvalheim & Revzen 2021, Prop. 5 — for a principal Koopman eigenfunction $\psi$ with exponent $\mu_K$ on the basin of a $\tau$-periodic orbit, $D_{x_0}\psi$ is a *left* eigenvector of $D_{x_0}\Phi^\tau$ with eigenvalue $e^{\mu_K\tau}$, "in particular, $e^{\mu_K\tau}$ is a Floquet multiplier" (*cited*). Mauroy & Mezić 2018 (Chaos 28, 073108) build the amplitude coordinate from $\Lambda_1 = \log(\mu_1)\thinspace\omega/2\pi$ with $\mu_1$ "the dominant eigenvalue (different from 1) of the monodromy matrix" (§IV, *cited*). The learned conjugacy $h$ between FitzHugh–Nagumo and a normal form transports the variational equation, so the multipliers of the two cycles must agree: $M' = Dh\thinspace M\thinspace Dh^{-1}$ for a $C^1$ conjugacy (*deduced*, from the chain rule); under a time reparametrization (orbital equivalence) the return maps stay $C^1$-conjugate, so $\mu = e^{\kappa T}$ is still invariant while $\kappa$ and $T$ separately are not (*deduced*). O2 therefore produces the one number that fixes the normal form's amplitude parameters given its period, a check on training as well as a diagnostic of the ODE.

**Routes to $\mu$.**

| Route | Computes | Where it stands | Cost / accuracy |
|---|---|---|---|
| (a) Monodromy by the variational equation, eigenvalues | $M$, both multipliers | the textbook method; Mauroy & Mezić 2018, §III.B integrate $\dot M = \partial_x F\thinspace M$ alongside the orbit (*cited*); AUTO86's outside-the-code practice (Guckenheimer & Meloon §5.4, *cited*) | one $(n + n^2)$-dimensional solve; the error in the trivial multiplier $\vert\lambda_1 - 1\vert$ "is a measure for the accuracy of the computation of the periodic orbit and its monodromy matrix" (G&M §5.4, *cited*; $6\cdot10^{-15}$ in their Josephson example) |
| (a′) Monodromy by AD of the time-$T$ flow map | same $M$ | `jax.jacfwd` of `flow_to` with `diffrax.ForwardMode`: "discretise then optimise" forward mode that "operates through the internal numerics of the solver" (diffrax documentation, *cited*) — the exact derivative of the *discrete* flow, the property G&M credit for their accuracy (§6) | one forward-mode solve carrying $n$ tangents; equivalent to (a) with the solver's own discretization of the variational equation (*deduced*) <!-- codespell:ignore: cited quotation -->|
| (b) Product of segment Jacobians from multiple shooting | $M = M_m\cdots M_1$, $M_i = \partial_x\Phi_{T/m}(x_i)$ | BifurcationKit's `FloquetQaD` (default; "not numerically very precise for large / small Floquet exponents", *cited*) and the `multiflap` package (*cited*); the factors are what the shooting Jacobian already contains | free after shooting; the product has the same conditioning problem as (a) |
| (c) Eigenvalue of the return-map Jacobian | $DP$, $\lbrace 0, \mu\rbrace$ | O1's custom JVP (Parker & Chua Eq. 5.21); BifurcationKit's Poincaré shooting: "the (non trivial) Floquet exponents are eigenvalues of the Poincaré return map" (*cited*) | the same solve as (a′) plus the projection; no trivial multiplier to carry |
| (d) Contraction of return-map iterates | $\mu$ as the ratio of successive distances to the fixed point | Parker & Chua §5.1.1 brute force read as a power iteration; O1's `FixedPoint.contraction` | linear convergence, polluted by the quadratic terms of $P$ until the iterates are close; a sanity check, not a result |
| (e) Liouville / Abel–Jacobi: $\det M = \exp\int_0^T\mathrm{tr}\thinspace Df(\gamma(t))\thinspace dt$ | in the plane, with $\lambda_1 = 1$ exactly: $\mu = \exp\int_0^T\mathrm{div}\thinspace f(\gamma(t))\thinspace dt$, i.e. $\kappa = T^{-1}\int_0^T\mathrm{div} f\thinspace dt$ | Andronov et al. (p. 118) and Perko 3rd ed. pp. 210–214, as cited by Giacomini & Grau 2005 (arXiv math/0505027, Theorem 3: $v\cdot DP(p_0) = \exp(\int_0^T\mathrm{div}\thinspace dt)\thinspace v$ for $v$ normal to $F(p_0)$, *cited* there); Mauroy & Mezić 2018, Remark 4 use it for planar systems, "$\lambda = (1/T)\int\mathrm{div} F(x_\gamma(t))\thinspace dt$" (*cited*) | a scalar quadrature along the orbit, in **log space**: no eigenvalue problem, no cancellation, exact in the plane |
| (f) Periodic Schur decomposition of $M_m\cdots M_1$ | all multipliers without forming the product | Fairgrieve & Jepson 1991, "O.K. Floquet multipliers" (SIAM J. Numer. Anal. 28:1446–1462) for collocation; Lust 2001 (Int. J. Bifurcation Chaos 11(9):2389–2410): common techniques "including AUTO97's, can be very inaccurate when Floquet multipliers are very large or very small", remedied by the periodic Schur decomposition (*cited*, abstract); BifurcationKit's `FloquetPQZ` (*cited*) | the right tool in $n \geq 3$ with multipliers of very different magnitude; in the plane (e) already gives $\ln\mu$ directly |

**The conditioning problem, in our terms.** A relaxation oscillator is *strongly* attracting: for our FitzHugh–Nagumo ($\dot x = c(y + x - x^3/3 + z)$, $\dot y = -(x - a + by)/c$, $c = 3$) $\mathrm{div} f = c(1 - x^2) - b/c$ runs from $-8.9$ on the slow branches to $+2.7$ on the jumps, and the implementation gives $\kappa T = \int\mathrm{div} f\thinspace dt = -27.106$, $\kappa = -2.4142$, $\mu = 1.6906\cdot10^{-12}$ (⟂ the survey's estimate of $10^{-17}$ was off: the jumps recover more than it allowed for). The explicit monodromy has eigenvalues $1$ and $\mu$ in a matrix with $O(1)$ entries, so by Bauer–Fike $\mu$ is resolved only to the absolute rounding of the product, a relative error $\varepsilon\Vert M\Vert/\mu$: Lust 2001's Eq. 10, $d = \log_{10}(\max\vert\lambda\vert/\vert\mu\vert)$ digits lost (*cited*). ⟂ The survey had claimed $O(1)$ relative error from the *solver's* $10^{-9}$; that was wrong: discretize-then-differentiate AD gives the exact Jacobian of the discrete flow, whose own small multiplier is carried multiplicatively, so the solver error lands on the trivial multiplier (period mismatch along $f$), not on $\mu$. Measured: the AD monodromy's eigenvalue agrees with Liouville to $6\cdot10^{-5}$, exactly $\varepsilon/\mu$, and the Cassini oval below, with $\mu \approx 10^{-16}$, is beyond its reach altogether. The same floor applies to the return-map Jacobian $DP$ (route (c)), and worse: its two eigenvalues are $0$ and $\mu$, both below the solver's *absolute* error on $DP$ ($\sim10^{-9}$), and its *trace*, algebraically $\mathrm{tr}\thinspace M - n^{\top}Mv/\langle n, v\rangle = (1 + \mu) - 1$, leaks the trivial multiplier's error additively ($3.5\cdot10^{-9}$ for $\mu = 1.7\cdot10^{-12}$ in Joon's prototype). Route (e) computes $\ln\mu$ with the quadrature's relative accuracy ($\sim10^{-9}$), and route (f), the subject of the next paragraph, with the per-segment factors' relative accuracy. For the normal forms ($\mu = e^{-2\cdot 2\pi/2} \approx 0.002$ for Hopf with $a = 1$, $w = 2$) every route is fine, and (a′)/(c) are tested in O1. Two side facts (*deduced* from (e)): in the plane $\mu = \det M > 0$ always, so a planar cycle has no negative multiplier (no period doubling of planar cycles, as Poincaré–Bendixson implies anyway); and the trivial-multiplier error of (a′)/(f) is a free, independent accuracy estimate of *the orbit and the solver together*, which (e) alone does not provide.

**Why the periodic Schur chain buys accuracy, and the planar shortcut.** Bojanczyk et al. 1992 (Theorem 1, Corollary 2) give orthogonal $Q_i$ with every $\hat A_i = Q_{i+1}^{\top}M_iQ_i$ upper triangular, cyclically, so the multipliers are $\prod_i \hat A_{i,jj}$: per-segment factors of moderate size ($\approx e^{\kappa T/m}$), each to relative accuracy, accumulated as a sum of logarithms, and the product $M$ is never formed. Lust 2001, §5: "the accuracy is related to the size of the diagonal elements of the upper triangular factors … and not to the size of the diagonal elements in the Schur decomposition of the product" (*cited*); his Table 5, five factors with eigenvalues $10^{10}\ldots10^{-10}$: explicit product $3.5\cdot10^{4}$ relative error on the smallest, periodic Schur $2.9\cdot10^{-11}$. This is the discrete QR method for Lyapunov exponents (Dieci et al. 1997, §3: $Z_j = Q_{j+1} \thinspace R_{j+1}$ with positive diagonals, exponents from $\sum \log R_{k,ii}$; their Claim 2: for a periodic system it "computes the real parts of the Floquet exponents of the computed approximation to the monodromy matrix exactly", *cited*), with the re-orthonormalization that Geist et al. 1990, §1 call mandatory because unorthonormalized columns collapse onto the most expanding direction (*cited*). Their general algorithm (periodic Hessenberg reduction by Givens rotations, implicit shifted periodic QR, deflation, reordering, Bojanczyk et al. §3; SLICOT `MB03VD`/`MB03WD`; 8 000 lines in `PeriodicSchurDecompositions.jl`, which BifurcationKit's `FloquetPQZ` calls on the shooting blocks with no use of the known eigenvector, *read in the source*) exists to *find* the Schur basis. Here one Schur vector is known analytically: the flow direction $f(x_1)$, eigenvector of the trivial multiplier and the *dominant* one for an attracting cycle. Seed $Q_1$ with it (Guckenheimer & Meloon's transformation of the monodromy to cross-section coordinates, §2 algorithm, is the same deflation) and Bojanczyk's constructive chain of QR factorizations (§2 proof) completes the decomposition with no iteration; in the plane the transverse block is $1\times1$. Two things the implementation taught (*deduced*, checked numerically, no literature statement found): the chain must run *forward*, because it propagates the first Schur vector, which forward is the dominant direction and self-correcting, while backward through the transposes (Joon's first RQ chain) it is the subdominant one and the chain does not close ($\Vert Z_m - Q_1\Vert = 0.63$ on FHN); and the seed must be a seed only, because using the analytic frame at every mesh point instead of the QR's $Z_i$ leaves a sub-diagonal residue ($5\cdot10^{-11}$: the solver's end-point velocity is not exactly the field's) that the $O(1)$ shear in $R_{12}$ multiplies past $\mu$. Measured on FHN against Liouville: $m = 1$ at the $\varepsilon/\mu$ floor ($1.4\cdot10^{-5}$), $m \geq 2$ at the discretization floor ($10^{-9}$), closure $10^{-10}$. In the plane the three log-space routes coincide: the chain's $\prod(R_i)_{22}$, the product of segment determinants, and Liouville's integral as the $m \to \infty$ limit of the latter. For a *repelling* cycle $f$ is the subdominant direction of the forward monodromy, so the chain runs in the direction in which the cycle attracts (the reversed field for a `reverse_time` cycle) and the result is inverted (ADR-0015 §2).

**The level-set family as an oracle** (`systems.LevelSetCycle`, $f = s(R\nabla g - g\nabla g)$). On $g = 0$: $\mathrm{div}(R\nabla g) = \partial_x(-g_y) + \partial_y(g_x) = 0$ and $\mathrm{div}(g\nabla g) = \Vert\nabla g\Vert^2 + g\Delta g = \Vert\nabla g\Vert^2$, so $\mathrm{div} f\vert_{\gamma} = -s\Vert\nabla g\Vert^2$ and, with the speed $\Vert f\Vert = s\Vert\nabla g\Vert$ on $\gamma$,

$$
\kappa T = -s\int_0^T\Vert\nabla g(\gamma(t))\Vert^2\thinspace dt = -\oint_{\gamma}\Vert\nabla g\Vert\thinspace d\ell, \qquad T = \oint_{\gamma}\frac{d\ell}{s\thinspace\Vert\nabla g\Vert}
$$

(*deduced*; consistent with the transverse equation $\dot g = -s\thinspace g\thinspace\Vert\nabla g\Vert^2$ of §2.3, whose linearization along $\gamma$ is the same rate). Both are line integrals over the curve $\lbrace g = 0\rbrace$ that need **no ODE solve**: for the Cassini oval the curve has the polar parametrization $r(\theta)^2 = a^2\cos 2\theta + \sqrt{a^4\cos^2 2\theta - a^4 + b^4}$, and the trapezoidal rule on a periodic analytic integrand converges exponentially, so $\kappa$ and $T$ are available to roundoff as oracles independent of diffrax. (Guckenheimer & Meloon's cubic has no such parametrization; its oracle stays $\max\vert g\vert$ and the Cassini-derived checks.)

**What the packages do** (*cited* from documentation): BifurcationKit uses `FloquetQaD` on the product of shooting Jacobians by default, `FloquetPQZ` (periodic Schur) for accuracy, and the return-map eigenvalues for Poincaré shooting; AUTO takes the multipliers from the collocation Jacobian (Fairgrieve & Jepson), improved from AUTO86 to AUTO97 (G&M §5.4); `multiflap` takes the product of segment Jacobians through `numpy.linalg.eig`. None of them exploits the planar Liouville identity, because none is planar-specific.

### 3.2 The method (`analysis/ode/floquet.py`, ADR-0015)

Functions over a `LimitCycle` (which, as of O2, is a fixed-shape `eqx.Module` carrying its `ode`, the shooting mesh `x0s`, `period`, `config`, `reverse_time` and a light `FixedPoint`; analysis functions take the cycle — ADR-0015 §5):

- `segment_jacobians(cycle) -> (m, dim, dim)`: $M_i = \partial_x\Phi_{T/m}(x_i)$ by `vmap(jacfwd(flow_to))` with `ForwardMode`, in the time direction in which the orbit attracts.
- `periodic_schur(jacobians, q1) -> PeriodicSchur(Q, R, signs)`: the forward QR chain seeded with `q1` as a `lax.scan`; positive diagonals (rows of $R$ and columns of $Q$ flipped together — Joon's prototype flipped columns of $R$, invisible on the diagonal, wrong for the vectors); `signs` $= \mathrm{diag}(Q_1^{\top}Q_{m+1})$ rounded; `multipliers` $= D\exp\sum\log\mathrm{diag}R_i$, `closure` $= \Vert Q_{m+1} - Q_1D\Vert$. `NotImplementedError` for `dim > 2` (the transverse block would need the periodic QR iteration; `slycot.math.mb03wd` wraps SLICOT's).
- `monodromy(jacobians)`, `multipliers_from_monodromy(M)` (eigenvalues, the one closest to 1 first), `return_map_multiplier(pmap, x)` (the eigenvalue of `jacfwd(P)`, not its trace): the diagnostics, with their floors in the docstrings.
- `divergence_integral(cycle)`: the augmented solve $(\dot x, \dot z) = (f, \mathrm{tr}\thinspace Df)$ through `systems.base.diffeqsolve` at `cycle.config`; planar only (`ValueError`).
- `floquet_periodic_schur`, `floquet_monodromy`, `floquet_liouville`: each `(cycle) -> Floquet` with `multipliers`, `exponents` (forward-time; index 0 trivial), `trivial_error`, `closure` and `vectors` (the right Floquet vectors at every mesh point from the triangular factors' cyclic products, O5's hook; `None` for the other routes); `floquet(cycle, *, method=)` dispatches over them. A `Literal` now; strategy objects when a method grows options. All are jit- and vmap-compatible functions of the cycle, left unjitted (ADR-0015 §6): a batch of cycles of one mesh size is one `vmap`.

`LimitCycle.sample(n_points)` re-integrates the orbit from `x0s[0]` (the sampled orbit is derived, not stored); `orientation()` from the sampled polygon (`analysis.data.cycle.orientation`).

### 3.3 Oracles (`tests/test_analysis_ode.py`, O2 block)

| System | Exact | Pinned |
|---|---|---|
| Hopf, Bautin, Winfree ($a < 0$), `integration="cartesian"` | $\kappa = \rho'(1)$, $\mu = e^{\kappa T}$ | all three routes to $10^{-8}$; trivial error and closure $< 10^{-8}$; Hopf's Floquet vectors radial at every mesh point (constant $\omega$: the radial direction is invariant) |
| Winfree $a = 0.25$, reverse time | $\kappa_u = a(1 - a)$ | all three routes report the forward-time $+\kappa_u$, $\mu > 1$ |
| Cassini oval (`systems.cassini`) | $\kappa T = -\oint\Vert\nabla g\Vert\thinspace d\ell = -36.797$, $T = \oint d\ell/(s\Vert\nabla g\Vert) = 1.69509$ by the trapezoidal rule in $\theta$ (no ODE solve) | chain and Liouville to $10^{-8}$; $T$ to $10^{-8}$ (a solver-free oracle for O1's period); $\mu \approx 10^{-16}$ is beyond the explicit monodromy's reach — its eigenvalue is pure rounding while its trivial multiplier is still fine |
| Guckenheimer & Meloon's cubic | — | chain and Liouville agree; trivial error at the solver floor |
| FitzHugh–Nagumo (slow) | — | $\mu \in (10^{-13}, 10^{-11})$; chain vs Liouville to $10^{-7}$ relative; the monodromy's eigenvalue between $10^{-7}$ and $10^{-3}$ relative (the $\varepsilon/\mu$ floor, asserted as a band); the return-map eigenvalue and the trace of $DP$ wrong by more than 100 %; trivial errors $< 10^{-8}$ |
| Invariance | — | $\kappa$ independent of the mesh base point (rolled `x0s`) and of $m \in \lbrace 1, 2, 8, 16\rbrace$ |
| The chain | — | $Q_{i+1}R_i = M_iQ_i$ exactly, $R_i$ upper triangular with positive diagonal, the seed is $Q_1$'s first column, $D = I$, `monodromy` is the ordered product; `periodic_schur` under `vmap` |
| Refusals | — | `periodic_schur` on $3\times3$ factors: `NotImplementedError`; Liouville and `orientation` on a 3-D cycle: `ValueError` |

### 3.4 Limits and consequences

- **Planar only, by construction**: the seed fixes one Schur vector, which in the plane is all there is; `dim \geq 3` needs the periodic QR iteration on the transverse block (`slycot`) or BifurcationKit-style section-reduced products — open.
- **The orbit's error is the floor**: $\kappa$ inherits O1's $10^{-8}$; the trivial multiplier and the closure report it (Lust 2001, Figs. 5, 9: the trivial multiplier's error is the same with and without the periodic Schur decomposition — it is set by the discretization).
- **Per-segment contraction must stay above rounding**: Dieci et al.'s Remark 4.5 from the other side — $m$ large enough that $e^{\kappa T/m}$ is not tiny ($m \geq 2$ sufficed for FHN; the default $m = 8$ leaves margin).
- **The forward-direction stability argument** is deduced and measured, not cited.
- **For training**: with $T$ fixed by O1, $\mu = e^{\kappa T}$ is what a conjugacy must match; for the Bautin normal form $\kappa = \rho'(1)$ gives one equation in $(a, b)$ — the first place O2's number feeds the model side (roadmap M).

## 4. O3 — `analysis.ode.asymptotic_phase`: the phase field of the basin

**Status: survey (2026-10-20), delivered before implementation (ADR-0012 amendment). §4.2 and §5.2 are proposals; the decisions marked ▶ are Joon's. Conventions already fixed with Joon: the public phase is an angle $\Theta \in [0, 2\pi)$, $\Theta = 0$ at `cycle.x0s[0]` and increasing with time ($\Theta(\gamma(t)) = \omega t$, $\omega = 2\pi/T$); everything is computed in time along `LimitCycle.sample` and rescaled by $\omega$ at the end; the per-point function is vmappable and a `phase_grid` utility handles the grid (masking, chunking).**

**The problem.** Given a `LimitCycle` of a planar `AbstractODE`, compute the asymptotic phase $\Theta(x)$ of any $x$ in the basin — Winfree's definition, $\lim_{t\to\infty}\vert\Phi_t(x) - \Phi_{t + \Theta(x)/\omega}(x_0)\vert = 0$ (Winfree 1974; Langfield, Krauskopf & Osinga 2014 §II.A, *cited*) — to the accuracy of the orbit, with `nan` outside the basin and on the phaseless set, and cheaply enough for a grid. This is the field a learned model's phase is compared against (roadmap M; Yawata et al. 2024 Figs. 4 and 8 compare on exactly such a grid, *cited*). Claims are *cited*, *secondary* or *deduced* as in §2.

### 4.1 What exists

**The object.** Isochrons are the level sets of $\Theta$; Guckenheimer 1975 (Theorem A, Appendix A, *cited*) shows each isochron $I(c)$ is the stable manifold of $c \in \gamma$ under the time-$T$ map $\Phi_T$, a cross-section of $\gamma$ diffeomorphic to $\mathbb R^{n-1}$, with the isochrons permuted by the flow ($I(\gamma(t)) = \Phi_t(I(\gamma(0)))$) and their union the basin. Two consequences we use: $D\Phi_T(c)$ has the eigenvalue $1$ along $f(c)$ and the stable eigenvalues transverse to it, and "the Stable Manifold Theorem ensures that $I(c)$ is tangent at $c$ to the stable linear eigenspace $E^s(c)$ that is spanned by the $n - 1$ stable eigenvectors of $D\Phi_T(c)$" (Langfield et al. 2014 §II.B, *cited*) — so in the plane the isochron through a mesh point is tangent to O2's Floquet vector $v(c)$; and the phaseless set is the frontier of the basin (Guckenheimer Appendix C: for planar flows every point of the frontier is phaseless, Theorem C1, *cited*) — FitzHugh–Nagumo's source equilibrium, Winfree's hole $r \leq a$. Isochrons accumulate on the phaseless set, and for slow–fast systems also in an *almost phaseless* region along the repelling slow manifold, where the phase is "very sensitive, meaning that a slight perturbation from any point will result in a completely different phase" (Langfield et al. §IV, *cited*; Osinga & Moehlis 2010 §4.4 "almost phaseless set", *cited*). No method resolves the phase there; the grid utility must report it, not hide it.

**Routes to $\Theta(x)$.**

| Route | Computes | Where it stands | Cost / accuracy |
|---|---|---|---|
| (a) Forward integration, phase read off the cycle | $\Phi_{nT}(x)$ for $n$ periods, then the phase of the cycle point it lands beside | Winfree's original computation: "asymptotic phase from forward-time integrations", which "was not accurate enough to resolve the isochron structure" in the sensitive regions (Langfield et al. §I, §III, *cited*); Monga et al. 2019 Eq. (2) is this definition (*cited*); Yawata et al.'s "direct numerical calculations" for the FHN truth (§IV.C, *cited*) | one solve of length $nT$ per point, $n \approx \log(\varepsilon/d_0)/\log\mu$ (one period for FHN, $\sim 10^2$ near a Hopf point); the nearest-point read-off is only $O(d)$ accurate because isochrons are transverse, not orthogonal, to $\gamma$ (*deduced*) |
| (b) Return times to a section | iterate the one-sided `PoincareMap` from $x$: $t_n - nT \to c(x)$ and $\Theta(x) = -\omega\thinspace c(x)$ | Guckenheimer 1975 §1: "the existence of isochrons is equivalent to the existence of cross-sections to $\gamma$ for which the time of first return is identically the period" (*cited*) — the offset $c(x)$ is what the section through $p$ misses; the return-time formulation is ours (*deduced*), on O1's map | the same solves as (a) split at the crossings; converges geometrically at rate $\mu$, the remainder after $n$ returns is $O(\vert y_n - p\vert)$; `nan` when no crossing within `t_max` |
| (b′) (b) with a first-order correction at the section | $\Theta(x) = -\omega t_n + \langle Z(p), y_n - p\rangle$ | $\Theta(y_n) = \Theta(p) + \nabla\Theta(p)\cdot(y_n - p) + O(\vert y_n - p\vert^2)$ with $Z(p)$ from O4 (*deduced*) | error $O(\vert y_n - p\vert^2) = O(\mu^{2n})$: halves the number of returns |
| (c) Fourier averages | $\Theta(x) = \arg\lim\frac1\tau\int_0^\tau g(\Phi_t x)e^{-i\omega t}dt$ for almost any observable $g$ | Mauroy & Mezić 2012 (Chaos 22:033112), Eq. (2), Prop. 1: the average is the Koopman eigenfunction of eigenvalue $i\omega$ and "the isochrons are the level sets of the Fourier averages" (*cited*) | forward integration only, "not limited by numerical issues", but convergence is $O(1/\tau)$ — they use 20–300 periods (*cited*); a quadrature state in our augmented-solve style |
| (d) Backward integration of a local isochron | extend the linear isochron $E^s(c)$ by the reversed flow | Sherwood & Guckenheimer 2010; Guillamon & Huguet 2009 (*secondary* via Langfield et al. §I and Osinga & Moehlis §3); Winfree's second method | gives *curves*, not a field; "requires exponentially small variations in the initial condition", impossible in double precision for slow–fast systems (Osinga & Moehlis §3, *cited*) |
| (e) BVP continuation | orbit segments ending on $E^s(c)$, continued in AUTO | Osinga & Moehlis 2010 (SIADS 9:1201); Langfield et al. 2014 (the FHN isochrons, "Winfree's puzzle") (*cited*) | the method of record for isochron *geometry* in sensitive regions; a continuation code, not a field evaluator |
| (f) Parametrization method | $K(\theta, \sigma)$ with $\partial_\theta K/T + \lambda\sigma\partial_\sigma K = X(K)$ (Eq. 9), the conjugacy of the basin to $\dot\theta = 1/T$, $\dot\sigma = \lambda\sigma$ (Eq. 10); $\Theta(x) = \theta$ and $\Sigma(x) = \sigma$ where $x = K(\theta, \sigma)$ (Eqs. 13, 15); $\nabla\Theta, \nabla\Sigma$ as the rows of $[DK]^{-1}$ (Eq. 23) | Guillamon & Huguet 2009 (SIADS 8:1005), Huguet & de la Llave 2013 (SIADS 12:1763) for planar systems; Pérez-Cervera, Seara & Huguet 2020 (Chaos 30:083117) in $d \geq 3$ (*cited*; see §4.5) | $K$ as a Fourier–Taylor series to order $L$ in $\sigma$, valid on a domain $\Omega_{\rm loc}$ where the invariance residual is $< E_{\rm tol}$ (their §III.C); $\Theta$ anywhere by flowing *into* $\Omega_{\rm loc}$ and inverting $K$ (*deduced* from Eq. 12 — their globalization runs the other way, backward, to draw curves); the same $K$ gives $\Sigma$ (O5) and both response functions (O4, O6): the shared engine of §4.5 |
| (g) Eikonal / level-set PDE | $\nabla\Theta\cdot f = \omega$ solved on a grid | Detrixhe, Doubeck, Moehlis & Gibou 2016 (*secondary*, cited by Pérez-Cervera et al.) | grid-native; the PDE has infinitely many solutions without the asymptotic condition (Brown et al. 2004 §2.2.2, *cited*) |

**What the learned-model comparison needs** is a field on a grid with the sensitive regions marked, not isochron curves: (a)/(b) with $n$ chosen from $\mu$, vmapped. The normal forms give the field in closed form (`phase`, with its $\Psi(r)$ shift for $\omega(r) \neq$ const); FHN's truth is (a) run to convergence, which is what Yawata et al. did and what Langfield et al. show qualitatively (their Figs. 3–4: the isochrons, the spiral accumulation at $q$, the boomerang turns near the repelling slow manifold).

### 4.2 Proposed method (`analysis/ode/phase.py`)

- `asymptotic_phase(cycle, x, *, n_returns=None, t_max=None, throw=False) -> Θ`: route (b′) — which is route (f) truncated at $L = 1$ (§4.5), so the function can later take a higher-order $K$ without changing its contract. The map is `PoincareMap(cycle.ode, cycle.section, cycle.config, t_max, reverse_time=cycle.reverse_time)`; $n$ returns by `iterate` (static $n$, so a `lax.scan`; default from $\mu$: $n = \lceil\log(\varepsilon_{\rm tol}/d_0)/\log\mu\rceil$ with $d_0$ a basin-scale guess, capped), then $\Theta = -\omega\sum_i(\tau_i - T) + \langle Z(p), y_n - p\rangle$, reduced mod $2\pi$; `nan` where a return is missing (`throw=False`, matching `PoincareMap`). Vmappable; jit left to the caller. ▶ The correction term ties O3 to O4 (`phase_sensitivity` at $p$ is one number per cycle, computed once); without it the function is (b), one order slower in $n$ — include or not?
- `phase_grid(cycle, xs, ys, *, batch_size, **kw) -> (Θ (ny, nx), mask)`: builds the grid, pre-masks by `ode.in_basin`, runs `asymptotic_phase` through `jax.lax.map(..., batch_size=)` over the flattened points (chunked vmap; the solves per point are the cost, $\sim 4\cdot10^4$ for $200\times200$), masks non-finite results, returns the field and the mask. ▶ Its home: `analysis.ode` (the field is an analysis product) or next to the plot code.
- **Reverse time**: for a repelling cycle the phase field of the *reversed* flow is what exists (the forward basin is the complement); `reverse_time` cycles get the reversed map and $\Theta$ of the reversed flow, documented.
- **Sensitivity diagnostic**: $\Vert\nabla\Theta(x)\Vert$ by `jax.grad` through the custom JVP, one tangent per point; `phase_grid` flags the almost-phaseless set $\lbrace\vert\nabla\Theta\vert > \tau\rbrace$ (Detrixhe et al. 2016's definition, §4.6). ▶ on by default, or on request (it doubles the cost of a grid).
- Rejected for now: (c) Fourier averages ($O(1/\tau)$; only wins in high dimension), (d)–(e) (curves, continuation codes), (f)–(g) (O5/O6 or a grid PDE we do not need).

### 4.3 Oracles (`tests/test_analysis_ode.py`, O3 block)

| System | Exact | Pinned |
|---|---|---|
| Hopf, Bautin, Winfree ($a < 0$), `integration="cartesian"` | `nf.phase(x)` (Bautin and Winfree carry the $\Psi(r)$ shift: isochrons are *not* radial) | $\Theta(x) - $ `nf.phase(x)` $=$ const mod $2\pi$ on a set of basin points, to $10^{-8}$; the constant is $-$`nf.phase(x0s[0])` |
| Any cycle | $\Theta(\Phi_t x) = \Theta(x) + \omega t$ | the flow invariance on random points and times; $\Theta(\gamma(t)) = \omega t$ on `sample` |
| Winfree $a = 0.25$ | the hole is phaseless | `nan` inside $r \leq a$, finite outside; the map's `t_max` respected |
| FitzHugh–Nagumo (slow) | — | $n = 1$ suffices ($\mu = 1.7\cdot10^{-12}$); flow invariance; `nan` at the equilibrium; the sensitivity diagnostic large along the repelling slow manifold (Langfield et al. Fig. 5, qualitative) |
| `phase_grid` | — | the mask equals `in_basin` ∧ finite; chunking changes nothing; shape |

### 4.4 Limits

- No isochron *curves*; the field near the almost-phaseless set is reported with its sensitivity, not resolved (that is (e)'s job, not planned).
- One section for the whole basin: a trajectory's $n$-th positive crossing of the *global* hyperplane $\Sigma$ is the $n$-th return near $p$ only once it is close to a convex-enough cycle; a Cassini-like cycle (period-2 map) would miscount — planar convex cycles only, with the period-$k$ detection of O1 as the guard (`k = 1` asserted).

### 4.5 The parametrization method as the shared engine for O3–O6 (Pérez-Cervera, Seara & Huguet 2020, read in full)

**What it is.** $K(\theta, \sigma)$ conjugates the basin to the linear flow $\dot\theta = 1/T$, $\dot\sigma = \lambda\sigma$ (Eqs. 9–10): $\Phi_t(K(\theta, \sigma)) = K(\theta + t/T, e^{\lambda t}\sigma)$ (Eq. 12). In the plane $\sigma$ is one amplitude and $\lambda = \kappa$ (their $\lambda_1$ is O2's exponent). $K$ is sought as $\sum_m K_m(\theta)\sigma^m$ (Eq. 27) with $2\pi$-periodic coefficients held as $N$ Fourier modes. Their §III.B, step by step, against what O1–O2 already provide:

| Step | Their computation | Ours |
|---|---|---|
| $m = 0$ | $K_0 = \gamma$: Poincaré section + Newton, then the orbit and the variational matrix $\Phi(\theta T)$ stored at $N$ equispaced $\theta$ (FFT between real and Fourier space) | **done**: `find_limit_cycle`, `LimitCycle.sample(N)`; $\Phi(t_j)$ by `jacfwd(flow_to)` per sample (or O2's factors chained) |
| $m = 1$ | $K_1(\theta) = \Phi(\theta T)\thinspace e^{-\lambda\theta T}\thinspace v$, $v$ the Floquet vector of the monodromy (Eq. 30); the scale $b$ of $v$ is free and sets the series' numerical range (Remark 3.2, §III.C, Table II) | **done up to the scale**: `floquet(cycle).vectors` is $K_1$ at the mesh points; between them, transport as in §5.1(e) |
| $m \geq 2$ | the homological equations (31): $\frac1T K_m' + m\lambda K_m = DX(K_0)K_m + B_m$, with $B_m$ the degree-$m$ Taylor coefficient of $X(K(\theta, \sigma))$ in $\sigma$, obtained by *Taylor-series automatic differentiation* (Appendix B); solved through the Floquet normal form $\Phi(t) = Q(t)e^{tR}$ (§III.A.1, Appendix A), which makes the equation diagonal in Fourier space, $u_k^{(j)} = A_k^{(j)}/(2\pi ik/T + m\lambda - \lambda_{j-1})$ (Eq. 43) | **not done; every ingredient exists in the stack**: $B_m$ is `jax.experimental.jet` (truncated Taylor-mode AD of $X$ along the one-dimensional series $K(\theta, \cdot)$ — exactly their Appendix B, in the plane the univariate case `jet` covers); $Q(\theta)$ from $\Phi(\theta T)$ and O2's $(1, \mu)$ by Eq. A4; the Fourier solve is `jnp.fft` and one division; the non-resonance condition (Remark 3.3) is automatic in the plane ($\lambda < 0$, $m \geq 2$) |
| accuracy and domain | $N$ by the tail criterion (47), $L$ and $b$ by experiment so that the monomials stay $O(1)$; $\Omega_{\rm loc}$ by the residual $E(\theta, \sigma)$ of the invariance equation (48–49) | the residual is one `jet`/`jacfwd` evaluation of Eq. 9 on a $(\theta, \sigma)$ grid |
| inversion | not needed for their curves (they evaluate $K$) | $\Theta(x), \Sigma(x)$ by a $2\times2$ Newton on $K(\theta, \sigma) = x$ inside $\Omega_{\rm loc}$ (optimistix), seeded by the nearest sample and $\sigma = 0$ |
| globalization | backward from $\Omega_{\rm loc}$, along the slow manifold, to draw isochrons and isostables (Algorithms 3.5, 3.8); response functions carried by the adjoint equations (24) | for a *field* on a grid, forward instead: flow $x$ for $nT$ until $\Phi_{nT}(x) \in \Omega_{\rm loc}$ (one period for FHN), invert $K$ there, then $\Theta(x) = \Theta(\Phi_{nT}x)$, $\Sigma(x) = e^{-\lambda nT}\Sigma(\Phi_{nT}x)$ (Eq. 12 read backward; *deduced*) — the read-off error is $O(\sigma^{L+1})$ instead of route (a)'s $O(\sigma)$, so $L$ trades against $n$ |

**Effort.** With `jet`, `fft`, `jacfwd` and optimistix in hand, the planar $K$ to order $L$ is one module of roughly the size of `floquet.py`: the homological solve is a dozen lines once $Q(\theta)$ is formed; the engineering is in the three tuning knobs ($N$, $L$, $b$) and in the residual-based $\Omega_{\rm loc}$ test, which they set by experiment (§III.C, Table II: for their slow–fast models the monomials decay to zero by degree 5, so $L > 5$ adds nothing and $\Omega_{\rm loc}$ stays thin — expect the same for FHN, where forward flow into $\Omega_{\rm loc}$ then does most of the work anyway; for a near-Hopf Bautin cycle the series is the win). Their timing: $L = 10$, $N = 2048$ in about 10 s on a laptop. A realistic split: $L = 1$ now (O3's (b′): everything is already computed), the series engine as its own item between O4 and O5, since O5/O6 are where it pays twice.

**Why it is the right long-term engine.** One object yields $\Theta$ (O3), $\nabla\Theta$ (O4), $\Sigma$ (O5), $\nabla\Sigma$ (O6) and the isochron/isostable curves — all from one conjugacy $K$, which is also exactly what the learned INN is supposed to be: the comparison of the trained model with $K$ is the comparison of two conjugacies to the same linear normal form, not of a model with a stack of separately computed curves. The alternatives for the later items, for the record: Koopman/Laplace averages for $\Sigma$ (Mauroy & Mezić 2018, Chaos 28:073108 §III: forward integration only, $O(1/\tau)$ convergence, and the amplitude eigenfunction needs the Floquet exponent fed in); the direct exponential fit of the distance to the cycle (roadmap O5's first plan: cheap, $O(\sigma^2)$-biased by the isochron geometry); the return-map eigencoordinate of Wilson & Moehlis 2016 Eq. 5 (first order in the distance to the $\theta = 0$ isochron, i.e. $L = 1$ again); the Eikonal solver of Detrixhe et al. 2016 for isochron fields in high dimension (grid PDE, not planar-specific). None of them gives the response functions off the cycle; (f) does.

**On the worry that direct integration is noisy.** The error of route (b) at a basin point is the solver's, not random: the return times carry the solver's global error ($\sim10^{-9}$ at the shooting tolerances), the event location is Newton-polished, and the remainder after $n$ returns is $O(\mu^n)$; the result is a smooth function of $x$ to $\sim10^{-8}$ in angle, as the O1 tests show for the return map itself. What *looks* like noise on a grid is real: in the almost-phaseless regions the true $\Theta$ varies by $2\pi$ across a grid cell, and no method — (e) included, whose authors compute the isochron *curves* there precisely because the field is unresolvable (Langfield et al. §IV) — turns that into a smooth picture. Route (f) does not change this (its $\Omega_{\rm loc}$ excludes those regions, and flowing into it is the same integration); what it changes is the read-off accuracy near the cycle and the number of periods needed when $\mu$ is close to 1.

### 4.6 A second sweep: approaches outside the first search

Done to guard against lock-in; what each adds, and whether it changes the plan.

| Approach | Reference | What it is | Bearing on us |
|---|---|---|---|
| Eulerian / fast-sweeping solution of the phase PDE | Detrixhe, Doubeck, Moehlis & Gibou 2016, SIADS 15:1501 (*cited*) | $\nabla\Theta\cdot F = 2\pi/T$ as a static Hamilton–Jacobi boundary-value problem, $\Theta$ prescribed on a thin band around $\gamma$, Godunov upwind differences on a uniform grid, fast sweeping ($O(N)$ claimed), "all isochrons are known implicitly"; first-order accurate ($\approx 10^{-3}$ against Osinga & Moehlis's continuation on the 2-D HH model), "$\theta$ will not converge near the phaseless set"; examples up to 4-D HH on a $160^4$ grid | the one method that *is* a grid field by construction, but at $10^{-3}$ with $O(\Delta x)$ convergence — two to five orders short of what an oracle needs. Their definition of the **almost phaseless set**, $A = \lbrace x : \vert\nabla\Theta(x)\vert > \tau\rbrace$, is the right diagnostic for `phase_grid` (adopted; it was my ▶ "sensitivity diagnostic", now with a citation) |
| Oscillator phase noise: the perturbation projection vector | Demir, Mehrotra & Roychowdhury 2000, IEEE TCAS-I 47:655 (*secondary*); Demir & Roychowdhury 2003, IEEE TCAD 22(2), DOI 10.1109/TCAD.2002.806599 (*cited*, abstract) | the circuit community's name for $Z$: the PPV, "the periodic function that links noise sources to oscillator phase noise", classically from "the monodromy matrix with time-domain simulation and then full or partial eigenanalysis", improved to "a single linear solution of the oscillator's time- or frequency-domain steady-state Jacobian matrix" — the shooting or harmonic-balance Jacobian — which "can be more accurate than explicit eigenanalysis" and avoids "having to pick the correct eigenfunction from a potentially large set of candidates" | an independent, twenty-year-old precedent for §5.1(c): $Z$ as the null vector of the transposed shooting Jacobian rather than an eigenvalue problem on the explicit monodromy. Our periodic Schur factors give the same thing with the conditioning already handled; worth citing as the origin |
| Higher-order *local* isochron expansions | Takeshita & Feres 2010, Nonlinearity 23:1303; Suvak & Demir 2010, IEEE TCAD 29:1215 (*secondary*: abstracts) | Taylor expansions of $\Theta$ about $\gamma$ to second and higher order ("quadratic approximations for the isochrons … and accurate phase computations") | subsumed by the parametrization method (§4.5), which is the same expansion organized as a conjugacy with the amplitude included |
| Floquet normal form in Fourier space, rigorously | Castelli, Lessard & Mireles James 2015, SIADS 14:132 (*secondary*); Castelli & Lessard, *Rigorous numerics in Floquet theory*, arXiv 1112.4874 | $\Phi(t) = Q(t)e^{tR}$ with $Q$ computed as a Fourier series from the periodic linear system directly (and validated), rather than from an integrated $\Phi(\theta T)$ | the alternative route to the $Q(\theta)$ of §4.5 if forming it from `jacfwd(flow_to)` at $N$ phases turns out noisy for strongly contracting cycles (the same $\varepsilon/\mu$ issue as O2, now along the orbit): a spectral solve of a $2\times2$ periodic linear ODE |
| Newton on the homological equations | Wilson 2020, Phys. Rev. E 101:022220 (*secondary*, via Pérez-Cervera et al. §VI) | the same expansion of $K$ (and of $\nabla\Theta$, $\nabla\Sigma$ in the amplitude) with each periodic coefficient found by a Newton iteration instead of the diagonal Fourier solve | a fallback if the Floquet-normal-form solve misbehaves; Pérez-Cervera et al. argue theirs is cheaper ($O(N\log N)$ per order) |
| Isochrons as invariant curves of the time-$T$ map | Castejón & Guillamon 2020, CNSNS 81:105008 (*secondary*) | Newton/continuation for invariant curves and Arnold tongues of periodically forced oscillators, in phase–amplitude coordinates | downstream (forced oscillators), not O3 |
| Mean-return-time isophases | Schwabedal & Pikovsky 2013, Phys. Rev. Lett. 110:204102 (*cited*, arXiv 1301.6852); Thomas & Lindner 2014, Phys. Rev. Lett. 113:254101 (*secondary*) | for *stochastic* oscillators, the isophase is the section "for which the mean first return time" equals the period; "in the noise-free case, isophases are also isochrones"; computed by iteratively deforming a section until all return times are equal | the stochastic sibling of route (b) — the deterministic statement is Guckenheimer's cross-section characterization — and the natural definition for the data half if noisy trajectories ever matter (D-items) |
| Data-driven phase/amplitude functions | Namura, Takata, Yamaguchi, Kobayashi & Nakao 2022 (arXiv 2203.01663, Phys. Rev. E): polynomial regression as a convex problem; Takata, Kato & Nakao 2025 (arXiv 2501.09340): EDMD; Yawata et al. 2024: the autoencoder (*cited*, abstracts) | $\Theta$ and $\Sigma$ fitted from time series without the equations, as Koopman eigenfunctions (EDMD) or by regression | not oracles — they are *competitors* of the learned INN, and the two Nakao-group methods are the baselines M should report next to Yawata's |
| Cell-mapping / set-oriented and Galerkin discretizations of the Koopman operator | (not found applied to isochrons beyond EDMD; Mauroy & Mezić 2012 §I note no efficient high-dimensional method at the time) | — | nothing to adopt |

**Verdict.** Nothing displaces the plan. Three things are adopted from the sweep: Detrixhe et al.'s almost-phaseless-set criterion $\vert\nabla\Theta\vert > \tau$ as `phase_grid`'s mask-and-flag diagnostic; the PPV literature as the citation for computing $Z$ from the shooting structure rather than an eigenanalysis; and Castelli et al.'s spectral $Q(\theta)$ as the fallback for §4.5. One thing is a caution: Detrixhe's first-order field is what a PDE-on-a-grid approach buys, so the field must come from integration (or from $K$), with the grid only as the sampling.

## 5. O4 — `analysis.ode.phase_sensitivity`: the PRC $Z(\theta) = \nabla\Theta$ on the cycle

**The problem.** $Z(\theta) = \nabla\Theta(\gamma(\theta))$, the infinitesimal phase response curve (phase sensitivity function), normalized by $\langle Z, f\rangle = \omega$ (Brown, Moehlis & Holmes 2004 Eq. 2.4/A.6; Nakao 2016 Eqs. 12–13 with $Z\cdot dX_0/d\theta = 1$, the same normalization in angle, *cited*): what enters every phase reduction $\dot\theta = \omega + Z(\theta)\cdot G$, and the second thing a learned model's phase is assessed on (Yawata et al. Figs. 5, 9, 12, 15: estimated vs "true PSF calculated by solving the adjoint equation", *cited*).

### 5.1 What exists

| Route | Computes | Where it stands | Cost / accuracy |
|---|---|---|---|
| (a) Direct (finite perturbation) | $\Delta\Theta/\Delta x$ for a small kick at $\gamma(\theta)$, measured after the transient | Winfree; Brown et al. §2.2.1 (*cited*): "the difference $\Delta\theta$ is preserved under the flow", so measure it as $t \to \infty$; Monga et al. §2.1 Fig. 1 compares it with the adjoint (*cited*) | $O(\Delta x)$ bias, one O3 evaluation per kick; the sanity check every paper shows |
| (b) Adjoint equation | $\dot Z = -Df(\gamma(t))^{\top}Z$, the $T$-periodic solution with $\langle Z(0), f(\gamma(0))\rangle = \omega$ | Ermentrout & Kopell 1991; Hoppensteadt & Izhikevich 1997 Eqs. 9.16–17; Brown et al. App. A, Eqs. A.4–A.7 (*cited*: $\nabla\theta = \omega Q_{\rm XPP}$); Monga et al. Eqs. 10–11 (*cited*); Izhikevich 2007 ch. 10 (Malkin's theorem, *secondary*) | XPP integrates it *backward*. Nakao 2016 (Contemp. Phys. 57:188, Appendix 5, *cited*): forward in time the adjoint equation "is linearly unstable … because of the minus sign before $J^{\dagger}$", the periodic solution "is a neutrally stable mode and therefore easily buried in other unstable modes that grow quickly", whereas backward "only the neutrally stable periodic mode remains, because all other modes are now linearly stable and decay quickly" — "by recording the oscillator state $X_0(\theta)$ on $\chi$ for one period of oscillation, and by evolving the adjoint equation backward in time using the recorded oscillator state for a long time, $Z(\theta)$ can be obtained" (his Fig. E1 is a C program doing exactly this for FHN). In our terms: the adjoint's multipliers are $\lbrace 1, 1/\mu\rbrace$, contamination decays by $\mu$ per backward period (*deduced*); the recorded orbit is a dense `Dopri8` interpolant, since the orbit itself repels backward and cannot be co-integrated |
| (c) Left eigenvector of the monodromy | $Z(c)^{\top}M(c) = Z(c)^{\top}$, i.e. $Z(c)$ is the left eigenvector of $D\Phi_T(c)$ for the eigenvalue $1$ | $\Theta\circ\Phi_T = \Theta$ differentiated (*deduced*); Kvalheim & Revzen 2021 Prop. 5 for the eigenfunction $e^{i\Theta}$ with exponent $i\omega$ (*cited*); Osinga & Moehlis §2.1: "in higher dimensions … the left eigenvector for the eigenvalue 1, found by the power method" (*cited*) | one linear-algebra step per point once $M(c)$ is known — and O2's periodic Schur chain *has* $M(c)$ in triangular form at every mesh point: $Z \propto Q_i\thinspace(1, T_{01}/(T_{00} - T_{11}))$ from the cyclic product $T_i$ of the factors (*deduced*; the planar triangular left eigenproblem) |
| (c′) Planar closed form from the Floquet vector | $Z \perp v$ (the isochron is tangent to $E^s = \mathrm{span}\thinspace v$), so $Z = \omega\thinspace v^{\perp}/\langle v^{\perp}, f\rangle$ | Langfield et al. §II.B for the tangency (*cited*); the rest *deduced* (biorthogonality of left and right eigenvectors of $M$) | the same as (c) in the plane, written with O2's `vectors` |
| (d) AD of the phase function | $\nabla\Theta(x)$ by `jax.grad` of `asymptotic_phase` through the custom JVP of the return map | Brown et al. §2.2.2: the adjoint "is equivalent to the direct method" (*cited*); AD makes the direct method exact in $\Delta x$ (*deduced*); Pérez-Cervera et al. Eq. 23: $\nabla\Theta$ as a row of $[DK]^{-1}$ — the gradient of the inverse parametrization (*cited*) | one tangent per point, on *or off* the cycle (the phase response *function* of Pérez-Cervera et al., Eq. 20–21); its accuracy is O3's |
| (e) Transport along the cycle | $Z(\Phi_t c) = D\Phi_t(c)^{-\top}Z(c)$ | $Z(x)^{\top} = Z(\Phi_t x)^{\top}D\Phi_t(x)$ from $\Theta\circ\Phi_t = \Theta + \omega t$ (*deduced*; the integrated form of (b)) | forward solves only; a $2\times2$ solve per point; conditioning $e^{\vert\kappa\vert t}$ — fine from the nearest mesh point ($t \leq T/m$), useless over a full period ($1/\mu$) |

**Conventions.** $Z\cdot f = \omega$ fixes the scale (Brown et al. A.6; XPP's $Q$ is $Z/\omega$, A.7); $\theta = 0$ at $x_0$; sign conventions of the components follow the state coordinates (Yawata's $Z_1, Z_2$). For the Stuart–Landau/Hopf form with shear $\beta$, $Z(\theta) = (-\sin\theta - \beta\cos\theta,\ \cos\theta - \beta\sin\theta)$ (Yawata et al. Eq. 31, *cited*) — the radial component is the shear; our Bautin and Winfree have $\omega(r) \neq$ const and hence a nonzero radial PRC, Hopf's is tangential.

### 5.2 Proposed method (`analysis/ode/phase.py`, same module as O3)

- `phase_sensitivity(cycle, n_points) -> (θ (n,), Z (n, dim))` by route (c) at the mesh points (from `floquet_periodic_schur`'s factors, with the normalization $\langle Z, f\rangle = \omega$) and route (e) from the nearest mesh point to each of the `n_points` uniform times (forward solves with `ForwardMode`, a `vmap`). Exact in the sense of the discrete flow, no backward integration, no interpolant. Output in angle: $\theta_j = \omega t_j$, $Z$ scaled by $\omega$.
- `phase_response(cycle, x) -> ∇Θ(x)`: route (d), `jax.grad(asymptotic_phase)` — the response *function* anywhere in the basin; on the cycle a cross-check of (c)+(e) to O3's accuracy.
- Cross-checks in tests, not in the API: (b) by backward integration over $k$ periods on a dense `Dopri8` interpolant of the orbit — Nakao's Appendix 5 recipe (▶ keep as a test-only reference, or expose as `method="adjoint"`?); (a) finite perturbation at a few points.
- ▶ The $Z(p)$ needed by O3's correction (b′) is `phase_sensitivity` at mesh point 0 — computed once per cycle; where to cache it (a field on `Floquet`? recomputed?) is a design point.

### 5.3 Oracles (O4 block)

| System | Exact | Pinned |
|---|---|---|
| Hopf | $Z(\theta) = (-\sin\theta, \cos\theta)\thinspace\omega/r$ at $r = 1$: tangential | (c)+(e) and (d) to $10^{-8}$; $\langle Z, f\rangle = \omega$; $\langle Z, v\rangle = 0$ |
| Bautin, Winfree ($a < 0$) | `jax.grad(nf.phase)` on the cycle (nonzero radial component from $\omega'(r)$) | same, to $10^{-8}$; the radial component's sign against $\omega'(1)$ |
| Any cycle | $\langle Z(\theta), f(\gamma(\theta))\rangle = \omega$ for all $\theta$; $Z$ is $2\pi$-periodic; $Z$ is $\perp$ O2's Floquet vector at every mesh point | identities |
| FitzHugh–Nagumo (slow) | — | (c)+(e), (d) and the backward adjoint (b) agree to $10^{-6}$; the direct method (a) within $O(\Delta x)$; shape against Yawata et al. Fig. 9 and Nakao 2016 Fig. 3(c) only qualitatively (both use $\varepsilon = 0.08$, $a = 0.7$, $b = 0.8$, $I = 0.8$, $T = 36.56$; ours is FitzHugh's $c = 3$ form, $T = 11.23$ — a test with *their* parameters, where Nakao's Fig. E1 code is the reference, is cheap and worth adding) |
| Conditioning | — | transport (e) over $T/m$ vs over $T$: the latter loses $\log_{10}(1/\mu)$ digits (asserted on FHN) |

### 5.4 Limits and consequences

- Planar in the closed form (c′) only; (c), (d), (e) are dimension-free, (c) needs the general periodic Schur for `dim > 2` (ADR-0015 open issue).
- $Z$ is as accurate as the orbit and the segment Jacobians ($10^{-8}$); near the almost-phaseless set, `phase_response` off the cycle inherits O3's sensitivity.
- For training (roadmap M): the PRC error in the first $k$ Fourier modes and the residual $\langle Z, f\rangle - \omega$ of the *learned* phase are now computable against this $Z$.

## 6–. O5–O6 and the data items (D2–D3)

Written as each item lands (roadmap Phase E): the parametrization engine of §4.5 (proposed as its own item), then O5 `isostable`, O6 `isostable_response` (Mauroy, Mezić & Moehlis 2013; Wilson & Moehlis 2016; Shirasaka, Kurebayashi & Nakao 2017; Wilson & Ermentrout 2018; Pérez-Cervera, Seara & Huguet 2020), then D2/D3 on the data half. Each survey is delivered to Joon before implementation (ADR-0012, amendment).
