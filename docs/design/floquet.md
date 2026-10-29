---
type: design
status: tentative
updated: 2026-10-29
verified_by: pending (Joon); the O2 tests in tests/test_analysis_ode.py pass
sources:
  - "ADR-0012 (analysis layout)"
  - "ADR-0015 (Floquet by seeded periodic Schur)"
  - "roadmap Phase E"
  - "Lust 2001"
  - "Fairgrieve & Jepson 1991"
  - "Mauroy & Mezić 2018"
  - "Wilson & Moehlis 2016"
  - "Kvalheim & Revzen 2021"
  - "Giacomini & Grau 2005"
  - "Bojanczyk et al. 1992"
  - "Dieci et al. 1997"
  - "Geist et al. 1990"
  - "BifurcationKit documentation"
---

# `analysis.ode.floquet` — the Floquet exponent of the orbit (O2)

A method document in the form of [`document-conventions.md`](../document-conventions.md)
§ Design documents: the problem, what exists (the literature survey, with what we take
from it and what we do differently and why), the method as implemented, what is exact
and what approximated, the oracles (what the tests pin down and against which exact
answers) and the limits (what the function does not do, and which later item does).
Claims are labeled [cited] (read in the source), [cited via X] (read in X, which cites
it) or [deduced] (our inference from the source or from the mathematics). The analysis package as a whole is laid out by
[ADR-0012](../decisions/0012-analysis-layout.md); the other analyses are
[`cycle-estimation.md`](cycle-estimation.md) (E1), [`limit-cycle.md`](limit-cycle.md)
(O1), this document (O2), [`phase-field.md`](phase-field.md) (O3–O4) and
[`isochrons.md`](isochrons.md) (O7).

## 1. The problem

**The problem.** Given a `LimitCycle` of a planar `AbstractODE` (O1: the shooting mesh $x_1, \dots, x_m \in \gamma$, $T$), compute the nontrivial Floquet multiplier $\mu$ and exponent $\kappa = T^{-1}\ln\mu$ of $\gamma$ to the accuracy of the orbit itself, with its own error estimate; $\kappa$ is the rate at which the amplitude (isostable) coordinate decays, $\dot\psi = \kappa\psi$, and the quantity a learned conjugacy must reproduce: the normal forms carry it in closed form ($\kappa = \rho'(1)$, `floquet_exponent`), so $\kappa$ of FitzHugh–Nagumo pins the normal form's amplitude parameters (§6).

## 2. What exists

The survey was delivered on 2026-10-19 and the method implemented on 2026-10-20 after
Joon's prototype (ADR-0015); the two corrections the implementation forced are marked ⟂.

**The object.** The monodromy $M = \partial_x\Phi_T(x^\ast)$, the fundamental solution of the variational equation $\dot Y = Df(\gamma(t))\thinspace Y$, $Y(0) = I$, at $t = T$. Its eigenvalues are the Floquet multipliers; they do not depend on the point $x^\ast \in \gamma$ or on the coordinates; one of them is $1$ with eigenvector $f(x^\ast)$ (the trivial multiplier), and the orbit is elementary (hyperbolic) when $1$ is simple (Guckenheimer & Meloon 2000, §1, [cited]; classical Floquet theory, Hale 1980, §III.7 [cited via Kvalheim & Revzen 2021, §3.1]). The Jacobian of a return map $P$ on a section $\Sigma$ at its fixed point is the quotient of $M$ by the flow direction ("the Jacobian of the return map can be obtained as a quotient of the monodromy matrix by projecting out the flow direction", Guckenheimer & Meloon §1, [cited]), so $DP$ has the eigenvalues of $M$ except the trivial one, which $DP$ replaces by $0$ when written on the ambient space (O1's `jacfwd(P)`, eigenvalues $\lbrace 0, \mu\rbrace$, [deduced] in §2.2 and tested). Wilson & Moehlis 2016 define the isostable coordinates from exactly this $DP$ and its eigenvectors (their Eqs. 3–5, [cited]) and set $\kappa_i = \log(\lambda_i)/T$ (Eq. 7).

**Why it matters to this project.** The Floquet exponent is the Koopman eigenvalue of the principal eigenfunction transverse to the cycle: Kvalheim & Revzen 2021, Prop. 5 — for a principal Koopman eigenfunction $\psi$ with exponent $\mu_K$ on the basin of a $\tau$-periodic orbit, $D_{x_0}\psi$ is a *left* eigenvector of $D_{x_0}\Phi^\tau$ with eigenvalue $e^{\mu_K\tau}$, "in particular, $e^{\mu_K\tau}$ is a Floquet multiplier" [cited]. Mauroy & Mezić 2018 (Chaos 28, 073108) build the amplitude coordinate from $\Lambda_1 = \log(\mu_1)\thinspace\omega/2\pi$ with $\mu_1$ "the dominant eigenvalue (different from 1) of the monodromy matrix" (§IV, [cited]). The learned conjugacy $h$ between FitzHugh–Nagumo and a normal form transports the variational equation, so the multipliers of the two cycles must agree: $M' = Dh\thinspace M\thinspace Dh^{-1}$ for a $C^1$ conjugacy ([deduced], from the chain rule); under a time reparametrization (orbital equivalence) the return maps stay $C^1$-conjugate, so $\mu = e^{\kappa T}$ is still invariant while $\kappa$ and $T$ separately are not [deduced]. O2 therefore produces the one number that fixes the normal form's amplitude parameters given its period, a check on training as well as a diagnostic of the ODE.

**Routes to $\mu$.**

| Route | Computes | Where it stands | Cost / accuracy |
|---|---|---|---|
| (a) Monodromy by the variational equation, eigenvalues | $M$, both multipliers | the textbook method; Mauroy & Mezić 2018, §III.B integrate $\dot M = \partial_x F\thinspace M$ alongside the orbit [cited]; AUTO86's outside-the-code practice (Guckenheimer & Meloon §5.4, [cited]) | one $(n + n^2)$-dimensional solve; the error in the trivial multiplier $\vert\lambda_1 - 1\vert$ "is a measure for the accuracy of the computation of the periodic orbit and its monodromy matrix" (G&M §5.4, [cited]; $6\cdot10^{-15}$ in their Josephson example) |
| (a′) Monodromy by AD of the time-$T$ flow map | same $M$ | `jax.jacfwd` of `flow_to` with `diffrax.ForwardMode`: "discretise then optimise" forward mode that "operates through the internal numerics of the solver" (diffrax documentation, [cited]) — the exact derivative of the *discrete* flow, the property G&M credit for their accuracy (§6) | one forward-mode solve carrying $n$ tangents; equivalent to (a) with the solver's own discretization of the variational equation [deduced] <!-- codespell:ignore: cited quotation -->|
| (b) Product of segment Jacobians from multiple shooting | $M = M_m\cdots M_1$, $M_i = \partial_x\Phi_{T/m}(x_i)$ | BifurcationKit's `FloquetQaD` (default; "not numerically very precise for large / small Floquet exponents", [cited]) and the `multiflap` package [cited]; the factors are what the shooting Jacobian already contains | free after shooting; the product has the same conditioning problem as (a) |
| (c) Eigenvalue of the return-map Jacobian | $DP$, $\lbrace 0, \mu\rbrace$ | O1's custom JVP (Parker & Chua Eq. 5.21); BifurcationKit's Poincaré shooting: "the (non trivial) Floquet exponents are eigenvalues of the Poincaré return map" [cited] | the same solve as (a′) plus the projection; no trivial multiplier to carry |
| (d) Contraction of return-map iterates | $\mu$ as the ratio of successive distances to the fixed point | Parker & Chua §5.1.1 brute force read as a power iteration; O1's `FixedPoint.contraction` | linear convergence, polluted by the quadratic terms of $P$ until the iterates are close; a sanity check, not a result |
| (e) Liouville / Abel–Jacobi: $\det M = \exp\int_0^T\mathrm{tr}\thinspace Df(\gamma(t))\thinspace dt$ | in the plane, with $\lambda_1 = 1$ exactly: $\mu = \exp\int_0^T\mathrm{div}\thinspace f(\gamma(t))\thinspace dt$, i.e. $\kappa = T^{-1}\int_0^T\mathrm{div} f\thinspace dt$ | Andronov et al. (p. 118) and Perko 3rd ed. pp. 210–214, as cited by Giacomini & Grau 2005 (arXiv math/0505027, Theorem 3: $v\cdot DP(p_0) = \exp(\int_0^T\mathrm{div}\thinspace dt)\thinspace v$ for $v$ normal to $F(p_0)$, [cited] there); Mauroy & Mezić 2018, Remark 4 use it for planar systems, "$\lambda = (1/T)\int\mathrm{div} F(x_\gamma(t))\thinspace dt$" [cited] | a scalar quadrature along the orbit, in **log space**: no eigenvalue problem, no cancellation, exact in the plane |
| (f) Periodic Schur decomposition of $M_m\cdots M_1$ | all multipliers without forming the product | Fairgrieve & Jepson 1991, "O.K. Floquet multipliers" (SIAM J. Numer. Anal. 28:1446–1462) for collocation; Lust 2001 (Int. J. Bifurcation Chaos 11(9):2389–2410): common techniques "including AUTO97's, can be very inaccurate when Floquet multipliers are very large or very small", remedied by the periodic Schur decomposition ([cited], abstract); BifurcationKit's `FloquetPQZ` [cited] | the right tool in $n \geq 3$ with multipliers of very different magnitude; in the plane (e) already gives $\ln\mu$ directly |

**The conditioning problem, in our terms.** A relaxation oscillator is *strongly* attracting: for our FitzHugh–Nagumo ($\dot x = c(y + x - x^3/3 + z)$, $\dot y = -(x - a + by)/c$, $c = 3$) $\mathrm{div} f = c(1 - x^2) - b/c$ runs from $-8.9$ on the slow branches to $+2.7$ on the jumps, and the implementation gives $\kappa T = \int\mathrm{div} f\thinspace dt = -27.106$, $\kappa = -2.4142$, $\mu = 1.6906\cdot10^{-12}$ (⟂ the survey's estimate of $10^{-17}$ was off: the jumps recover more than it allowed for). The explicit monodromy has eigenvalues $1$ and $\mu$ in a matrix with $O(1)$ entries, so by Bauer–Fike $\mu$ is resolved only to the absolute rounding of the product, a relative error $\varepsilon\Vert M\Vert/\mu$: Lust 2001's Eq. 10, $d = \log_{10}(\max\vert\lambda\vert/\vert\mu\vert)$ digits lost [cited]. ⟂ The survey had claimed $O(1)$ relative error from the *solver's* $10^{-9}$; that was wrong: discretize-then-differentiate AD gives the exact Jacobian of the discrete flow, whose own small multiplier is carried multiplicatively, so the solver error lands on the trivial multiplier (period mismatch along $f$), not on $\mu$. Measured: the AD monodromy's eigenvalue agrees with Liouville to $6\cdot10^{-5}$, exactly $\varepsilon/\mu$, and the Cassini oval below, with $\mu \approx 10^{-16}$, is beyond its reach altogether. The same floor applies to the return-map Jacobian $DP$ (route (c)), and worse: its two eigenvalues are $0$ and $\mu$, both below the solver's *absolute* error on $DP$ ($\sim10^{-9}$), and its *trace*, algebraically $\mathrm{tr}\thinspace M - n^{\top}Mv/\langle n, v\rangle = (1 + \mu) - 1$, leaks the trivial multiplier's error additively ($3.5\cdot10^{-9}$ for $\mu = 1.7\cdot10^{-12}$ in Joon's prototype). Route (e) computes $\ln\mu$ with the quadrature's relative accuracy ($\sim10^{-9}$), and route (f), the subject of the next paragraph, with the per-segment factors' relative accuracy. For the normal forms ($\mu = e^{-2\cdot 2\pi/2} \approx 0.002$ for Hopf with $a = 1$, $w = 2$) every route is fine, and (a′)/(c) are tested in O1. Two side facts ([deduced] from (e)): in the plane $\mu = \det M > 0$ always, so a planar cycle has no negative multiplier (no period doubling of planar cycles, as Poincaré–Bendixson implies anyway); and the trivial-multiplier error of (a′)/(f) is a free, independent accuracy estimate of *the orbit and the solver together*, which (e) alone does not provide.

**Why the periodic Schur chain buys accuracy, and the planar shortcut.** Bojanczyk et al. 1992 (Theorem 1, Corollary 2) give orthogonal $Q_i$ with every $\hat A_i = Q_{i+1}^{\top}M_iQ_i$ upper triangular, cyclically, so the multipliers are $\prod_i \hat A_{i,jj}$: per-segment factors of moderate size ($\approx e^{\kappa T/m}$), each to relative accuracy, accumulated as a sum of logarithms, and the product $M$ is never formed. Lust 2001, §5: "the accuracy is related to the size of the diagonal elements of the upper triangular factors … and not to the size of the diagonal elements in the Schur decomposition of the product" [cited]; his Table 5, five factors with eigenvalues $10^{10}\ldots10^{-10}$: explicit product $3.5\cdot10^{4}$ relative error on the smallest, periodic Schur $2.9\cdot10^{-11}$. This is the discrete QR method for Lyapunov exponents (Dieci et al. 1997, §3: $Z_j = Q_{j+1} \thinspace R_{j+1}$ with positive diagonals, exponents from $\sum \log R_{k,ii}$; their Claim 2: for a periodic system it "computes the real parts of the Floquet exponents of the computed approximation to the monodromy matrix exactly", [cited]), with the re-orthonormalization that Geist et al. 1990, §1 call mandatory because unorthonormalized columns collapse onto the most expanding direction [cited]. Their general algorithm (periodic Hessenberg reduction by Givens rotations, implicit shifted periodic QR, deflation, reordering, Bojanczyk et al. §3; SLICOT `MB03VD`/`MB03WD`; 8 000 lines in `PeriodicSchurDecompositions.jl`, which BifurcationKit's `FloquetPQZ` calls on the shooting blocks with no use of the known eigenvector, *read in the source*) exists to *find* the Schur basis. Here one Schur vector is known analytically: the flow direction $f(x_1)$, eigenvector of the trivial multiplier and the *dominant* one for an attracting cycle. Seed $Q_1$ with it (Guckenheimer & Meloon's transformation of the monodromy to cross-section coordinates, §2 algorithm, is the same deflation) and Bojanczyk's constructive chain of QR factorizations (§2 proof) completes the decomposition with no iteration; in the plane the transverse block is $1\times1$. Two things the implementation taught ([deduced], checked numerically, no literature statement found): the chain must run *forward*, because it propagates the first Schur vector, which forward is the dominant direction and self-correcting, while backward through the transposes (Joon's first RQ chain) it is the subdominant one and the chain does not close ($\Vert Z_m - Q_1\Vert = 0.63$ on FHN); and the seed must be a seed only, because using the analytic frame at every mesh point instead of the QR's $Z_i$ leaves a sub-diagonal residue ($5\cdot10^{-11}$: the solver's end-point velocity is not exactly the field's) that the $O(1)$ shear in $R_{12}$ multiplies past $\mu$. Measured on FHN against Liouville: $m = 1$ at the $\varepsilon/\mu$ floor ($1.4\cdot10^{-5}$), $m \geq 2$ at the discretization floor ($10^{-9}$), closure $10^{-10}$. In the plane the three log-space routes coincide: the chain's $\prod(R_i)_{22}$, the product of segment determinants, and Liouville's integral as the $m \to \infty$ limit of the latter. For a *repelling* cycle $f$ is the subdominant direction of the forward monodromy, so the chain runs in the direction in which the cycle attracts (the reversed field for a `reverse_time` cycle) and the result is inverted (ADR-0015 §2).

**The level-set family as an oracle** (`systems.LevelSetCycle`, $f = s(R\nabla g - g\nabla g)$). On $g = 0$: $\mathrm{div}(R\nabla g) = \partial_x(-g_y) + \partial_y(g_x) = 0$ and $\mathrm{div}(g\nabla g) = \Vert\nabla g\Vert^2 + g\Delta g = \Vert\nabla g\Vert^2$, so $\mathrm{div} f\vert_{\gamma} = -s\Vert\nabla g\Vert^2$ and, with the speed $\Vert f\Vert = s\Vert\nabla g\Vert$ on $\gamma$,

$$
\kappa T = -s\int_0^T\Vert\nabla g(\gamma(t))\Vert^2\thinspace dt = -\oint_{\gamma}\Vert\nabla g\Vert\thinspace d\ell, \qquad T = \oint_{\gamma}\frac{d\ell}{s\thinspace\Vert\nabla g\Vert}
$$

([deduced]; consistent with the transverse equation $\dot g = -s\thinspace g\thinspace\Vert\nabla g\Vert^2$ of [`limit-cycle.md`](limit-cycle.md) §5, whose linearization along $\gamma$ is the same rate). Both are line integrals over the curve $\lbrace g = 0\rbrace$ that need **no ODE solve**: for the Cassini oval the curve has the polar parametrization $r(\theta)^2 = a^2\cos 2\theta + \sqrt{a^4\cos^2 2\theta - a^4 + b^4}$, and the trapezoidal rule on a periodic analytic integrand converges exponentially, so $\kappa$ and $T$ are available to roundoff as oracles independent of diffrax. (Guckenheimer & Meloon's cubic has no such parametrization; its oracle stays $\max\vert g\vert$ and the Cassini-derived checks.)

**What the packages do** ([cited] from documentation): BifurcationKit uses `FloquetQaD` on the product of shooting Jacobians by default, `FloquetPQZ` (periodic Schur) for accuracy, and the return-map eigenvalues for Poincaré shooting; AUTO takes the multipliers from the collocation Jacobian (Fairgrieve & Jepson), improved from AUTO86 to AUTO97 (G&M §5.4); `multiflap` takes the product of segment Jacobians through `numpy.linalg.eig`. None of them exploits the planar Liouville identity, because none is planar-specific.

## 3. The method (`analysis/ode/floquet.py`, ADR-0015)

Functions over a `LimitCycle` (which, as of O2, is a fixed-shape `eqx.Module` carrying its `ode`, the shooting mesh `x0s`, `period`, `config`, `reverse_time` and a light `FixedPoint`; analysis functions take the cycle — ADR-0015 §5):

- `segment_jacobians(cycle) -> (m, dim, dim)`: $M_i = \partial_x\Phi_{T/m}(x_i)$ by `vmap(jacfwd(flow_to))` with `ForwardMode`, in the time direction in which the orbit attracts.
- `periodic_schur(jacobians, q1) -> PeriodicSchur(R, Q, signs)`: the forward QR chain seeded with `q1` as a `lax.scan`; positive diagonals (rows of $R$ and columns of $Q$ flipped together — Joon's prototype flipped columns of $R$, invisible on the diagonal, wrong for the vectors); `signs` $= \mathrm{diag}(Q_1^{\top}Q_{m+1})$ rounded; `multipliers` $= D\exp\sum\log\mathrm{diag}R_i$, `closure` $= \Vert Q_{m+1} - Q_1D\Vert$. `NotImplementedError` for `dim > 2` (the transverse block would need the periodic QR iteration; `slycot.math.mb03wd` wraps SLICOT's).
- `monodromy(jacobians)`, `multipliers_from_monodromy(M)` (eigenvalues, the one closest to 1 first), `return_map_multiplier(pmap, x)` (the eigenvalue of `jacfwd(P)`, not its trace): the diagnostics, with their floors in the docstrings.
- `divergence_integral(cycle)`: the augmented solve $(\dot x, \dot z) = (f, \mathrm{tr}\thinspace Df)$ through `systems.base.diffeqsolve` at `cycle.config`; planar only (`ValueError`).
- `floquet_periodic_schur`, `floquet_monodromy`, `floquet_liouville`: each `(cycle) -> Floquet` with `multipliers`, `exponents` (forward-time; index 0 trivial), `trivial_error`, `closure` and `vectors` (the right Floquet vectors at every mesh point from the triangular factors' cyclic products, O5's hook; `None` for the other routes); `floquet(cycle, *, method=)` dispatches over them. A `Literal` now; strategy objects when a method grows options. All are jit- and vmap-compatible functions of the cycle, left unjitted (ADR-0015 §6): a batch of cycles of one mesh size is one `vmap`.

`LimitCycle.sample(n_points)` re-integrates the orbit from `x0s[0]` (the sampled orbit is derived, not stored); `orientation()` from the sampled polygon (`analysis.data.cycle.orientation`).

## 4. Exact and approximated

The periodic Schur chain is exact arithmetic on the segment Jacobians, and the Liouville
identity is exact for a planar field; what is approximated is each Jacobian and the
divergence integral, both by the solver at `cycle.config`, so $\kappa$ carries the
solver's global error over a period, about $10^{-9}$ (§3), and §5 pins all three routes
at $10^{-8}$. The monodromy and return-map routes are diagnostics whose floor is the
resolution of $1$ in double precision, $\mu - 1 \approx 10^{-16}$, which is why a strongly
attracting cycle gets its multiplier from the chain and not from $M$ (§2). The trivial
multiplier's distance from $1$ and the chain's closure are reported as the error
estimate.

## 5. Oracles (`tests/test_analysis_ode.py`, O2 block)

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

## 6. Limits and consequences

- **Planar only, by construction**: the seed fixes one Schur vector, which in the plane is all there is; `dim \geq 3` needs the periodic QR iteration on the transverse block (`slycot`) or BifurcationKit-style section-reduced products — open.
- **The orbit's error is the floor**: $\kappa$ inherits O1's $10^{-8}$; the trivial multiplier and the closure report it (Lust 2001, Figs. 5, 9: the trivial multiplier's error is the same with and without the periodic Schur decomposition — it is set by the discretization).
- **Per-segment contraction must stay above rounding**: Dieci et al.'s Remark 4.5 from the other side — $m$ large enough that $e^{\kappa T/m}$ is not tiny ($m \geq 2$ sufficed for FHN; the default $m = 8$ leaves margin).
- **The forward-direction stability argument** is deduced and measured, not cited.
- **For training**: with $T$ fixed by O1, $\mu = e^{\kappa T}$ is what a conjugacy must match; for the Bautin normal form $\kappa = \rho'(1)$ gives one equation in $(a, b)$ — the first place O2's number feeds the model side (roadmap M).
