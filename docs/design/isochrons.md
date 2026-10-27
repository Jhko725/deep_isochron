---
type: design
status: draft for review; the decisions of §11 settled 2026-10-27
updated: 2026-10-27
verified_by: pending (Joon); the measurements in §3, §5 and §7 were made on 2026-10-27 with AUTO-07p at commit 44cc1c4 and are recorded in the branch's change document
sources: [Langfield et al. 2014, Osinga & Moehlis 2010, Hannam et al. 2022, Krauskopf & Rieß 2008, Guckenheimer 1975, Nakao 2016, AUTO-07p (manual doc/auto.tex and source at 44cc1c4), the coworker's src/Isochron2PBVP, ADR-0014, ADR-0015, ADR-0016]
---

# Isochrons by continuation in AUTO-07p: the method, the coworker's version and ours

This document is the derivation behind roadmap item O7, which computes the isochrons of
a planar limit cycle with AUTO-07p. It describes three versions of one method:

- the method as Langfield et al. publish it (2014, §II.B), §4;
- the coworker's reproduction in [`src/Isochron2PBVP`](https://github.com/Jhko725/deep_isochron/tree/3af99f2/src/Isochron2PBVP), removed from the tree
  once the pipeline reproduced it, which follows the rotating
  multi-segment set-up of Hannam et al. (2022), §5;
- the version that `analysis/ode/auto/` will implement, §6.

Sections 1–3 give the background that all three share: what an isochron is (§1), how
AUTO continues a boundary value problem (§2), and how the stable direction at a point of
the cycle is computed (§3). Section 3 also records a defect in the coworker's version:
its bundle stage integrates the wrong linear equation. Sections 7–10 cover accuracy,
oracles, limits and tests, and §11 lists the decisions that are open. Claims are labeled
**[cited]** (read in the named source), **[code]** (read in, or measured with, the code
on 2026-10-27) or **[deduced]** (our derivation).

The implementation plan is the Project document "Isochrons from AUTO-07p" of
2026-10-27; ADR-0017 will record the structural decisions once §11 is settled.

## 1. What is computed

**An isochron is a level set of the asymptotic phase.** Let $\dot x = F(x)$ be a planar
system with an attracting periodic orbit $\Gamma$ of period $T$, and let
$\gamma_0 \in \Gamma$ be the reference point of phase 0. The point of phase $\theta$ on the cycle is
$\gamma_\theta = \varphi(\theta T / 2\pi, \gamma_0)$, where $\varphi(t, x)$ is the flow.
The asymptotic phase $\Theta(x)$ of a point $x$ in the basin is the phase of the cycle
point that the orbit of $x$ converges to [cited] (Langfield et al. 2014, §I.A, who use
the fraction $\vartheta = \theta / 2\pi \in [0, 1)$):

$$
\lim_{t \to \infty} \bigl\vert \varphi(t, x) - \varphi\bigl(t + \Theta(x) T / 2\pi, \gamma_0\bigr) \bigr\vert = 0 .
$$

The isochron of $\gamma_\theta$ is $I(\gamma_\theta) = \lbrace x : \Theta(x) = \theta \rbrace$.
This document measures $\theta$ in radians in $(-\pi, \pi]$, increasing with time, as
`model/base.py` and `systems/normal_forms/base.py` already do (decision 1 of the plan).
The reference point is `cycle.x0s[0]` of an O1 `LimitCycle`, and the O1 function
`reanchor(cycle, point)` moves it (decision 2).

**Each isochron is the stable manifold of its base point under the time-$T$ map.**
Guckenheimer (1975) shows this, and it has three consequences that the method uses
[cited] (Langfield et al. 2014, §I.B):

1. The Jacobian $D\varphi_T(\gamma_\theta)$ has the eigenvalue 1, with the eigenvector
   $F(\gamma_\theta)$ tangent to the cycle, and the stable eigenvalue $\mu$, with an
   eigenvector $v_s(\theta)$ transverse to it. In the plane, $\mu$ is the cycle's
   nontrivial Floquet multiplier, the same at every $\theta$.
2. $I(\gamma_\theta)$ is a smooth curve tangent at $\gamma_\theta$ to the line
   $E^s(\gamma_\theta) = \gamma_\theta + \mathbb R\thinspace v_s(\theta)$.
3. The flow maps isochrons onto isochrons:
   $I(\gamma_{\theta}) = \varphi\bigl(\theta T / 2\pi, I(\gamma_0)\bigr)$
   (their Eq. 2).

For FitzHugh–Nagumo with Langfield's parameters, $T = 11.2279$ and
$\mu = 1.6906\times 10^{-12}$ [code] (the variational transition matrix integrated with
`Dopri8` at `rtol = 1e-11`). The Floquet exponent is $\ln\mu / T = -2.414$, so a
perturbation transverse to the cycle shrinks by a factor of about $6\times 10^{11}$ per
period.

**All three versions trace an isochron as the start points of orbit segments.** An orbit
segment of duration $\tau$ that ends on $E^s(\gamma_\theta)$ at a small signed distance
$\eta$ from $\gamma_\theta$ starts at a point whose phase is the end point's phase minus
$2\pi\tau / T$. With $\tau$ a whole number $k$ of periods, the start point lies on
$I(\gamma_\theta)$ up to the error of the linear approximation, which §7 bounds.
Continuing the segment in $\eta$ moves the end point along $E^s$ and makes the start
point trace the isochron as a curve parametrized by arclength. Backward integration from
$E^s$ would reach the same points, but its error grows by the factor $1/\mu$ per period,
and the BVP formulation spreads that sensitivity along the whole segment [cited]
(Langfield et al. 2014, §II.A–B).

## 2. Continuation in AUTO, from the ground up

This section explains the AUTO vocabulary that §4–6 use. Readers who know AUTO can skip
it.

**AUTO solves a boundary value problem on the unit interval.** Each orbit segment is
rescaled to $t \in [0, 1]$, so that $\dot u = \tau F(u)$ with the duration $\tau$ as a
parameter (Langfield et al. 2014, Eq. 8). The equation file supplies four routines:
`FUNC`, the right-hand side; `BCND`, the `NBC` boundary conditions, which relate
$u(0)$, $u(1)$ and the parameters; `ICND`, the `NINT` integral conditions; and `STPNT`,
which builds a start solution. `NDIM` is the number of components of $u$. Several
planar pieces are stacked into one $u$: the coworker's problem with `NDIM = 10` carries
two orbit segments, the bundle along each, and the isochron segment.

**The number of free parameters is fixed by a count.** Given $u(0)$, the ODE fixes the
whole segment, so a solution has `NDIM` unknowns from $u(0)$ plus the free parameters,
and `NBC + NINT` equations. A one-parameter family, a curve of solutions, needs one more
unknown than equations, so AUTO frees

$$
\mathtt{NFPR} = \mathtt{NBC} + \mathtt{NINT} - \mathtt{NDIM} + 1
$$

parameters [code] (`src/toolboxbv.f90:43` of AUTO-07p). The first `NFPR` entries of the
run constant `ICP` are the free parameters. Further entries are printed but held fixed;
the AUTO manual calls them "overspecified" and recommends them for printing quantities
that `PVLS` computes [cited] (AUTO-07p manual, §"Parameter overspecification"). The
coworker uses this to print the base point and the traced isochron point.

**The segments are represented by orthogonal collocation.** AUTO divides $[0, 1]$ into
`NTST` mesh intervals, represents $u$ on each by a polynomial of degree `NCOL`, and
imposes the ODE at `NCOL` collocation points per interval; with `IAD > 0` it adapts the
mesh to the solution [cited] (AUTO-07p manual, §NTST, §NCOL, §IAD; Doedel 2007). A stored
solution holds the polynomial's values at `NTST · NCOL + 1` points, so the solution
between them is recovered exactly by Lagrange interpolation within each mesh interval
[cited] (Langfield et al. 2014, §II.B, who use this to shift the orbit's base point).
The coworker's constants are `NTST = 300` and `NCOL = 5`, with Newton tolerances
`EPSL = EPSU = 1e-10` and `EPSS = 1e-9` [code] (`c.fncMultiSeg`).

**A continuation step predicts along the curve and corrects back onto it.** AUTO steps a
distance `DS` along the tangent of the solution curve, measured in a norm over the
solution and the free parameters, and then corrects with Newton's method orthogonally to
the step (pseudo-arclength continuation). The step adapts between `DSMIN` and `DSMAX`.
The weights `THL` remove parameters from that norm; the coworker sets the weights of the
three time parameters to 0, so that the period and segment lengths do not take up the
step [code] (`c.fncMultiSeg`).

**A run stops at a special point or a user condition.** AUTO labels special points
along a branch: `BP`, a branch point, where two solution curves cross; `LP`, a fold in
the continuation parameter; `UZ`, a user point where a parameter takes a value listed in
`UZR`; and `EP`, the end of the run. `UZSTOP` lists parameter values at which the run
stops. A later run restarts from a labeled solution, for example `run(sol('UZ1'), ...)`,
and `ISW = -1` at a `BP` switches to the other branch. When a restart increases `NDIM`,
AUTO copies the existing components and the new ones start from `STPNT` or zero [cited]
(manual, §"User-supplied routines": `STPNT` is called "when a system is manually
extended"; [code] `src/io.f90:1489-1495` copies the first components only).

**Branch points are how the linear eigenvector problem is solved.** The bundle stage of
§5 appends a linear equation $\dot w = \tau B(t) w$ with the boundary condition
"$w$ at the end equals $\lambda$ times $w$ at the start". The zero solution $w = 0$
solves it for every $\lambda$, so continuing in $\lambda$ along $w = 0$ is a
continuation along a trivial branch. A nonzero solution exists exactly where $\lambda$
is an eigenvalue of the transition matrix of $B$ over one period, and there AUTO detects
a `BP`. Switching branch there with a normalization parameter that grows from 0 to 1
yields the normalized eigenvector [deduced from the coworker's runs 4–5, §5].

## 3. The stable direction

The isochron segment of §5 ends on the line through the base point along a direction
$d(\theta)$, and §1 requires $d(\theta) = v_s(\theta)$. Three linear equations along the
cycle give access to it.

**The variational equation transports tangent vectors.** Let $A(t) = DF(\gamma(t))$ and
let $\Phi(t)$ solve $\dot\Phi = A\Phi$ with $\Phi(0) = I$. The monodromy
$\Phi(T)$ at $\gamma_\theta$ has the eigenvalues 1 and $\mu$, with eigenvectors
$F(\gamma_\theta)$ and $v_s(\theta)$, and $\mu = \exp\int_0^T \operatorname{tr} A\thinspace dt$
in the plane (Liouville's formula). Langfield et al. (2014, §II.B) obtain $v_s$ this way:
they "solve the first variational equation together with the periodic orbit as an
extended system", following Krauskopf & Rieß (2008), and find the bundle of stable
vectors on the orbit's mesh [cited].

**The adjoint equation transports the phase sensitivity.** The phase sensitivity
$Z(\theta) = \nabla\Theta(\gamma_\theta)$ is the periodic solution of
$\dot Z = -A^{\mathsf T} Z$, normalized by $Z\cdot F = 2\pi / T$ [cited] (Nakao 2016,
Eqs. 12–13, with $\omega = 2\pi / T$). The transition matrix of the adjoint equation is
$\Phi(t)^{-\mathsf T}$, so its multipliers are 1 and $1/\mu$. For any solution $y$ of the
adjoint equation and $z$ of the variational equation, $y\cdot z$ is constant in time.
Taking $y = Z$, which is periodic, and $z$ the solution through $v_s$, which shrinks by
$\mu$ per period, gives $Z(\theta)\cdot v_s(\theta) = \mu\thinspace Z(\theta)\cdot v_s(\theta)$. Since $\mu \neq 1$,

$$
Z(\theta)\cdot v_s(\theta) = 0 \quad\text{for every } \theta \qquad [\text{deduced}].
$$

In the plane this makes the stable direction the rotation of $Z$ by a right angle,
$v_s(\theta) \parallel J Z(\theta)$, with

$$
J = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix} .
$$

$J w = (-w_2, w_1)$ is exactly the end direction that the coworker's `BCND` uses:
`(/ -U0(8), U0(7) /)` (`isochron_multiseg.f90`, `FB(12:13)`). The set-up is therefore
correct when the bundle $w$ solves the adjoint equation.

**The coworker's bundle solves $\dot w = +A^{\mathsf T} w$, which is neither equation.**
`FUNC` sets `F(5:6) = PAR(12)*MATMUL(TRANSPOSE(A),U(5:6))` and the same for `F(7:8)`
[code] (`fncMultiSegIso.f90:49-50`; the generic file kept it until D1). The script's
own header calls the slots the "Adjoint Floquet multipliers" and the "Adjoint Floquet
Bundle" (`mainIsochronMultiSegFNC.py:20-21`), so the sign is a slip, not a different
method. Its transition matrix
$\Psi(t)$ has no fixed relation to $\Phi(t)$ in general, and two measurements show the
consequence [code]:

- For `HopfNormalForm(1.3, 1.5, 0.4)` the multipliers of $\Psi(T)$ are
  $-0.0043 \pm 0.0004\thinspace i$, and for `BautinNormalForm(1.0, 0.3, 1.5, 0.5)` they
  are $-0.0039 \pm 0.0018\thinspace i$. Neither has the multiplier 1, so the bundle
  stage, which looks for a branch point at $\lambda = 1$, has nothing to find. Run
  unchanged in AUTO on the generated problems, the coworker's runs 4–5 walk the trivial
  branch from $\lambda = 1.2$ to $-8.7$ without a branch point, for both forms.
- For FitzHugh–Nagumo, $\Psi(T)$ does have the multipliers 1 and $\mu$, and AUTO's run
  4 detects branch points at $\lambda = 1$ and at $\lambda = 1.68948\times 10^{-12}$.

**FitzHugh–Nagumo is an exception because its Jacobian is similar to its transpose.**
The Jacobian of Eq. 15 of Langfield et al. (2014) is

$$
A = \begin{pmatrix} c(1 - x^2) & c \\ -1/c & -b/c \end{pmatrix},
\qquad
D = \begin{pmatrix} 1 & 0 \\ 0 & -c^2 \end{pmatrix},
\qquad
D A D^{-1} = A^{\mathsf T} ,
$$

with a constant $D$, so $\Psi(t) = D\Phi(t)D^{-1}$ [deduced]. Its eigenvector for the
multiplier 1 is $D F(\gamma_\theta)$, a rescaled tangent vector, not the phase
sensitivity. At the base point $(-0.19007, -0.12319)$ that AUTO's runs 0–5 produce, the
bundle vector is $w = (-0.58385, -0.81186)$, which equals $DF/\vert DF\vert$ to eight
digits [code]. The end direction $Jw$ is then $58.0°$ from the true stable direction
$v_s = (0.92511, 0.37971)$ and $26.9°$ from the flow [code]. It is transverse to the
cycle, which is why the coworker's isochrons are still close to the true ones (§7), but
it is not $v_s$.

**Changing the sign gives the adjoint equation, and AUTO then finds $v_s$ exactly.**
With `F(5:6)` and `F(7:8)` negated and every run constant of runs 4–5 unchanged, AUTO
finds the branch point at $\lambda = 1$, and the normalized bundle at the same base
point is $w = (-0.37971, 0.92511)$. Its rotation $Jw$ agrees with $-v_s$ to eight digits
[code]. The bundle vector $w$ itself is the phase sensitivity up to its scale:
$w\cdot F = 1.1147$ for $\vert w\vert = 1$, against $Z\cdot F = 2\pi/T = 0.5596$. On
the Hopf and Bautin forms of the previous paragraph, the negated runs find the branch
point at $\lambda = 0.99999999998$, and $w$ equals the direction of the closed-form
$\nabla\Theta$ at the base point to round-off [code].

**Three ways to get the stable direction were weighed; D1 chose (a).**

| Option | Equation in `FUNC` | Branch point at | What $w$ is | Cost |
|---|---|---|---|---|
| (a) adjoint | $\dot w = -\tau A^{\mathsf T} w$ | $\lambda = 1$, for every system | the direction of $Z(\theta)$ | one sign; `BCND` unchanged |
| (b) variational | $\dot w = \tau A w$ | $\lambda = \mu$ ($1.7\times 10^{-12}$ for FitzHugh–Nagumo) | $v_s(\theta)$ itself | `BCND` uses $w$, not $Jw$; the continuation in $\lambda$ must pass $\lambda = 1$ and travel twelve orders of magnitude |
| (c) from O2 | none; the bundle stage is removed | — | O2's `Floquet.vectors`, carried to each base point | new slots for $d$; $d$ must follow the rotation of stage 3 |

Option (a) keeps every stage of the coworker's pipeline and makes $w$ double as the
phase sensitivity that O4 needs; `isochron_multiseg.f90` integrates it since D1. Option (b) is the published route. AUTO does detect the
branch point at $1.69\times 10^{-12}$ in the coworker's run 4, so (b) is feasible, but
its branch point moves with the system's contraction rate. Option (c) duplicates O2 inside
the pipeline and loses AUTO's own consistency between the bundle and the collocated orbit.

## 4. The published method (Langfield et al. 2014, §II.B)

**Langfield et al. trace every isochron from one stable direction, at the reference
point.** They use $v_s$ at $\gamma_0$ only, because the bundle is computed on the whole
mesh but "its accuracy is only guaranteed at the point $c_0$", and map isochrons with
Eq. 6, $I(\gamma_0) = \varphi\bigl((1 - \vartheta)T, I(\gamma_\vartheta)\bigr)$
[cited]. An isochron of phase $\vartheta$ is then the set of start points of segments
that end on $E^s(\gamma_0)$ after the time

$$
\tau = \bigl(k + (1 - \vartheta)\bigr) T, \qquad k = 1, 2, \dots
$$

(their Eqs. 11 and 14), with the boundary conditions

$$
\bigl(u(1) - \gamma_0\bigr)\cdot n_s = 0, \qquad \bigl(u(1) - \gamma_0\bigr)\cdot v_s = \eta ,
$$

where $n_s \perp v_s$ (their Eqs. 9–10) [cited].

**The start solution is the shifted cycle, extended in time.** For phase $\vartheta$
they shift the collocated cycle so that $u(0) = u(1) = \gamma_\vartheta$, locating
$\gamma_\vartheta$ by Lagrange interpolation, and correct it with the periodicity
condition and $(u(0) - \gamma_\vartheta)\cdot F(u(0)) = 0$ in one step of a dummy
parameter (their Eqs. 12–13). They then free $u(1)$ and continue in $\tau$ until Eq. 11
holds; that segment, with $\eta = 0$, starts the isochron run [cited].

**The isochron is extended in stages.** The run continues $\eta$ from 0 to
$\eta_{\max} \ll 1$. To go further, the final segment is concatenated with one more copy
of the cycle, rescaled to $[0, 1]$, and continued again with $k$ increased by 1; the
small gap at the junction closes in the first Newton step. Each stage stretches the
interval of length $\eta_{\max}$ on $E^s$ by the backward expansion over the added
period, and the staging keeps the mesh small [cited] (§II.B–C and Fig. 2).

**Two other set-ups move the base point instead.** Langfield et al. note that one can
"use continuation to shift the base point and/or the stable eigenbundle", as Osinga &
Moehlis (2010) did, and that they "found small differences in the computed isochrons"
between the two [cited]. Hannam et al. (2022, supplementary §SM1) split the orbit into a
phase segment and an anchoring segment, rotate both the orbit and its Floquet bundle by
continuation in $\theta$, and trace each isochron from the rotated base point and its
own Floquet vector $w_\theta$, in arcs of one period each [cited]. The coworker's code
follows this rotating set-up.

## 5. The coworker's algorithm, run by run

**The coworker's script is a sequence of ten AUTO runs and one loop.** It lives in
[`src/Isochron2PBVP/fncMultiSeg/`](https://github.com/Jhko725/deep_isochron/tree/3af99f2/src/Isochron2PBVP/fncMultiSeg) at commit `3af99f2` (`mainIsochronMultiSegFNC.py`, `fncMultiSegIso.f90`,
`c.fncMultiSeg`, and `isoFunctions.py` one level up). It numbers its parameters as
follows [code]:

| `PAR` | Meaning | Our `Idx` name |
|---|---|---|
| 1–4 | $a$, $b$, $c$, $z$ | the system's slots |
| 11 | period $T$ | `period` |
| 12 | duration of segment 1, from the anchor to the junction | `seg1` |
| 13 | duration of segment 2, from the junction back to the anchor | `seg2` |
| 14 | duration of the isochron segment | `copies` |
| 34 | bundle multiplier $\lambda$ | `mult` |
| 35 | $\vert w\vert^2$ at the junction | `norm` |
| 36 | distance $\eta$ along the end direction | `eta` |
| 70–71 | anchor point | `anchor` |
| 72–73 | base point, the junction | `base` |
| 74–75 | start point of the isochron segment, the traced point | `point` |
| 80 | AUTO's stability flag | `stability` |

**The extended system grows with `NDIM`.** The state stacks planar pieces [code]
(`FUNC` and `BCND` of `fncMultiSegIso.f90`):

- `U(1:2)` is segment 1 and `U(3:4)` segment 2 of the cycle, with
  $\dot u = \tau_1 F(u)$ and $\dot u = \tau_2 F(u)$.
- `U(5:6)` and `U(7:8)` are the bundle on the two segments, with
  $\dot w = +\tau_i A^{\mathsf T} w$ (§3).
- `U(9:10)` is the isochron segment, with $\dot u = \tau_{\rm iso} F(u)$ and
  $\tau_{\rm iso} = 5T$.

The boundary conditions close the two cycle segments into one orbit, `FB(1:4)`; fix
$T = \tau_1 + \tau_2$, `FB(5)`; anchor the start of segment 1 on the horizontal line
$y = $ `PAR(71)`, `FB(6)`; close the bundle up to the multiplier, $w_2(1) = w_1(0)$ and
$w_1(1) = \lambda w_2(0)$, `FB(7:10)`; normalize $\vert w_2(0)\vert^2 = $ `PAR(35)`,
`FB(11)`; and end the isochron segment at the junction plus $\eta\thinspace Jw_2(0)$,
`FB(12:13)`. The junction $u_2(0)$ is the base point, and the start of segment 1 is the
anchor.

**The runs, in order** [code] (`mainIsochronMultiSegFNC.py:39-55`; free parameters from
the count of §2):

| Run | Problem | Free parameters | Start | Stop | Purpose |
|---|---|---|---|---|---|
| 0 | equilibria (`IPS = 1`, `NDIM = 2`) | $c$ | the source $(0.9066, -0.2582)$ at $c = 3$ | the Hopf point (`HB1`, $c = 2.1192$) | find where the cycle is born |
| 1 | periodic orbits (`IPS = 2`) | $c$, $T$ | `HB1`, `ISW = -1` | $c = 3$ (`UZSTOP`) | the cycle at the target parameters |
| 2 | two segments (`IPS = 4`, `NDIM = 4`, `NBC = 6`) | $\tau_2$, $\tau_1$, $T$ | the run-1 cycle as segment 1 with $\tau_1 = T$; `STPNT` puts segment 2 at the anchor, which `PVLS` stored, with $\tau_2 = 0$ | $\tau_2 = 1.0$ | grow segment 2 out of the anchor |
| 3 | the same | anchor $y$, $\tau_1$, $T$ | run 2 | anchor $y = 0.5$ | move phase 0 to the crossing with $y = 0.5$ |
| 4 | bundle (`NDIM = 8`, `NBC = 11`) | $\lambda$, `norm`, $\tau_1$, $T$ | $w = 0$, $\lambda = 1.2$, `norm` $= 0$, `DS` $= -0.1$ | the second `BP` | walk the trivial branch to the branch points at $\lambda = 1$ and $\lambda = \mu$ |
| 5 | the same | the same | the first `BP` ($\lambda = 1$), `ISW = -1` | `norm` $= 1$ | the normalized bundle |
| 6, 7 | the same | $\tau_2$, $\lambda$, $\tau_1$, $T$ | run 5 | $\tau_2$ at the bounds $[0, 1.01T]$ | rotate the junction, in both directions, with `UZR` at $\tau_2 = T\cdot\lbrace 0, 0.01, \dots, 1.00\rbrace$ |
| 8 | — | — | runs 6 and 7 | — | `merge` and `rl` join and relabel the two branches into 100 `UZ` solutions |
| per phase, set-up | isochron (`NDIM = 10`, `NBC = 13`) | $\eta$, $\lambda$, $\tau_1$, $T$ | the start of `createMultSegPeriodCopy2D` | one step (`NMX = 1`) | correct the start solution at $\eta = 0$ |
| per phase, $\pm$ | the same | the same | the set-up, `DS` $= \pm 10^{-5}$ | $\vert\eta\vert = 10^{-6}$ or $\vert x\vert = 10$ | trace the two halves of the isochron |

**The start solution of the isochron runs is built by spline resampling.**
`createMultSegPeriodCopy2D` takes the rotated solution, sets $\tau_{\rm iso} = 5T$, and
resamples `U(1:8)` with SciPy's `splrep` onto $5(m - 1) + 1$ points of $[0, 1]$, where
$m$ is the number of stored points. It fills `U(9:10)` with the cycle started at the
base point, segment 2 followed by segment 1, repeated five times [code]
(`isoFunctions.py`). The fivefold resampling is needed because `U(9:10)` covers five
periods on the same $[0, 1]$ as the one-period pieces.

**Two of the stop conditions do not act as written.** The dictionary
`UZSTOP={36:[1e-06,-1e-6],74:[10,-10],74:[10,-10]}` repeats the key 74, so the intended
bound on the $y$ coordinate (`PAR(75)`) never applies [code]. And the outward half of
each isochron stops at $\vert x\vert = 10$ while $\eta$ is still about $10^{-14}$ (for
example $9.6\times 10^{-15}$, $1.5\times 10^{-15}$ and $5.8\times 10^{-14}$ for the
isochrons stored as files 1, 26 and 51), because one period of the backward flow
stretches the segment by $1/\mu$. Only the inward half, which spirals toward the
source, reaches $\vert\eta\vert = 10^{-6}$, after 3510 to 6803 steps; isochron 26 then
comes within $1.7\times 10^{-4}$ of the source [code] (the AUTO output of the rerun of
2026-10-27).

**The stored files are numbered against the phase.** File $k$ is the `UZ` solution at
$\tau_2 = kT/100$, so its base point has the phase fraction $1 - k/100$ after the anchor:
file 100 starts at the anchor $(-1.7217, 0.5)$, and files 1, 26 and 51 at
$\tau_1 = 11.1156$, $8.3086$ and $5.5017$ [code]. The value $\tau_2 = 0$ would repeat phase 0;
the run reaches it only as an end point, so it has no file.

**The run takes hours.** On this session's two cores, runs 0–5 take 3 s, runs 6–8
take 45 s, and one isochron takes 140–290 s, almost all of it in the inward half; the
100 isochrons take 5–6 hours in sequence [code]. A rerun reproduces the stored files 1
and 26 with the same point counts. The largest distance from a point of one curve to
the polyline of the other is $3.6\times 10^{-6}$ [code]. The distance between nearest
points, $7.8\times 10^{-4}$, is larger only because the points lie up to 0.19 apart; it
measures the spacing, not a difference between the curves.

## 6. Our algorithm, stage by stage

**Our pipeline keeps the coworker's rotating structure and replaces its first four runs.**
The mapping is:

| Coworker | Ours | What changes |
|---|---|---|
| runs 0–1, Hopf continuation | O1's `find_limit_cycle` | the cycle comes from shooting; runs 0–1 survive only as a test (decision 12) |
| run 2, growing segment 2 | stage 1, `_periodic_orbit` | the two segments are cut from O1's cycle at half the period and corrected once |
| run 3, sliding the anchor | `reanchor` before stage 1 | phase 0 is set on the `LimitCycle`, not in AUTO |
| `FB(6)`, $y = $ const | $(u_1(0) - p)\cdot n = 0$ | the anchor is O1's section through phase 0 (decision 3) |
| runs 4–5 | stage 2, `_adjoint_bundle` | the adjoint equation, option (a) of §3 (D1) |
| runs 6–8 | stage 3, `_rotate_phase` | `UZR` at the requested phases only |
| `createMultSegPeriodCopy2D` | the start of stage 4 | on AUTO's stored grid: `U(1:8)` as stored, `U(9:10)` integrated by O1's solver; no splines (D2) |
| the two isochron runs | stage 4, `_isochron` | stop when the traced point leaves a box relative to the cycle (decision 15) |
| the loop over phases | `isochron_foliation` | a process pool (decision 17) |

**Stage 1 corrects O1's cycle onto AUTO's mesh.** The cycle is integrated from
`x0s[0]` with `cycle.config` over one period and split at $T/2$, so
$\tau_1 = \tau_2 = T/2$. The pieces become segments 1 and 2 on a uniform grid of $[0, 1]$, and one run with
`NDIM = 4`, `NBC = 6` and the free parameters $\tau_2$, $\tau_1$, $T$ corrects them. The
split point does not matter, because stage 3 rotates the junction to every requested
phase. The coworker's runs 2–3 left $\tau_2 = 1.0$, which was as arbitrary.

**The new anchor condition computes the same orbit with a better-conditioned equation.**
`FB(6)` removes the time-shift freedom of the periodic orbit: without it, every shift of
the same orbit solves the other conditions, and the Newton matrix is singular. A linear
condition $(u_1(0) - p)\cdot n = 0$ removes that freedom when the flow crosses it,
$n\cdot F(u_1(0)) \neq 0$. The coworker's horizontal line has $n = (0, 1)$ and needs
$\dot y \neq 0$ at the anchor, which fails where the cycle has a horizontal tangent. O1's
section is normal to the flow at phase 0, so $n\cdot F = \vert F\vert$, the largest value
a unit normal can give [deduced]. The condition fixes where phase 0 is pinned, not the
orbit, so after `reanchor` to the coworker's anchor the isochrons are the same.

**Stage 3 maps a phase to a segment length.** The base point is the junction, the end
of segment 1, so it lies $\tau_1$ after the anchor: its phase is $2\pi\tau_1/T$. A
requested $\theta$, wrapped to $[0, 2\pi)$, is therefore reached at

$$
\tau_2 = \bigl(1 - \theta / 2\pi\bigr)\thinspace T ,
$$

and stage 3 places one `UZR` value per requested phase [deduced; consistent with the
coworker's files, §5]. Phase 0 is $\tau_2 = T$; stage 3 continues to $1.01T$, as the
coworker does, so that it is an interior `UZ` point.

**Stage 4 builds its start solution on AUTO's stored grid.** `U(1:8)` is the rotated
solution as AUTO stored it, and `U(9:10)` is integrated from the base point with O1's
solver at the same points $t$, scaled to $n$ periods (D2). No interpolation is needed,
because every component shares AUTO's one mesh: with `NTST` intervals over $[0, 1]$, the
isochron segment gets `NTST`$/n$ intervals per period whatever start data it is given,
and AUTO adapts the mesh to the data. The coworker's fivefold spline resampling
therefore adds nothing the mesh keeps. Isochron 26 traced from this start agrees with
the spline start to $3.3\times 10^{-6}$, the size of AUTO's rerun-to-rerun difference,
with 7064 against 7065 points [code]. Evaluating AUTO's piecewise polynomials exactly by
Lagrange interpolation on the `NCOL + 1` equally spaced points of each interval
(`src/mesh.f90:163` of AUTO-07p) was the other candidate; it is not needed.

**Stage 4 stops on the cycle's own scale and says why.** The two runs trace the halves
of the isochron with `DS` $= \pm 10^{-5}$ and stop at $\vert\eta\vert$ = `eta_stop`,
when the traced point leaves the cycle's bounding box enlarged `box_scale` times about
its center in either coordinate, or at the step limit `NMX`. Each `Isochron` records
which of `"eta"`, `"box"` and `"max_steps"` ended each half (D7); a half that ran out of
steps is correct as far as it goes, so it is flagged rather than raised. For
FitzHugh–Nagumo, whose cycle has $\vert x\vert$ up to about 2, `box_scale = 5`
reproduces the coworker's bound of 10 and adds the lost bound on $y$.

**The mesh grows with the number of periods.** `IsochronSettings.ntst` defaults to
`60 * n_periods`, 300 for the default 5, so that the isochron segment keeps 60 intervals
per period when `n_periods` changes (D6). With a fixed `NTST`, doubling `n_periods`
would halve the mesh per period without any sign of it.

## 7. What is exact and what is approximated

**The linear approximation sets the phase error of every traced point.** Near the base
point, $\Theta(\gamma_\theta + \delta) = \theta + Z(\theta)\cdot\delta + O(\vert\delta\vert^2)$.
The segment ends at $\gamma_\theta + \eta d$, and phase is carried exactly along orbits,
so the traced start point has the phase error

$$
\Delta\Theta = \eta\thinspace Z(\theta)\cdot d + O(\eta^2) \qquad [\text{deduced}].
$$

With $d = v_s$, the first-order term vanishes by §3, and the error is $O(\eta^2)$. With
the coworker's $d = Jw$, it is first order in $\eta$. The measured effect is small
because $\eta$ itself is small: below $10^{-13}$ on the outward halves and at most
$10^{-6}$ on the inward ones (§5). Isochron 26 computed with option (a) lies within
$8.1\times 10^{-6}$ of the coworker's stored file, measured from points to the other
curve's polyline in both directions, and both reach to $1.7\times 10^{-4}$ of the source
[code]. The corrected run takes 272 s against 141 s and returns 7065 points against
3690, because its inward half takes more continuation steps.

**The number of periods sets how close to the source an isochron reaches.** Near the
source the forward flow expands at the rate of the source's eigenvalues' real part,
$0.1339$ for FitzHugh–Nagumo (Langfield et al. 2014, §III). A segment of $n$ periods
therefore connects a start point at distance $\delta$ from the source to an end point
near the cycle only if $\delta \gtrsim e^{-0.1339 n T}$ times the size of the source's
neighborhood. That factor is $5\times 10^{-4}$ for $n = 5$, the same order as the
$1.7\times 10^{-4}$ that isochron 26 reaches (§5) [deduced]. More of the spiral needs
a larger `n_periods` or Langfield's staged extension (D4).

**Collocation and Newton set the accuracy of each solution.** The orbit, the bundle and
the segments are piecewise polynomials on `NTST = 300` intervals of degree `NCOL = 5`,
corrected to `EPSU = 1e-10`. The isochron segment winds $n$ times around the cycle on
the same number of intervals, so its mesh is $n$ times coarser per period than the
orbit's; the adaptive mesh redistributes the intervals, and the coworker's results show
that this is enough for FitzHugh–Nagumo. D6 keeps that resolution per period when
`n_periods` changes; a system that needs more raises `ntst` itself.

**Phase is extremely sensitive near the source, which limits forward-time checks.**
Integrating the coworker's points forward for four periods and comparing with the base
point's orbit gives phase errors near $10^{-9}$ rad for points away from the source and
up to 1 rad for points within $10^{-3}$ of it [code]. The isochrons accumulate near the
source (Langfield et al. 2014, §IV), so there a small error in the stored position or
in the integration becomes a large error in phase, and this check cannot yet tell
AUTO's error from its own (§8) [deduced].

## 8. Oracles and the tolerances they justify

| Oracle | What it checks | Tolerance and its reason |
|---|---|---|
| The coworker's isochrons, files 100, 26 and 51 | the whole pipeline on FitzHugh–Nagumo, with phase 0 at the coworker's anchor and each stored curve's phase from its base point | a two-way distance from points to the other curve's polyline of $2\times 10^{-5}$, above the $3.6\times 10^{-6}$ of AUTO's own rerun and the $8.1\times 10^{-6}$ between the coworker's bundle and option (a) (D3) |
| O1's cycle | the period of stage 1 (11.2279; Langfield et al. 2014, §III) and its points | AUTO's correction tolerance against O1's shooting tolerance |
| The Hopf route (runs 0–1) | O1's cycle by an independent method (decision 12) | the same |
| O2's Floquet vectors | $Jw$ parallel to `floquet(cycle).vectors` at the points `x0s` | the angle at the accuracy of O2's vectors |
| The closed forms of Bautin and Winfree | `nf.phase(point)` $= \theta$ along every traced isochron, with phase 0 at $(1, 0)$ (decision 14) | $O(\eta^2)$ plus collocation error, since the end direction is $v_s$ |
| Forward integration | for traced points farther than a set distance from the phaseless set, the phase after $k$ periods against the base point's orbit | scaled by the local phase sensitivity; points near the source are excluded |
| The bundle's scale | $w\cdot F$ constant along the cycle, since $Z\cdot F = 2\pi/T$ | round-off times the condition of the bundle |

O3's phase field and O4's $Z(\theta)$ add two more oracles when they exist: the phase
field is constant along every AUTO isochron, and $w$ is parallel to $Z(\theta)$.

## 9. Limits and deferred work

- **Planar systems only.** The extended system stacks planar pieces, and the end
  direction $Jw$ is a planar construction. A system with more states needs a stable
  subspace and a two-dimensional isochron, which Langfield et al. (2014, §V) describe as
  open.
- **Forward-time isochrons of attracting cycles only.** Backward-time isochrons of
  repelling cycles, which Langfield et al. (2014, §V) mention, are deferred.
- **One `n_periods` for the whole isochron.** Langfield's staged extension (§4) is
  deferred (D4) until a system's isochrons stop at `"max_steps"` or fall short of its
  phaseless set. The outward halves run with $\eta$ near $10^{-14}$ and points up to
  0.19 apart; resampling by arclength is a follow-up if a consumer needs even spacing.
- **No parameter sweeps.** Following a family of cycles in a parameter is not part of
  O7; the Hopf-route test of decision 12 is its starting point.
- **The bundle and O4.** Replacing the bundle stage by O4's $Z(\theta)$ is deferred to
  O4 (decision 11).

## 10. The tests this document promises

- `rhs.inc`, compiled by the gfortran driver, against `rhs` and `jacfwd(rhs)`; the
  generic file compiled with each planar system's includes; every wrap cut at a blank;
  `indices.inc` one slot per line; the bundle's adjoint sign (Step A,
  `tests/test_auto_codegen.py`).
- The FitzHugh–Nagumo problem builds with AUTO and finds the Hopf point at $c = 2.1192$
  (Step B smoke test).
- `reanchor`: the period is unchanged, the new `x0s[0]` lies on the cycle, and
  reanchoring at a cycle point returns that point.
- Stage 1: the corrected period equals O1's; the anchor lies on the section.
- Stage 2: the branch point is at $\lambda = 1$ for FitzHugh–Nagumo, Hopf, Bautin and
  Winfree; the end direction is parallel to O2's $v_s$ at the mesh points.
- Stage 3: the base point of each `UZ` solution has the requested phase on O1's cycle.
- Stage 4: the oracles of §8, on three phases of each system; the stop reason of each
  half; `ntst` scaling with `n_periods`.
- The Hopf route against O1's cycle.
- `IsochronFoliation.save` and `load` round-trip every field.
- Refusals: a non-planar system, an unknown phase type, a missing AUTO tree.

## 11. Decisions

Joon settled these on 2026-10-27; each was phrased so that "yes" accepted the
recommendation, and each was accepted.

- **D1 — The adjoint bundle.** Stage 2 integrates $\dot w = -\tau A^{\mathsf T}w$,
  option (a) of §3; the coworker's $+A^{\mathsf T}$ was a sign slip, confirmed by its
  failure on the Hopf and Bautin forms. Decision 11 of the plan (keep AUTO's bundle)
  stands with the corrected equation. Recorded in ADR-0017.
- **D2 — The stage-4 start on AUTO's grid.** No interpolation and no SciPy (§6).
  Lagrange evaluation within AUTO's mesh intervals and the coworker's splines were the
  alternatives.
- **D3 — The regression oracle's tolerance.** The coworker's files stay the oracle,
  compared by the two-way distance from points to the other curve's polyline at
  $2\times 10^{-5}$. This replaces decision 13's $2\times 10^{-3}$, which rested on a
  point-to-point distance that measured the spacing of the points (§5).
- **D4 — One segment per isochron.** Stage 4 keeps the coworker's single segment of
  `n_periods` periods; Langfield's staged extension is deferred (§9).
- **D5 — The rotating base point.** The pipeline keeps the coworker's rotation of the
  base point and the bundle (Hannam et al. 2022), rather than Langfield's single base
  point with segments of $(k + 1 - \vartheta)T$. With the corrected bundle, each base
  point has its own accurate stable direction, and every isochron gets the same
  treatment. Recorded in ADR-0017.
- **D6 — The mesh against the number of periods.** `ntst` defaults to `60 * n_periods`
  (§6).
- **D7 — The stop reason of each half.** `"eta"`, `"box"` or `"max_steps"`, flagged
  rather than raised (§6).
- **D8 — Wrapping between tokens.** `codegen._wrap` cuts only at blanks, which SymPy's
  printer puts only between tokens. The earlier cut at a fixed width split `PAR` in
  Bautin's `rhs.inc`, which then did not compile.
- **D9 — The coworker.** Claude drafts a note on the sign slip; Joon decides whether to
  send it.

## Acknowledgments

The AUTO-07p reproduction that this pipeline is built from, `src/Isochron2PBVP` (in the
tree until commit `3af99f2`), is the
work of Andrus Giraldo (Department of Computational Sciences, Korea Institute for
Advanced Study). §5 describes it run by run, and the pipeline of §6 keeps its rotating
multi-segment structure and its run constants.
