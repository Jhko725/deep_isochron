---
type: design
status: agreed; §5.3–5.4 tentative
updated: 2026-10-04
verified_by: joon (2026-10-03; §5.3–5.4 pending)
sources: [Wilson & Moehlis 2016, Yawata et al. 2024, Kvalheim & Revzen 2021, Langfield et al. 2014]
---

# Normal forms — conventions, derivations, and the API they imply

**Status: agreed (roadmap B10), 2026-10-03; implemented in B11 (2026-10-03).** Amended 2026-10-03: §5.1 corollary sign (confirmed); §5.3 Yawata paragraph rewritten from the paper's equations and §5.4 (Kvalheim & Revzen) added — both *tentative*, to be revisited by Joon as the claims are checked. This document merges the
B10 design note with Joon's earlier derivation notes (LaTeX, "Tractable periodic
dynamics") and resolves the conflicts between them. It is the single reference for the
mathematics of the analytic base systems; `systems/normal_forms/` is implemented against
it (B11), and ADR-0007 points here. All design decisions are settled
and listed in §11; §9 is the implementation contract.

Companion files: `systems/normal_forms/{base,hopf,bautin,integration}.py`;
`tests/test_normal_forms.py`. Related: ADR-0007 (hierarchy, `SolverConfig`, integrations
as objects).

All closed forms below were re-verified symbolically/numerically on 2026-10-03
(residuals of the defining identities at machine precision, including the regime
$-1 < b < 0$).

---

## 1. Setting and symbols

A **normal form** here is a planar ODE with an attracting limit cycle, rotationally
symmetric about the origin, so that in polar coordinates
$\mathbf{u} = (x, y) = (r\cos\theta, r\sin\theta)$ the radial and angular motions decouple.

| symbol | meaning | convention |
|---|---|---|
| $r \ge 0$, $\theta \in (-\pi, \pi]$ | polar coordinates about the origin | `to_polar(u) = (r, θ)`, $\theta = \mathrm{atan2}(y, x)$; $\theta$ is *unwrapped* ($\in\mathbb{R}$) whenever it is integrated |
| $\rho(r)$ | **log growth rate** of the radius, $\rho = \dot r/r = \tfrac{d}{dt}\ln r$ | even in $r$; $\rho(1) = 0$, $\rho'(1) < 0$ |
| $\omega(r)$ | **angular rate**, $\dot\theta = \omega(r)$ | even in $r$; $\omega(1) =: \omega_1 \ne 0$ |
| $\Gamma$ | the limit cycle $r = 1$ | the only attracting cycle |
| $T = 2\pi/\omega_1$ | period | sign of $\omega_1$ is the sense of rotation |
| $\kappa = \rho'(1) < 0$ | non-trivial **Floquet exponent** | **[decided]** no factor 2 — see §4.2 |
| $\mu = e^{\kappa T}$ | non-trivial **Floquet multiplier** | $0 < \mu < 1$ |
| $\Theta(\mathbf{u}) \in (-\pi, \pi]$ | **asymptotic phase**: $\dot\Theta = \omega_1$ along every trajectory | $\Theta = \theta$ on $\Gamma$ |
| $h(r)$ | **phase shift**: $\Theta = \theta + h(r)$ | $h(1) = 0$ |
| $\Psi(\mathbf{u})$ | **isostable coordinate** (amplitude): $\dot\Psi = \kappa\Psi$ | **[decided]** $\Psi = 0$ on $\Gamma$, $\nabla\Psi\cdot\mathbf e_r = 1$ on $\Gamma$ (§5.2) |
| $\mathsf{J}$ | rotation by $+\pi/2$, $\mathsf{J}(x, y) = (-y, x)$ | |

**Notation [decided].** Uppercase $\Theta, \Psi$ are the two *coordinate functions* on the
basin; lowercase $\theta$ is the polar angle. $\Psi$ and $\kappa$ follow Wilson & Moehlis
(2016), who write the isostable coordinate as $\psi$ and the exponent as $\kappa$; we
capitalise $\psi\to\Psi$ so that both coordinate functions $(\Theta,\Psi)$ are uppercase.
Langfield, Krauskopf & Osinga (2014) use a period-normalized phase $\vartheta\in[0,1)$;
we use radians, $\Theta = 2\pi\vartheta$.

"Even in $r$" means $\rho(r) = \tilde\rho(r^2)$ for a smooth $\tilde\rho$ — equivalently,
$\rho$ has only even powers in its Taylor expansion at $0$. This is what makes the
cartesian vector field smooth at the origin (§2) and is a *requirement* on the defining
data, not a convenience.

**Why these quantities.** Isochrons are the level sets of $\Theta$ and isostables the
level sets of $\Psi$; the infinitesimal phase and amplitude response curves are
$\nabla\Theta$ and $\nabla\Psi$; $\kappa$, $T$ and the eigenvalues at the enclosed fixed
point are the smooth-conjugacy invariants the learned map must match (`learnings`);
$(\Theta,\Psi)$ is the chart in which the flow is *linear*. For the observed systems (FHN)
none of this is closed form, which is the reason the normal forms carry it.

---

## 2. Vector fields in the three charts

**Polar** (the defining chart):

$$
\dot r = r\rho(r), \qquad \dot\theta = \omega(r).
$$

**Cartesian** (what the data and the bijection see): with $r = |\mathbf{u}|$,

$$
\dot{\mathbf{u}} = \rho(r)\,\mathbf{u} + \omega(r)\,\mathsf{J}\mathbf{u},
\qquad\text{i.e.}\qquad
\dot x = \rho x - \omega y,\quad \dot y = \rho y + \omega x .
$$

*Derivation.* $\mathbf{u} = r\mathbf e_r$,
$\dot{\mathbf{u}} = \dot r\thinspace\mathbf e_r +r\dot\theta\thinspace\mathbf e_\theta = r\rho\thinspace\mathbf e_r + r\omega\thinspace\mathbf e_\theta = \rho\mathbf{u}+ \omega\mathsf{J}\mathbf{u}$, since $r\mathbf e_\theta = \mathsf{J}\mathbf{u}$. Because $\rho,\omega$
are even, $\rho(|\mathbf{u}|) = \tilde\rho(x^2+y^2)$ is a smooth function of $\mathbf{u}$ and the
field is smooth at the origin. *Numerical note*: $|\mathbf{u}| = \sqrt{x^2+y^2}$ has no
derivative at exactly $\mathbf{u}=0$, so the code never forms it — the cartesian `rhs` is
evaluated as $\tilde\rho(x^2+y^2)\thinspace\mathbf u + \tilde\omega(x^2+y^2)\thinspace\mathsf J\mathbf u$ (§9).

**Cartesian Jacobian.** For general even $\rho,\omega$,

$$
\mathsf{D}\dot{\mathbf{u}} = \rho(r)\,\mathsf{I} + \omega(r)\,\mathsf{J}
+ \frac{1}{r}\bigl(\rho'(r)\,\mathbf{u} + \omega'(r)\,\mathsf{J}\mathbf{u}\bigr)\mathbf{u}^{\mathsf T}.
$$

Written out for the Hopf form ($\rho = a(1-r^2)$, $\omega = \omega_0 + (\omega_1-\omega_0)r^2$),

$$
\mathsf{D}\dot{\mathbf{u}} = \begin{bmatrix}
a(1-r^2) - 2ax^2 - 2(\omega_1-\omega_0)xy &
-2axy - \omega_0 - (\omega_1-\omega_0)r^2 - 2(\omega_1-\omega_0)y^2 \\
-2axy + \omega_0 + (\omega_1-\omega_0)r^2 + 2(\omega_1-\omega_0)x^2 &
a(1-r^2) - 2ay^2 + 2(\omega_1-\omega_0)xy
\end{bmatrix}.
$$

(The $(1,2)$ entry had a sign error, $+(\omega_1-\omega_0)r^2$, in the earlier LaTeX notes;
it does not affect the value at the origin.) Since $\rho'(r), \omega'(r) = O(r)$ for even
functions, the rank-one term is $O(r^2)$ and the Jacobian at the origin is

$$
\mathsf{D}\dot{\mathbf{u}}(0) = \rho(0)\,\mathsf{I} + \omega(0)\,\mathsf{J}.
$$

**Phase–amplitude** $(\Theta, \Psi)$: $\dot\Theta = \omega_1$, $\dot\Psi = \kappa\Psi$ — the
flow is *linear*:

$$
\Theta(t) = \Theta_0 + \omega_1 t, \qquad \Psi(t) = \Psi_0\, e^{\kappa t}.
$$

This chart is defined on the basin minus the origin; its inverse needs the inverse of
$\Psi(r)$ (§7).

**Polar-to-cartesian conversion** (general planar flow $\dot r = f(r,\theta)$,
$\dot\theta = g(r,\theta)$):
$\dot x = x f/r - y g$, $\dot y = y f/r + x g$. For the two concrete forms this gives

$$
\begin{aligned}
\dot x &= a\,x\,(1-x^2-y^2)\bigl(1+b(x^2+y^2)\bigr) - y\bigl(\omega_0 + (\omega_1-\omega_0)(x^2+y^2)\bigr),\\
\dot y &= a\,y\,(1-x^2-y^2)\bigl(1+b(x^2+y^2)\bigr) + x\bigl(\omega_0 + (\omega_1-\omega_0)(x^2+y^2)\bigr),
\end{aligned}
$$

with $b = 0$ for Hopf.

---

## 3. The two concrete forms

Both have $\omega(r) = \omega_0 + (\omega_1 - \omega_0)\thinspace r^2$, so $\omega(1) = \omega_1$,
$\omega(0) = \omega_0$. Write $c := (\omega_1 - \omega_0)/a$.

**Hopf (Stuart–Landau).** The generic planar Hopf normal form is
$\dot r = r(\alpha + \beta r^2)$, $\dot\theta = \gamma + \delta r^2$; for $\alpha > 0$,
$\beta < 0$ it has an unstable focus at $r=0$ and a stable cycle at
$r^\ast = \sqrt{-\alpha/\beta}$. The cycle radius is immaterial for a conjugacy target, so we
fix $r^\ast=1$ by $\beta = -\alpha =: -a$, and reparametrise the angular rate by its value on
the cycle, $\omega_1 = \gamma+\delta$, and at the origin, $\omega_0 = \gamma$:

$$
\dot r = a\,r\,(1 - r^2), \qquad \dot\theta = \omega_0 + (\omega_1 - \omega_0)\,r^2 .
$$

**Bautin (generalized Hopf).** Adds a quintic term. Keeping the same stability pattern
(unstable focus at $0$, stable cycle at $r=1$) suggests the factorised parametrization

$$
\dot r = a\,r\,(1 - r^2)(1 + b r^2), \qquad \dot\theta = \omega_0 + (\omega_1 - \omega_0)\,r^2 ,
$$

so that $\rho(r) = a(1-r^2)(1+br^2) = a\bigl(1 + (b-1)r^2 - br^4\bigr)$ and

$$
\rho'(r) = 2ar\,(b - 1 - 2br^2), \qquad \rho'(1) = -2a(1+b).
$$

| | Hopf | Bautin |
|---|---|---|
| $\rho(r)$ | $a(1-r^2)$ | $a(1-r^2)(1+br^2)$ |
| $\kappa = \rho'(1)$ | $-2a$ | $-2a(1+b)$ |
| $\mu = e^{\kappa T}$ | $e^{-4\pi a/\omega_1}$ | $e^{-4\pi a(1+b)/\omega_1}$ |
| $h(r)$ | $c\ln r$ | $c\left[\ln r - \tfrac12\ln\dfrac{1+br^2}{1+b}\right]$ |
| $\Psi(r)$ | $\dfrac{r^2-1}{2r^2}$ | $\dfrac{(r^2-1)(1+br^2)^{b}}{2(1+b)^{b}\thinspace r^{2(1+b)}}$ |
| eigenvalues at $0$ | $a \pm i\omega_0$ | $a \pm i\omega_0$ |
| basin of $\Gamma$ | $\mathbb{R}^2\setminus\lbrace0\rbrace$ | $\mathbb{R}^2\setminus\lbrace0\rbrace$ if $b \ge 0$; $0 < r < 1/\sqrt{-b}$ if $-1 < b < 0$ |
| $\Psi$ as $r\to\infty$ (or $\to$ basin edge) | $\tfrac12$ | $\tfrac12\bigl(\tfrac{b}{1+b}\bigr)^{b}$ if $b > 0$; $+\infty$ at $r^2 = -1/b$ if $b < 0$ |

Stability condition: $\rho'(1) < 0 \iff b > -1$. The earlier notes required $a, b > 0$;
that is sufficient but stricter than needed. For $-1 < b < 0$ the factor $(1 + br^2)$ has a
second zero at $r^2 = -1/b$ with $\rho' > 0$ there: an **unstable outer cycle** bounding
the basin of $\Gamma$ (the two-cycle regime of the Bautin bifurcation, Kuznetsov Ch. 8).
Outside it $\rho > 0$ and $r\to\infty$ in finite time (quintic growth, $\dot r \sim a|b|r^5$).
As a conjugacy target $b \ge 0$ is used (`learnings`: $b$ decouples the fixed-point
instability $a$ from the cycle's contraction $\kappa = -2a(1+b)$, which is why Bautin was
chosen over cubic Hopf). Hopf is Bautin with $b = 0$ throughout.

---

## 4. Linear invariants

### 4.1 The fixed point at the origin

From §2, $\mathsf{D}\dot{\mathbf{u}}(0) = \rho(0)\mathsf{I} + \omega(0)\mathsf{J}$ with
eigenvalues $\rho(0) \pm i\thinspace\omega(0)$. For both forms this is

$$
\mathsf{D}\dot{\mathbf{u}}(0) = \begin{bmatrix} a & -\omega_0 \\ \omega_0 & a \end{bmatrix},
\qquad \lambda_{1,2} = a \pm i\omega_0 ,
$$

an unstable **focus** (spiral) for $a > 0$, $\omega_0 \ne 0$. The Bautin factor $(1+br^2)$
is $1 + O(r^2)$, so it cannot change the linearization at $0$. A smooth conjugacy
preserves these eigenvalues; a topological one preserves only the stability type.

### 4.2 Stability of the cycle: Floquet exponent and multiplier

Linearise the polar system about $r = 1$ (write $\delta r$, $\delta\theta$):

$$
\begin{bmatrix}\dot{\delta r}\\ \dot{\delta\theta}\end{bmatrix}
= \begin{bmatrix}\partial_r(r\rho) & 0\\ \omega'(1) & 0\end{bmatrix}_{r=1}
\begin{bmatrix}\delta r\\ \delta\theta\end{bmatrix}
= \begin{bmatrix}\rho'(1) & 0\\ \omega'(1) & 0\end{bmatrix}
\begin{bmatrix}\delta r\\ \delta\theta\end{bmatrix} =: \mathsf{A}
\begin{bmatrix}\delta r\\ \delta\theta\end{bmatrix},
$$

using $\partial_r(r\rho)\vert_{r=1} = \rho(1) + \rho'(1) = \rho'(1)$. For the two forms
$\mathsf{A} = \begin{bmatrix}-2a(1+b) & 0\cr 2(\omega_1-\omega_0) & 0\end{bmatrix}$.

**Convention [decided]: $\kappa = \rho'(1)$ with $\rho$ a function of $r$ — no factor 2.**
The same number written in $s = r^2$, $\dot s = 2s\tilde\rho(s)$, linearizes to
$2\tilde\rho'(1)$, and since $\rho'(r) = 2r\tilde\rho'(r^2)$ both agree
(Hopf: $\rho'(1) = -2a = 2\tilde\rho'(1)$). The pre-B11 code computed
`2·grad(radial_rate)(1)` *because* `radial_rate` took $s$; the factor of two was a chart
artefact.

**Multiplier.** The multipliers are the eigenvalues of the monodromy matrix $\mathsf{M}(T)$,
$\dot{\mathsf{M}} = \mathsf{A}(t)\mathsf{M}$, $\mathsf{M}(0) = \mathsf{I}$. Liouville's formula
(Chicone, Prop. 2.20) gives, for any planar cycle,

$$
\det\mathsf{M}(T) = \exp\!\int_0^T \mathrm{tr}\,\mathsf{A}(s)\,ds = \mu_1\mu_2,
$$

and since one multiplier is always $\mu_1 = 1$ (the direction along the flow),
$\mu_2 = \exp\int_0^T\mathrm{tr}\thinspace\mathsf{A}$. Here rotational symmetry makes $\mathsf{A}$
*constant* with $\mathrm{tr}\thinspace\mathsf{A} = \rho'(1) = \kappa$, so

$$
\mu = e^{\kappa T} = e^{2\pi\kappa/\omega_1}, \qquad
\mu_{\text{Hopf}} = e^{-4\pi a/\omega_1},\quad \mu_{\text{Bautin}} = e^{-4\pi a(1+b)/\omega_1}.
$$

(The earlier LaTeX notes dropped the $\pi$ in the Bautin multiplier.) Equivalently, since
$\mathsf{A}$ is constant, $\mathsf{M}(T) = e^{\mathsf{A}T}$ with eigenvalues $e^{0} = 1$ and
$e^{\kappa T}$ directly. Floquet multipliers are invariant under the smooth change of chart
polar $\leftrightarrow$ cartesian away from the origin, so these are the multipliers of the
cartesian field too.

---

## 5. Nonlinear coordinates

### 5.1 Asymptotic phase $\Theta$, the shift $h$, and isochrons

Seek $\Theta = \theta + h(r)$ with $\dot\Theta = \omega_1$ everywhere in the basin:

$$
\dot\Theta = \dot\theta + h'(r)\dot r = \omega(r) + h'(r)\,r\rho(r) = \omega_1
\quad\Longrightarrow\quad
h'(r) = \frac{\omega_1 - \omega(r)}{r\rho(r)},
$$

with $h(1) = 0$ so that $\Theta = \theta$ on the cycle. (Equivalently, with the earlier
notes' sign convention $\Theta = \theta - f(r)$, $f' = (\omega(r) - \omega_1)/\dot r$;
the general formula there was written with the opposite sign, though the substituted
result was right.) The integrand is finite at $r = 1$ — numerator and denominator vanish
linearly, ratio $\to -\omega'(1)/\rho'(1)$ — and the integral from $1$ to $r$ converges
for $r$ in the basin.

For the concrete forms, $\omega_1 - \omega(r) = (\omega_1-\omega_0)(1-r^2)$ cancels the
$(1-r^2)$ in $\rho$:

$$
h'(r) = \frac{c}{r(1+br^2)} = c\left(\frac1r - \frac{br}{1+br^2}\right),\qquad
h(r) = c\left[\ln r - \tfrac12\ln(1+br^2)\right] + C,
$$

and $h(1)=0$ gives $C = \tfrac{c}{2}\ln(1+b)$, i.e. the table in §3. Thus

$$
\Theta(r,\theta) = \theta + \frac{\omega_1-\omega_0}{2a}\,\ln\frac{(1+b)\,r^2}{1+br^2}
\qquad(\text{Hopf: } \theta + \tfrac{\omega_1-\omega_0}{a}\ln r).
$$

For Hopf this is the Stuart–Landau phase function of Nakao (2016, App. D).

**Isochrons** are $\Theta = \Theta_0$, i.e. the curves

$$
\theta = \Theta_0 - h(r).
$$

This is the primary representation (`isochron(Θ, r)` returns these points in cartesian
coordinates; `limit_cycle(Θ) = (cos Θ, sin Θ)`). For $\omega \equiv \omega_1$ ($c = 0$) they
are the radial lines $\theta = \Theta_0$; $\omega(r)\ne\omega_1$ shears them into spirals.
Amplitude dependence of the frequency is what the synchronization literature calls
*nonisochronicity* (Kuramoto; Pikovsky, Rosenblum & Kurths) or *shear*; for the standard
$\omega(r)$ the nonisochronicity parameter is $\omega_1-\omega_0$.

*Corollary — radius as a function of angle.* When $c \ne 0$ one can solve for $r$. On
the isochron $\Theta_0$, $h(r) = \Theta_0 - \theta$; let
$A^2 := \exp\negthinspace\bigl(2(\Theta_0 - \theta)/c\bigr)$ with $\theta$ unwrapped; then
$(1+b)r^2 = A^2(1+br^2)$ and

$$
r(\theta;\Theta_0) = \frac{A}{\sqrt{1 + b(1 - A^2)}},\qquad
A = \exp\frac{a(\Theta_0-\theta)}{\omega_1-\omega_0}
\qquad(\text{Hopf: } r = A).
$$

(Sign corrected 2026-10-03: the first version had $A^2 = \exp(2(\theta-\Theta_0)/c)$, the
reciprocal, which recovers $1/r$-like values; checked numerically against
$\theta = \Theta_0 - h(r)$.) Caveats: (i) it is singular at $c = 0$ (radial isochrons);
(ii) for $b > 0$ each isochron reaches $r\to\infty$ at the *finite* angle where
$A^2 = (1+b)/b$, because $h(r)\to \tfrac{c}{2}\ln\tfrac{1+b}{b}$ as $r\to\infty$, so the
formula is defined only for $A^2 < (1+b)/b$; (iii) for $-1 < b < 0$, $A^2 \to 0$ as $r\to 0$
and $A^2\to\infty$ is not reached — the isochron ends at the outer cycle. The form
$\theta = \Theta_0 - h(r)$ has none of these problems.

### 5.2 Isostable coordinate $\Psi$

**Definition.** $\Psi$ is the Koopman eigenfunction of the generator with eigenvalue
$\kappa$ (Mauroy, Mezić & Moehlis 2013; Kato et al. 2021, Eq. (2)/(12)):

$$
\dot{\mathbf{u}}\cdot\nabla\Psi = \kappa\,\Psi \quad\text{on the basin.}
$$

Wilson & Moehlis (2016) define the same object for periodic orbits through the Poincaré
map (their Eq. (5)) and fix its scale by **$\nabla\Psi\cdot\mathbf{v} = 1$ on $\Gamma$**
(their Eq. (14)), where $\mathbf{v}$ is the right Floquet eigenvector for $\mu$. Any
$\Psi\mapsto C\Psi$ satisfies the same equation (Kato et al.: "the scale of the amplitude
function is arbitrary"), so a normalization is a convention, not a modeling choice; it
matters only when comparing $\nabla\Psi$ (the isostable response curve) with values
reported elsewhere.

**Normalisation [decided]: Wilson–Moehlis.** For the normal forms the right Floquet vector
at $r=1$ is radial; with $\mathbf{v} = \mathbf e_r$ the condition reads

$$
\partial_r\Psi\big|_{r=1} = 1,\qquad\text{i.e.}\qquad \Psi \approx r - 1 \text{ near }\Gamma,
$$

so $\Psi < 0$ inside the cycle, $\Psi > 0$ outside, and $\Psi$ is the linear Floquet
coordinate $\delta r$ to first order. (The B10 draft proposed $\Psi \approx 1 - r^2$,
positive inside; that is the Hopf expression $(1-r^2)/r^2$ taken as reference. It is
superseded. Code and docstrings must agree with the sign here.)

**Derivation.** Seek $\Psi = \Psi(r)$: $\Psi'(r)\thinspace r\rho(r) = \kappa\Psi$, i.e.

$$
\frac{d\ln|\Psi|}{dr} = \frac{\kappa}{r\rho(r)}.
$$

Near the cycle $r\rho(r)\approx\rho'(1)(r-1) = \kappa(r-1)$, so $d\ln|\Psi|/dr \approx 1/(r-1)$ and $\Psi\propto (r-1)$: the isostable coordinate is **linear** in the distance to
the cycle, as it must be (its gradient on $\Gamma$ is the left Floquet vector).

For Bautin, in $s = r^2$ ($ds = 2r\thinspace dr$):

$$
\frac{d\ln|\Psi|}{ds} = \frac{-(1+b)}{s(1-s)(1+bs)},\qquad
\frac{1}{s(1-s)(1+bs)} = \frac1s + \frac{1}{(1+b)(1-s)} - \frac{b^2}{(1+b)(1+bs)},
$$

so $\ln|\Psi| = -(1+b)\ln s + \ln|1-s| + b\ln(1+bs) + \text{const}$, i.e.

$$
\Psi = C\,\frac{(s-1)(1+bs)^{b}}{s^{1+b}}.
$$

The leading behavior at $s=1$ is $C(1+b)^b (s-1) \approx 2C(1+b)^b (r-1)$, so the
normalization $\partial_r\Psi(1) = 1$ fixes $C = 1/\bigl(2(1+b)^b\bigr)$:

$$
\boxed{\;\Psi(r) = \frac{(r^2-1)\,(1+br^2)^{b}}{2\,(1+b)^{b}\,r^{2(1+b)}}\;}
\qquad\Bigl(\text{Hopf: } \Psi = \frac{r^2-1}{2r^2} = \tfrac12\bigl(1 - r^{-2}\bigr)\Bigr).
$$

**Monotonicity and range.** $\Psi' = \kappa\Psi/(r\rho)$ has the sign of $\kappa\Psi/\rho$,
which is positive both inside ($\Psi < 0$, $\rho > 0$) and outside ($\Psi > 0$, $\rho < 0$): $\Psi$
is strictly increasing on the basin, from $-\infty$ at $r = 0$ (where
$\Psi\sim -r^{-2(1+b)}/(2(1+b)^b)$, exponent $\kappa/a$) to

- $\tfrac12\bigl(\tfrac{b}{1+b}\bigr)^{b}$ as $r\to\infty$ for $b \ge 0$ (Hopf: $\tfrac12$);
- $+\infty$ at the outer cycle $r^2 = -1/b$ for $-1 < b < 0$, through the factor $(1+br^2)^{b}$.

So $r(\Psi)$ exists and is unique for every reachable $\Psi$ (§7). **Isostables** are the
circles $r = \text{const}$ — rotational symmetry puts *all* non-trivial isostable (and
isochron) geometry of a target system into the conjugating map, exactly as `learnings`
records for isochrons.

**Transport through a conjugacy.** If $H$ conjugates a target field $F$ to the normal form
with matched $\kappa$, then by the chain rule $(F\cdot\nabla)(\Psi\circ H) = \kappa\thinspace(\Psi\circ H)$,
so $\Psi\circ H$ is the target's isostable coordinate and $\mathsf{D}H^{\mathsf T}\thinspace\nabla\Psi\circ H$
its isostable response curve; likewise $\Theta\circ H$ and the phase response. The
Wilson–Moehlis scale is inherited only if the matched Floquet vector is normalized the same
way on the target side — this is what makes the normalization choice matter downstream.

### 5.3 The family generated by $\Psi$; the first-order member; Yawata et al.'s latent space

**$\Psi$ determines the radial field.** Reading the eigen-equation
$\Psi'(r)\thinspace r\rho(r) = \kappa\Psi(r)$ backwards, any strictly increasing $\Psi$ with
$\Psi(1) = 0$, $\Psi'(1) = 1$ defines a rotationally symmetric normal form

$$
\dot r = \kappa\,\frac{\Psi(r)}{\Psi'(r)}, \qquad \dot\theta = \omega(r),
$$

whose isostable coordinate is $\Psi$ by construction and whose Floquet exponent is $\kappa$.
Hopf and Bautin are the members with $\Psi = \tfrac12(1 - r^{-2})$ and the boxed
expression above (verified: substituting them recovers $\dot r = ar(1-r^2)$ and
$ar(1-r^2)(1+br^2)$). The polynomial degree of $\dot r$ in $r$ — cubic for Hopf, quintic
for Bautin — is a property of the chosen $\Psi$, not of the construction. Note that an
*odd polynomial* $\dot r = r(\alpha + \beta r^2 + \cdots)$ needs degree $\ge 3$ to have a
cycle at all: the degree-1 member $\dot r = kr$, $\dot\theta = \omega$ has no limit cycle
(the origin is a global focus for $k < 0$), so it cannot be a conjugacy target for an
oscillator. "First order" has to be understood as *linear in some coordinate other than
$r$*, which is exactly what the chart $(\Theta,\Psi)$ provides.

**The first-order (log-polar) member.** The simplest admissible $\Psi$ is

$$
\Psi(r) = \ln r:\qquad \dot r = \kappa\, r\ln r,\quad \dot\theta = \omega(r),\qquad
\rho(r) = \kappa\ln r .
$$

In log-polar coordinates $(\ln r, \theta)$ the flow is literally linear, $\tfrac{d}{dt}\ln r = \kappa\ln r$, so the $(\Theta,\Psi)$ chart *is* the log-polar chart: $h\equiv 0$ when
$\omega\equiv\omega_1$, isochrons are the rays $\theta = \Theta_0$, isostables the circles,
$r(t) = r_0^{\thinspace e^{\kappa t}}$. The alternative "affine" choice $\Psi = r - 1$ gives
$\dot r = \kappa(r-1)$, which is linear in $r$ but has $\dot r = -\kappa > 0$ at the origin:
the origin is not a fixed point and the field is not even continuous there as a planar
vector field, so it is excluded. The log-polar member keeps the origin fixed
($r\ln r\to 0$) and its basin is $\mathbb{R}^2\setminus\lbrace0\rbrace$, but it violates the "even in
$r$" requirement of §1: $\rho(0) = \kappa\ln 0 = +\infty$, the cartesian field
$\dot{\mathbf u} = \kappa\ln|\mathbf u|\thinspace\mathbf u + \omega\mathsf{J}\mathbf u$ is continuous
but not $C^1$ at the origin, and **the eigenvalues at the origin do not exist**. Its
isostable coordinate diverges only logarithmically at the focus, whereas for a
hyperbolic unstable focus with eigenvalues $\alpha\pm i\beta$ (Hopf: $\alpha = a$) the
isostable coordinate diverges like $\mathrm{dist}^{\kappa/\alpha}$ (Hopf:
$\Psi\sim -\tfrac12 r^{-2} = -\tfrac12 r^{\kappa/a}$). Consequences for this project:

- the invariant "eigenvalues at the enclosed fixed point" (`learnings`) cannot be matched,
  so a conjugacy to FHN can only be a diffeomorphism of the *punctured* basin, and must
  blow up (exponentially in $\Psi$) as it approaches the equilibrium;
- in exchange the member has one parameter fewer ($a$ disappears; only $\kappa$,
  $\omega_1$, $\omega_0$) and the smallest possible nonlinearity in the base.

It is admissible as a *third* subclass (`LogPolarNormalForm`) provided the base class
does not assume finite `eigenvalues_origin()`; it is **deferred** (§11). The row for the §3 table would be: $\rho = \kappa\ln r$; $\kappa$ free;
$\mu = e^{\kappa T}$; $\Psi = \ln r$; eigenvalues at $0$: none; basin
$\mathbb{R}^2\setminus\lbrace0\rbrace$; $\Psi\to\pm\infty$ as $r\to\infty, 0$. The angular rate
should *not* be the standard $\omega_0 + (\omega_1-\omega_0)r^2$ here: the $(1-r^2)$ no
longer cancels against $\rho$ and $h' = (\omega_1-\omega_0)(1-r^2)/(\kappa r\ln r)$
integrates to $\ln|\ln r| - \mathrm{Ei}(2\ln r)$, not elementary. The natural shear
for this member is linear in the log chart, $\omega(r) = \omega_1 + \delta\ln r$, giving
$h(r) = -(\delta/\kappa)\ln r$ (verified), isochrons $\theta = \Theta_0 + (\delta/\kappa)\ln r$ — logarithmic spirals, as for Hopf — and $\omega(0)$ undefined,
consistent with the missing eigenvalues.

**Relation to Yawata et al. (2024).** Their phase autoencoder maps the oscillator state
$X$ to a three-dimensional latent $\mathbf Y = (Y_1, Y_2, Y_3) = f_{\rm enc}(X)$ with
$Y_1^2 + Y_2^2 = 1$ (Eq. (11), enforced by normalizing the first two encoder outputs,
Eqs. (15)–(16)), $(Y_1, Y_2)$ rotating at a constant learned frequency $\omega$ and $Y_3$
decaying as $e^{\lambda\tau}$ per sampling interval $\tau$ with learned $\lambda < 0$
(Eqs. (12)–(14)), "for all $\tau > 0$". The phase is $\Theta(X) = \arctan(Y_2/Y_1)$
(Eq. (19)) and the phase sensitivity function is $Z(\theta) = \nabla_X\Theta\vert_{X = f_{\rm dec}(\theta)}$ by autodiff (Eq. (20)). Sec. VI identifies $Y_1 + iY_2$ with the
Koopman eigenfunction of exponent $i\omega$ and, for planar oscillators, $Y_3$ with the
eigenfunction of exponent $\lambda$ — i.e. with $e^{i\Theta}$ and $\Psi$, $\lambda = \kappa$
— while noting that the trained $Y_3$ is "closely related, though not equivalent" to it
(their learned $\lambda$ differs from the true second Floquet exponent because Eq. (14)
"is not strictly satisfied by the trained phase autoencoder"; they report it is "not easy
to realize in practice"). So their latent space is exactly the chart of §7,

$$
\mathbf Y = (\cos\Theta, \sin\Theta, \Psi),
$$

a cylinder embedded in $\mathbb R^3$, and their latent dynamics is the linear flow
$\dot\Theta = \omega_1$, $\dot\Psi = \kappa\Psi$ of §2 discretized at $\tau$.

Two things about this chart that the baseline inherits from §5.4 rather than from the
autoencoder: the exact $(\Theta, \Psi)$ **exists and is unique on the whole basin**, as a
$C^{k,\alpha}$ *diffeomorphism* $Q \to S^1 \times \mathbb R$ (Kvalheim & Revzen 2021,
Prop. 3 — see §5.4), so the cylinder is not a near-cycle approximation but the global
structure of every planar oscillator's basin; and the two conventions the autoencoder
cannot learn — the phase origin and the scale of $\Psi$ — are precisely the two
normalizations in that uniqueness statement ($\psi_\theta(x_0) = 1$ and
$\mathsf D_{x_0}\psi_z\vert_{E^s} = \mathrm{id}$, the latter being the Wilson–Moehlis choice
of §5.2). What is *approximate* in Yawata et al. is the learned map, and it is trained
near the cycle by construction: initial states are points of the cycle perturbed by
Gaussian noise of $\gamma_2 = 0.5$ cycle-standard-deviations and evolved for
$\gamma_1 = 3$ periods ($N_s = 1000$ orbits; Eqs. (27)–(28)), which is why their
$\lambda$ is poorly determined. Which planar normal form one associates with the chart is
a matter of choosing $\Psi(r)$: the log-polar member corresponds to $Y_3 = \ln r$, Hopf to
$Y_3 = \tfrac12(1 - r^{-2})$, and so on.

What distinguishes this project's setting is topological. The chart $(\Theta, \Psi)$ maps
the *punctured* plane onto the cylinder; there is no diffeomorphism of $\mathbb R^2$ onto
$S^1\times\mathbb R$. The autoencoder sidesteps this by not being invertible (separate
encoder and decoder, the cylinder embedded in $\mathbb R^3$). The conjugacy approach keeps
a diffeomorphism of the plane, $H:\mathbb R^2\to\mathbb R^2$, and lets the *normal form's
own* chart $\psi_{\rm NF}: \mathbb R^2\setminus\lbrace0\rbrace\to S^1\times\mathbb R$ carry the
topology: $\psi_{\rm FHN} = \psi_{\rm NF}\circ H$ (§5.2, transport). This is the precise
sense in which Hopf/Bautin "add" the fixed point to Yawata's picture (§5.4 for what that
costs), and why the log-polar member sits between the two.

*Implementation mapping (roadmap B12).* The baseline lives in `model/`, not in
`systems/normal_forms/`:

| Yawata et al. | this document | `deep_isochron` |
|---|---|---|
| $(Y_1, Y_2)/R$, $Y_3$ (Eqs. (15)–(16)) | $(\cos\Theta, \sin\Theta, \Psi)$ | `PhaseAmplitudeAutoencoder.encoder`: MLP, first two outputs normalized |
| $f_{\rm step}$ with learned $\omega$, $\lambda$ (Eq. (18)) | $\Theta(t+\tau) = \Theta + \omega_1\tau$, $\Psi(t+\tau) = e^{\kappa\tau}\Psi$ | `PhaseAmplitudeLatentDynamics(omega, kappa)`: closed-form flow on $\mathbb R^3$, continuous in $t$ (replaces `LinearLatentDynamics`) |
| $L_{\rm recon}$ (21); $L_{\rm pha}$, $L_{\rm dev}$ over $k = 1..K$ steps with $\alpha_k = k^{-\min(1, L_{\rm pha})}$ (22)–(24); $L_{\rm aux}$ = center of mass of $(Y_1,Y_2)$ in the batch (25), to escape the $\omega = 0$ solution; weights $(1, 0.5, 0.5, 2)$ then $(1, 5, 0.5, 0)$ once $L_{\rm pha} < 0.01$, $L_{\rm aux} < 0.05$ (26) | — | `PhaseAutoencoderLoss` over a window: reconstruction, latent consistency split into phase/amplitude parts, center-of-mass term; the $\alpha_k$ and weight schedules as config |
| ICs on the cycle $+\thinspace\gamma_2\sigma\odot\xi$, evolved $\gamma_1 T$ (27)–(28) | near-cycle training distribution | an `AbstractICSampler` `OnCycleGaussian(cycle_points, gamma2)`; `limit_cycle(Θ)` for normal forms, a long integration for FHN |
| $\Theta = \arctan(Y_2/Y_1)$, $Z = \nabla\Theta$ at $f_{\rm dec}(\theta)$ (19)–(20) | $\Theta$, $\nabla\Theta$ (§1) | `phase(x)`, `jax.grad` |
| evaluation: Stuart–Landau analytic phase | §7 `to_phase_amplitude` on Hopf/Bautin data | learned $\Theta$ vs exact up to a constant; $\Psi$ up to scale; $\kappa$ *reported*, not asserted |

### 5.4 (*tentative*) Global existence and uniqueness of the chart (Kvalheim & Revzen 2021)

The closed forms of §5.1–5.2 are special to the normal forms, but the *objects* they
compute exist for every oscillator in the study, on the whole basin, and are unique. This
is the content of Kvalheim & Revzen's global Floquet normal form, specialised here to the
plane.

**Statement** (their Prop. 3, specialised). Let $\Phi$ be a $C_{\rm loc}^{k,\alpha}$ flow on
the basin $Q$ of an attracting hyperbolic $\tau$-periodic orbit $\Gamma$, $x_0\in\Gamma$,
$E_{x_0}^s$ the $\mathsf D_{x_0}\Phi^\tau$-invariant complement of $\mathsf T_{x_0}\Gamma$,
and assume the spectral-spread condition $\nu(\mathsf D\Phi^\tau\vert_{E^s}, \mathsf D\Phi^\tau\vert_{E^s}) < k+\alpha$
and $k$-nonresonance of $\mathsf D\Phi^\tau\vert_{E^s}$ with itself. Then there is a **unique
proper $C_{\rm loc}^{k,\alpha}$ embedding** $\psi = (\psi_\theta, \psi_z): Q\to S^1\times (E_{x_0}^s\otimes\mathbb C)$ with $\psi_\theta(x_0) = 1$ and $\mathsf D_{x_0}\psi_z\vert_{E^s} = \mathrm{id}$ such that

$$
\psi_\theta\circ\Phi^t = e^{2\pi i t/\tau}\,\psi_\theta,\qquad \psi_z\circ\Phi^t = e^{tA}\psi_z
\qquad(\text{their Eq. (21)}),
$$

and when $A$ is real, $\psi: Q\to S^1\times E_{x_0}^s$ is a **diffeomorphism**. Their
Prop. 7 states the same for the principal eigenfunction (the isostable coordinate) alone:
existence and uniqueness modulo scalar multiplication, given the derivative along $E^s$.

**In the plane.** $\dim E^s = 1$, $\mathsf D\Phi^\tau\vert_{E^s} = \mu = e^{\kappa\tau}\in(0,1)$ is
real, so the spectral spread is $\nu(\mu,\mu) = \ln\mu/\ln\mu = 1 < k+\alpha$ for any
$k\ge 2$ or ($k = 1$, $\alpha > 0$), and $k$-nonresonance ($\mu\ne\mu^m$, $m\ge2$) holds
because $0 < \mu < 1$. For a $C^\infty$ field (FHN is polynomial) the conditions are
automatic (their Remark 6). Hence, *for every planar oscillator in this project*:

$$
(\Theta, \Psi) := \bigl(\arg\psi_\theta,\ \psi_z\bigr): \ Q \to S^1\times\mathbb R
\quad\text{is a $C^\infty$ diffeomorphism of the whole basin},\qquad
\dot\Theta = \omega_1,\ \dot\Psi = \kappa\Psi .
$$

For the normal forms this is §5.1–5.2 with $Q = \mathbb R^2\setminus\lbrace0\rbrace$ ($b\ge0$) and
the uniqueness explains why there was nothing to choose beyond $h(1) = 0$ and
$\partial_r\Psi(1) = 1$: those are exactly the two normalizations in the statement. For FHN
it is Langfield et al.'s isochron foliation together with the (there uncomputed)
amplitude coordinate; $Q = \mathbb R^2\setminus\lbrace x^\ast\rbrace$, the plane minus the equilibrium
(Winfree's phaseless set).

**Consequence for the conjugacy (deduced, not in the paper).** If the target and the
normal form have the same period $T$ and Floquet exponent $\kappa$, then

$$
H := \psi_{\rm NF}^{-1}\circ\psi_{\rm target}: \ \mathbb R^2\setminus\{x^*\}\ \to\ \mathbb R^2\setminus\{0\}
$$

is a $C^\infty$ diffeomorphism conjugating the two flows on the punctured basins, and it is
**unique up to the two-parameter group** generated by the normalizations — a phase shift
$\Theta\mapsto\Theta + c$ (rotation of the normal form) and a scale $\Psi\mapsto C\Psi$ —
because any two such $H$ differ by a self-conjugacy of the normal form on its basin, and
uniqueness pins those down. Conversely, if $T$ or $\kappa$ differ no conjugacy exists
(they are invariants). This is the existence/uniqueness theory behind the project's
"co-optimize the invariants with the INN" (`learnings`): once the invariants match, the
map the INN is asked to learn exists, is smooth on the punctured plane, and is determined
up to a rotation and an amplitude scale.

**The fixed point.** $H$ above is defined on the punctured plane. Extending it *smoothly*
across $x^\ast\mapsto 0$ is a separate question, governed by the linearization at the
equilibrium: in reverse time $x^\ast$ is a hyperbolic sink whose basin is the open disk
bounded by $\Gamma$, and their Prop. 2 (global Sternberg linearization) gives a unique
$C^{k,\alpha}$ linearizing diffeomorphism of that disk, determined by its derivative at
$x^\ast$, provided the eigenvalues $\alpha\pm i\beta$ at $x^\ast$ are nonresonant. So a smooth
conjugacy *including the fixed point* requires, in addition, the eigenvalues at $x^\ast$ to
equal $a\pm i\omega_0$ — three further invariants are therefore one period, one Floquet
exponent and one complex eigenvalue, i.e. four real numbers, which is exactly the number
of parameters $(a, b, \omega_1, \omega_0)$ of the Bautin form and one more than Hopf has.
That the two local conjugacies (basin-of-$\Gamma$ and disk-of-$x^\ast$) glue into one
diffeomorphism of the plane is plausible by the uniqueness statements on the overlap (the
symmetry groups coincide: rotation $\leftrightarrow$ $\Theta$-shift, complex scaling at the
focus $\leftrightarrow$ $\Psi$-scale, since $\Psi\sim -\tfrac12 r^{\kappa/a}$ near $0$) but is
**not proved here**; it is the statement the project's "eigenvalue invariant" rests on
and should be either proved or cited before the paper. Without the eigenvalue match, $H$
is still a diffeomorphism of the punctured planes extending to a *homeomorphism* of the
planes (both ends compactify to a point), which is the NCF-style topological conjugacy
(`learnings`: homeomorphism vs diffeomorphism trade-off).

**For the baseline (B12).** The autoencoder learns $\psi_{\rm target}$ directly — the
cylinder *is* the global target, Prop. 3 guarantees it exists and is smooth — without
the planar intermediate; its two free conventions are the two normalizations; and the
quality of its $\Psi$ is limited by the near-cycle training distribution, not by the
theory.

---

## 6. Trajectories

### 6.1 Closed form via the phase–amplitude chart

Since $\Psi(t) = \Psi_0 e^{\kappa t}$ and $\Theta(t) = \Theta_0 + \omega_1 t$,

$$
r(t) = \Psi^{-1}\!\bigl(\Psi(r_0)\,e^{\kappa t}\bigr), \qquad
\theta(t) = \theta_0 + h(r_0) + \omega_1 t - h\bigl(r(t)\bigr).
$$

This is what `learnings` records as "the radial ODE admits a closed-form antiderivative
via partial fractions; the angular integral has an exact closed form": $\ln|\Psi|$ *is*
that antiderivative, and $h(r_0) - h(r(t))$ *is* the angular integral
$(\omega_1-\omega_0)\int_0^t (r^2 - 1)\thinspace dt'$.

**Hopf is explicit.** $\Psi = \tfrac12(1 - r^{-2})$ inverts in closed form:
$1 - r^{-2} = (1 - r_0^{-2})e^{-2at}$, hence

$$
r(t) = \sqrt{\frac{r_0^2}{r_0^2 + (1 - r_0^2)\,e^{-2at}}},\qquad
\theta(t) = \theta_0 + \omega_1 t + \frac{\omega_1-\omega_0}{a}\ln\frac{r_0}{r(t)} .
$$

(The earlier notes derive the same $r(t)$ for the unscaled form $\dot r = \alpha r - r^3$,
$\dot\theta = 1$, as $r = r_0\sqrt{\alpha/(r_0^2 + (\alpha - r_0^2)e^{-2\alpha t})}$;
rescaling $r\mapsto r/\sqrt\alpha$ gives $a = \alpha$, $\omega_1=\omega_0=1$ and the
formula above.)

**Bautin needs one scalar root solve** per output time: $\Psi$ is strictly monotone (§5.2),
so $r(\Psi)$ is a bracketed 1-D root on $(0, r_{\max})$ with $r_{\max} = 1/\sqrt{-b}$ for
$b < 0$ and a growing bracket otherwise. No ODE solver, no step-size error; exactly the map
the conjugacy learns. Implemented in B11 as `ClosedFormIntegration`, bisection-safeguarded
Newton as in `CubicBSpline` (ADR-0006).

### 6.2 Numerical integrations

All take cartesian $\mathbf u_0$ and return a `diffrax.Solution` with cartesian `.ys`.

| name | state integrated | singular at | notes |
|---|---|---|---|
| `cartesian` | $\mathbf{u}\in\mathbb{R}^2$, `rhs` | nowhere | the only one that can start at $0$ |
| `polar` | $(r,\theta)$, `rhs_polar` | $r=0$ | $\theta$ unwrapped |
| `r_squared` | $(s,\theta)$: $\dot s = 2s\tilde\rho(s)$, $\dot\theta = \omega(\sqrt s)$ | $r=0$ (via $\theta_0$, $\sqrt s$) | polynomial RHS for Hopf/Bautin; the former `BautinNormalForm.solve` |
| `closed_form` (B11) | none — $(\Theta,\Psi)$ chart | $r=0$ | §6.1; needs $\Psi^{-1}$ |

For Bautin the `r_squared` system is
$\dot s = 2a\thinspace s(1-s)(1+bs)$, $\dot\theta = \omega_0 + (\omega_1-\omega_0)s$ — the earlier
notes used the equivalent pair $(u, v) = (r^2, \int_0^t r^2)$ and reported that adaptive
explicit Runge–Kutta methods sometimes failed to converge for larger $b$ in this chart,
falling back to the implicit `Kvaerno5` (rtol $10^{-4}$, atol $10^{-6}$). The current
default is `Tsit5` + `PIDController` in double precision; the closed-form integration
removes the question for the normal forms and is the natural reference the three numerical
integrations are tested against.

---

## 7. The phase–amplitude chart as API

```
to_phase_amplitude(u)       = (Θ(u), Ψ(u))                                  # closed form
from_phase_amplitude(Θ, Ψ)  = (r(Ψ) cos(Θ − h(r)), r(Ψ) sin(Θ − h(r)))      # r(Ψ): monotone root
```

With $r(\Psi)$ in hand, `from_phase_amplitude` and `ClosedFormIntegration` are the same
code.

$(\cos\Theta, \sin\Theta, \Psi)$ is the latent space of the phase autoencoder of
Yawata et al. (2024); see §5.3.

---

## 8. Parameters

Unconstrained leaves, constrained on read (ADR-0001 in spirit):
`a = GreaterThan(0)(raw_a)` (raw $0\mapsto a=1$),
`b = GreaterThan(-1)(raw_b)` (raw $0\mapsto b=0$, the Hopf form);
$\omega_1$ (`w`), $\omega_0$ (`w0`) free; `w0` defaults to `w` (no shear). `params()`
reports the constrained values. $b > -1$ is exactly the stability condition $\rho'(1) < 0$
(§3).

---

## 9. API implied (for B11)

**Decided: Option B.** The defining data are even functions of $r$, which is enforced
*by construction* by having subclasses supply them as smooth functions of $s = r^2$; the
public API is entirely in $r$. The $s$-chart appears in exactly two places: the abstract
`*_sq` hooks and the `r_squared` integration.

```python
class AbstractNormalForm(AbstractODE):
    # defining data — abstract, in s = r²  (ρ̃(s) = ρ(√s), ω̃(s) = ω(√s))
    def _log_growth_rate_sq(self, s): ...   # ρ̃(s): ṙ = r ρ̃(r²); ρ̃(1) = 0, ρ̃'(1) < 0
    def _angular_rate_sq(self, s):    ...   # ω̃(s): θ̇ = ω̃(r²)
    def phase_shift(self, r):        ...   # h(r), h(1) = 0            (closed form, in r)
    def isostable(self, r):          ...   # Ψ(r), Ψ(1) = 0, Ψ'(1) = 1 (closed form, in r)

    # public r-chart views of the defining data (final)
    def log_growth_rate(self, r): return self._log_growth_rate_sq(r * r)   # ρ(r)
    def angular_rate(self, r):    return self._angular_rate_sq(r * r)      # ω(r)

    # derived (final)
    omega() = angular_rate(1.0)      period()
    floquet_exponent() = grad(log_growth_rate)(1.0)      floquet_multiplier() = exp(κ T)
    eigenvalues_origin() = _log_growth_rate_sq(0.0) ± i _angular_rate_sq(0.0)
    rhs(t, u)      # s = u·u;  _log_growth_rate_sq(s) * u + _angular_rate_sq(s) * J u
    rhs_polar(t, (r, θ))
    phase(u)  amplitude(u)  limit_cycle(Θ)  isochron(Θ, r)
    to_polar / from_polar (polar);  to_phase_amplitude / from_phase_amplitude (§7)
    flow(ts, u0, *, config, integration="r_squared")
```

- **Names.** `log_growth_rate` is $\rho = \dot r/r = d\ln r/dt$; the literature names its
  value at the origin (the *linear growth rate*), its cubic coefficient (the *first
  Lyapunov coefficient*) and its slope at the cycle (the Floquet exponent), but not the
  function itself, so the name states what it is. `radial_rate` (ambiguous between $\dot r$
  and $\dot r/r$) and `growth_rate` (invites the $\dot r$ reading) are rejected.
  `angular_rate` is $\omega(r)$; "frequency" is avoided because of the $\omega$ vs
  $\omega/2\pi$ ambiguity. The `_sq` suffix marks the $s$-chart hooks.
- **Why the data live in $s$.** With $\tilde\rho,\tilde\omega$ smooth in $s$, the cartesian
  field $\tilde\rho(|\mathbf u|^2)\thinspace\mathbf u + \tilde\omega(|\mathbf u|^2)\thinspace\mathsf J\mathbf u$
  is smooth at the origin with no $\sqrt{}$ anywhere: for Hopf/Bautin it is literally a
  polynomial in $(x,y)$, and *all* its derivatives at the origin are exact under autodiff.
  The rejected Option A (data in $r$, cartesian `rhs` recovering $r=|\mathbf u|$ through a
  double-`where` safe-$\sqrt{}$ device) gives the right Jacobian at exactly $\mathbf u = 0$
  only because of the device, the right second derivatives there only by coincidence (both
  are zero), wrong third and higher derivatives there, and cannot stop a subclass from
  supplying a non-even $\rho$ and silently producing a non-smooth field. Away from the
  origin the two are identical. Nothing in the project evaluates the base field at exactly
  the origin, so the practical difference is small; the structural guarantee is the
  reason for B.
- **Factor 2.** `floquet_exponent` is `grad(log_growth_rate)(1.0)` — the $r$-chart
  derivative, $\kappa = \rho'(1)$, by §4.2. Autodiff through `r * r` supplies
  $\rho'(1) = 2\tilde\rho'(1)$; no manual factor anywhere. Concrete on the base class, with
  the subclasses' closed forms tested against it.
- **`r_squared` integration** calls `_log_growth_rate_sq(s)` and `_angular_rate_sq(s)`
  directly: polynomial RHS, no $\sqrt{}$; the chart is singular at the origin only through
  $\theta_0$.
- **`polar` integration, `phase_shift`, `isostable`, the root solve** are all in $r$ via the
  public views.
- **`phase_shift` and `isostable` stay in $r$** because their closed forms (§3) are most
  legible there and they are never evaluated at the origin (both diverge). They are
  abstract per subclass; `test_phase_and_isostable_identities` checks them against the
  `_sq` data by autodiff, so a subclass cannot get them inconsistent unnoticed.
- The `AbstractNormalForm` docstring points to this document, §1 (evenness) and §9.

---

## 10. Tests the document implies

Existing tests carry over with $\kappa = $ `grad(log_growth_rate)(1)`. New or changed:

- $\Psi$ normalization: `grad(isostable)(1.0) == 1` and $\Psi(1) = 0$ for both forms, and
  $\Psi < 0$ for $r < 1$;
- the defining identities $\dot{\mathbf{u}}\cdot\nabla\Theta = \omega_1$ and
  $\dot{\mathbf{u}}\cdot\nabla\Psi = \kappa\Psi$ by autodiff on the basin, including
  $-1 < b < 0$ inside the outer cycle (`test_phase_and_isostable_identities`), and along
  integrated trajectories (`test_phase_and_amplitude_along_the_flow`);
- Jacobian of `rhs` at the origin equals $\rho(0)\mathsf{I} + \omega(0)\mathsf{J}$, and
  matches the explicit Hopf Jacobian of §2 at a few generic points; `log_growth_rate(r)
  == _log_growth_rate_sq(r*r)` (trivial, but pins the public/private contract);
- `floquet_multiplier == exp(floquet_exponent · period)`, and for Hopf equals the
  monodromy of the polar linearization integrated over one period;
- `to_phase_amplitude ∘ from_phase_amplitude = id` on the basin;
- `closed_form` agrees with `cartesian` to `TOL["flow"]`; Hopf `closed_form` agrees with
  the explicit $r(t)$ of §6.1;
- the $r\to\infty$ limit of $\Psi$ (table in §3), and $\Psi\to+\infty$ at the outer cycle
  for $b < 0$.

---

## 11. Decisions

**Decided 2026-10-03.**

- Notation: $\Theta$ phase, $\Psi$ isostable (uppercase coordinate functions), $\theta$
  polar angle, $\kappa$ exponent, $\mu$ multiplier, $\omega_1$ (`w`) / $\omega_0$ (`w0`).
- $\kappa = \rho'(1)$ with $\rho(r)$; no factor 2.
- $\Psi$ normalized à la Wilson & Moehlis: $\partial_r\Psi(1) = 1$, negative inside.
  Bautin's expression is divided by $2(1+b)^b$ and sign-flipped relative to the B10 draft.
- This document lives in the repo as Markdown with LaTeX math; the LaTeX manuscript
  section is to be regenerated from it, not maintained separately.
- `ClosedFormIntegration` and the $(\Theta,\Psi)$ chart go into B11 (§6.1, §7); the Hopf
  trajectories were already computed from the analytical formula before, so this
  restores that and extends it to Bautin via the root solve.
- The log-polar member of §5.3 is **deferred**: it cannot match the fixed-point invariant
  the project relies on, and has no concrete use until a Yawata-style baseline (roadmap
  B12) needs it. If added later it is a third subclass with non-finite
  `eigenvalues_origin()` and a non-standard `angular_rate`.
- Names: `log_growth_rate` for $\rho$, `angular_rate` for $\omega$, with `_sq` suffix for
  the $s$-chart hooks (§9).
- Defining data in $s$ with the public API in $r$ (**Option B**, §9); Option A and its
  safe-$\sqrt{}$ device are rejected.

Nothing remains open. Changes to any of the above go through this document first.

---

## References

- A. T. Winfree, *The Geometry of Biological Time*, 2nd ed., Springer (2001) — isochrons,
  asymptotic phase.
- J. Guckenheimer, Isochrons and phaseless sets, *J. Math. Biol.* 1, 259–273 (1975).
- J. Guckenheimer, P. Holmes, *Nonlinear Oscillations, Dynamical Systems, and Bifurcations
  of Vector Fields*, Springer (1983) — Floquet theory.
- C. Chicone, *Ordinary Differential Equations with Applications*, 3rd ed., Springer
  (2024) — Liouville's formula, Prop. 2.20.
- Yu. A. Kuznetsov, *Elements of Applied Bifurcation Theory*, 3rd ed., Springer (2004),
  Ch. 8 — the Bautin (generalized Hopf) normal form and its two-cycle regime.
- H. Nakao, Phase reduction approach to synchronization of nonlinear oscillators,
  *Contemp. Phys.* 57(2), 188–214 (2016) — phase function $\Theta(\mathbf{X})$; Stuart–Landau
  isochrons (App. D).
- A. Mauroy, I. Mezić, J. Moehlis, Isostables, isochrons, and Koopman spectrum for the
  action–angle representation of stable fixed point dynamics, *Physica D* 261, 19–30
  (2013) — isostables as level sets of a Koopman eigenfunction.
- D. Wilson, J. Moehlis, Isostable reduction of periodic orbits, *Phys. Rev. E* 94, 052213
  (2016) — isostable coordinates $\psi_i$ of limit cycles, Eq. (5); normalization
  $\nabla\psi_i\cdot\mathbf v_i = 1$, Eq. (14).
- Y. Kato, J. Zhu, W. Kurebayashi, H. Nakao, Asymptotic phase and amplitude for classical
  and semiclassical stochastic oscillators via Koopman operator theory, *Mathematics*
  9(18), 2188 (2021) — eigenfunction characterization of phase and amplitude, Eqs. (2),
  (12); arbitrariness of the amplitude scale.
- P. Langfield, B. Krauskopf, H. M. Osinga, Solving Winfree's puzzle: the isochrons in the
  FitzHugh–Nagumo model, *Chaos* 24, 013131 (2014) — FHN isochrons; period-normalized
  phase $\vartheta\in[0,1)$.
- K. Yawata, K. Fukami, K. Taira, H. Nakao, Phase autoencoder for limit-cycle
  oscillators, *Chaos* 34, 063111 (2024) — latent variables $(Y_1,Y_2,Y_3)$, Eqs.
  (11)–(14); relation to Koopman eigenfunctions, App. B and Sec. VI.
- M. D. Kvalheim, S. Revzen, Existence and uniqueness of global Koopman eigenfunctions
  for stable fixed points and periodic orbits, *Physica D* 425, 132959 (2021);
  arXiv:1911.11996v4 — Theorem 2 / Prop. 3 (global Floquet normal form, unique proper
  embedding $Q\to S^1\times E^s$), Prop. 7 (principal eigenfunctions of a limit cycle),
  Prop. 2 (global Sternberg linearization of a sink), Remark 6 ($C^\infty$ case).
- P. Kidger, *On Neural Differential Equations*, DPhil thesis, Oxford (2021) — diffrax.
