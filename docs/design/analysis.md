---
type: design
status: tentative
updated: 2026-10-06
verified_by: pending (Joon); the E1 oracle tests in tests/test_analysis.py pass
sources: [ADR-0012 (analysis layout), roadmap Phase E, docs/design/normal-forms.md (phase conventions)]
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
frozen dataclass holding NumPy arrays — the analysis functions are host-side,
non-differentiable post-processing, so JAX buys nothing here (ADR-0012).

`Cycle` provides the curve ($\gamma$, $\gamma'$), sampling (`phases`, `points`), point
queries (`nearest_phase`, `nearest_point`, `distance`: dense argmin over `resolution`
samples followed by one Newton step on the stationarity condition
$(\gamma(\varphi) - x) \cdot \gamma'(\varphi) = 0$) and two constructors: `from_points(phases, points, period)` (least
squares in the $1 + 2K$ coefficients — any order, any spacing, repeated phases, which is
what pooled trajectories produce) and `circle` (the normal forms' cycle in their own
coordinates). E3 adds `winding` and `contains`.

## 1. E1 — `analysis.data.estimate_cycle`: the cycle from trajectory tails

**The problem.** From one or several trajectories of a planar oscillator sampled on a
common time grid — converging to the cycle from initial conditions anywhere, possibly
with measurement noise — find the limit cycle as a `Cycle` (curve, period, phase
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
- **Kralemann, Cimponeriu, Rosenblum, Pikovsky & Mrowka 2008**, *Phase dynamics of
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

  i.e. the cumulative density rescaled to $2\pi$. Our `phase_from_protophase` is this
  formula with the $k \ne 0$ sum written as $2\thinspace\mathrm{Re}\sum_{k \ge 1}$.
- **Gengel & Pikovsky 2021/2022**, *Phase reconstruction with iterated Hilbert
  transforms* (chapter in *Physics of Biological Oscillators*, Springer 2021; arXiv
  2111.10300). *Cited*: the state of the art for protophases from a *scalar* signal;
  the PTP step is unchanged. Not needed here (§1.2).
- **Zielinski, Hahn, Lipp, Cuvelier & Brunette 2014**, *Strengths and limitations of
  period estimation methods for circadian data*, PLoS ONE **9**(5), e96462. *Cited* from
  the survey: among period estimators (Fourier/periodogram, autocorrelation, peak-to-peak
  timing, curve fitting), the curve-fitting family that uses every sample is the most
  robust to noise, and timing-based estimators (two samples per cycle) degrade fastest.
  *Deduced*: this is the reason our period comes from a regression over all tail samples
  rather than from crossing intervals (§1.2, step 5).
- **Namura, Takata, Yamaguchi, Kato & Nakao 2022**, *Estimating asymptotic phase and
  amplitude functions of limit-cycle oscillators from time series data*, Phys. Rev. E
  **106**, 014204. *Cited*: phase and amplitude functions are fitted *off* the cycle by
  polynomial regression on time-series data, with the cycle and Floquet exponent as
  inputs. Relevant to E4/E7 (comparison with the learned phase), not to E1.
- **Mauroy & Mezić 2012**, Chaos **22**, 033112; **Mauroy, Mezić & Moehlis 2013**,
  Physica D **261**, 19–30: isochrons and isostables by Fourier–Laplace averages of
  observables along trajectories — E5's method when the vector field is known.
- **Langfield, Krauskopf & Osinga 2014**, Chaos **24**, 013131: the published FitzHugh–
  Nagumo isochrons; their period $T_\Gamma \approx 11.2279$ and clockwise orientation are
  E1's FitzHugh–Nagumo oracle (`tests/test_systems.py` already pins the period).

**What we do differently, and why.** The phase-reconstruction literature starts from a
*scalar* observable, so its first and hardest step is the protophase (Hilbert transform,
embedding, iterated transforms). Here the **full planar state is observed**, so the polar
angle about any interior point of the loop is a protophase for free (*deduced*: the angle
about an interior point winds once per revolution, which is all a protophase needs), and
the PTP transformation of Kralemann et al. — a density correction, not a signal-processing
step — carries over unchanged. Second, the literature assumes the data lie on or near the
cycle; our trajectories start anywhere and converge, so the estimate must pick its own
window (the **tail**) and *check* that the window is on the cycle. Third, the period
estimate exploits the phase: once $\varphi$ is uniform in time, the period is a slope.

### 1.2 The method

Input: trajectories `ys (N, T, 2)` (or one, `(T, 2)`) on a common grid `ts (T,)`;
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

## 2–5. E2–E5

Written when each item lands (roadmap Phase E): E3 `Cycle.winding`/`contains`, E2
`settling_time` and the E1/E2 iteration, E4 `floquet_multiplier` from data, E5
`analysis.ode`.
