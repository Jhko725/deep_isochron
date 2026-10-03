# Normal forms — conventions, derivations, and the API they imply

**Status: draft for review (roadmap B10).** Once agreed, B11 reimplements
`systems/normal_forms/` against this document and ADR-0007 is amended to point here. The
current code is correct but mixes `r` and `s = r²` in its public names; the open decisions
are collected at the end.

Companion files: `systems/normal_forms/{base,hopf,bautin,integration}.py`;
`tests/test_normal_forms.py`. Related: ADR-0007 (hierarchy, `SolverConfig`, integrations
as objects).

## 1. Setting and symbols

A **normal form** here is a planar ODE with an attracting limit cycle, rotationally
symmetric about the origin, so that in polar coordinates `u = (x, y) = (r cos θ, r sin θ)`
the radial and angular motions decouple:

| symbol | meaning | convention |
|---|---|---|
| `r ≥ 0`, `θ ∈ (−π, π]` | polar coordinates about the origin | `to_chart(u) = (r, θ)`, `θ = atan2(y, x)` |
| `ρ(r)` | **growth rate** of the radius, `ṙ = r ρ(r)` | even in `r`; `ρ(1) = 0`, `ρ'(1) < 0` |
| `ω(r)` | **angular rate**, `θ̇ = ω(r)` | even in `r`; `ω(1) =: ω₁ ≠ 0` |
| `Γ` | the limit cycle `r = 1` | the only attracting cycle |
| `T = 2π/ω₁` | period | sign of `ω₁` is the sense of rotation |
| `κ = 2ρ'(1) < 0` | non-trivial Floquet exponent | contraction rate of the amplitude |
| `φ(u) ∈ (−π, π]` | **asymptotic phase**: `φ̇ = ω₁` along every trajectory | `φ = θ` on `Γ` |
| `h(r)` | **phase shift**: `φ = θ + h(r)` | `h(1) = 0` |
| `ψ(u)` | **isostable coordinate** (amplitude): `ψ̇ = κ ψ` | `ψ = 0` on `Γ`, `ψ > 0` inside |
| `J` | rotation by `+π/2`, `J(x, y) = (−y, x)` | |

"Even in `r`" means `ρ(r) = ρ̃(r²)` for a smooth `ρ̃` — equivalently, `ρ` has only even
powers in its Taylor expansion at 0. This is what makes the cartesian vector field smooth
at the origin (§2) and is a *requirement* on the defining data, not a convenience.

**Why these quantities.** Isochrons are the level sets of `φ`; the phase response of the
oscillator is `∇φ`; `κ` and `T` are the smooth-conjugacy invariants the learned map must
match (`learnings`: Floquet exponent, period and the eigenvalue structure at the enclosed
fixed point are hard constraints); `ψ` gives the amplitude coordinate in which the
conjugacy target is linear (`ψ̇ = κψ`). For the observed systems none of this is closed
form, which is the reason the normal forms carry it.

## 2. Vector fields in the three charts

**Polar** (the defining chart): `ṙ = r ρ(r)`, `θ̇ = ω(r)`.

**Cartesian** (what the data and the bijection see): with `r = |u|`,

```
u̇ = ρ(|u|) u + ω(|u|) J u ,   i.e.  ẋ = ρ x − ω y ,  ẏ = ρ y + ω x .
```

*Derivation.* `u = r e_r`, `u̇ = ṙ e_r + r θ̇ e_θ = rρ e_r + rω e_θ = ρ u + ω J u` since
`r e_θ = J u`. Because `ρ`, `ω` are even, `ρ(|u|) = ρ̃(x² + y²)` is a smooth function of
`u`, and `u̇` is smooth at the origin with Jacobian `ρ(0) I + ω(0) J` there — eigenvalues
`ρ(0) ± i ω(0)` (`eigenvalues_origin`). *Numerical caveat*: computing `|u| = √(x²+y²)` has
no derivative at exactly `u = 0`; §7 says how the code avoids a NaN there.

**Phase–amplitude** `(φ, ψ)`: `φ̇ = ω₁`, `ψ̇ = κψ` — the flow is *linear*: `φ(t) = φ₀ +
ω₁ t`, `ψ(t) = ψ₀ e^{κt}`. This chart is defined on the basin minus the origin; its inverse
needs the inverse of `ψ(r)` (§6).

## 3. Derivations

### 3.1 Asymptotic phase and the phase shift `h`

Seek `φ = θ + h(r)` with `φ̇ = ω₁` everywhere in the basin. Then

```
φ̇ = θ̇ + h'(r) ṙ = ω(r) + h'(r) r ρ(r) = ω₁   ⟹   h'(r) = (ω₁ − ω(r)) / (r ρ(r)) ,
```

with `h(1) = 0` so that `φ = θ` on the cycle. The integrand is finite at `r = 1` (both
numerator and denominator vanish linearly, ratio `→ −ω'(1)/(ρ'(1))`) and the integral from
`1` to `r` converges for `r` in the basin. **Isochrons** are `φ = const`, i.e. the curves
`θ = φ − h(r)`; for `ω ≡ ω₁` they are the radial lines `θ = φ`, and `ω(r) ≠ ω₁` shears
them into spirals (the Bautin form with `w₀ ≠ w`). `isochron(φ, r)` returns these points
in cartesian coordinates; `limit_cycle(φ) = (cos φ, sin φ)`.

### 3.2 Isostable coordinate `ψ` and the Floquet exponent `κ`

**`κ` first, with the convention fixed.** The amplitude equation is `ṙ = g(r) := r ρ(r)`;
its linearisation at the cycle is `g'(1) = ρ(1) + ρ'(1) = ρ'(1)`. So, **with `ρ` a function
of `r`, `κ = ρ'(1)`** — no factor. The same quantity written in `s = r²`, `ṡ = 2sρ̃(s)`,
linearises to `2ρ̃'(1)`; and since `ρ'(r) = 2rρ̃'(r²)`, both give the same number (Hopf:
`ρ(r) = a(1 − r²)`, `ρ'(1) = −2a`; `ρ̃(s) = a(1 − s)`, `2ρ̃'(1) = −2a`). The current code
computes `2·grad(radial_rate)(1)` *because* `radial_rate` takes `s`; after B11 it is
`grad(growth_rate)(1)` with `growth_rate(r)`. The factor of two is a chart artefact — the
kind of thing this document exists to pin down.

**`ψ`.** Seek `ψ = ψ(r)` with `ψ̇ = κψ`: `ψ'(r) r ρ(r) = κ ψ(r)`, i.e.

```
d ln ψ / dr = κ / (r ρ(r)) .
```

Near the cycle `rρ(r) ≈ ρ'(1)(r − 1) = κ(r − 1)`, so `d ln ψ/dr ≈ 1/(r − 1)` and `ψ ∝
(r − 1)`: the isostable coordinate is **linear** in the distance to the cycle, as it must
be (it is the Koopman eigenfunction of eigenvalue `κ`, and its gradient on `Γ` is the
left Floquet vector). Sign and normalisation, fixed here: `ψ > 0` inside the cycle, `ψ < 0`
outside, `ψ → +∞` as `r → 0`, and `ψ ≈ 1 − r²` (`≈ 2(1 − r)`) to first order at the
cycle. Any other choice is `ψ ↦ Cψ`, which leaves `ψ̇ = κψ` unchanged; `C = 1` makes
Hopf's `ψ = (1 − r²)/r²` the reference.

### 3.3 The two concrete forms

Both have `ω(r) = w₀ + (w − w₀) r²`, so `ω₁ = w`, `ω(0) = w₀`, and `c := (w − w₀)/a`.

| | Hopf (Stuart–Landau) | Bautin |
|---|---|---|
| `ρ(r)` | `a(1 − r²)` | `a(1 − r²)(1 + b r²)` |
| `κ = ρ'(1)` | `−2a` | `−2a(1 + b)` |
| `h(r)` | `c ln r` | `c [ln r − ½ ln((1 + b r²)/(1 + b))]` |
| `ψ(r)` | `(1 − r²)/r²` | `(1 − r²)(1 + b r²)^b / r^{2(1+b)}` |
| eigenvalues at 0 | `a ± i w₀` | `a ± i w₀` |
| basin of `Γ` | `R² \ {0}` | `R² \ {0}` if `b ≥ 0`; `0 < r < 1/√(−b)` if `−1 < b < 0` |

*Derivations (Bautin; Hopf is `b = 0`).* With `s = r²` and `ds = 2r dr`:

- `h`: `h' = (ω₁ − ω)/(rρ) = (w − w₀)(1 − r²) / (a r (1 − r²)(1 + b r²)) = c / (r(1 + b
  r²))`. `∫ dr/(r(1+br²)) = ln r − ½ ln(1 + b r²)` (check: derivative `1/r − br/(1+br²) =
  1/(r(1+br²))`). Normalising `h(1) = 0` adds `+½ ln(1+b)`.
- `ψ`: `d ln ψ/dr = κ/(rρ) = −2(1+b) / (r(1−r²)(1+br²))`. In `s`: `−(1+b) ds /
  (s(1−s)(1+bs))`, partial fractions `1/(s(1−s)(1+bs)) = 1/s + 1/((1+b)(1−s)) −
  b²/((1+b)(1+bs))`, integrate: `ln ψ = −(1+b) ln s + ln(1−s) + b ln(1+bs)`, i.e. `ψ =
  (1−s)(1+bs)^b / s^{1+b}`. Leading behaviour at `s = 1`: `(1 − s)·(1+b)^b`, so the
  normalisation `ψ ≈ 1 − s` divides by `(1 + b)^b` — **the current code omits this
  factor** (harmless for the laws, which are invariant under `ψ ↦ Cψ`, but not the
  normalisation stated in §3.2; B11 fixes it). The table above shows the unnormalised
  form; the normalised one is `(1 − r²)(1 + b r²)^b / ((1 + b)^b r^{2(1+b)})`.
- Both identities are verified exactly by autodiff in `test_phase_and_isostable_identities`
  and along integrated trajectories in `test_phase_and_amplitude_along_the_flow`.

*Bautin with `b < 0`.* `ρ(r) = a(1 − r²)(1 + b r²)` has a second zero at `r² = −1/b` where
`ρ' > 0`: an unstable cycle bounding the basin of `Γ` (the two-cycle regime of the Bautin
bifurcation). Outside it `ρ > 0` and `r → ∞` in finite time (cubic growth). `ψ` involves
`(1 + b r²)^b`, undefined past the outer cycle. `b > −1` is exactly `ρ'(1) < 0`, so the
constraint `B_CONSTRAINT = GreaterThan(−1)` is the stability condition; as a conjugacy
target `b ≥ 0` is used (`learnings`: `b` decouples the fixed-point instability `a` from the
cycle's contraction `κ = −2a(1+b)`, which is why Bautin was chosen over cubic Hopf).

## 4. Parameters

Unconstrained leaves, constrained on read (ADR-0001 in spirit): `a = GreaterThan(0)(raw_a)`
(`raw = 0 ↦ a = 1`), `b = GreaterThan(−1)(raw_b)` (`raw = 0 ↦ b = 0`, the Hopf form); `w`,
`w₀` free; `w₀` defaults to `w` (no shear). `params()` reports the constrained values.

## 5. Integrations (`flow(..., integration=)`)

All take cartesian `u₀`, return a `diffrax.Solution` with cartesian `.ys`.

| name | state integrated | singular at | notes |
|---|---|---|---|
| `cartesian` | `u ∈ R²`, `rhs` | nowhere | the only one that can start at `0` |
| `polar` | `(r, θ)`, `rhs_polar` | `r = 0` | `θ` unwrapped |
| `r_squared` | `(s, θ)`: `ṡ = 2sρ(√s)`, `θ̇ = ω(√s)` | `r = 0` (via `θ₀`, `√s`) | polynomial RHS for Hopf/Bautin; the former `BautinNormalForm.solve` |
| `closed_form` (**proposed**) | none — `(φ, ψ)` chart | `r = 0` | `ψ(t) = ψ₀e^{κt}`, `φ(t) = φ₀ + ω₁t`; needs `ψ⁻¹` by a 1-D monotone root solve (§6) |

The closed-form integration is what `learnings` records as "the radial ODE admits a
closed-form antiderivative via partial fractions; the angular integral has an exact closed
form" — in the present language, `ln ψ` *is* that antiderivative (`F(s) = −ln ψ/(1+b)`
satisfies `F(s(t)) = F(s₀) + 2at`), and `θ(t) = θ₀ + h(r₀) + ω₁ t − h(r(t))` *is* the
angular integral. It costs one Newton solve per output time instead of an ODE solve, has no
step-size error, and is exactly the map the conjugacy learns; it is the natural reference
for the three numerical integrations. Proposed for B11 as `ClosedFormIntegration`, with
the root solve bisection-safeguarded on `r ∈ (0, r_max)` as in `CubicBSpline` (ADR-0006).

## 6. The phase–amplitude chart as API

Given §3, the normal form can expose a third chart:

```
to_phase_amplitude(u)   = (φ(u), ψ(u))          # closed form
from_phase_amplitude(φ, ψ) = (r(ψ) cos(φ − h(r)), r(ψ) sin(φ − h(r)))   # r(ψ): monotone root
```

`ψ(r)` is strictly decreasing on the basin (`ψ' = κψ/(rρ)` has the sign of `κψ/ρ`, which
is negative both inside, `ψ > 0, ρ > 0`, and outside, `ψ < 0, ρ < 0`), from `+∞` at `r =
0` to `−b^b` (Bautin, `b > 0`) or `−1` (Hopf) as `r → ∞`, so `r(ψ)` exists for every
reachable `ψ`. With `r(ψ)` in hand, `from_phase_amplitude` and `ClosedFormIntegration` are
the same code.

## 7. API implied (for B11)

Public names in `r`; the defining data in `r`; the `s = r²` form confined to the `r_squared`
integration and to the smoothness argument.

```python
class AbstractNormalForm(AbstractODE):
    # defining data — even functions of r (Taylor series in r² at 0)
    def growth_rate(self, r):   ...      # ρ(r): ṙ = r ρ(r); ρ(1) = 0, ρ'(1) < 0
    def angular_rate(self, r):  ...      # ω(r): θ̇ = ω(r)
    def phase_shift(self, r):   ...      # h(r), h(1) = 0
    def isostable(self, r):     ...      # ψ(r), ψ(1) = 0, ψ ≈ 1 − r² near the cycle

    # derived (final)
    omega()  period()  floquet_exponent() = grad(growth_rate)(1.0)  floquet_multiplier()
    eigenvalues_origin() = growth_rate(0) ± i angular_rate(0)
    rhs(t, u) / rhs_polar(t, (r, θ))
    phase(u)  amplitude(u)  limit_cycle(φ)  isochron(φ, r)
    to_chart / from_chart (polar);  to_phase_amplitude / from_phase_amplitude (§6)
    flow(ts, u0, *, config, integration="r_squared")
```

- `growth_rate` is the proposed name for `ρ = ṙ/r` (it is the exponential growth rate of
  the radius, `d ln r/dt`); `radial_rate` was ambiguous between `ṙ` and `ṙ/r`.
- `floquet_exponent` becomes `grad(growth_rate)(1.0)` — no factor 2, by §3.2 — and stays
  concrete on the base class with the subclasses' closed forms tested against it.
- The cartesian `rhs` computes `r = |u|` with the standard safe pattern (`r = √s` where
  `s > 0`, else `0`) so that autodiff at exactly the origin gives the correct Jacobian
  `ρ(0)I + ω(0)J` rather than NaN; second derivatives *at exactly the origin* are not
  reproduced by this device (they are zero by symmetry for the true field and come out as
  zero from the device too for even `ρ`, `ω` — but this is not relied upon anywhere).
  Away from the origin everything is exact. The alternative — keeping the defining data in
  `s` — avoids the device entirely at the cost of the `r`/`s` split the review objected to.
- `r_squared` integration calls `growth_rate(√s)`; since `ρ` is even this is exactly
  `ρ̃(s)`, and `√s` only appears in a place that is already singular at the origin.

## 8. Tests the document implies

Existing tests carry over with `κ = grad(growth_rate)(1)`; new or changed:
`ψ` normalisation (`ψ(r) ≈ 1 − r²` near 1 for both forms); Jacobian of `rhs` at the origin
equals `ρ(0)I + ω(0)J`; `to_phase_amplitude ∘ from_phase_amplitude = id` on the basin;
`closed_form` agrees with `cartesian` to `TOL["flow"]`; the `r → ∞` limit of `ψ`.

## 9. Open decisions for review

1. **Name of `ρ`**: `growth_rate` (proposed) vs `radial_rate` (current, ambiguous) vs
   `rho`.
2. **Convention `κ = ρ'(1)` with `ρ(r)`** (proposed, no factor 2) — confirm.
3. **`ψ` normalisation `ψ ≈ 1 − r²` near the cycle** (divide Bautin's by `(1+b)^b`) —
   confirm, or keep the unnormalised current form.
4. **Defining data in `r` with the safe-`√` device** (Option A) vs **defining data in `s`,
   public API in `r`** (Option B). Proposed: A.
5. **Add `ClosedFormIntegration` and the `(φ, ψ)` chart in B11**, or defer to Phase E.
   Proposed: B11 — they are a few dozen lines once `r(ψ)` exists and give the exact
   reference the numerical integrations are compared against.

## References

- A. T. Winfree, *The Geometry of Biological Time*, 2nd ed., Springer (2001) — isochrons,
  asymptotic phase.
- J. Guckenheimer, Isochrons and phaseless sets, *J. Math. Biol.* 1, 259–273 (1975).
- A. Mauroy, J. Moehlis, I. Mezić, Isostables, isochrons, and Koopman spectrum for the
  action-angle representation of stable fixed point dynamics, *Physica D* 261, 19–30
  (2013) — isostable coordinates as Koopman eigenfunctions (`ψ̇ = κψ`).
- D. Wilson, J. Moehlis, Isostable reduction of periodic orbits, *Phys. Rev. E* 94, 052213
  (2016) — isostables of limit cycles and the Floquet normalisation.
- Yu. A. Kuznetsov, *Elements of Applied Bifurcation Theory*, Ch. 8 — the Bautin
  (generalised Hopf) normal form and its two-cycle regime.
