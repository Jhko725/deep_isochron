---
type: glossary
status: current
updated: 2026-11-25
sources:
  - "design/normal-forms.md §1"
  - "ADR-0020 (the phase in turns)"
---

# Glossary

Every quantity is named here once, and a name used in code, a document or a figure label is the name in this table. *Symbol* is the symbol of [`design/normal-forms.md`](design/normal-forms.md) § 1, *Code name* is the identifier in the package, and *Unit or range* gives the unit of the quantity or the set that it lies in. *Avoid* lists the names that are not used for the quantity.

The table is seeded from § 1 of the normal-forms document and from [ADR-0020](decisions/0020-phase-in-turns.md). A quantity outside them gets its row when its code is next touched. A row that carries the marker is Claude's reading, which Joon has not confirmed.

| Term | Symbol | Code name | Unit or range | Definition | Avoid |
|---|---|---|---|---|---|
| polar radius | $r$ | `r`, first output of `to_polar` | dimensionless, $r \ge 0$ | The distance of a point of the plane from the origin of a normal form. | |
| polar angle | $\theta$ | second output of `to_polar` | radians, in $(-\pi, \pi]$, unwrapped when integrated | The angle $\mathrm{atan2}(y, x)$ of a point of the plane. | "phase" for this angle |
| log growth rate | $\rho(r)$ | `log_growth_rate` | inverse time | The rate $\dot r / r$ of a normal form, with $\rho(1) = 0$. | "radial rate" |
| angular rate | $\omega(r)$ | `angular_rate` | radians per unit time | The rate $\dot\theta$ of a normal form at radius $r$. Its value on the cycle is $\omega_1$. | "frequency" |
| limit cycle | $\Gamma$ | `limit_cycle`, `LimitCycle` | a closed curve | The attracting periodic orbit, $r = 1$ for a normal form. | "orbit" for the attractor |
| basin | $(r_{\rm in}, r_{\rm out})$ | `basin_radii` | an open annulus | The set of points whose trajectories converge to the limit cycle. The asymptotic phase and the isostable coordinate exist on the basin only. | "domain" |
| period | $T$ | `period` | time, $T > 0$ | The time of one revolution of the limit cycle, $2\pi / \lvert\omega_1\rvert$ for a normal form. | |
| Floquet exponent | $\kappa$ | `floquet_exponent` | inverse time, $\kappa < 0$ for an attracting cycle | The non-trivial exponent of the limit cycle, $\rho'(1)$ for a normal form, with no factor 2. | "decay rate" |
| Floquet multiplier | $\mu$ | `floquet_multiplier` | dimensionless, $0 < \mu < 1$ for an attracting cycle | The non-trivial multiplier $e^{\kappa T}$ of the limit cycle. | |
| phase angle | $\phi$ | none; an intermediate of `to_phase_amplitude` **[unconfirmed]** | radians | The polar angle shifted by the phase shift, $\phi = \theta + h(r)$, which advances at the rate $\omega_1$ along every trajectory. | "phase" for this angle |
| phase shift | $h(r)$ | `phase_shift` | radians | The angle added to the polar angle to give the phase angle, with $h(1) = 0$. | |
| asymptotic phase | $\Theta$ | `phase` | turns, in $[0, 1)$ | The fraction of the period at which a point's trajectory joins the limit cycle. It increases at the rate $1/T$ along every trajectory, clockwise or counterclockwise. | "angle", a phase in radians |
| phase difference | none | `phase_difference` | turns, in $(-\tfrac12, \tfrac12]$ | The signed shortest arc from one phase to another, positive when the first is ahead. | a difference taken by subtraction |
| isostable coordinate | $\Psi$ | `isostable` (of a radius), `amplitude` (of a point) **[unconfirmed]** | dimensionless, 0 on the cycle | The amplitude coordinate that decays as $\dot\Psi = \kappa\Psi$, with $\nabla\Psi\cdot\mathbf e_r = 1$ on the cycle of a normal form. | |
| isochron | level set of $\Theta$ | `isochron` | a curve in the plane | The set of points of one asymptotic phase. | |
| phase sensitivity | $Z(\Theta)$ | `phase_sensitivity` | turns per unit of state | The gradient of the asymptotic phase on the limit cycle, normalized by $Z \cdot F = 1/T$ for the vector field $F$. | "PRC" for the vector, a normalization $Z \cdot F = \omega$ |
