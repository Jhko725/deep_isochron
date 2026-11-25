---
type: verification
status: current
updated: 2026-11-25
---

# Verification

One row per numerical claim: the test that carries it, the oracle it is compared against, the tolerance, the tier, and its state. A claim with no row is unverified. A row is added when the claim is made, and it is marked green, with the commit, when the test passes.

The table was seeded on 2026-11-25 from § 5 of the five `method` design documents. A claim outside them gets its row when its code is next touched. A tolerance is named by its entry in `TOL` of `tests/helpers.py`, where its reason is written.

| # | Claim | Test | Oracle | Tolerance | Tier | State |
|---|---|---|---|---|---|---|
| 1 | `estimate_cycle` recovers the curve, the period and the phase of a normal form's cycle from trajectory data | `tests/test_analysis.py::test_estimate_cycle_recovers_the_unit_circle_and_period` | the closed forms of Bautin and Hopf, [`design/cycle-estimation.md`](design/cycle-estimation.md) § 5 | `cycle_curve`, 2e-3; `cycle_period`, 1e-6 relative; `cycle_phase`, 3e-3 turns | quick | green, `7964fcc` |
| 2 | `estimate_cycle` gives the period of the FitzHugh–Nagumo cycle | `tests/test_analysis.py::test_fitzhugh_nagumo_cycle_matches_langfield_et_al` | the published period 11.2279 (Langfield et al. 2014, §III) | 2e-3 | quick | green, `7964fcc` |
| 3 | `limit_cycle` finds the period and the radius of a normal form's cycle by shooting | `tests/test_analysis_ode.py::test_limit_cycle_matches_the_normal_form` | the closed forms $T = 2\pi/\lvert\omega_1\rvert$ and $r = 1$, [`design/limit-cycle.md`](design/limit-cycle.md) § 5 | `limit_cycle`, 1e-8 | quick | green, `7964fcc` |
| 4 | `limit_cycle` stays on the algebraic orbit of Guckenheimer & Meloon | `tests/test_analysis_ode.py::test_limit_cycle_on_the_guckenheimer_meloon_algebraic_orbit` | the level set $g = 0$ of the designed field (Guckenheimer & Meloon 2000, §5.1) | `level_set`, 1e-8 | quick | green, `7964fcc` |
| 5 | `limit_cycle` gives the period of the FitzHugh–Nagumo cycle | `tests/test_analysis_ode.py::test_limit_cycle_fitzhugh_nagumo_against_langfield` | the published period 11.2279 (Langfield et al. 2014, §III) | 2e-3 | quick | green, `7964fcc` |
| 6 | `floquet` gives the non-trivial exponent of a normal form's cycle by all three routes | `tests/test_analysis_ode.py::test_floquet_exponent_matches_the_normal_form` | the closed form $\kappa = \rho'(1)$, [`design/floquet.md`](design/floquet.md) § 5 | `floquet`, 1e-8 | quick | green, `7964fcc` |
| 7 | `floquet` and `limit_cycle` give the exponent and the period of the Cassini oval | `tests/test_analysis_ode.py::test_floquet_and_period_of_the_cassini_oval_against_line_integrals` | two line integrals by the trapezoidal rule, with no ODE solve | `floquet`, 1e-8 | quick | green, `7964fcc` |
| 8 | on FitzHugh–Nagumo, the chain and Liouville routes agree while the explicit monodromy and the return map lose the multiplier | `tests/test_analysis_ode.py::test_floquet_fitzhugh_nagumo_shows_the_conditioning_floors` | the two independent routes against each other | 1e-7 relative between the chain and Liouville | quick | green, `7964fcc` |
| 9 | the asymptotic phase by forward integration equals a normal form's phase up to one constant | `tests/test_phase.py::test_asymptotic_phase_is_the_closed_form_phase` | the closed form `nf.phase`, [`design/phase-field.md`](design/phase-field.md) § 5 | `phase`, 2e-9 turns | quick | green, `7964fcc` |
| 10 | the asymptotic phase advances by $t/T$ along the flow | `tests/test_phase.py::test_asymptotic_phase_advances_with_the_flow` | the law $\Theta(\Phi_t x) = \Theta(x) + t/T \bmod 1$ | `phase`, 2e-9 turns | quick | green, `7964fcc` |
| 11 | the FitzHugh–Nagumo phase field is constant on the stored isochrons near the cycle | `tests/test_phase.py::test_fitzhugh_nagumo_phase_on_the_coworkers_isochrons` | Andrus Giraldo's stored isochrons, files 26, 51 and 100 | `golden_phase`, 1e-8 turns | slow | not run on `conventions-sync` |
| 12 | AUTO's orbit and adjoint bundle agree with the shooting cycle and its Floquet vectors | `tests/test_auto_stages.py::test_stages_one_to_three_agree_with_o1_and_o2` | `limit_cycle` and `floquet`, [`design/isochrons.md`](design/isochrons.md) § 5 | `auto_orbit`, `bundle_direction` and `bundle_scale`, 1e-8 each | auto | not run on `conventions-sync`: the session had no AUTO-07p tree |
| 13 | a traced isochron of a normal form carries the requested phase | `tests/test_auto_stages.py::test_normal_form_isochrons_carry_the_closed_form_phase` | the closed form `nf.phase` | `isochron_phase`, 1e-8 turns | auto | not run on `conventions-sync`: the session had no AUTO-07p tree |
| 14 | the traced FitzHugh–Nagumo isochrons lie on the stored ones | `tests/test_isochron.py::test_fitzhugh_nagumo_isochrons_lie_on_the_coworkers` | Andrus Giraldo's stored isochrons | `isochron_curve`, 2e-5 | auto | not run on `conventions-sync`: the session had no AUTO-07p tree |
