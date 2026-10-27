"""``analysis.ode``, the limit cycle and its Floquet exponent from the vector field,
against exact oracles.

The five sections follow the module: `CrossSection`; `PoincareMap`, the first-return
map on the normal forms with its custom JVP (Parker & Chua 1989, Eq. 5.21), its
two-sided and reverse-time variants and its missing-crossing behavior;
`multiple_shooting` from a perturbed guess; `find_limit_cycle` on Hopf, Bautin and
Winfree (exact period and radius), the Guckenheimer–Meloon algebraic orbit, the
FitzHugh–Nagumo cycle of Langfield et al. 2014 (slow), and its guards; `floquet`, the
three routes against ``κ = ρ'(1)`` on the normal forms, solver-free line integrals on
the Cassini oval, and the conditioning floors on FitzHugh–Nagumo (slow).

The normal forms are constructed with ``integration="cartesian"`` so that the analysis
integrates their vector field numerically and their closed forms are the oracle; with
the default closed-form ``flow`` the comparison would be circular, and one test checks
that the two agree.
"""

import importlib

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from deep_isochron.analysis import Cycle, NotConvergedError
from deep_isochron.analysis.ode import (
    CrossSection,
    find_limit_cycle,
    FixedPoint,
    floquet,
    LimitCycle,
    monodromy,
    multiple_shooting,
    periodic_schur,
    PoincareMap,
    reanchor,
    return_map_multiplier,
    segment_jacobians,
    SHOOTING_SOLVER_CONFIG,
)
from deep_isochron.systems import (
    AbstractODE,
    cassini,
    FitzhughNagumo,
    guckenheimer_meloon,
    SolverConfig,
)
from deep_isochron.systems.normal_forms import (
    BautinNormalForm,
    HopfNormalForm,
    WinfreeNormalForm,
)
from hypothesis import given, settings, strategies as st

from .helpers import assert_close, TOL


CFG = SHOOTING_SOLVER_CONFIG
INTEGRATION = {
    "integration": "cartesian"
}  # analyze the vector field, not the closed form
HOPF = HopfNormalForm(a=1.0, w=2.0, **INTEGRATION)
X0 = jnp.array([0.3, 0.1])


def _on_cycle(nf, angle=0.0):
    return jnp.array([jnp.cos(angle), jnp.sin(angle)])


def _multiplier(nf) -> float:
    """e^{κT} with the *unsigned* period (the normal forms' ``period()`` is signed)."""
    return float(jnp.exp(nf.floquet_exponent() * jnp.abs(nf.period())))


# --------------------------------------------------------------- 1. CrossSection --
def test_cross_section_signed_distance_and_unit_normal():
    """A `CrossSection` normalizes its normal, its signed distance is zero at the
    center and flips with `flip`, and a zero normal raises `ValueError`."""
    plane = CrossSection(jnp.array([1.0, 2.0]), jnp.array([0.0, 3.0]))
    assert_close(jnp.linalg.norm(plane.normal), 1.0, atol=TOL["identity"])
    assert_close(
        plane.signed_distance(jnp.array([1.0, 2.0])), 0.0, atol=TOL["identity"]
    )
    assert_close(
        plane.signed_distance(jnp.array([5.0, 2.5])), 0.5, atol=TOL["identity"]
    )
    assert_close(
        plane.flip().signed_distance(jnp.array([5.0, 2.5])), -0.5, atol=TOL["identity"]
    )
    with pytest.raises(ValueError):
        CrossSection(jnp.zeros(2), jnp.zeros(2))


def test_flow_normal_section_points_along_the_flow():
    """`CrossSection.normal_to_flow` has a positive component along the vector field
    at its center and raises at an equilibrium."""
    p = _on_cycle(HOPF)
    plane = CrossSection.normal_to_flow(HOPF, p)
    assert float(jnp.dot(plane.normal, HOPF.rhs(0.0, p))) > 0
    assert_close(plane.signed_distance(p), 0.0, atol=TOL["identity"])
    with pytest.raises(ValueError):  # the origin is an equilibrium
        CrossSection.normal_to_flow(HOPF, jnp.zeros(2))


# ----------------------------------------------------------------- 2. PoincareMap --
@pytest.mark.parametrize(
    "nf",
    [
        HOPF,
        BautinNormalForm(a=1.0, b=0.1, w=-1.5, **INTEGRATION),
        WinfreeNormalForm(a=-0.25, w=-0.5, **INTEGRATION),
    ],
    ids=["hopf", "bautin_cw", "winfree"],
)
def test_return_map_on_the_cycle_returns_in_one_period(nf):
    """From a point of the cycle the first-return map returns the same point after
    the normal form's unsigned period, to `TOL["return_map"]`."""
    p = _on_cycle(nf, 0.4)
    P = PoincareMap(nf, CrossSection.normal_to_flow(nf, p), CFG, t_max=20.0)
    y, tau = P(p)
    assert_close(tau, abs(float(nf.period())), atol=TOL["return_map"])
    assert_close(y, p, atol=TOL["return_map"])


def test_return_map_is_jittable_and_vmappable():
    """`PoincareMap` gives the same return under `filter_jit`, and `vmap` over three
    radii lands every point on the section after one period (Hopf's ``ω`` is
    constant)."""
    p = _on_cycle(HOPF, 0.4)
    P = PoincareMap(HOPF, CrossSection.normal_to_flow(HOPF, p), CFG, t_max=20.0)
    y_eager, tau_eager = P(p)
    y_jit, tau_jit = eqx.filter_jit(P)(p)
    assert_close(y_jit, y_eager, atol=TOL["jit_eager"])
    assert_close(tau_jit, tau_eager, atol=TOL["jit_eager"])
    # several starting points in one call: points off the cycle return to the section
    radii = jnp.array([0.8, 1.0, 1.2])
    xs = radii[:, None] * p[None]
    ys, taus = jax.vmap(P)(xs)
    assert_close(jax.vmap(P.section.signed_distance)(ys), 0.0, atol=TOL["return_map"])
    assert_close(taus, abs(float(HOPF.period())), atol=TOL["return_map"])  # ω constant


def test_negative_map_is_the_positive_map_of_the_flipped_section():
    """The negative-direction map on a section equals the positive map on the flipped
    section, and on Hopf it reaches the antipodal point after half a period."""
    p = _on_cycle(HOPF, 1.0)
    plane = CrossSection.normal_to_flow(HOPF, p)
    neg = PoincareMap(HOPF, plane, CFG, t_max=20.0, direction="negative")
    pos_flipped = PoincareMap(HOPF, plane.flip(), CFG, t_max=20.0)
    y1, t1 = neg(p)
    y2, t2 = pos_flipped(p)
    assert_close(y1, y2, atol=TOL["return_map"])
    assert_close(t1, t2, atol=TOL["return_map"])
    # the halfway point: half a period, on the opposite side of the circle
    assert_close(t1, 0.5 * abs(float(HOPF.period())), atol=TOL["return_map"])
    assert_close(y1, -p, atol=TOL["return_map"])


def _cassini_point_at_height(ode, y0=0.5, x_guess=1.3):
    """A point of the Cassini cycle at height ``y0`` (Newton in ``x``)."""
    x = x_guess
    for _ in range(50):
        u = jnp.array([x, y0])
        g, dg = float(ode.level(u)), float(jax.grad(ode.level)(u)[0])
        x -= g / dg
    return jnp.array([x, y0])


def test_two_sided_map_visits_the_four_intersections_of_a_non_convex_cycle():
    """On the Cassini oval a horizontal section above the waist has four crossings:
    the two-sided map visits all four in one period, the one-sided map's fixed point
    has ``k = 2``, and `fixed_point` on a two-sided map raises `ValueError`."""
    ode = cassini()
    p = _cassini_point_at_height(ode)
    plane = CrossSection(p, jnp.array([0.0, 1.0]))  # horizontal, above the waist
    both = PoincareMap(ode, plane, CFG, t_max=50.0, direction="both")
    xs, taus = both.iterate(p, 4)
    assert_close(np.asarray(xs[:, 1]), 0.5, atol=TOL["return_map"])
    # four distinct intersections, then back to the start
    assert len(np.unique(np.round(np.asarray(xs[:4, 0]), 6))) == 4
    assert_close(xs[4], p, atol=TOL["return_map"])
    # the one-sided map sees a period-2 orbit
    pos = PoincareMap(ode, plane, CFG, t_max=50.0)
    fp, it = pos.fixed_point(p, tol=TOL["return_map"])
    assert fp.k == 2 and it.iterates.shape[0] == len(it.taus) + 1
    assert_close(fp.tau, float(jnp.sum(taus)), atol=TOL["return_map"])
    with pytest.raises(ValueError):
        both.fixed_point(p)


def test_fixed_point_contracts_at_the_floquet_multiplier():
    """`fixed_point` from off the Hopf cycle converges to radius 1 with the reported
    contraction ``e^{κT}`` within 2 %, and a one-iteration budget raises
    `NotConvergedError` with stage ``"settle"``."""
    P = PoincareMap(
        HOPF, CrossSection.normal_to_flow(HOPF, _on_cycle(HOPF)), CFG, t_max=20.0
    )
    fp, _ = P.fixed_point(jnp.array([1.05, 0.0]), tol=1e-6)
    assert fp.converged and fp.k == 1
    assert_close(jnp.linalg.norm(fp.x), 1.0, atol=1e-6)
    # successive differences shrink by e^{κT} per return once in the linear regime
    # (κ = ρ'(1) = -2a, T = 2π/w)
    assert_close(fp.contraction, _multiplier(HOPF), rtol=2e-2)
    with pytest.raises(NotConvergedError) as info:
        P.fixed_point(jnp.array([1.3, 0.0]), tol=1e-12, max_iter=1)
    assert info.value.stage == "settle" and not info.value.last[0].converged


def test_return_map_derivative_has_eigenvalues_zero_and_the_multiplier():
    """``jax.jacfwd`` through the custom JVP (Parker & Chua Eq. 5.21)."""
    nf = BautinNormalForm(a=1.0, b=0.2, w=1.0, **INTEGRATION)
    p = _on_cycle(nf, 0.7)
    P = PoincareMap(nf, CrossSection.normal_to_flow(nf, p), CFG, t_max=20.0)
    DP = jax.jacfwd(lambda x: P(x)[0])(p)
    eig = np.sort(np.abs(np.linalg.eigvals(np.asarray(DP))))
    assert_close(eig[0], 0.0, atol=TOL["trivial_multiplier"])
    assert_close(eig[1], _multiplier(nf), atol=TOL["trivial_multiplier"])


def test_return_time_derivative_matches_finite_differences():
    """Winfree's angular rate depends on r, so the return time varies with the point:
    the JVP's dτ (implicit differentiation of the crossing) against central
    differences; for Hopf (constant ω) dτ is zero."""
    w = WinfreeNormalForm(a=-0.25, w=-0.5, **INTEGRATION)
    p = _on_cycle(w)
    P = PoincareMap(w, CrossSection.normal_to_flow(w, p), CFG, t_max=20.0)
    x = jnp.array([1.1, 0.0])
    d = jnp.array([1.0, 0.0])
    _, dtau = jax.jvp(lambda z: P(z)[1], (x,), (d,))
    h = 1e-5
    fd = (P(x + h * d)[1] - P(x - h * d)[1]) / (2 * h)
    assert_close(dtau, fd, rtol=1e-5, atol=1e-7)
    Ph = PoincareMap(HOPF, CrossSection.normal_to_flow(HOPF, p), CFG, t_max=20.0)
    _, dtau_h = jax.jvp(lambda z: Ph(z)[1], (x,), (d,))
    assert_close(dtau_h, 0.0, atol=TOL["return_map"])


def test_return_map_without_a_crossing_raises_or_returns_inf():
    """A section the Hopf cycle never reaches raises `EquinoxRuntimeError`, returns
    an infinite ``tau`` with ``throw=False``, and makes `fixed_point` raise
    `NotConvergedError` with stage ``"return"``."""
    # a section the Hopf cycle never reaches: through (3, 0) with normal (1, 0)
    plane = CrossSection(jnp.array([3.0, 0.0]), jnp.array([1.0, 0.0]))
    with pytest.raises(eqx.EquinoxRuntimeError):
        PoincareMap(HOPF, plane, CFG, t_max=5.0)(jnp.array([1.0, 0.0]))
    _, tau = PoincareMap(HOPF, plane, CFG, t_max=5.0, throw=False)(
        jnp.array([1.0, 0.0])
    )
    assert not bool(jnp.isfinite(tau))
    with pytest.raises(NotConvergedError) as info:
        PoincareMap(HOPF, plane, CFG, t_max=5.0).fixed_point(jnp.array([1.0, 0.0]))
    assert info.value.stage == "return"


def test_reverse_time_map_contracts_onto_winfrees_repelling_inner_cycle():
    """In reverse time the repelling cycle r = a attracts (Parker & Chua §5.1.1,
    footnote 1): one return (a full period, 2π/|ω(a)| with ω(a) = −(1 + w(1 − a)))
    brings the radius closer to a by e^{−κ_u T}, κ_u = a(1 − a); the derivative of the
    reversed map has the eigenvalues {0, e^{−κ_u T}}."""
    w = WinfreeNormalForm(a=0.25, w=-0.5, **INTEGRATION)
    p = w.flow_to(-20.0, jnp.array([0.3, 0.0]), config=CFG)  # partly settled, reversed
    P = PoincareMap(
        w, CrossSection.normal_to_flow(w, p), CFG, t_max=30.0, reverse_time=True
    )
    y, tau = P(p)
    T = 2 * np.pi / 0.625
    assert_close(tau, T, atol=1e-2)  # off the cycle ω depends on r: a brute-force guess
    d_before = abs(float(jnp.linalg.norm(p)) - 0.25)
    d_after = abs(float(jnp.linalg.norm(y)) - 0.25)
    kappa_u = float(w.inner_floquet_exponent())
    assert_close(d_after / d_before, np.exp(-kappa_u * T), rtol=5e-2)
    # the fixed-point tolerance cannot go below the solver's noise (~1e-10 per return),
    # and ω varies with r here (dT/dr ≈ 8), so the return time is only known to ~1e-8
    fp, _ = P.fixed_point(p, tol=1e-9)
    assert fp.k == 1
    assert_close(jnp.linalg.norm(fp.x), 0.25, atol=TOL["return_map"])
    assert_close(fp.tau, T, atol=1e-7)
    eig = np.sort(
        np.abs(np.linalg.eigvals(np.asarray(jax.jacfwd(lambda x: P(x)[0])(fp.x))))
    )
    assert_close(eig[1], np.exp(-kappa_u * fp.tau), rtol=1e-7)


# ----------------------------------------------------------- 3. multiple_shooting --
@pytest.mark.parametrize("m", [1, 4])
def test_shooting_polishes_a_perturbed_guess(m):
    """`multiple_shooting` with ``m`` segments recovers Hopf's period and unit radius
    from a guess perturbed by ``1e-2``, eagerly and under `filter_jit`."""
    nf = HOPF
    T = abs(float(nf.period()))
    p = _on_cycle(nf)
    plane = CrossSection.normal_to_flow(nf, p)
    times = T * jnp.arange(m) / m
    exact = jnp.stack([_on_cycle(nf, float(nf.omega()) * t) for t in times])
    guess = exact + 1e-2 * jnp.cos(7.0 * jnp.arange(2 * m).reshape(m, 2))
    guess = guess.at[0].set(p)  # the phase condition holds at the guess
    res = multiple_shooting(nf, guess, 1.05 * T, plane, config=CFG)
    assert_close(res.period, T, atol=TOL["limit_cycle"])
    assert_close(jnp.linalg.norm(res.points, axis=1), 1.0, atol=TOL["limit_cycle"])
    assert float(res.residual) < TOL["limit_cycle"]
    # the same under jit
    jitted = eqx.filter_jit(
        lambda pts, T0: multiple_shooting(nf, pts, T0, plane, config=CFG)
    )
    res_jit = jitted(guess, 1.05 * T)
    assert_close(res_jit.period, res.period, atol=TOL["limit_cycle"])


def test_shooting_is_more_accurate_with_tighter_solver_tolerances():
    """The period error of `multiple_shooting` is smaller with ``rtol = 1e-10`` than
    with ``1e-6`` on the inner solver."""
    nf = HOPF
    T = abs(float(nf.period()))
    p = _on_cycle(nf)
    plane = CrossSection.normal_to_flow(nf, p)
    out = []
    for rtol in (1e-6, 1e-10):
        cfg = SolverConfig(
            solver=dfx.Dopri8(), rtol=rtol, atol=rtol * 1e-2, adjoint=dfx.ForwardMode()
        )
        res = multiple_shooting(
            nf, p[None], 1.02 * T, plane, config=cfg, rtol=1e-12, atol=1e-14
        )
        out.append(abs(float(res.period) - T))
    assert out[1] < out[0]


def test_shooting_rejects_bad_input():
    """`multiple_shooting` raises `ValueError` for an empty mesh."""
    plane = CrossSection.normal_to_flow(HOPF, _on_cycle(HOPF))
    with pytest.raises(ValueError):
        multiple_shooting(HOPF, jnp.zeros((0, 2)), 1.0, plane)


# ------------------------------------------------------------- 4. find_limit_cycle --
def _check_invariants(lc: LimitCycle, orientation: int, n: int = 64):
    """The output invariants of `find_limit_cycle`: positive period, the first sample
    on the section, times ``i T / n``, the orientation, the residual, and the shooting
    mesh on the orbit."""
    assert lc.period > 0
    times, points = lc.sample(n)
    assert_close(points[0], lc.x0s[0], atol=TOL["identity"])
    assert_close(lc.section.signed_distance(lc.x0s[0]), 0.0, atol=TOL["return_map"])
    assert_close(times, lc.period * np.arange(n) / n)
    assert lc.orientation() == orientation
    assert lc.residual < TOL["limit_cycle"]
    # the shooting mesh lies on the orbit at times i T / m
    _, mesh = lc.sample(lc.m)
    assert_close(mesh, lc.x0s, atol=TOL["limit_cycle"])


@pytest.mark.parametrize(
    "nf, orientation",
    [
        (HOPF, 1),
        (BautinNormalForm(a=1.0, b=0.1, w=-1.5, **INTEGRATION), -1),
        (WinfreeNormalForm(a=-0.25, w=-0.5, **INTEGRATION), -1),
    ],
    ids=["hopf", "bautin_cw", "winfree"],
)
def test_limit_cycle_matches_the_normal_form(nf, orientation):
    """`find_limit_cycle` recovers the unit circle and the period of the normal form
    with the expected orientation, and `sample` is uniform in phase."""
    lc = find_limit_cycle(nf, X0, t_settle=5.0, m=2, config=CFG)
    assert_close(lc.period, abs(float(nf.period())), atol=TOL["limit_cycle"])
    _, points = lc.sample(64)
    assert_close(np.linalg.norm(points, axis=1), 1.0, atol=TOL["limit_cycle"])
    _check_invariants(lc, orientation)
    # the phase-uniform sampling: the cycle's phase advances uniformly in time
    angles = np.unwrap(np.arctan2(points[:, 1], points[:, 0]))
    assert_close(np.diff(angles), orientation * 2 * np.pi / 64, atol=1e-8)


def test_closed_form_flow_and_numerical_flow_give_the_same_cycle():
    """A normal form with its default closed-form ``flow`` is analyzed exactly (the
    chart *is* the answer); with ``integration="cartesian"`` the solver is analyzed
    against it. Both must agree — and only the second is a test of the numerics."""
    exact = find_limit_cycle(HopfNormalForm(a=1.0, w=2.0), X0, t_settle=5.0, m=2)
    numerical = find_limit_cycle(HOPF, X0, t_settle=5.0, m=2)
    assert_close(exact.period, numerical.period, atol=TOL["limit_cycle"])
    assert_close(exact.sample(32)[1], numerical.sample(32)[1], atol=TOL["limit_cycle"])
    # the closed form solves itself to Levenberg–Marquardt's atol (1e-2 tol), with no
    # solver-error floor underneath — the mark of the circularity
    assert exact.residual < 1e-10


def test_limit_cycle_points_are_a_phase_uniform_cycle():
    """The output is what ``Cycle.from_points`` expects: points at phases uniform in
    time; the fitted curve reproduces them (the two halves of ``analysis`` meet)."""
    lc = find_limit_cycle(
        BautinNormalForm(a=1.0, b=0.1, w=-1.5, **INTEGRATION), X0, t_settle=5.0, m=2
    )
    times, points = lc.sample(256)
    phases = 2 * np.pi * np.asarray(times) / float(lc.period)
    cyc = Cycle.from_points(phases, np.asarray(points), float(lc.period))
    assert cyc.residual < TOL["cycle_query"]
    assert_close(cyc(phases), points, atol=TOL["cycle_query"])


def test_limit_cycle_on_the_guckenheimer_meloon_algebraic_orbit():
    """On the Guckenheimer & Meloon 2000 orbit the sampled points lie on the level set
    ``g = 0`` to `TOL["level_set"]` and the period is near 7.7, as they report."""
    ode = guckenheimer_meloon(0.07)
    start = jnp.array([0.0, 0.2952161257895192])  # on the orbit (their p. 970)
    lc = find_limit_cycle(ode, start, t_settle=2.0, m=4, config=CFG)
    _, points = lc.sample(128)
    g = np.array([float(ode.level(jnp.asarray(pt))) for pt in points])
    assert np.abs(g).max() < TOL["level_set"]
    assert 7.6 < lc.period < 7.8  # "period approximately 7.7"
    assert_close(lc.settle.tau, lc.period, atol=TOL["return_map"])
    _check_invariants(lc, 1)


@pytest.fixture(scope="module")
def hopf_cycle() -> LimitCycle:
    """The unit-circle cycle of `HOPF` by `find_limit_cycle`, counterclockwise."""
    return find_limit_cycle(HOPF, X0, t_settle=5.0, m=4, config=CFG)


@settings(max_examples=10)
@given(angle=st.floats(-np.pi, np.pi), radius=st.floats(0.6, 1.6))
def test_reanchor_moves_phase_zero_to_the_nearest_cycle_point(
    hopf_cycle, angle, radius
):
    """`reanchor` at the point at `angle` and distance `radius` from the center of
    Hopf's unit-circle cycle puts ``x0s[0]`` at ``(cos angle, sin angle)``, the nearest
    cycle point, and leaves the orbit and the period as they were; the result has every
    invariant of `find_limit_cycle`."""
    on_cycle = np.array([np.cos(angle), np.sin(angle)])
    lc = reanchor(hopf_cycle, jnp.asarray(radius * on_cycle))
    assert_close(lc.x0s[0], on_cycle, atol=TOL["limit_cycle"])
    assert_close(lc.period, hopf_cycle.period, atol=TOL["limit_cycle"])
    assert_close(np.linalg.norm(lc.x0s, axis=1), 1.0, atol=TOL["limit_cycle"])
    _check_invariants(lc, 1)
    # the section is normal to the flow at the new origin
    f = np.asarray(HOPF.rhs(0.0, lc.x0s[0]))
    assert_close(np.asarray(lc.section.normal), f / np.linalg.norm(f), atol=1e-12)


def test_reanchor_at_the_origin_returns_the_same_mesh(hopf_cycle):
    """Reanchoring at ``x0s[0]`` returns the cycle's own mesh: a cycle point is its own
    nearest point."""
    lc = reanchor(hopf_cycle, hopf_cycle.x0s[0])
    assert_close(lc.x0s, hopf_cycle.x0s, atol=TOL["limit_cycle"])


def test_reanchor_refuses_a_point_of_the_wrong_shape(hopf_cycle):
    """`reanchor` raises `ValueError` for a point that is not a vector of the
    cycle's dimension."""
    with pytest.raises(ValueError, match="shape"):
        reanchor(hopf_cycle, jnp.zeros(3))


@pytest.mark.slow
def test_limit_cycle_fitzhugh_nagumo_against_langfield():
    """Langfield et al. 2014, Sec. III: T_Γ ≈ 11.2279, clockwise."""
    lc = find_limit_cycle(FitzhughNagumo(), jnp.array([3.0, 3.0]), t_settle=10.0, m=8)
    assert_close(lc.period, 11.2279, atol=2e-3)
    assert_close(lc.period, lc.settle.tau, atol=1e-6)  # brute force agrees with polish
    _check_invariants(lc, -1)


def test_limit_cycle_near_hopf_needs_the_polish():
    """A Bautin cycle with κ → 0: the return map barely contracts, so the fixed-point
    stage runs out of iterations at a tight tolerance, while a loose settle plus the
    shooting polish reaches the exact answer."""
    nf = BautinNormalForm(a=0.02, b=0.1, w=1.0, **INTEGRATION)  # κ = ρ'(1) small
    with pytest.raises(NotConvergedError):
        find_limit_cycle(nf, X0, t_settle=5.0, m=2, settle_tol=1e-10, max_returns=3)
    lc = find_limit_cycle(nf, X0, t_settle=5.0, m=2, settle_tol=1e-2, max_returns=40)
    assert_close(lc.period, abs(float(nf.period())), atol=TOL["limit_cycle"])
    assert_close(np.linalg.norm(lc.x0s, axis=1), 1.0, atol=TOL["limit_cycle"])


def test_limit_cycle_reverse_time_finds_the_repelling_cycle():
    """With ``reverse_time=True`` `find_limit_cycle` finds Winfree's repelling inner
    cycle ``r = a`` with its period ``2π/|ω(a)|``."""
    w = WinfreeNormalForm(a=0.25, w=-0.5, **INTEGRATION)
    lc = find_limit_cycle(
        w, jnp.array([0.3, 0.0]), t_settle=20.0, m=2, reverse_time=True
    )
    assert_close(lc.period, 2 * np.pi / 0.625, atol=TOL["limit_cycle"])
    assert_close(
        np.linalg.norm(lc.sample(32)[1], axis=1), 0.25, atol=TOL["limit_cycle"]
    )
    assert lc.reverse_time
    _check_invariants(lc, -1)


def test_limit_cycle_refuses_bad_starts():
    """`find_limit_cycle` raises `ValueError` from inside Winfree's hole and from
    Hopf's equilibrium."""
    with pytest.raises(ValueError):  # inside Winfree's hole: outside the basin
        find_limit_cycle(
            WinfreeNormalForm(a=0.25, w=-0.5, **INTEGRATION),
            jnp.array([0.1, 0.0]),
            t_settle=5.0,
        )
    with pytest.raises(ValueError):  # the origin is an equilibrium of Hopf
        find_limit_cycle(HOPF, jnp.zeros(2), t_settle=5.0)


def test_limit_cycle_guards_against_a_collapsed_period(monkeypatch):
    """The shooting equations are solved by T → 0 (Parker & Chua §5.5, "Extraneous
    solutions"): a guess with a tiny period converges there; ``find_limit_cycle``
    compares the polished period with the return-map estimate and rejects it."""
    module = importlib.import_module("deep_isochron.analysis.ode.limit_cycle")
    p = _on_cycle(HOPF)
    plane = CrossSection.normal_to_flow(HOPF, p)
    res = multiple_shooting(HOPF, p[None], 1e-3, plane, config=CFG)
    assert float(res.period) < 1e-2  # the extraneous solution exists
    monkeypatch.setattr(module, "multiple_shooting", lambda *a, **k: res)
    with pytest.raises(NotConvergedError) as info:
        module.find_limit_cycle(HOPF, X0, t_settle=5.0, m=1, config=CFG)
    assert info.value.stage == "shooting" and "collapsed" in str(info.value)


@given(
    a=st.floats(0.5, 2.0),
    w=st.floats(0.5, 3.0),
    sign=st.sampled_from([-1.0, 1.0]),
    r0=st.floats(0.4, 1.8),
    angle=st.floats(0.0, 6.2),
)
def test_limit_cycle_on_random_hopf_forms(a, w, sign, r0, angle):
    """Over Hypothesis-drawn Hopf parameters and starts, `find_limit_cycle` gives
    period ``2π/w``, unit radius and the orientation of ``sign``."""
    nf = HopfNormalForm(a=a, w=sign * w, **INTEGRATION)
    x0 = r0 * jnp.array([jnp.cos(angle), jnp.sin(angle)])
    lc = find_limit_cycle(nf, x0, t_settle=8.0 / a, m=2, config=CFG)
    assert_close(lc.period, 2 * np.pi / w, atol=TOL["limit_cycle"])
    assert_close(np.linalg.norm(lc.x0s, axis=1), 1.0, atol=TOL["limit_cycle"])
    assert lc.orientation(16) == int(sign)


# ------------------------------------------------------------------- 5. floquet --
METHODS = ("periodic_schur", "monodromy", "liouville")


def _exponent(fl) -> float:
    """The non-trivial Floquet exponent of a `Floquet` result."""
    return float(fl.exponents[1])


@pytest.mark.parametrize(
    "nf",
    [
        HOPF,
        BautinNormalForm(a=1.0, b=0.1, w=-1.5, **INTEGRATION),
        WinfreeNormalForm(a=-0.25, w=-0.5, **INTEGRATION),
    ],
    ids=["hopf", "bautin_cw", "winfree"],
)
def test_floquet_exponent_matches_the_normal_form(nf):
    """All three routes give κ = ρ'(1); the trivial multiplier is 1 and the periodic
    Schur chain closes."""
    lc = find_limit_cycle(nf, X0, t_settle=5.0, m=4, config=CFG)
    kappa = float(nf.floquet_exponent())
    for method in METHODS:
        fl = floquet(lc, method=method)
        assert_close(_exponent(fl), kappa, atol=TOL["floquet"])
        assert_close(fl.multipliers[1], np.exp(kappa * float(lc.period)), rtol=1e-7)
        if method != "liouville":
            assert float(fl.trivial_error) < TOL["floquet"]
    ps = floquet(lc, method="periodic_schur")
    assert float(ps.closure) < TOL["floquet"]
    assert fl.method == "liouville" and ps.vectors is not None


def test_floquet_vectors_of_hopf_are_radial():
    """Hopf's angular rate is constant, so the radial direction is invariant under
    the monodromy: the right Floquet vector at every mesh point is radial."""
    lc = find_limit_cycle(HOPF, X0, t_settle=5.0, m=8, config=CFG)
    v = floquet(lc).vectors[:, :, 0]
    radial = lc.x0s / jnp.linalg.norm(lc.x0s, axis=1, keepdims=True)
    assert_close(jnp.abs(jnp.sum(v * radial, axis=1)), 1.0, atol=TOL["floquet"])


def test_floquet_reverse_time_reports_the_forward_exponent():
    """Winfree's inner cycle r = a repels with κ_u = a(1 − a); found in reverse time,
    its multipliers and exponents are reported forward-time (ADR-0014 §4)."""
    a = 0.25
    w = WinfreeNormalForm(a=a, w=-0.5, **INTEGRATION)
    lc = find_limit_cycle(
        w, jnp.array([0.3, 0.0]), t_settle=20.0, m=4, reverse_time=True
    )
    for method in METHODS:
        fl = floquet(lc, method=method)
        assert_close(_exponent(fl), a * (1 - a), atol=TOL["floquet"])
        assert float(fl.multipliers[1]) > 1 and abs(float(fl.multipliers[0]) - 1) < 1e-7


def _cassini_line_integrals(ode, a: float, b: float, n: int = 2048):
    """T = ∮ dℓ / (s‖∇g‖) and log μ = −∮ ‖∇g‖ dℓ on the oval r(θ)² = a² cos 2θ +
    √(b⁴ − a⁴ sin² 2θ), by the trapezoidal rule in θ (exponentially convergent for
    the analytic integrand): oracles that involve no ODE solve."""

    def point(theta):
        r2 = a**2 * jnp.cos(2 * theta) + jnp.sqrt(b**4 - a**4 * jnp.sin(2 * theta) ** 2)
        r = jnp.sqrt(r2)
        return jnp.array([r * jnp.cos(theta), r * jnp.sin(theta)])

    thetas = jnp.linspace(0.0, 2 * jnp.pi, n, endpoint=False)
    speed = jax.vmap(lambda t: jnp.linalg.norm(jax.jacfwd(point)(t)))(thetas)
    grad_norm = jax.vmap(lambda t: jnp.linalg.norm(jax.grad(ode.level)(point(t))))(
        thetas
    )
    dtheta = 2 * jnp.pi / n
    period = float(jnp.sum(speed / (ode.speed * grad_norm)) * dtheta)
    log_mu = -float(jnp.sum(grad_norm * speed) * dtheta)
    return period, log_mu


def test_floquet_and_period_of_the_cassini_oval_against_line_integrals():
    """A level-set cycle's exponent κT = −∮‖∇g‖ dℓ (div f = −s‖∇g‖² on g = 0) and its
    period T = ∮ dℓ/(s‖∇g‖) are line integrals over the curve: a solver-free oracle
    for O2's exponent and, in passing, for O1's period."""
    a, b = 1.0, 1.1
    ode = cassini(a, b)
    period, log_mu = _cassini_line_integrals(ode, a, b)
    lc = find_limit_cycle(ode, jnp.array([1.3, 0.3]), t_settle=3.0, m=8, config=CFG)
    assert_close(float(lc.period), period, atol=TOL["floquet"])
    for method in ("periodic_schur", "liouville"):
        fl = floquet(lc, method=method)
        assert_close(_exponent(fl) * float(lc.period), log_mu, atol=TOL["floquet"])
    # κT ≈ −36.8 over T ≈ 1.7: μ ≈ 1e-16 is at machine resolution of the trivial
    # multiplier, so the explicit monodromy's eigenvalue is rounding noise — Bauer–Fike
    # bounds its error by ε‖M‖ ≈ μ, and the realized error (6e-2 to 5e-1 relative,
    # depending on the association order) is orders beyond the chain's 1e-9 — while
    # its trivial multiplier is still fine: the failure the chain exists to avoid
    mo = floquet(lc, method="monodromy")
    assert float(mo.trivial_error) < TOL["floquet"]
    mu = np.exp(log_mu)
    assert abs(float(mo.multipliers[1]) / mu - 1) > 1e-5


def test_floquet_on_the_guckenheimer_meloon_orbit():
    """The periodic Schur and Liouville routes agree on the Guckenheimer–Meloon orbit
    and the trivial multiplier is accurate."""
    ode = guckenheimer_meloon(0.07)
    lc = find_limit_cycle(
        ode, jnp.array([0.0, 0.2952161257895192]), t_settle=2.0, m=8, config=CFG
    )
    ps, li = floquet(lc, method="periodic_schur"), floquet(lc, method="liouville")
    assert_close(_exponent(ps), _exponent(li), atol=TOL["floquet"])
    assert float(ps.trivial_error) < TOL["floquet"]


@pytest.mark.slow
def test_floquet_fitzhugh_nagumo_shows_the_conditioning_floors():
    """FHN's cycle is strongly attracting (μ ≈ 1.7e-12): the periodic Schur chain and
    Liouville agree to 1e-8 relative; the explicit monodromy's eigenvalue is at its
    ε‖M‖/μ floor (≈ 1e-4 relative, Lust 2001, Eq. 10); the return-map Jacobian's
    eigenvalue is at the solver's absolute floor and wrong by orders of magnitude;
    the trivial multiplier reports the orbit's accuracy regardless."""
    lc = find_limit_cycle(FitzhughNagumo(), jnp.array([3.0, 3.0]), t_settle=10.0, m=8)
    ps, mo, li = (floquet(lc, method=m) for m in METHODS)
    mu = float(li.multipliers[1])
    assert 1e-13 < mu < 1e-11
    assert abs(float(ps.multipliers[1]) / mu - 1) < 1e-7
    assert 1e-7 < abs(float(mo.multipliers[1]) / mu - 1) < 1e-3
    assert float(ps.trivial_error) < 1e-8 and float(mo.trivial_error) < 1e-8
    pm = PoincareMap(lc.ode, lc.section, CFG, t_max=2 * float(lc.period))
    assert abs(float(return_map_multiplier(pm, lc.x0s[0])) / mu - 1) > 1
    # the leakage of the trace (not used): tr DP = (1 + μ) − 1 numerically
    DP = jax.jacfwd(lambda u: pm(u)[0])(lc.x0s[0])
    assert abs(float(jnp.trace(DP)) / mu - 1) > 1


def test_floquet_return_map_route_on_a_moderate_cycle():
    """`return_map_multiplier` matches ``e^{κT}`` to ``1e-8`` on a Bautin cycle with a
    moderate multiplier."""
    nf = BautinNormalForm(a=1.0, b=0.1, w=-1.5, **INTEGRATION)
    p = _on_cycle(nf, 0.4)
    pm = PoincareMap(nf, CrossSection.normal_to_flow(nf, p), CFG, t_max=20.0)
    assert_close(float(return_map_multiplier(pm, p)), _multiplier(nf), atol=1e-8)


def test_floquet_is_invariant_under_the_base_point_and_the_mesh():
    """The exponent from `floquet` is the same for a rolled mesh (base point moved)
    and for ``m`` in 1, 2, 16."""
    nf = BautinNormalForm(a=1.0, b=0.1, w=-1.5, **INTEGRATION)
    kappa = float(nf.floquet_exponent())
    lc = find_limit_cycle(nf, X0, t_settle=5.0, m=8, config=CFG)
    for k in (1, 3, 5):
        rolled = eqx.tree_at(lambda c: c.x0s, lc, jnp.roll(lc.x0s, -k, axis=0))
        assert_close(_exponent(floquet(rolled)), kappa, atol=TOL["floquet"])
    for m in (1, 2, 16):
        lc_m = find_limit_cycle(nf, X0, t_settle=5.0, m=m, config=CFG)
        assert_close(_exponent(floquet(lc_m)), kappa, atol=TOL["floquet"])


def test_periodic_schur_factorization_is_exact_and_vmappable():
    """``Q[i+1] R[i] = M_i Q[i]`` for every segment (pins the sign convention: rows
    of R and columns of Q flipped together), the seed is the first column of Q[0],
    and the jitted chain vmaps over a batch of products."""
    lc = find_limit_cycle(HOPF, X0, t_settle=5.0, m=4, config=CFG)
    J = segment_jacobians(lc)
    f = HOPF.rhs(0.0, lc.x0s[0])
    ps = periodic_schur(J, f / jnp.linalg.norm(f))
    for i in range(lc.m):
        assert_close(ps.Q[i + 1] @ ps.R[i], J[i] @ ps.Q[i], atol=TOL["identity"])
        assert_close(jnp.tril(ps.R[i], -1), 0.0, atol=TOL["identity"])
        assert bool(jnp.all(jnp.diag(ps.R[i]) > 0))
    assert_close(ps.Q[0][:, 0], f / jnp.linalg.norm(f), atol=TOL["identity"])
    assert_close(ps.signs, 1.0, atol=TOL["identity"])
    assert_close(monodromy(J), J[3] @ J[2] @ J[1] @ J[0], atol=TOL["identity"])
    batch = jnp.stack([J, J[::-1]])  # a second product: the same factors reversed
    mults = jax.vmap(lambda Js: periodic_schur(Js, f / jnp.linalg.norm(f)).multipliers)(
        batch
    )
    assert mults.shape == (2, 2)
    assert_close(mults[0], ps.multipliers, atol=TOL["identity"])


def test_floquet_functions_jit_and_vmap_over_cycles():
    """A ``LimitCycle`` is a fixed-shape pytree: ``floquet`` jits as it stands, and
    vmaps over a batch of cycles of the same mesh size (the ODE's parameters are
    leaves of the cycle, so a batch of normal forms is one call)."""
    cycles = [
        find_limit_cycle(
            HopfNormalForm(a=a, w=2.0, **INTEGRATION), X0, t_settle=5.0, m=2
        )
        for a in (0.5, 1.0)
    ]
    jitted = eqx.filter_jit(floquet)
    assert_close(jitted(cycles[1]).exponents[1], -2.0, atol=TOL["floquet"])
    batch = jax.tree.map(lambda *xs: jnp.stack(xs), *cycles)
    out = eqx.filter_vmap(floquet)(batch)
    assert_close(out.exponents[:, 1], jnp.array([-1.0, -2.0]), atol=TOL["floquet"])
    _, points = eqx.filter_jit(cycles[0].sample)(8)
    assert points.shape == (8, 2)


class _Linear3D(AbstractODE):
    """A three-dimensional linear ODE, for the non-planar refusals."""

    dim: int = eqx.field(static=True, default=3, init=False)

    def rhs(self, t, u, args=None):
        return -u


def test_floquet_refuses_non_planar_systems():
    """`periodic_schur` raises `NotImplementedError` and `floquet` and
    `LimitCycle.orientation` raise `ValueError` on a three-dimensional cycle."""
    with pytest.raises(NotImplementedError):
        periodic_schur(jnp.tile(jnp.eye(3), (2, 1, 1)), jnp.array([1.0, 0.0, 0.0]))
    fp = FixedPoint(
        x=jnp.zeros(3),
        tau=jnp.asarray(1.0),
        k=1,
        contraction=jnp.asarray(float("nan")),
        converged=True,
    )
    fake = LimitCycle(
        ode=_Linear3D(),
        x0s=jnp.zeros((2, 3)),
        period=jnp.asarray(1.0),
        residual=jnp.asarray(0.0),
        settle=fp,
        section=CrossSection(jnp.zeros(3), jnp.array([1.0, 0.0, 0.0])),
        config=CFG,
    )
    with pytest.raises(ValueError):
        floquet(fake, method="liouville")
    with pytest.raises(ValueError):
        fake.orientation()
