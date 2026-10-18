"""``analysis.ode`` — O1, the limit cycle from the vector field, against exact oracles.

1. Hyperplane        signed distance, unit normal, the flow-normal section
2. PoincareMap       the first-return map on the normal forms (exact return time and
                     point); P₋ on (p, n) = P₊ on (p, −n); the two-sided map on a
                     four-crossing (Cassini) cycle visits the four intersections;
                     fixed_point with period 2 there, contraction → e^{κT}; the
                     Jacobian's eigenvalues {0, e^{κT}} (Parker & Chua Eq. 5.21); the
                     basin and horizon failures; reverse time on Winfree's repelling
                     inner cycle
3. multiple_shooting from a perturbed guess, m = 1 and m = 4 agree; tolerance
                     monotonicity; the guards
4. limit_cycle       Hopf/Bautin/Winfree (exact T, radius, multiplier), the
                     Guckenheimer–Meloon algebraic orbit (max|g|), FitzHugh–Nagumo
                     against Langfield et al. (slow), near-Hopf Bautin, reverse time,
                     output invariants; Hypothesis over Hopf parameters
"""

import diffrax as dfx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from deep_isochron.analysis import Cycle, NotConvergedError
from deep_isochron.analysis.ode import (
    Hyperplane,
    limit_cycle,
    LimitCycle,
    multiple_shooting,
    PoincareMap,
    segment_flow,
    SHOOTING_SOLVER_CONFIG,
)
from deep_isochron.systems import FitzhughNagumo, SolverConfig
from deep_isochron.systems.normal_forms import (
    BautinNormalForm,
    HopfNormalForm,
    WinfreeNormalForm,
)
from hypothesis import given, strategies as st

from .helpers import (
    assert_close,
    cassini_cycle,
    guckenheimer_meloon_cycle,
    TOL,
)


CFG = SHOOTING_SOLVER_CONFIG
HOPF = HopfNormalForm(a=1.0, w=2.0)
X0 = jnp.array([0.3, 0.1])


def _on_cycle(nf, angle=0.0):
    return jnp.array([jnp.cos(angle), jnp.sin(angle)])


def _multiplier(nf) -> float:
    """e^{κT} with the *unsigned* period (the normal forms' ``period()`` is signed)."""
    return float(jnp.exp(nf.floquet_exponent() * jnp.abs(nf.period())))


# ------------------------------------------------------------------ 1. Hyperplane --
def test_hyperplane_signed_distance_and_unit_normal():
    plane = Hyperplane(jnp.array([1.0, 2.0]), jnp.array([0.0, 3.0]))
    assert_close(jnp.linalg.norm(plane.normal), 1.0, atol=TOL["identity"])
    assert_close(plane(jnp.array([1.0, 2.0])), 0.0, atol=TOL["identity"])
    assert_close(plane(jnp.array([5.0, 2.5])), 0.5, atol=TOL["identity"])
    assert_close(plane.flip()(jnp.array([5.0, 2.5])), -0.5, atol=TOL["identity"])
    with pytest.raises(ValueError):
        Hyperplane(jnp.zeros(2), jnp.zeros(2))


def test_flow_normal_section_points_along_the_flow():
    p = _on_cycle(HOPF)
    plane = Hyperplane.through(HOPF, p)
    assert float(jnp.dot(plane.normal, HOPF.rhs(0.0, p))) > 0
    assert_close(plane(p), 0.0, atol=TOL["identity"])
    with pytest.raises(ValueError):  # the origin is an equilibrium
        Hyperplane.through(HOPF, jnp.zeros(2))


# ----------------------------------------------------------------- 2. PoincareMap --
@pytest.mark.parametrize(
    "nf",
    [HOPF, BautinNormalForm(a=1.0, b=0.1, w=-1.5), WinfreeNormalForm(a=-0.25, w=-0.5)],
    ids=["hopf", "bautin_cw", "winfree"],
)
def test_return_map_on_the_cycle_returns_in_one_period(nf):
    p = _on_cycle(nf, 0.4)
    P = PoincareMap(nf, Hyperplane.through(nf, p), CFG, t_max=20.0)
    y, tau = P(p)
    assert_close(tau, abs(float(nf.period())), atol=TOL["return_map"])
    assert_close(y, p, atol=TOL["return_map"])


def test_negative_map_is_the_positive_map_of_the_flipped_section():
    p = _on_cycle(HOPF, 1.0)
    plane = Hyperplane.through(HOPF, p)
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
    ode = cassini_cycle()
    p = _cassini_point_at_height(ode)
    plane = Hyperplane(p, jnp.array([0.0, 1.0]))  # horizontal section above the waist
    both = PoincareMap(ode, plane, CFG, t_max=50.0, direction="both")
    xs, taus = both.iterate(p, 4)
    heights = np.asarray(xs[:, 1])
    assert_close(heights, 0.5, atol=TOL["return_map"])
    # four distinct intersections, then back to the start
    xcoords = np.asarray(xs[:4, 0])
    assert len(np.unique(np.round(xcoords, 6))) == 4
    assert_close(xs[4], p, atol=TOL["return_map"])
    # the one-sided map sees a period-2 orbit
    pos = PoincareMap(ode, plane, CFG, t_max=50.0)
    fp = pos.fixed_point(p, tol=TOL["return_map"])
    assert fp.k == 2
    assert_close(fp.tau, float(jnp.sum(taus)), atol=TOL["return_map"])
    with pytest.raises(ValueError):
        both.fixed_point(p)


def test_fixed_point_contracts_at_the_floquet_multiplier():
    P = PoincareMap(HOPF, Hyperplane.through(HOPF, _on_cycle(HOPF)), CFG, t_max=20.0)
    fp = P.fixed_point(jnp.array([1.05, 0.0]), tol=1e-6)
    assert fp.converged and fp.k == 1
    assert_close(jnp.linalg.norm(fp.x), 1.0, atol=1e-6)
    # successive differences shrink by e^{κT} per return once in the linear regime
    # (κ = ρ'(1) = -2a, T = 2π/w)
    assert_close(fp.contraction, _multiplier(HOPF), rtol=2e-2)
    with pytest.raises(NotConvergedError) as info:
        P.fixed_point(jnp.array([1.3, 0.0]), tol=1e-12, max_iter=1)
    assert info.value.stage == "settle" and not info.value.last.converged


def test_return_map_jacobian_has_eigenvalues_zero_and_the_multiplier():
    nf = BautinNormalForm(a=1.0, b=0.2, w=1.0)
    p = _on_cycle(nf, 0.7)
    P = PoincareMap(nf, Hyperplane.through(nf, p), CFG, t_max=20.0)
    eig = np.sort(np.abs(np.linalg.eigvals(np.asarray(P.jacobian(p)))))
    assert_close(eig[0], 0.0, atol=TOL["trivial_multiplier"])
    assert_close(eig[1], _multiplier(nf), atol=TOL["trivial_multiplier"])


def test_return_map_fails_loudly_without_a_crossing():
    # a section the Hopf cycle never reaches: through (3, 0) with normal (1, 0)
    plane = Hyperplane(jnp.array([3.0, 0.0]), jnp.array([1.0, 0.0]))
    P = PoincareMap(HOPF, plane, CFG, t_max=5.0)
    with pytest.raises(NotConvergedError) as info:
        P(jnp.array([1.0, 0.0]))
    assert info.value.stage == "return"


def test_reverse_time_map_contracts_onto_winfrees_repelling_inner_cycle():
    """In reverse time the repelling cycle r = a attracts (Parker & Chua §5.1.1,
    footnote 1): one return (a full period, 2π/|ω(a)| with ω(a) = −(1 + w(1 − a)))
    brings the radius closer to a by e^{−κ_u T}, κ_u = a(1 − a)."""
    w = WinfreeNormalForm(a=0.25, w=-0.5)
    x0 = jnp.array([0.3, 0.0])  # just outside the hole
    p = segment_flow(w, x0, jnp.asarray(20.0), CFG, -1.0)  # partly settled, reversed
    P = PoincareMap(w, Hyperplane.through(w, p), CFG, t_max=30.0, reverse_time=True)
    y, tau = P(p)
    T = 2 * np.pi / 0.625
    assert_close(tau, T, atol=1e-2)  # off the cycle ω depends on r: a brute-force guess
    d_before = abs(float(jnp.linalg.norm(p)) - 0.25)
    d_after = abs(float(jnp.linalg.norm(y)) - 0.25)
    kappa_u = float(w.inner_floquet_exponent())
    assert_close(d_after / d_before, np.exp(-kappa_u * T), rtol=5e-2)
    fp = P.fixed_point(p, tol=TOL["return_map"])
    assert fp.k == 1
    assert_close(jnp.linalg.norm(fp.x), 0.25, atol=TOL["return_map"])
    assert_close(fp.tau, T, atol=TOL["return_map"])


# ----------------------------------------------------------- 3. multiple_shooting --
@pytest.mark.parametrize("m", [1, 4])
def test_shooting_polishes_a_perturbed_guess(m):
    nf = HOPF
    T = abs(float(nf.period()))
    p = _on_cycle(nf)
    plane = Hyperplane.through(nf, p)
    times = T * jnp.arange(m) / m
    exact = jnp.stack([_on_cycle(nf, float(nf.omega()) * t) for t in times])
    guess = exact + 1e-2 * jnp.cos(7.0 * jnp.arange(2 * m).reshape(m, 2))
    guess = guess.at[0].set(p)  # the phase condition holds at the guess
    res = multiple_shooting(nf, guess, 1.05 * T, plane, config=CFG, coarsen=False)
    assert_close(res.period, T, atol=TOL["limit_cycle"])
    assert_close(jnp.linalg.norm(res.points, axis=1), 1.0, atol=TOL["limit_cycle"])
    assert res.residual < TOL["limit_cycle"]


def test_shooting_is_more_accurate_with_tighter_solver_tolerances():
    nf = HOPF
    T = abs(float(nf.period()))
    p = _on_cycle(nf)
    plane = Hyperplane.through(nf, p)
    out = []
    for rtol in (1e-6, 1e-10):
        cfg = SolverConfig(
            solver=dfx.Dopri8(), rtol=rtol, atol=rtol * 1e-2, adjoint=dfx.ForwardMode()
        )
        res = multiple_shooting(
            nf, p[None], 1.02 * T, plane, config=cfg, rtol=1e-12, atol=1e-14
        )
        out.append(abs(res.period - T))
    assert out[1] < out[0]


def test_shooting_rejects_bad_input():
    plane = Hyperplane.through(HOPF, _on_cycle(HOPF))
    with pytest.raises(ValueError):
        multiple_shooting(HOPF, jnp.zeros((0, 2)), 1.0, plane)
    with pytest.raises(ValueError):
        multiple_shooting(HOPF, _on_cycle(HOPF)[None], -1.0, plane)


# ------------------------------------------------------------------ 4. limit_cycle --
def _check_invariants(lc: LimitCycle, orientation: int):
    assert lc.period > 0
    assert_close(lc.points[0], lc.x0, atol=TOL["identity"])
    assert_close(lc.section(jnp.asarray(lc.x0)), 0.0, atol=TOL["return_map"])
    assert_close(lc.times, lc.period * np.arange(len(lc.times)) / len(lc.times))
    assert lc.orientation == orientation
    assert lc.trivial_multiplier_error < TOL["trivial_multiplier"]
    assert lc.residual < TOL["limit_cycle"]


@pytest.mark.parametrize(
    "nf, orientation",
    [
        (HOPF, 1),
        (BautinNormalForm(a=1.0, b=0.1, w=-1.5), -1),
        (WinfreeNormalForm(a=-0.25, w=-0.5), -1),
    ],
    ids=["hopf", "bautin_cw", "winfree"],
)
def test_limit_cycle_matches_the_normal_form(nf, orientation):
    lc = limit_cycle(nf, X0, t_settle=5.0, m=2, n_points=64, config=CFG)
    assert_close(lc.period, abs(float(nf.period())), atol=TOL["limit_cycle"])
    assert_close(np.linalg.norm(lc.points, axis=1), 1.0, atol=TOL["limit_cycle"])
    eig = np.sort(np.abs(np.linalg.eigvals(lc.monodromy)))
    assert_close(eig[1], 1.0, atol=TOL["trivial_multiplier"])
    assert_close(eig[0], _multiplier(nf), atol=TOL["trivial_multiplier"])
    _check_invariants(lc, orientation)
    # the phase-uniform sampling: the cycle's phase advances uniformly in time
    angles = np.unwrap(np.arctan2(lc.points[:, 1], lc.points[:, 0]))
    assert_close(np.diff(angles), orientation * 2 * np.pi / len(lc.times), atol=1e-8)


def test_limit_cycle_points_are_a_phase_uniform_cycle():
    """The output is what ``Cycle.from_points`` expects: points at phases uniform in
    time; the fitted curve reproduces them (the two halves of ``analysis`` meet)."""
    lc = limit_cycle(BautinNormalForm(a=1.0, b=0.1, w=-1.5), X0, t_settle=5.0, m=2)
    phases = 2 * np.pi * lc.times / lc.period
    cyc = Cycle.from_points(phases, lc.points, lc.period)
    assert cyc.residual < TOL["cycle_query"]
    assert_close(cyc(phases), lc.points, atol=TOL["cycle_query"])


def test_limit_cycle_on_the_guckenheimer_meloon_algebraic_orbit():
    ode = guckenheimer_meloon_cycle(0.07)
    start = jnp.array([0.0, 0.2952161257895192])  # on the orbit (their p. 970)
    lc = limit_cycle(ode, start, t_settle=2.0, m=4, n_points=128, config=CFG)
    g = np.array([float(ode.level(jnp.asarray(pt))) for pt in lc.points])
    assert np.abs(g).max() < TOL["level_set"]
    assert 7.6 < lc.period < 7.8  # "period approximately 7.7"
    assert_close(lc.settle.tau, lc.period, atol=TOL["return_map"])
    _check_invariants(lc, 1)


@pytest.mark.slow
def test_limit_cycle_fitzhugh_nagumo_against_langfield():
    """Langfield, Krauskopf & Osinga (2014), Sec. III: T_Γ ≈ 11.2279, clockwise."""
    lc = limit_cycle(FitzhughNagumo(), jnp.array([3.0, 3.0]), t_settle=10.0, m=8)
    assert_close(lc.period, 11.2279, atol=2e-3)
    assert_close(lc.period, lc.settle.tau, atol=1e-6)  # brute force agrees with polish
    _check_invariants(lc, -1)


def test_limit_cycle_near_hopf_needs_the_polish():
    """A Bautin cycle with κ → 0: the return map barely contracts, so the fixed-point
    stage runs out of iterations at a tight tolerance, while a loose settle plus the
    shooting polish reaches the exact answer."""
    nf = BautinNormalForm(a=0.02, b=0.1, w=1.0)  # κ = ρ'(1) small
    with pytest.raises(NotConvergedError):
        limit_cycle(nf, X0, t_settle=5.0, m=2, settle_tol=1e-10, max_returns=3)
    lc = limit_cycle(nf, X0, t_settle=5.0, m=2, settle_tol=1e-2, max_returns=40)
    assert_close(lc.period, abs(float(nf.period())), atol=TOL["limit_cycle"])
    assert_close(np.linalg.norm(lc.points, axis=1), 1.0, atol=TOL["limit_cycle"])


def test_limit_cycle_reverse_time_finds_the_repelling_cycle():
    w = WinfreeNormalForm(a=0.25, w=-0.5)
    lc = limit_cycle(w, jnp.array([0.3, 0.0]), t_settle=20.0, m=2, reverse_time=True)
    assert_close(lc.period, 2 * np.pi / 0.625, atol=TOL["limit_cycle"])
    assert_close(np.linalg.norm(lc.points, axis=1), 0.25, atol=TOL["limit_cycle"])
    eig = np.sort(np.abs(np.linalg.eigvals(lc.monodromy)))
    kappa_u = float(w.inner_floquet_exponent())
    assert_close(eig[1], np.exp(kappa_u * lc.period), rtol=1e-8)  # repelling: > 1
    assert lc.reverse_time and lc.orientation == -1


def test_limit_cycle_refuses_bad_starts():
    with pytest.raises(ValueError):  # inside Winfree's hole: outside the basin
        limit_cycle(
            WinfreeNormalForm(a=0.25, w=-0.5), jnp.array([0.1, 0.0]), t_settle=5.0
        )
    with pytest.raises(ValueError):  # the origin is an equilibrium of Hopf
        limit_cycle(HOPF, jnp.zeros(2), t_settle=5.0)


def test_limit_cycle_guards_against_a_collapsed_period(monkeypatch):
    """The shooting equations are solved by T → 0 (Parker & Chua §5.5, "Extraneous
    solutions"): a guess with a tiny period converges there; ``limit_cycle`` compares
    the polished period with the return-map estimate and rejects it."""
    import importlib

    module = importlib.import_module("deep_isochron.analysis.ode.limit_cycle")

    p = _on_cycle(HOPF)
    plane = Hyperplane.through(HOPF, p)
    res = multiple_shooting(HOPF, p[None], 1e-3, plane, config=CFG, coarsen=False)
    assert res.period < 1e-2  # the extraneous solution exists
    monkeypatch.setattr(
        module,
        "multiple_shooting",
        lambda *a, **k: res,  # pretend the polish found it
    )
    with pytest.raises(NotConvergedError) as info:
        module.limit_cycle(HOPF, X0, t_settle=5.0, m=1, config=CFG)
    assert info.value.stage == "shooting" and "collapsed" in str(info.value)


@given(
    a=st.floats(0.5, 2.0),
    w=st.floats(0.5, 3.0),
    sign=st.sampled_from([-1.0, 1.0]),
    r0=st.floats(0.4, 1.8),
    angle=st.floats(0.0, 6.2),
)
def test_limit_cycle_on_random_hopf_forms(a, w, sign, r0, angle):
    nf = HopfNormalForm(a=a, w=sign * w)
    x0 = r0 * jnp.array([jnp.cos(angle), jnp.sin(angle)])
    lc = limit_cycle(nf, x0, t_settle=8.0 / a, m=2, n_points=16, config=CFG)
    assert_close(lc.period, 2 * np.pi / w, atol=TOL["limit_cycle"])
    assert_close(np.linalg.norm(lc.points, axis=1), 1.0, atol=TOL["limit_cycle"])
    assert lc.orientation == int(sign)
    assert lc.trivial_multiplier_error < TOL["trivial_multiplier"]
