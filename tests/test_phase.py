"""The asymptotic phase (``deep_isochron.analysis.ode.phase``, roadmap O3) and
``wrap_phase``.

The oracles: ``math.remainder`` for ``wrap_phase``; the exact nearest point of Hopf's
unit circle for `nearest_cycle_time`; the closed-form phase of the Bautin and Winfree
normal forms, integrated with ``integration="cartesian"`` so that the solver is what is
tested; the phase along the cycle, $\\omega t$; the flow law
$\\Theta(\\Phi_s x) = \\Theta(x) + \\omega s$; and (``slow``) the coworker's stored
FitzHugh–Nagumo isochrons near the cycle.
"""

import copy
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from deep_isochron.analysis.ode import (
    asymptotic_phase,
    find_limit_cycle,
    LimitCycle,
    nearest_cycle_time,
    phase_grid,
    reanchor,
)
from deep_isochron.systems import FitzhughNagumo
from deep_isochron.systems.normal_forms import (
    BautinNormalForm,
    HopfNormalForm,
    WinfreeNormalForm,
)
from deep_isochron.utils.numerics import wrap_phase
from hypothesis import given, settings, strategies as st
from jaxtyping import Array, Float

from .helpers import (
    assert_close,
    FHN_ANCHOR,
    golden_isochron,
    GOLDEN_NEAR_CYCLE,
    golden_phase,
    PHASE_T_END,
    TOL,
)


INTEGRATION = {"integration": "cartesian"}  # the solver is under test, not the chart
NORMAL_FORMS = {
    # counterclockwise, with a radius-dependent angular rate (w0 != w)
    "bautin": BautinNormalForm(a=1.0, b=0.3, w=1.5, w0=0.5, **INTEGRATION),
    # clockwise, with Winfree's twist
    "winfree": WinfreeNormalForm(a=-0.25, w=-0.5, **INTEGRATION),
}


def _cycle(ode, x0) -> LimitCycle:
    """The cycle of `ode` with phase 0 at ``(1, 0)``."""
    cycle = find_limit_cycle(ode, jnp.asarray(x0), t_settle=10.0, m=4)
    return reanchor(cycle, jnp.array([1.0, 0.0]))


@pytest.fixture(scope="module", params=list(NORMAL_FORMS))
def normal_form_cycle(request) -> LimitCycle:
    """The unit-circle cycle of a normal form of `NORMAL_FORMS`."""
    return _cycle(NORMAL_FORMS[request.param], [1.2, 0.1])


def _points(radii, angles) -> Float[Array, "n 2"]:
    """The points at every pair of `radii` and `angles`."""
    r, a = np.meshgrid(radii, angles)
    return jnp.asarray(np.stack([r * np.cos(a), r * np.sin(a)], axis=-1).reshape(-1, 2))


def _phase(cycle: LimitCycle, xs: Float[Array, "n 2"], t_end: float):
    return jax.jit(jax.vmap(lambda x: asymptotic_phase(cycle, x, t_end=t_end)))(xs)


def _gap(a, b) -> np.ndarray:
    """The absolute wrapped difference of two phases."""
    return np.abs(np.asarray(wrap_phase(jnp.asarray(a) - jnp.asarray(b))))


# --------------------------------------------------------------- 1. wrap_phase --
def test_wrap_phase_equals_the_ieee_remainder():
    """`wrap_phase` equals ``math.remainder(θ, 2π)`` bit for bit, with $-\\pi$ sent to
    $\\pi$, on multiples of $2\\pi/n$ and on random angles, so that the isochron phases
    wrapped with it keep their file hashes; its derivative is 1."""
    rng = np.random.default_rng(0)
    angles = [2 * math.pi * k / n for n in (7, 64, 100) for k in range(-3 * n, 3 * n)]
    angles += rng.uniform(-1e3, 1e3, 1000).tolist()
    angles += [math.pi, -math.pi, 3 * math.pi, -3 * math.pi, 0.0]
    got = np.asarray(wrap_phase(jnp.asarray(angles)))
    for angle, value in zip(angles, got, strict=True):
        expected = math.remainder(angle, 2 * math.pi)
        assert value == (math.pi if expected == -math.pi else expected), angle
    assert float(jax.grad(wrap_phase)(7.0)) == 1.0


# ------------------------------------------------------- 2. nearest_cycle_time --
@pytest.fixture(scope="module")
def hopf_cycle() -> LimitCycle:
    """Hopf's unit circle, counterclockwise at angular rate 2, phase 0 at ``(1, 0)``."""
    return _cycle(HopfNormalForm(a=1.0, w=2.0, **INTEGRATION), [0.3, 0.1])


def test_nearest_cycle_time_finds_the_radial_projection(hopf_cycle):
    """Under `vmap`, for points on both sides of the circle and between samples, so
    that the Newton steps in time are traced and of either sign,
    `nearest_cycle_time` returns the radial projection, at the time that is its angle
    as a fraction of the period, and reports convergence."""
    angles = np.linspace(-np.pi, np.pi, 13, endpoint=False) + 1e-3
    xs = _points([0.5, 1.0, 1.7], angles)
    t, point, converged = jax.vmap(lambda y: nearest_cycle_time(hopf_cycle, y))(xs)
    on_circle = np.asarray(xs) / np.linalg.norm(np.asarray(xs), axis=1, keepdims=True)
    assert_close(point, on_circle, atol=TOL["limit_cycle"])
    expected = np.remainder(np.arctan2(on_circle[:, 1], on_circle[:, 0]), 2 * np.pi)
    period = float(hopf_cycle.period)
    assert _gap(2 * np.pi * np.asarray(t) / period, expected).max() < TOL["limit_cycle"]
    assert np.all(np.asarray(converged))


# -------------------------------------------------------- 3. asymptotic_phase --
def test_asymptotic_phase_is_the_closed_form_phase(normal_form_cycle):
    """On an annulus from 0.3 to 2.5 times the cycle radius, the asymptotic phase
    differs from the normal form's closed-form phase, signed by the sense of rotation,
    by one constant: phase 0 is at ``(1, 0)``, where the closed form is 0 too."""
    ode = normal_form_cycle.ode
    xs = _points([0.3, 0.7, 1.0, 1.6, 2.5], np.linspace(-np.pi, np.pi, 16))
    result = _phase(normal_form_cycle, xs, PHASE_T_END["normal_forms"])
    sense = math.copysign(1.0, float(ode.omega()))
    closed_form = sense * jax.vmap(ode.phase)(xs)
    offset = float(ode.phase(jnp.array([1.0, 0.0])))
    assert _gap(result.phase, closed_form - sense * offset).max() < TOL["phase"]
    assert np.asarray(result.discrepancy).max() < TOL["phase"]
    assert np.asarray(result.distance).max() < TOL["phase"]


def test_asymptotic_phase_on_the_cycle_is_omega_t(normal_form_cycle):
    """At the cycle point reached a time $t$ after phase 0, the phase is
    $2\\pi t/T$."""
    times, points = normal_form_cycle.sample(16)
    result = _phase(normal_form_cycle, points, PHASE_T_END["normal_forms"])
    expected = 2 * np.pi * np.asarray(times) / float(normal_form_cycle.period)
    assert _gap(result.phase, expected).max() < TOL["phase"]


@settings(max_examples=10, deadline=None)
@given(
    radius=st.floats(0.3, 2.5),
    angle=st.floats(-np.pi, np.pi),
    s=st.floats(0.0, 10.0),
)
def test_asymptotic_phase_advances_with_the_flow(hopf_cycle, radius, angle, s):
    """$\\Theta(\\Phi_s x) = \\Theta(x) + 2\\pi s/T$ on Hopf's basin."""
    x = jnp.array([radius * np.cos(angle), radius * np.sin(angle)])
    y = hopf_cycle.ode.flow_to(s, x, config=hopf_cycle.config)
    t_end = PHASE_T_END["normal_forms"]
    a, b = _phase(hopf_cycle, jnp.stack([x, y]), t_end).phase
    advance = 2 * np.pi * s / float(hopf_cycle.period)
    assert _gap(b - a, advance).max() < TOL["phase"]


def test_asymptotic_phase_refuses_a_repelling_cycle(hopf_cycle):
    """A cycle found in reverse time is repelling, and its phase is refused."""
    with pytest.raises(ValueError, match="attracting"):
        asymptotic_phase(
            copy.replace(hopf_cycle, reverse_time=True), jnp.ones(2), t_end=1.0
        )


# ------------------------------------------------------------- 4. phase_grid --
def test_phase_grid_orients_the_field_and_masks_the_equilibrium(hopf_cycle):
    """``field[i, j]`` is the phase at ``(xs[j], ys[i])``; the equilibrium at the
    origin, whose orbit never reaches the cycle, is the one masked point, with
    ``nan`` in the field and its diagnostics returned."""
    xs, ys = jnp.linspace(-1.5, 1.5, 5), jnp.linspace(-1.0, 1.0, 3)
    t_end = PHASE_T_END["normal_forms"]
    field, mask, result = phase_grid(hopf_cycle, xs, ys, t_end=t_end, batch_size=4)
    assert field.shape == mask.shape == result.phase.shape == (3, 5)
    expected = np.ones((3, 5), dtype=bool)
    expected[1, 2] = False
    np.testing.assert_array_equal(np.asarray(mask), expected)
    assert np.isnan(field[1, 2]) and float(result.distance[1, 2]) > 0.5
    one = asymptotic_phase(hopf_cycle, jnp.array([xs[4], ys[0]]), t_end=t_end)
    assert_close(field[0, 4], one.phase, atol=TOL["jit_eager"])


def test_phase_grid_refuses_float32(hopf_cycle):
    """`phase_grid` raises `RuntimeError` when float64 is off."""
    with jax.enable_x64(False), pytest.raises(RuntimeError, match="float64"):
        phase_grid(hopf_cycle, jnp.zeros(2), jnp.zeros(2), t_end=1.0)


# ------------------------------------------------------------ 5. FitzHugh–Nagumo --
@pytest.mark.slow
@pytest.mark.parametrize("k", [26, 51, 100])
def test_fitzhugh_nagumo_phase_on_the_coworkers_isochrons(k):
    """The points of the coworker's isochron file `k` within `GOLDEN_NEAR_CYCLE` of the
    cycle carry its phase. Farther out, near the source, a stored point's position
    error times the steep phase gradient there exceeds a radian; direct integration
    over five periods confirms the phase computed here, not the stored one."""
    ode = FitzhughNagumo()
    cycle = reanchor(
        find_limit_cycle(ode, jnp.array([1.0, 0.5]), t_settle=50.0), FHN_ANCHOR
    )
    _, samples = cycle.sample(4096)
    _, points = golden_isochron(k)
    distance = np.linalg.norm(points[:, None] - np.asarray(samples)[None], axis=2).min(
        1
    )
    near = jnp.asarray(points[distance < GOLDEN_NEAR_CYCLE])
    assert near.shape[0] >= 10
    result = _phase(cycle, near, PHASE_T_END["fitzhugh_nagumo"])
    assert _gap(result.phase, golden_phase(k)).max() < TOL["golden_phase"]
