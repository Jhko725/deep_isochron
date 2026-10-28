"""The stages of the isochron pipeline (``analysis.ode.auto.stages``) against the
oracles of ``docs/design/isochrons.md`` §5; every test but the stop reason runs AUTO
(marker ``auto``).

- Stages 1–3 against O1 and O2, for every planar system: the bundle's branch point at
  multiplier 1, which the coworker's ``+Aᵀ`` bundle did not have (§3.1); the orbit and
  period against O1's flow; ``Jw`` along O2's Floquet vector; ``w·F`` constant; each
  rotated base point at its phase on O1's cycle, phase π included; and, for the normal
  forms, ``w`` along the closed-form gradient of the phase.
- The coworker's Hopf route (his runs 0–1) against O1's cycle (decision 12).
- Stage 4 on the Bautin and Winfree forms against their closed-form phase
  (decision 14), and the stop reason of a half by the box edge.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from deep_isochron.analysis.ode import floquet, isochron_foliation, IsochronSettings
from deep_isochron.analysis.ode.auto import (
    adjoint_bundle,
    auto_session,
    AutoSolution,
    periodic_orbit,
    rotate_phase,
    Runner,
    stages,
)
from deep_isochron.analysis.ode.auto.codegen import AutoProblem, internal_indices
from jaxtyping import Float

from .helpers import anchored_cycle, ISOCHRON_SETTINGS, PLANAR_SYSTEMS, TOL


J = np.array([[0.0, -1.0], [1.0, 0.0]])
"""The quarter turn, which takes the bundle ``w`` to the stable direction."""


def _sine(a: Float[np.ndarray, "2"], b: Float[np.ndarray, "2"]) -> float:
    """The sine of the angle between the lines along `a` and `b`, accurate near 0."""
    return float(abs(a[0] * b[1] - a[1] * b[0]) / np.linalg.norm(a) / np.linalg.norm(b))


# ------------------------------------------------------------- stages 1–3 ----
@pytest.mark.auto
@pytest.mark.parametrize("name", list(PLANAR_SYSTEMS))
def test_stages_one_to_three_agree_with_o1_and_o2(auto, name):
    """Stage 2 branches at multiplier 1 and normalizes the bundle (decision D1); its
    orbit is O1's flow and its period O1's; ``Jw`` lies along O2's Floquet vector and
    ``w·F`` is constant along the cycle; stage 3 puts the base point of phase
    ``2πk/m`` at O1's mesh point ``x0s[k]``, with ``Jw`` along the Floquet vector
    there, phase π included, where the rotation starts. On the normal forms ``w``
    lies along the gradient of the closed-form phase, and that phase at each base point
    is the requested one, signed by the sense of rotation (decision 14)."""
    cycle = anchored_cycle(name)
    ode, m = cycle.ode, cycle.x0s.shape[0]
    vectors = np.asarray(floquet(cycle).vectors)
    assert m % 2 == 0  # so that k = m/2 is phase π, the start of stage 3
    ks = sorted({m // 8, m // 3, m // 2, (5 * m) // 8})
    phases = [math.remainder(2 * math.pi * k / m, 2 * math.pi) for k in ks]
    with auto_session(ode) as run:
        idx = run.idx
        bundle = adjoint_bundle(run, periodic_orbit(run, cycle))
        rotated = rotate_phase(run, bundle, phases)
    assert bundle.par[idx.mult] == pytest.approx(1.0, abs=TOL["branch_point"])
    assert bundle.par[idx.norm] == pytest.approx(1.0, abs=TOL["branch_point"])
    period, seg1 = bundle.par[idx.period], bundle.par[idx.seg1]
    assert period == pytest.approx(float(cycle.period), rel=TOL["auto_orbit"])
    # the orbit: segment 1 from phase 0, segment 2 from where segment 1 ends
    t, u = bundle.t, bundle.u
    first = np.asarray(
        ode.flow(jnp.asarray(t * seg1), cycle.x0s[0], config=cycle.config).ys
    )
    second = ode.flow(
        jnp.asarray(t * (period - seg1)), jnp.asarray(first[-1]), config=cycle.config
    )
    np.testing.assert_allclose(u[:, :2], first, atol=TOL["auto_orbit"])
    np.testing.assert_allclose(u[:, 2:4], np.asarray(second.ys), atol=TOL["auto_orbit"])
    # the bundle's direction and scale
    assert _sine(J @ u[0, 4:6], vectors[0, :, 0]) < TOL["bundle_direction"]
    field = jax.vmap(ode.rhs, (None, 0))
    w_dot_f = np.concatenate(
        [
            (
                u[:, 4 + j : 6 + j]
                * np.asarray(field(0.0, jnp.asarray(u[:, j : j + 2])))
            ).sum(1)
            for j in (0, 2)
        ]
    )
    assert np.ptp(w_dot_f) < TOL["bundle_scale"] * abs(w_dot_f.mean())
    # stage 3
    assert set(rotated) == set(phases)
    for k, theta in zip(ks, phases):
        base, w = rotated[theta].u[0, 2:4], rotated[theta].u[0, 6:8]
        np.testing.assert_allclose(
            base, np.asarray(cycle.x0s[k]), atol=TOL["auto_orbit"]
        )
        assert _sine(J @ w, vectors[k, :, 0]) < TOL["bundle_direction"]
        if name != "fitzhugh_nagumo":
            gradient = np.asarray(jax.grad(ode.phase)(jnp.asarray(base)))
            assert _sine(w, gradient) < TOL["bundle_direction"]
            sense = math.copysign(1.0, float(ode.omega()))
            gap = math.remainder(
                float(ode.phase(jnp.asarray(base))) - sense * theta, 2 * math.pi
            )
            assert abs(gap) < TOL["isochron_phase"]


def _hopf_route(run: Runner, problem: AutoProblem, c: float) -> AutoSolution:
    """The coworker's runs 0–1 for FitzHugh–Nagumo: the equilibrium continued in ``c``
    from 3 down to the Hopf point, then the branch of periodic orbits born there
    (``IPS = 2``, period in ``PAR(11)``) continued back up to ``c``.

    **Returns:**

    The orbit at ``c``.
    """
    slot, plain = problem.slot("c"), {"NDIM": 2, "NBC": 0, "NINT": 0}
    equilibria = run(
        [0.9066, -0.2582], PAR=problem.pars, ICP=[slot], IPS=1, DS=-0.01, NMX=200,
        SP=["HB1"], **plain,
    )  # fmt: skip
    orbits = run(
        equilibria("HB1"), IPS=2, ISW=-1, ICP=[slot, 11], DS=0.1, NMX=200,
        UZSTOP={slot: c}, **plain,
    )  # fmt: skip
    return run.labeled(orbits, "UZ")[0]


@pytest.mark.auto
def test_hopf_route_reaches_o1s_cycle(auto):
    """The coworker's route to the FitzHugh–Nagumo cycle, continuation from the Hopf
    point, ends at O1's period, and its orbit is O1's flow (decision 12)."""
    cycle = anchored_cycle("fitzhugh_nagumo")
    ode = cycle.ode
    with auto_session(ode) as run:
        orbit = _hopf_route(run, run.problem, ode.c)
    period = orbit.par[11]
    assert period == pytest.approx(float(cycle.period), rel=TOL["auto_orbit"])
    flowed = ode.flow(
        jnp.asarray(orbit.t * period), jnp.asarray(orbit.u[0]), config=cycle.config
    )
    np.testing.assert_allclose(orbit.u, np.asarray(flowed.ys), atol=TOL["auto_orbit"])


# ----------------------------------------------------------------- stage 4 ----
def test_a_user_point_on_the_box_is_a_box_stop():
    """A half that ends on an edge of the box stopped there; one that ends inside it
    stopped at ``eta_stop``, even when AUTO located that point with
    ``|eta|`` a little below ``eta_stop``."""
    idx = internal_indices(4)  # FitzHugh–Nagumo's four parameters come first
    box = (np.array([-2.0, -2.0]), np.array([2.0, 2.0]))
    settings = IsochronSettings(eta_stop=1e-6)
    empty = np.zeros((0,)), np.zeros((0, 2))

    def end(eta, x, y):
        par = {idx.eta: eta, idx.point[0]: x, idx.point[1]: y}
        return AutoSolution(*empty, par, "UZ", 7)

    reason = stages._stop_reason
    assert reason(end(-9.9999999532e-07, 0.8, 0.5), idx, settings, box) == "eta"
    assert reason(end(1.3e-10, 2.0, 0.1), idx, settings, box) == "box"
    assert reason(end(-5e-8, 0.2, -2.0), idx, settings, box) == "box"


@pytest.mark.auto
@pytest.mark.parametrize("name", ["bautin", "winfree"])
def test_normal_form_isochrons_carry_the_closed_form_phase(auto, name):
    """With phase 0 at ``(1, 0)``, the closed-form phase at every traced point equals
    the requested phase, signed by the sense of rotation (decision 14), on halves
    that reach from inside half the cycle's radius to outside one and a half."""
    cycle = anchored_cycle(name)
    ode = cycle.ode
    phases = [0.5, 2.0]
    foliation = isochron_foliation(cycle, phases, ISOCHRON_SETTINGS["reach"])
    assert not foliation.failures, foliation.failures
    sense = math.copysign(1.0, float(ode.omega()))
    for iso, theta in zip(foliation.isochrons, phases, strict=True):
        got = np.asarray(jax.vmap(ode.phase)(jnp.asarray(iso.points)))
        gap = np.remainder(got - sense * theta + np.pi, 2 * np.pi) - np.pi
        assert np.abs(gap).max() < TOL["isochron_phase"]
        radius = np.linalg.norm(iso.points, axis=1)
        assert radius.min() < 0.5 and radius.max() > 1.5
