"""Laws of ``deep_isochron.systems`` that are not normal-form mathematics: the flow
machinery, and facts about the specific observed systems at their default parameters.

1. flow               vmap over initial conditions == loop; SolverConfig honoured (a
                      loose tolerance changes the answer, a tight one does not); flow
                      reports failure in Solution.result with throw=False; SolverConfig
                      is hashable and leafless
2. observed systems   FitzHugh–Nagumo: equilibrium, its eigenvalues and the period of
                      the cycle, all from Langfield, Krauskopf & Osinga (2014) for
                      Winfree's parameters; Hodgkin–Huxley: gating variables in [0, 1],
                      the neuron spikes

The normal forms' own laws are in ``test_normal_forms.py``.
"""

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from deep_isochron.systems import (
    BautinNormalForm,
    FitzhughNagumo,
    HodgekinHuxley,
    SolverConfig,
)

from tests.helpers import assert_close, TOL


TIGHT = SolverConfig(solver=dfx.Dopri5(), rtol=1e-10, atol=1e-12, max_steps=16384)


# --------------------------------------------------------------------- 1. flow -----
@pytest.mark.parametrize(
    "ode",
    [FitzhughNagumo(), BautinNormalForm(1.0, 0.3, 1.5, 0.5)],
    ids=["fhn", "bautin"],
)
def test_vmap_over_initial_conditions_matches_loop(ode):
    ts = jnp.linspace(0.0, 2.0, 11)
    u0s = jax.random.normal(jax.random.key(0), (5, ode.dim))
    batched = eqx.filter_vmap(ode.flow, in_axes=(None, 0))(ts, u0s).ys
    tol = TOL["identity"]
    for i in range(5):
        assert_close(batched[i], ode.flow(ts, u0s[i]).ys, rtol=tol, atol=tol)


def test_solver_config_is_honoured():
    ode = FitzhughNagumo()
    ts = jnp.linspace(0.0, 20.0, 41)
    u0 = jnp.array([0.5, 0.1])
    tight = ode.flow(ts, u0, config=TIGHT).ys
    loose = ode.flow(ts, u0, config=SolverConfig(rtol=1e-2, atol=1e-2)).ys
    assert not jnp.allclose(tight, loose, rtol=1e-8, atol=1e-8), (
        "tolerances had no effect"
    )
    other = ode.flow(
        ts,
        u0,
        config=SolverConfig(
            solver=dfx.Kvaerno5(), rtol=1e-10, atol=1e-12, max_steps=16384
        ),
    ).ys
    assert_close(other, tight, rtol=TOL["flow"], atol=TOL["flow"])


def test_flow_reports_failure_without_raising():
    ode = FitzhughNagumo()
    ts = jnp.linspace(0.0, 50.0, 11)
    config = SolverConfig(max_steps=8, throw=False)
    sol = ode.flow(ts, jnp.array([0.5, 0.1]), config=config)
    assert sol.result != dfx.RESULTS.successful
    sol = ode.flow(ts, jnp.array([0.5, 0.1]), config=TIGHT)
    assert sol.result == dfx.RESULTS.successful


def test_solver_config_is_hashable_and_leafless():
    cfg = SolverConfig(solver=dfx.Kvaerno5(), rtol=1e-4)
    hash(cfg)
    assert jax.tree.leaves(eqx.filter(cfg, eqx.is_array)) == []


# ----------------------------------------------------------- 2. observed systems ---
def test_fitzhugh_nagumo_fixed_point_eigenvalues():
    """Langfield, Krauskopf & Osinga, Chaos 24, 013131 (2014), Sec. III: for Winfree's
    parameters a = 0.7, b = 0.8, c = 3, z = -0.4 (this class's defaults) the unique
    equilibrium is (0.9066, -0.2582), a source with eigenvalues 0.1339 ± 0.9163 i."""
    ode = FitzhughNagumo()
    u = jnp.array([0.90656707, -0.25820883])
    assert_close(ode.rhs(0.0, u), jnp.zeros(2), atol=1e-7)
    eig = jnp.linalg.eigvals(jax.jacfwd(lambda v: ode.rhs(0.0, v))(u))
    assert_close(jnp.sort(eig.real), jnp.array([0.13387089, 0.13387089]), atol=1e-6)
    assert_close(
        jnp.sort(jnp.abs(eig.imag)), jnp.array([0.91628034, 0.91628034]), atol=1e-6
    )


def test_fitzhugh_nagumo_period():
    """Langfield et al. (2014), Sec. III: the stable periodic orbit has period
    T_Γ ≈ 11.2279 (clockwise). Measured from upward zero crossings of x - x* after the
    transient, with linear interpolation of the crossing times."""
    ode = FitzhughNagumo()
    x_star = 0.90656707
    ts = jnp.linspace(0.0, 200.0, 20001)
    cfg = SolverConfig(solver=dfx.Dopri5(), rtol=1e-10, atol=1e-12, max_steps=1 << 17)
    x = ode.flow(ts, jnp.array([-1.0, 1.0]), config=cfg).ys[:, 0] - x_star
    up = jnp.flatnonzero((x[:-1] < 0) & (x[1:] >= 0))
    up = up[up > len(ts) // 2]  # after the transient
    t_cross = ts[up] - x[up] * (ts[up + 1] - ts[up]) / (x[up + 1] - x[up])
    periods = jnp.diff(t_cross)
    assert_close(periods, jnp.full_like(periods, 11.2279), atol=2e-3)


def test_hodgkin_huxley_gating_variables_stay_in_unit_interval():
    ode = HodgekinHuxley()
    ts = jnp.linspace(0.0, 50.0, 501)
    u0 = jnp.array([-65.0, 0.05, 0.6, 0.32])
    ys = ode.flow(
        ts,
        u0,
        config=SolverConfig(
            solver=dfx.Kvaerno5(), rtol=1e-8, atol=1e-10, max_steps=65536
        ),
    ).ys
    gates = ys[:, 1:]
    assert jnp.all(jnp.isfinite(ys))
    assert jnp.all((gates >= -1e-6) & (gates <= 1 + 1e-6))
    assert ys[:, 0].max() > 0.0, "no spike: current I=30 should drive firing"
