"""Laws of the normal forms (``deep_isochron.systems.normal_forms``), generic over the
parameters (Hypothesis draws both forms).

1. structure      phase advances at rate w; isostable decays at rate kappa — exactly, by
                  autodiff of the defining identities, and again along the integrated
                  flow; floquet_exponent == 2 rho'(1); h(1) = psi(1) = rho(1) = 0;
                  cartesian rhs == polar rhs pushed through the chart; isochron /
                  limit_cycle consistency; Hopf == Bautin(b=0); Bautin's unstable outer
                  cycle for -1 < b < 0
2. integrations   cartesian / polar / r_squared agree away from the origin; cartesian
                  is finite at the origin; unknown name rejected; each integration is
                  static under filter_jit (one trace each)

Facts about specific observed systems are in ``test_systems.py``.
"""

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from deep_isochron.systems import (
    BautinNormalForm,
    HopfNormalForm,
    INTEGRATIONS,
    SolverConfig,
)
from hypothesis import given, settings, strategies as st

from tests.helpers import assert_close, TOL


TIGHT = SolverConfig(solver=dfx.Dopri5(), rtol=1e-10, atol=1e-12, max_steps=16384)

normal_forms = st.one_of(
    st.builds(
        HopfNormalForm,
        a=st.floats(0.2, 3.0),
        w=st.floats(-3.0, 3.0).filter(lambda w: abs(w) > 0.1),
        w0=st.floats(-3.0, 3.0),
    ),
    st.builds(
        BautinNormalForm,
        a=st.floats(0.2, 3.0),
        b=st.floats(0.0, 2.0),
        w=st.floats(-3.0, 3.0).filter(lambda w: abs(w) > 0.1),
        w0=st.floats(-3.0, 3.0),
    ),
)
"""Both normal forms with drawn parameters (``w0 != w`` gives sheared isochrons);
``b >= 0`` so that the cycle attracts the whole plane (see ``BautinNormalForm``)."""

radii = st.floats(0.15, 2.5)
angles = st.floats(-3.1, 3.1)


def _point(r, theta):
    return jnp.array([r * jnp.cos(theta), r * jnp.sin(theta)])


# ------------------------------------------------------- 1. normal-form structure --
@given(nf=normal_forms, r=radii)
def test_phase_and_isostable_identities(nf, r):
    """Exact (autodiff) form of dphi/dt = w and dpsi/dt = kappa psi along the flow:
    w(r²) + h'(r) r rho(r²) == w(1)  and  psi'(r) r rho(r²) == kappa psi(r)."""
    r = jnp.asarray(r)
    rdot = r * nf.radial_rate(r**2)
    lhs_phase = nf.angular_rate(r**2) + jax.grad(nf.phase_shift)(r) * rdot
    assert_close(
        lhs_phase, nf.omega(), rtol=TOL["closed_form"], atol=TOL["closed_form"]
    )
    lhs_amp = jax.grad(nf.isostable)(r) * rdot
    assert_close(
        lhs_amp,
        nf.floquet_exponent() * nf.isostable(r),
        rtol=TOL["closed_form"],
        atol=TOL["closed_form"],
    )


@given(nf=normal_forms)
def test_normal_form_closed_forms_are_consistent(nf):
    one = jnp.asarray(1.0)
    assert_close(nf.phase_shift(one), 0.0, atol=TOL["identity"])
    assert_close(nf.isostable(one), 0.0, atol=TOL["identity"])
    assert_close(nf.radial_rate(one), 0.0, atol=TOL["identity"])
    # the class's closed-form kappa equals the generic autodiff definition 2 rho'(1)
    assert_close(
        nf.floquet_exponent(),
        2 * jax.grad(nf.radial_rate)(one),
        rtol=TOL["closed_form"],
        atol=TOL["closed_form"],
    )
    assert nf.floquet_exponent() < 0
    assert_close(nf.period(), 2 * jnp.pi / nf.omega(), rtol=TOL["identity"])
    assert_close(
        nf.floquet_multiplier(),
        jnp.exp(nf.floquet_exponent() * nf.period()),
        rtol=TOL["identity"],
    )
    # isostable sign convention: positive inside the cycle, negative outside
    assert nf.isostable(jnp.asarray(0.5)) > 0 > nf.isostable(jnp.asarray(1.5))


@given(nf=normal_forms, r=radii, theta=angles)
def test_cartesian_rhs_is_polar_rhs_in_the_chart(nf, r, theta):
    u = _point(r, theta)
    pushed = jax.jvp(
        nf.from_chart, (nf.to_chart(u),), (nf.rhs_polar(0.0, nf.to_chart(u)),)
    )[1]
    assert_close(
        nf.rhs(0.0, u), pushed, rtol=TOL["closed_form"], atol=TOL["closed_form"]
    )
    assert_close(
        nf.from_chart(nf.to_chart(u)),
        u,
        rtol=TOL["closed_form"],
        atol=TOL["closed_form"],
    )


@given(nf=normal_forms, r=radii, theta=angles)
def test_phase_and_amplitude_along_the_flow(nf, r, theta):
    """Integrated version: phase(u(t)) - phase(u0) == w t (mod 2pi) and
    amplitude(u(t)) == amplitude(u0) exp(kappa t)."""
    ts = jnp.linspace(0.0, 1.5, 7)
    ys = nf.flow(ts, _point(r, theta), config=TIGHT, integration="cartesian").ys
    dphi = jax.vmap(nf.phase)(ys) - nf.phase(ys[0]) - nf.omega() * ts
    dphi = jnp.arctan2(jnp.sin(dphi), jnp.cos(dphi))  # mod 2pi
    assert_close(dphi, jnp.zeros_like(ts), atol=TOL["flow"])
    amp = jax.vmap(nf.amplitude)(ys)
    expected = nf.amplitude(ys[0]) * jnp.exp(nf.floquet_exponent() * ts)
    assert_close(amp, expected, rtol=TOL["flow"], atol=TOL["flow"])


@given(nf=normal_forms, phi=angles, r=radii)
def test_isochron_and_limit_cycle(nf, phi, r):
    pts = nf.isochron(phi, jnp.array([r, 1.0]))
    assert_close(jax.vmap(nf.phase)(pts), jnp.full(2, phi), atol=TOL["closed_form"])
    assert_close(pts[1], nf.limit_cycle(jnp.asarray(phi)), atol=TOL["closed_form"])
    assert_close(
        nf.phase(nf.limit_cycle(jnp.asarray(phi))), phi, atol=TOL["closed_form"]
    )


@given(a=st.floats(0.2, 3.0), w=st.floats(0.5, 3.0), w0=st.floats(-2.0, 2.0), r=radii)
def test_hopf_is_bautin_with_b_zero(a, w, w0, r):
    hopf, bautin = HopfNormalForm(a, w, w0), BautinNormalForm(a, 0.0, w, w0)
    r = jnp.asarray(r)
    for f, g in (
        (hopf.phase_shift, bautin.phase_shift),
        (hopf.isostable, bautin.isostable),
    ):
        assert_close(f(r), g(r), rtol=TOL["closed_form"], atol=TOL["closed_form"])
    assert_close(
        hopf.floquet_exponent(), bautin.floquet_exponent(), rtol=TOL["identity"]
    )
    u = _point(r, 0.7)
    assert_close(
        hopf.rhs(0.0, u), bautin.rhs(0.0, u), rtol=TOL["identity"], atol=TOL["identity"]
    )


@given(b=st.floats(-0.9, -0.1), a=st.floats(0.2, 3.0))
def test_bautin_negative_b_has_unstable_outer_cycle(b, a):
    """Bautin bifurcation structure: for -1 < b < 0 a second cycle at r² = -1/b, with
    rho' > 0 there (unstable); r = 1 stays stable."""
    nf = BautinNormalForm(a, b, 1.0)
    s_outer = jnp.asarray(-1.0 / b)
    assert_close(nf.radial_rate(s_outer), 0.0, atol=TOL["closed_form"])
    assert jax.grad(nf.radial_rate)(s_outer) > 0
    assert nf.floquet_exponent() < 0


def test_bautin_rejects_unstable_cycle():
    with pytest.raises(ValueError):
        BautinNormalForm(b=-1.0)
    with pytest.raises(ValueError):
        HopfNormalForm(a=0.0)


# ------------------------------------------------------------- 2. integrations -----
@given(nf=normal_forms, r=radii, theta=angles)
@settings(deadline=None)
def test_integrations_agree(nf, r, theta):
    ts = jnp.linspace(0.0, 2.0, 9)
    u0 = _point(r, theta)
    ref = nf.flow(ts, u0, config=TIGHT, integration="cartesian").ys
    for name in ("polar", "r_squared"):
        ys = nf.flow(ts, u0, config=TIGHT, integration=name).ys
        assert_close(ys, ref, rtol=TOL["flow"], atol=TOL["flow"], msg=name)
        assert_close(ys[0], u0, atol=TOL["closed_form"], msg=f"{name}: y(0) != u0")


def test_cartesian_integration_is_finite_at_the_origin():
    nf = BautinNormalForm(1.0, 0.5, 2.0, 0.5)
    ts = jnp.linspace(0.0, 1.0, 5)
    ys = nf.flow(ts, jnp.zeros(2), integration="cartesian").ys
    assert jnp.all(jnp.isfinite(ys)) and jnp.all(
        ys == 0.0
    )  # the origin is a fixed point
    g = jax.jacrev(lambda u0: nf.flow(ts, u0, integration="cartesian").ys[-1])(
        jnp.zeros(2)
    )
    assert jnp.all(jnp.isfinite(g))


def test_unknown_integration_rejected():
    with pytest.raises(ValueError, match="Unknown integration"):
        BautinNormalForm().flow(
            jnp.linspace(0, 1, 3), jnp.ones(2), integration="spherical"
        )


def test_integrations_are_static_under_filter_jit():
    """An integration has no array leaves, so it is a static argument: distinct
    integrations compile distinct traces and a repeated call does not retrace."""
    nf = BautinNormalForm()
    ts = jnp.linspace(0.0, 1.0, 5)
    traces = []

    @eqx.filter_jit
    def flow(u0, method):
        traces.append(type(method).__name__)
        return nf.flow(ts, u0, integration=method).ys

    for s in INTEGRATIONS.values():
        flow(jnp.array([0.5, 0.2]), s)
        flow(jnp.array([0.7, -0.1]), s)
    assert sorted(traces) == sorted(type(s).__name__ for s in INTEGRATIONS.values())
