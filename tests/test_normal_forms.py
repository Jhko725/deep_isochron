"""Laws of the normal forms (``deep_isochron.systems.normal_forms``), generic over the
parameters (Hypothesis draws both forms). The tests implement §10 of
``docs/design/normal-forms.md``.

1. structure      Θ advances at rate ω₁; Ψ decays at rate κ — exactly, by autodiff of
                  the defining identities, and again along the integrated flow;
                  κ = ρ'(1) = grad(log_growth_rate)(1) (no factor 2); h(1) = Ψ(1) = 0;
                  Ψ'(1) = 1 and Ψ < 0 inside (Wilson–Moehlis); public r-views equal the
                  _sq hooks; cartesian rhs == polar rhs pushed through the chart; the
                  Jacobian at the origin is ρ(0) I + ω(0) J and the explicit Hopf
                  Jacobian of §2; isochron / limit_cycle consistency; Ψ's limits; Hopf
                  == Bautin(b=0); Bautin's unstable outer cycle for -1 < b < 0
2. charts         to_phase_amplitude ∘ from_phase_amplitude == id on the basin; the root
                  solve is differentiable in Ψ and in the parameters (implicit function
                  theorem vs finite differences); unreachable Ψ gives nan
3. integrations   cartesian / polar / r_squared / closed_form agree away from the
                  origin; Hopf closed_form equals the explicit r(t) of §6.1; cartesian
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
)
from hypothesis import given, settings, strategies as st

from tests.helpers import assert_close, SOLVERS, TOL


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
    """Exact (autodiff) form of dΘ/dt = ω₁ and dΨ/dt = κΨ along the flow:
    ω(r) + h'(r) r ρ(r) == ω₁  and  Ψ'(r) r ρ(r) == κ Ψ(r)  (design document §4, §5)."""
    _check_identities(nf, jnp.asarray(r))


@given(
    a=st.floats(0.2, 3.0),
    b=st.floats(-0.9, -0.1),
    w=st.floats(0.3, 3.0),
    frac=st.floats(0.05, 0.95),
)
def test_identities_hold_inside_the_outer_cycle(a, b, w, frac):
    """§10: the identities also hold on the basin bounded by the unstable outer cycle
    r² = -1/b when -1 < b < 0 (the normal form is a valid target there too)."""
    nf = BautinNormalForm(a, b, w, 0.5 * w)
    _check_identities(nf, frac * jnp.sqrt(-1.0 / b))


def _check_identities(nf, r):
    rdot = r * nf.log_growth_rate(r)
    lhs_phase = nf.angular_rate(r) + jax.grad(nf.phase_shift)(r) * rdot
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


@given(a=st.floats(0.2, 3.0), w=st.floats(0.3, 3.0), w0=st.floats(-3.0, 3.0))
def test_hopf_floquet_multiplier_is_the_polar_monodromy(a, w, w0):
    """§10: the Floquet multiplier equals the monodromy of the radial linearization
    integrated over one period; for Hopf ṙ = a r (1 − r²) gives δṙ = −2a δr on the
    cycle."""
    nf = HopfNormalForm(a, w, w0)
    sol = dfx.diffeqsolve(
        dfx.ODETerm(
            lambda t, dr, _: jax.grad(lambda r: r * nf.log_growth_rate(r))(1.0) * dr
        ),
        dfx.Tsit5(),
        0.0,
        nf.period(),
        None,
        jnp.asarray(1.0),
        stepsize_controller=dfx.PIDController(rtol=1e-10, atol=1e-12),
        max_steps=16384,
    )
    assert sol.ys is not None
    assert_close(sol.ys[-1], nf.floquet_multiplier(), rtol=TOL["flow"])
    assert_close(nf.floquet_exponent(), -2 * a, rtol=TOL["closed_form"])


@given(nf=normal_forms)
def test_normal_form_closed_forms_are_consistent(nf):
    one = jnp.asarray(1.0)
    assert_close(nf.phase_shift(one), 0.0, atol=TOL["identity"])
    assert_close(nf.isostable(one), 0.0, atol=TOL["identity"])
    assert_close(nf.log_growth_rate(one), 0.0, atol=TOL["identity"])
    # κ = ρ'(1) in the r-chart (design document §4.2), no factor 2
    assert_close(
        nf.floquet_exponent(),
        jax.grad(nf.log_growth_rate)(one),
        rtol=TOL["closed_form"],
        atol=TOL["closed_form"],
    )
    assert nf.floquet_exponent() < 0
    # Wilson–Moehlis normalization: ∂ᵣΨ(1) = 1
    assert_close(jax.grad(nf.isostable)(one), 1.0, rtol=TOL["closed_form"])
    # public r-views are the _sq hooks composed with r²
    for r in (0.3, 1.0, 2.2):
        r = jnp.asarray(r)
        assert_close(
            nf.log_growth_rate(r), nf._log_growth_rate_sq(r * r), rtol=TOL["identity"]
        )
        assert_close(
            nf.angular_rate(r), nf._angular_rate_sq(r * r), rtol=TOL["identity"]
        )
    assert_close(nf.period(), 2 * jnp.pi / nf.omega(), rtol=TOL["identity"])
    assert_close(
        nf.floquet_multiplier(),
        jnp.exp(nf.floquet_exponent() * nf.period()),
        rtol=TOL["identity"],
    )
    # isostable sign convention (§5.2): negative inside the cycle, positive outside
    assert nf.isostable(jnp.asarray(0.5)) < 0 < nf.isostable(jnp.asarray(1.5))
    # eigenvalues at the origin: ρ(0) ± i ω(0)
    lam = nf.eigenvalues_origin()
    zero = jnp.asarray(0.0)
    assert_close(
        lam.real, jnp.full(2, nf._log_growth_rate_sq(zero)), rtol=TOL["identity"]
    )
    assert_close(
        jnp.abs(lam.imag),
        jnp.full(2, jnp.abs(nf._angular_rate_sq(zero))),
        rtol=TOL["identity"],
    )


@given(a=st.floats(0.2, 3.0), b=st.floats(0.0, 2.0), w=st.floats(0.5, 3.0))
def test_closed_form_kappa_and_psi_limits(a, b, w):
    """Table of §3: κ = -2a(1+b), μ = exp(κT); Ψ(r→∞) = ½ (b/(1+b))^b (Hopf: ½)."""
    nf = BautinNormalForm(a, b, w)
    assert_close(nf.floquet_exponent(), -2 * a * (1 + b), rtol=TOL["closed_form"])
    assert_close(
        nf.floquet_multiplier(),
        jnp.exp(-4 * jnp.pi * a * (1 + b) / w),
        rtol=TOL["closed_form"],
    )
    far = nf.isostable(jnp.asarray(1e8))
    assert_close(far, 0.5 * (b / (1 + b)) ** b if b > 0 else 0.5, rtol=1e-6, atol=1e-7)
    assert nf.isostable(jnp.asarray(1e-3)) < -1e3


@given(nf=normal_forms)
def test_jacobian_at_the_origin(nf):
    """D rhs(0) = ρ(0) I + ω(0) J (design document §2), exact under autodiff because the
    cartesian field has no square root."""
    jac = jax.jacfwd(lambda u: nf.rhs(0.0, u))(jnp.zeros(2))
    zero = jnp.asarray(0.0)
    rho0, w0 = nf._log_growth_rate_sq(zero), nf._angular_rate_sq(zero)
    expected = jnp.array([[rho0, -w0], [w0, rho0]])
    assert_close(jac, expected, rtol=TOL["identity"], atol=TOL["identity"])
    assert jnp.all(jnp.isfinite(jax.hessian(lambda u: nf.rhs(0.0, u)[0])(jnp.zeros(2))))


@given(
    a=st.floats(0.2, 3.0),
    w=st.floats(0.5, 3.0),
    w0=st.floats(-2.0, 2.0),
    x=st.floats(-2, 2),
    y=st.floats(-2, 2),
)
def test_hopf_jacobian_explicit(a, w, w0, x, y):
    """The explicit Hopf Jacobian of design document §2 at a generic point."""
    nf = HopfNormalForm(a, w, w0)
    u = jnp.array([x, y])
    r2 = x * x + y * y
    d = w - w0
    expected = jnp.array(
        [
            [
                a * (1 - r2) - 2 * a * x * x - 2 * d * x * y,
                -2 * a * x * y - w0 - d * r2 - 2 * d * y * y,
            ],
            [
                -2 * a * x * y + w0 + d * r2 + 2 * d * x * x,
                a * (1 - r2) - 2 * a * y * y + 2 * d * x * y,
            ],
        ]
    )
    jac = jax.jacfwd(lambda v: nf.rhs(0.0, v))(u)
    assert_close(jac, expected, rtol=TOL["closed_form"], atol=TOL["closed_form"])


@given(nf=normal_forms, r=radii, theta=angles)
def test_cartesian_rhs_is_polar_rhs_in_the_chart(nf, r, theta):
    u = _point(r, theta)
    pushed = jax.jvp(
        nf.from_polar, (nf.to_polar(u),), (nf.rhs_polar(0.0, nf.to_polar(u)),)
    )[1]
    assert_close(
        nf.rhs(0.0, u), pushed, rtol=TOL["closed_form"], atol=TOL["closed_form"]
    )
    assert_close(
        nf.from_polar(nf.to_polar(u)),
        u,
        rtol=TOL["closed_form"],
        atol=TOL["closed_form"],
    )


@given(nf=normal_forms, r=radii, theta=angles)
def test_phase_and_amplitude_along_the_flow(nf, r, theta):
    """Integrated version: phase(u(t)) - phase(u0) == w t (mod 2pi) and
    amplitude(u(t)) == amplitude(u0) exp(kappa t)."""
    ts = jnp.linspace(0.0, 1.5, 7)
    ys = nf.flow(
        ts, _point(r, theta), config=SOLVERS["tight"], integration="cartesian"
    ).ys
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
    ρ' > 0 there (unstable); r = 1 stays stable; Ψ → +∞ at the outer cycle."""
    nf = BautinNormalForm(a, b, 1.0)
    r_outer = jnp.sqrt(-1.0 / b)
    assert_close(nf.log_growth_rate(r_outer), 0.0, atol=TOL["closed_form"])
    assert jax.grad(nf.log_growth_rate)(r_outer) > 0
    assert nf.floquet_exponent() < 0
    # Ψ ∝ (1 - r²/r_outer²)^b near the outer cycle, so Ψ → +∞ like ε^b (b < 0).
    near = nf.isostable(r_outer * (1 - 1e-3))
    nearer = nf.isostable(r_outer * (1 - 1e-12))
    assert 0 < near < nearer
    assert_close(jnp.log(nearer / near), -b * jnp.log(1e9), rtol=0.05)


def test_bautin_rejects_unstable_cycle():
    with pytest.raises(ValueError):
        BautinNormalForm(b=-1.0)
    with pytest.raises(ValueError):
        HopfNormalForm(a=0.0)


# ------------------------------------------------------------------- 2. charts -----
@given(nf=normal_forms, r=radii, theta=angles)
def test_phase_amplitude_chart_round_trip(nf, r, theta):
    u = _point(r, theta)
    z = nf.to_phase_amplitude(u)
    assert bool(jnp.abs(z[0]) <= jnp.pi)
    assert_close(
        nf.from_phase_amplitude(z), u, rtol=TOL["closed_form"], atol=TOL["closed_form"]
    )
    assert_close(
        nf.radius_from_isostable(nf.isostable(jnp.asarray(r))),
        r,
        rtol=TOL["closed_form"],
    )


@given(nf=normal_forms, r=radii)
def test_radius_from_isostable_is_differentiable(nf, r):
    """Implicit-function JVP: dr/dΨ = 1/Ψ'(r); and the parameter derivative matches
    finite differences (Bautin's b enters Ψ)."""
    psi = nf.isostable(jnp.asarray(r))
    assert_close(
        jax.grad(nf.radius_from_isostable)(psi)
        * jax.grad(nf.isostable)(jnp.asarray(r)),
        1.0,
        rtol=TOL["closed_form"],
    )
    if isinstance(nf, BautinNormalForm):
        raw_b = nf.raw_b

        def f(rb):
            return eqx.tree_at(lambda m: m.raw_b, nf, rb).radius_from_isostable(psi)

        eps = 1e-6
        fd = (f(raw_b + eps) - f(raw_b - eps)) / (2 * eps)
        assert_close(jax.grad(f)(raw_b), fd, rtol=1e-5, atol=1e-7)


def test_radius_from_isostable_unreachable_and_basin_edge():
    nf = BautinNormalForm(1.3, 0.5, 2.0, 0.7)
    assert jnp.isnan(nf.radius_from_isostable(jnp.asarray(0.3)))  # above Ψ(∞) ≈ 0.2887
    nf = BautinNormalForm(1.0, -0.5, 1.0)
    r = nf.radius_from_isostable(jnp.asarray(50.0))  # just inside the outer cycle
    assert 1.41 < r < jnp.sqrt(2.0)
    assert_close(nf.isostable(r), 50.0, rtol=1e-8)


# ------------------------------------------------------------- 3. integrations -----
@given(nf=normal_forms, r=radii, theta=angles)
@settings(deadline=None)
def test_integrations_agree(nf, r, theta):
    ts = jnp.linspace(0.0, 2.0, 9)
    u0 = _point(r, theta)
    ref = nf.flow(ts, u0, integration="closed_form").ys  # exact (§6)
    assert nf.default_integration == "closed_form"
    assert_close(nf.flow(ts, u0).ys, ref, rtol=TOL["identity"])  # the default
    for name in ("cartesian", "polar", "r_squared"):
        ys = nf.flow(ts, u0, config=SOLVERS["tight"], integration=name).ys
        assert_close(ys, ref, rtol=TOL["flow"], atol=TOL["flow"], msg=name)
        assert_close(ys[0], u0, atol=TOL["closed_form"], msg=f"{name}: y(0) != u0")


@given(a=st.floats(0.2, 3.0), w=st.floats(0.5, 3.0), w0=st.floats(-2.0, 2.0), r0=radii)
def test_hopf_closed_form_is_explicit(a, w, w0, r0):
    """Design document §6.1: r(t)⁻² = 1 + (r₀⁻² − 1) e^{−2at} and
    θ(t) = θ₀ + ω₁ t + c ln(r₀/r(t))."""
    nf = HopfNormalForm(a, w, w0)
    ts = jnp.array([0.0, 0.7, 1.9])
    ys = nf.flow(ts, jnp.array([r0, 0.0]), integration="closed_form").ys
    r = jnp.sqrt(jnp.sum(ys**2, axis=-1))
    assert_close(
        r ** (-2), 1 + (r0 ** (-2) - 1) * jnp.exp(-2 * a * ts), rtol=TOL["closed_form"]
    )
    theta = jnp.arctan2(ys[:, 1], ys[:, 0])
    expected = w * ts + (w - w0) / a * jnp.log(r0 / r)
    gap = jnp.arctan2(jnp.sin(theta - expected), jnp.cos(theta - expected))
    assert_close(gap, jnp.zeros_like(gap), atol=TOL["closed_form"])


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
