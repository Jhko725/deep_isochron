"""The symbolic layer (``deep_isochron.systems.symbolic``, ADR-0016): one SymPy
definition per system, the JAX ``rhs`` derived from it.

Section 1 checks the derivation: the derived ``rhs`` equals SymPy's own ``lambdify``
of the template on every system to round-off, and the normal forms' derived hooks
equal the template's rates. Section 2 checks the compilation: built once per
concrete class at class creation, as a class attribute that is neither a dataclass
field nor a pytree leaf, with abstract intermediates building nothing and a
malformed template failing at class creation. Section 3 checks symbolically that the
normal forms' hand-derived ``h(r)`` and ``Ψ(r)``, transcribed from their module
docstrings, satisfy ``h' = (ω₁ − ω)/(rρ)`` and ``rρΨ' = κΨ`` with ``κ = ρ'(1)``.
Section 4 converts ``symbolic()`` to polar coordinates: the normal forms' rates have
no ``θ`` dependence, and FitzHugh–Nagumo's field does not separate.
"""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import sympy as sp
from deep_isochron.systems import (
    AbstractEvenNormalForm,
    AbstractNormalForm,
    BautinNormalForm,
    FitzhughNagumo,
    HodgekinHuxley,
    HopfNormalForm,
    WinfreeNormalForm,
)
from deep_isochron.systems.symbolic import (
    AbstractSymbolicODE,
    SupportsSymbolic,
    SymbolicODE,
)
from hypothesis import given, settings, strategies as st

from tests.helpers import assert_close, TOL


SYSTEMS = [
    FitzhughNagumo(),
    FitzhughNagumo(a=0.5, b=0.9, c=2.0, z=-0.3),
    HodgekinHuxley(),
    HopfNormalForm(1.3, 1.5, 0.4),
    BautinNormalForm(1.0, 0.3, 1.5, 0.5),
    WinfreeNormalForm(-0.25, -0.5),
    WinfreeNormalForm(0.25, -0.5),
]


def _lambdified(ode):
    """SymPy's own evaluation of the template, as an oracle independent of sympy2jax."""
    sym = ode.symbolic()
    f = sp.lambdify(sym.states + tuple(sym.params.values()), sym.rhs, modules="numpy")
    values = [float(getattr(ode, name)) for name in sym.params]
    return lambda u: np.asarray(f(*np.asarray(u, dtype=float), *values), dtype=float)


def _points(ode, draw):
    """A Hypothesis-drawn state of `ode` in its domain: gates in ``(0, 1)`` and a
    potential away from the ``-40 mV`` singularity for Hodgkin–Huxley, a point of the
    basin otherwise."""
    if isinstance(ode, HodgekinHuxley):
        return jnp.array(
            [
                draw(st.floats(-80.0, 40.0).filter(lambda v: abs(v + 40) > 1e-3)),
                *[draw(st.floats(0.01, 0.99)) for _ in range(3)],
            ]
        )
    r_in = ode.basin_radii()[0] if isinstance(ode, AbstractNormalForm) else 0.0
    r = draw(st.floats(r_in + 0.05, 3.0))
    theta = draw(st.floats(-3.1, 3.1))
    return jnp.array([r * np.cos(theta), r * np.sin(theta)])


# ------------------------------------------------------------- 1. derivation -----
@pytest.mark.parametrize("ode", SYSTEMS, ids=repr)
@settings(max_examples=20)
@given(data=st.data())
def test_rhs_is_the_template(ode, data):
    """The derived `rhs` equals SymPy's `lambdify` of the template at a drawn state,
    to `TOL["identity"]`, on every system."""
    u = _points(ode, data.draw)
    assert isinstance(ode, SupportsSymbolic)
    assert_close(
        ode.rhs(0.0, u), _lambdified(ode)(u), rtol=TOL["identity"], atol=TOL["identity"]
    )


@pytest.mark.parametrize("ode", SYSTEMS[:2], ids=repr)
def test_rhs_jits_vmaps_and_differentiates(ode):
    """The derived `rhs` of FitzHugh–Nagumo gives the same values under `jit` and
    `vmap`, and `jacfwd` of it equals SymPy's Jacobian of the template."""
    us = jnp.array([[0.5, -0.2], [-1.0, 0.3], [0.0, 0.0]])
    batched = jax.jit(jax.vmap(ode.rhs, in_axes=(None, 0)))(0.0, us)
    for u, fu in zip(us, batched):
        assert_close(fu, ode.rhs(0.0, u), rtol=TOL["jit_eager"], atol=TOL["jit_eager"])
    jac = jax.jacfwd(ode.rhs, argnums=1)(0.0, us[0])
    sym = ode.symbolic()
    expected = sp.lambdify(
        sym.states + tuple(sym.params.values()),
        sp.Matrix(sym.rhs).jacobian(sym.states),
    )(*np.asarray(us[0]), *[float(getattr(ode, n)) for n in sym.params])
    assert_close(
        jac,
        np.asarray(expected, dtype=float),
        rtol=TOL["identity"],
        atol=TOL["identity"],
    )


@pytest.mark.parametrize(
    "nf",
    [HopfNormalForm(1.3, 1.5, 0.4), BautinNormalForm(1.0, 0.3, 1.5, 0.5)],
    ids=repr,
)
@given(s=st.floats(0.0, 9.0))
def test_even_hooks_are_the_template(nf, s):
    """The even forms' `_log_growth_rate_sq` and `_angular_rate_sq` equal the
    template's rates evaluated by SymPy at ``s = r²``."""
    _, rho, omega, params = type(nf)._symbolic_rates_sq()
    subs = {sym: float(getattr(nf, name)) for name, sym in params.items()}
    s_sym = next(iter(rho.free_symbols - set(params.values())))
    assert_close(
        nf._log_growth_rate_sq(jnp.asarray(s)),
        float(rho.subs(subs).subs(s_sym, s)),
        rtol=TOL["identity"],
        atol=TOL["identity"],
    )
    assert_close(
        nf._angular_rate_sq(jnp.asarray(s)),
        float(omega.subs(subs).subs(s_sym, s)),
        rtol=TOL["identity"],
        atol=TOL["identity"],
    )


# ----------------------------------------------------------- 2. compilation ------
@pytest.mark.parametrize(
    "cls", [type(ode) for ode in SYSTEMS], ids=lambda c: c.__name__
)
def test_compiled_form_is_a_class_attribute_not_a_field_or_leaf(cls):
    """The compiled form lives in the concrete class's ``__dict__``, is not a
    dataclass field, is not a pytree leaf, and is not rebuilt per instance."""
    name = "_compiled" if issubclass(cls, AbstractSymbolicODE) else "_compiled_rates"
    assert name in cls.__dict__, "built on the concrete class itself"
    assert name not in {f.name for f in dataclasses.fields(cls)}
    ode = SYSTEMS[[type(o) for o in SYSTEMS].index(cls)]
    for leaf in jax.tree.leaves(ode):
        assert not hasattr(leaf, "module"), "a compiled form leaked into the pytree"
    assert getattr(ode, name) is getattr(cls, name), "no per-instance rebuild"


def test_abstract_intermediates_build_nothing():
    """The abstract classes, and an abstract layer with no template of its own, have
    no compiled form."""
    assert "_compiled" not in AbstractSymbolicODE.__dict__
    assert "_compiled_rates" not in AbstractNormalForm.__dict__
    assert "_compiled_rates" not in AbstractEvenNormalForm.__dict__
    assert "_compiled_rates_sq" not in AbstractEvenNormalForm.__dict__

    class Mid(AbstractSymbolicODE):  # an abstract layer with no template of its own
        pass

    assert "_compiled" not in Mid.__dict__


def test_template_conventions_are_checked_at_class_creation():
    """A template with a parameter not named after its attribute, a free symbol, or
    too few expressions raises `ValueError` at class creation."""
    x, y, k = sp.symbols("x y k")
    with pytest.raises(ValueError, match="named after its attribute"):

        class Misnamed(AbstractSymbolicODE):
            dim: int = eqx.field(static=True, default=2, init=False)
            rate: float = 1.0

            @classmethod
            def _symbolic_template(cls):
                return SymbolicODE((x, y), {"rate": k}, (-k * y, k * x))

    with pytest.raises(ValueError, match="free symbols"):

        class Stray(AbstractSymbolicODE):
            dim: int = eqx.field(static=True, default=2, init=False)

            @classmethod
            def _symbolic_template(cls):
                return SymbolicODE((x, y), {}, (-k * y, k * x))

    with pytest.raises(ValueError, match="expressions for"):

        class Short(AbstractSymbolicODE):
            dim: int = eqx.field(static=True, default=2, init=False)

            @classmethod
            def _symbolic_template(cls):
                return SymbolicODE((x, y), {}, (-y,))


# ------------------------------------------------------------ 3. closed forms ----
def _closed_forms(cls):
    """`(r, h(r), Ψ(r), params)` transcribed from the module docstrings."""
    r = sp.Symbol("r", positive=True)
    if cls is HopfNormalForm:
        a, w, w0 = sp.symbols("a w w0", real=True)
        c = (w - w0) / a
        return r, c * sp.log(r), (r**2 - 1) / (2 * r**2), {"a": a, "w": w, "w0": w0}
    if cls is BautinNormalForm:
        a, b, w, w0 = sp.symbols("a b w w0", real=True)
        c = (w - w0) / a
        h = c * (sp.log(r) - sp.Rational(1, 2) * sp.log((1 + b * r**2) / (1 + b)))
        psi = (r**2 - 1) * (1 + b * r**2) ** b / (2 * (1 + b) ** b * r ** (2 * (1 + b)))
        return r, h, psi, {"a": a, "b": b, "w": w, "w0": w0}
    if cls is WinfreeNormalForm:
        a, w = sp.symbols("a w", real=True)
        h = (w / a) * (sp.log((r - a) / r) - sp.log(1 - a))
        psi = (r - 1) / r * ((1 - a) * r / (r - a)) ** (1 / a)
        return r, h, psi, {"a": a, "w": w}
    raise AssertionError(cls)


@pytest.mark.parametrize(
    "cls",
    [HopfNormalForm, BautinNormalForm, WinfreeNormalForm],
    ids=lambda c: c.__name__,
)
def test_closed_forms_satisfy_their_defining_odes_symbolically(cls):
    """SymPy verifies ``ω + h' r ρ = ω₁`` and ``(log Ψ)' r ρ = κ`` against the
    symbolic rates, with ``Ψ(1) = 0``, ``Ψ'(1) = 1`` and ``h(1) = 0``."""
    r_t, rho, omega, params_t = cls._symbolic_rates()
    r, h, psi, params = _closed_forms(cls)
    rho = rho.subs(r_t, r).subs({params_t[k]: params[k] for k in params})
    omega = omega.subs(r_t, r).subs({params_t[k]: params[k] for k in params})
    omega_1 = omega.subs(r, 1)
    kappa = sp.diff(rho, r).subs(r, 1)
    # Θ = θ + h(r) advances at ω₁:  ω + h' r ρ = ω₁
    assert sp.simplify(omega + sp.diff(h, r) * r * rho - omega_1) == 0
    # Ψ decays at κ:  Ψ' r ρ = κ Ψ,  Ψ(1) = 0,  Ψ'(1) = 1 — checked on the logarithmic
    # derivative (log Ψ)' r ρ = κ, whose left side is rational in r once the logs of the
    # products are expanded (`force=True`: the factors' signs do not matter formally),
    # where `simplify` succeeds; the product form with symbolic exponent b defeats it
    log_psi = sp.expand_log(sp.log(psi), force=True)
    assert sp.simplify(sp.diff(log_psi, r) * r * rho - kappa) == 0
    assert sp.simplify(psi.subs(r, 1)) == 0
    assert sp.simplify(sp.diff(psi, r).subs(r, 1)) == 1
    assert sp.simplify(h.subs(r, 1)) == 0


# ------------------------------------------------------------------ 4. charts ----
def _to_polar(sym: SymbolicODE):
    """``(r, θ, ṙ, θ̇)`` of a planar symbolic field in polar coordinates."""
    x, y = sym.states
    f, g = sym.rhs
    r, theta = sp.symbols("r theta", positive=True)
    polar = {x: r * sp.cos(theta), y: r * sp.sin(theta)}
    r_dot = ((x * f + y * g) / sp.sqrt(x**2 + y**2)).subs(polar)
    theta_dot = ((x * g - y * f) / (x**2 + y**2)).subs(polar)
    return r, theta, sp.simplify(r_dot), sp.simplify(theta_dot)


@pytest.mark.parametrize(
    "nf",
    [HopfNormalForm(1.3, 1.5, 0.4), BautinNormalForm(1.0, 0.3), WinfreeNormalForm()],
    ids=repr,
)
def test_cartesian_symbolic_field_recovers_the_rates(nf):
    """The cartesian field of a normal form in polar coordinates is ``ṙ = r ρ(r)``
    and ``θ̇ = ω(r)`` with no ``θ`` dependence."""
    r, theta, r_dot, theta_dot = _to_polar(nf.symbolic())
    r_t, rho, omega, _ = type(nf)._symbolic_rates()
    assert not r_dot.has(theta) and not theta_dot.has(theta)
    assert sp.simplify(r_dot - r * rho.subs(r_t, r)) == 0
    assert sp.simplify(theta_dot - omega.subs(r_t, r)) == 0


def test_fitzhugh_nagumo_is_not_rotationally_symmetric():
    """FitzHugh–Nagumo's ``ṙ`` and ``θ̇`` both depend on ``θ``."""
    _, theta, r_dot, theta_dot = _to_polar(FitzhughNagumo().symbolic())
    assert r_dot.has(theta) and theta_dot.has(theta)


def test_even_forms_have_polynomial_cartesian_fields():
    """Hopf's and Bautin's cartesian fields are polynomial in the states; Winfree's
    is not."""
    for nf in (HopfNormalForm(), BautinNormalForm()):
        for expr in nf.symbolic().rhs:
            assert expr.is_polynomial(*nf.symbolic().states)
    for expr in WinfreeNormalForm().symbolic().rhs:
        assert not expr.is_polynomial(*WinfreeNormalForm().symbolic().states)
