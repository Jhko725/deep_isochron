"""The symbolic layer (``deep_isochron.systems.symbolic``, ADR-0016): one SymPy
definition per system, with the JAX ``rhs`` and the normal forms' rate hooks derived
from it.

Section 1 checks the derivation. The derived ``rhs`` equals SymPy's own ``lambdify``
of the cartesian template on every system, the normal forms' rate hooks equal the
rates that SymPy reads off their polar templates, and ``params()`` returns the
constructor's values. Section 2 checks the compilation: it happens once per class that
has no abstract methods left, as a class attribute that is neither a dataclass field
nor a pytree leaf, and a template that breaks a convention fails at class creation.
Section 3 checks symbolically that the normal forms' hand-derived ``h(r)`` and
``Ψ(r)``, transcribed from their module docstrings, satisfy ``h' = (ω₁ − ω)/(rρ)``
and ``rρΨ' = κΨ`` with ``κ = ρ'(1)``. Section 4 converts the cartesian template back
to polar coordinates: a normal form's field is its polar template again, and
FitzHugh–Nagumo's field does not separate.
"""

import abc
import dataclasses
from collections.abc import Callable

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
from deep_isochron.systems.symbolic import AbstractSymbolicODE, SymbolicODE
from hypothesis import given, settings, strategies as st
from jaxtyping import Array, Float

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
NORMAL_FORMS = [ode for ode in SYSTEMS if isinstance(ode, AbstractNormalForm)]


def _lambdified(
    ode: AbstractSymbolicODE,
) -> Callable[[Float[Array, " dim"]], np.ndarray]:
    """SymPy's own evaluation of the cartesian template, as an oracle independent of
    sympy2jax."""
    sym = ode.symbolic()
    f = sp.lambdify(sym.states + tuple(sym.params.values()), sym.rhs, modules="numpy")
    values = [float(getattr(ode, name)) for name in sym.params]
    return lambda u: np.asarray(f(*np.asarray(u, dtype=float), *values), dtype=float)


def _points(ode: AbstractSymbolicODE, draw: Callable) -> Float[Array, " dim"]:
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


def _polar_rates(
    cls: type[AbstractNormalForm],
) -> tuple[sp.Symbol, sp.Expr, sp.Expr, dict[str, sp.Symbol]]:
    """`(r, ρ, ω, params)` read off the polar template by the test itself, as
    ``ρ = ṙ/r`` canceled and ``ω = θ̇``."""
    template = cls._symbolic_template_polar()
    r, _ = template.states
    r_dot, theta_dot = template.rhs
    return r, sp.cancel(r_dot / r), theta_dot, template.params


# ------------------------------------------------------------- 1. derivation -----
@pytest.mark.parametrize("ode", SYSTEMS, ids=repr)
@settings(max_examples=20)
@given(data=st.data())
def test_rhs_is_the_template(ode, data):
    """The derived `rhs` equals SymPy's `lambdify` of the cartesian template at a
    drawn state, to `TOL["identity"]`, on every system."""
    u = _points(ode, data.draw)
    assert isinstance(ode, AbstractSymbolicODE)
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


@pytest.mark.parametrize("nf", NORMAL_FORMS, ids=repr)
@given(fraction=st.floats(0.0, 1.0))
def test_rate_hooks_are_the_polar_template(nf, fraction):
    """`log_growth_rate` and `angular_rate` equal ``ṙ/r`` and ``θ̇`` of the polar
    template, evaluated by SymPy at a radius of the basin; the even forms'
    `_log_growth_rate_sq` and `_angular_rate_sq` equal the same rates at
    ``s = r²``."""
    r_in = nf.basin_radii()[0]
    radius = r_in + 0.05 + fraction * (3.0 - r_in)
    r, rho, omega, params = _polar_rates(type(nf))
    values = {symbol: float(getattr(nf, name)) for name, symbol in params.items()}
    expected_rho = float(rho.subs(values).subs(r, radius))
    expected_omega = float(omega.subs(values).subs(r, radius))
    tol = dict(rtol=TOL["identity"], atol=TOL["identity"])
    assert_close(nf.log_growth_rate(jnp.asarray(radius)), expected_rho, **tol)
    assert_close(nf.angular_rate(jnp.asarray(radius)), expected_omega, **tol)
    if isinstance(nf, AbstractEvenNormalForm):
        s = jnp.asarray(radius**2)
        assert_close(nf._log_growth_rate_sq(s), expected_rho, **tol)
        assert_close(nf._angular_rate_sq(s), expected_omega, **tol)


@pytest.mark.parametrize(
    ("ode", "expected"),
    [
        (FitzhughNagumo(), {"a": 0.7, "b": 0.8, "c": 3.0, "z": -0.4}),
        (HopfNormalForm(1.3, 1.5, 0.4), {"a": 1.3, "w": 1.5, "w0": 0.4}),
        (
            BautinNormalForm(1.0, 0.3, 1.5, 0.5),
            {"a": 1.0, "b": 0.3, "w": 1.5, "w0": 0.5},
        ),
        (WinfreeNormalForm(-0.25, -0.5), {"a": -0.25, "w": -0.5}),
    ],
    ids=lambda v: type(v).__name__ if not isinstance(v, dict) else "",
)
def test_params_are_the_constructor_values(ode, expected):
    """`params()` returns the values given to the constructor, constrained ones
    included, in the template's order."""
    params = ode.params()
    assert list(params) == list(expected)
    assert_close(
        np.array(list(params.values())),
        np.array(list(expected.values())),
        rtol=TOL["closed_form"],
        atol=TOL["closed_form"],
    )


# ----------------------------------------------------------- 2. compilation ------
@pytest.mark.parametrize(
    "ode", SYSTEMS, ids=lambda ode: type(ode).__name__ + "-" + repr(ode)[-12:]
)
def test_compiled_forms_are_class_attributes_not_fields_or_leaves(ode):
    """Every compiled form lives in the concrete class's ``__dict__``, is not a
    dataclass field, is not a pytree leaf, and is not rebuilt per instance."""
    cls = type(ode)
    names = ["_compiled"]
    if isinstance(ode, AbstractNormalForm):
        names.append("_compiled_rates")
    if isinstance(ode, AbstractEvenNormalForm):
        names.append("_compiled_rates_sq")
    fields = {f.name for f in dataclasses.fields(cls)}
    for name in names:
        assert name in cls.__dict__, f"{name} is built on the concrete class itself"
        assert name not in fields
        assert getattr(ode, name) is getattr(cls, name), "no per-instance rebuild"
    for leaf in jax.tree.leaves(ode):
        assert not hasattr(leaf, "module"), "a compiled form leaked into the pytree"


def test_classes_with_abstract_methods_build_nothing():
    """A class compiles only when no abstract method is left. The abstract bases
    build nothing; a layer that defines the template and adds an abstract method
    builds nothing; its concrete subclass, which inherits the template, builds."""
    for cls in (AbstractSymbolicODE, AbstractNormalForm, AbstractEvenNormalForm):
        for name in ("_compiled", "_compiled_rates", "_compiled_rates_sq"):
            assert name not in cls.__dict__, f"{cls.__name__} built {name}"
    x, y = sp.symbols("x y")

    class TemplateButAbstract(AbstractSymbolicODE):
        dim: int = eqx.field(static=True, default=2, init=False)

        @classmethod
        def _symbolic_template(cls):
            return SymbolicODE((x, y), {}, (-y, x))

        @abc.abstractmethod
        def extra(self) -> None: ...

    assert "_compiled" not in TemplateButAbstract.__dict__

    class Concrete(TemplateButAbstract):
        def extra(self) -> None:
            return None

    assert "_compiled" in Concrete.__dict__


def test_template_conventions_are_checked_at_class_creation():
    """A cartesian template with a parameter not named after its attribute, a free
    symbol, or too few expressions raises `ValueError` at class creation."""
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


def _normal_form_class(
    base: type[AbstractNormalForm], r_dot: Callable, theta_dot: Callable
) -> type[AbstractNormalForm]:
    """A concrete normal form whose polar template is ``(r_dot(r, θ), theta_dot(r,
    θ))``; its closed forms are placeholders, since only class creation is tested."""
    r = sp.Symbol("r", positive=True)
    theta = sp.Symbol("theta", real=True)

    class Planted(base):
        @classmethod
        def _symbolic_template_polar(cls):
            rhs = (sp.sympify(r_dot(r, theta)), sp.sympify(theta_dot(r, theta)))
            return SymbolicODE((r, theta), {}, rhs)

        def phase_shift(self, r):
            return 0.0 * r

        def isostable(self, r):
            return 0.0 * r

    return Planted


def test_polar_template_conventions_are_checked_at_class_creation():
    """A normal form whose polar field depends on ``θ``, whose ``ṙ`` lacks the factor
    ``r``, or whose template has three states raises `ValueError` at class creation;
    the well-formed control compiles."""
    good = _normal_form_class(
        AbstractNormalForm, lambda r, t: r * (1 - r), lambda r, t: 1
    )
    assert "_compiled_rates" in good.__dict__
    with pytest.raises(ValueError, match="rotationally symmetric"):
        _normal_form_class(
            AbstractNormalForm, lambda r, t: r * (1 - r) * sp.cos(t), lambda r, t: 1
        )
    with pytest.raises(ValueError, match="rotationally symmetric"):
        _normal_form_class(AbstractNormalForm, lambda r, t: r * (1 - r), lambda r, t: t)
    with pytest.raises(ValueError, match="lacks the factor r"):
        _normal_form_class(AbstractNormalForm, lambda r, t: 1 - r, lambda r, t: 1)
    with pytest.raises(ValueError, match="has two states"):
        r, theta, z = sp.symbols("r theta z")

        class ThreeStates(AbstractNormalForm):
            @classmethod
            def _symbolic_template_polar(cls):
                return SymbolicODE(
                    (r, theta, z), {}, (r * (1 - r), sp.Integer(1), sp.Integer(0))
                )

            def phase_shift(self, r):
                return 0.0 * r

            def isostable(self, r):
                return 0.0 * r


def test_even_family_membership_is_checked_at_class_creation():
    """A declared `AbstractEvenNormalForm` whose rates are not functions of ``r²``
    raises `ValueError` at class creation, for an odd power and for a function of
    ``r`` such as ``cos r``; the even control compiles its ``s``-form."""
    good = _normal_form_class(
        AbstractEvenNormalForm, lambda r, t: r * (1 - r**2), lambda r, t: 1 + r**2
    )
    assert "_compiled_rates_sq" in good.__dict__
    with pytest.raises(ValueError, match="is declared even"):
        _normal_form_class(
            AbstractEvenNormalForm, lambda r, t: r * (1 - r**3), lambda r, t: 1
        )
    with pytest.raises(ValueError, match="is declared even"):
        _normal_form_class(
            AbstractEvenNormalForm, lambda r, t: r * (1 - r), lambda r, t: 1
        )
    with pytest.raises(ValueError, match="is declared even"):
        _normal_form_class(
            AbstractEvenNormalForm, lambda r, t: r * (1 - r**2), lambda r, t: sp.cos(r)
        )


# ------------------------------------------------------------ 3. closed forms ----
def _closed_forms(
    cls: type[AbstractNormalForm],
) -> tuple[sp.Symbol, sp.Expr, sp.Expr, dict[str, sp.Symbol]]:
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
    rates of the polar template, with ``Ψ(1) = 0``, ``Ψ'(1) = 1`` and ``h(1) = 0``."""
    r_t, rho, omega, params_t = _polar_rates(cls)
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
def _to_polar(sym: SymbolicODE) -> tuple[sp.Symbol, sp.Symbol, sp.Expr, sp.Expr]:
    """``(r, θ, ṙ, θ̇)`` of a planar cartesian symbolic field in polar coordinates.

    `trigsimp` runs before `simplify`, because `simplify` alone leaves powers of
    ``sin θ`` and ``cos θ`` that sum to a constant in Bautin's expanded field."""
    x, y = sym.states
    f, g = sym.rhs
    r, theta = sp.symbols("r theta", positive=True)
    polar = {x: r * sp.cos(theta), y: r * sp.sin(theta)}
    r_dot = ((x * f + y * g) / sp.sqrt(x**2 + y**2)).subs(polar)
    theta_dot = ((x * g - y * f) / (x**2 + y**2)).subs(polar)
    return (
        r,
        theta,
        sp.simplify(sp.trigsimp(r_dot)),
        sp.simplify(sp.trigsimp(theta_dot)),
    )


@pytest.mark.parametrize("nf", NORMAL_FORMS[:3], ids=repr)
def test_cartesian_template_is_the_polar_template(nf):
    """The derived cartesian template of a normal form, converted back to polar
    coordinates, is its polar template ``(ṙ, θ̇)``, with no ``θ`` dependence."""
    r, theta, r_dot, theta_dot = _to_polar(nf.symbolic())
    template = type(nf)._symbolic_template_polar()
    r_t, _ = template.states
    expected_r_dot, expected_theta_dot = (e.subs(r_t, r) for e in template.rhs)
    assert not r_dot.has(theta) and not theta_dot.has(theta)
    assert sp.simplify(r_dot - expected_r_dot) == 0
    assert sp.simplify(theta_dot - expected_theta_dot) == 0


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
