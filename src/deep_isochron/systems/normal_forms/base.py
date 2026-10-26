r"""Planar oscillators with closed-form phase–amplitude structure.

A normal form is a planar ODE that separates in polar coordinates $(r, \theta)$ as
$\dot r = r\thinspace\rho(r)$, $\dot\theta = \omega(r)$, with the stable limit cycle
at $r = 1$. A concrete class declares the log growth rate $\rho$ and the angular rate
$\omega$ as SymPy expressions (`_symbolic_rates`) and supplies two closed forms in
$r$: the phase shift $h(r)$, with the asymptotic phase $\Theta = \theta + h(r)$, and the
isostable coordinate $\Psi(r)$, with $\dot\Psi = \kappa\Psi$. The base class derives
everything else: the JAX rates, the Floquet exponent $\kappa = \rho'(1)$ by autodiff,
the vector fields in both charts, ``phase``, ``amplitude``, ``limit_cycle``,
``isochron``, the three charts (polar, cartesian, phase–amplitude), and ``flow`` with a
choice of integration.

Two invariants span the class. Everything public takes and returns cartesian
coordinates, with closed forms in $r$. The chart exists on the cycle's basin only, the
annulus ``basin_radii()``; outside it, ``phase``, ``amplitude`` and the closed-form flow
return ``nan`` (ADR-0013).

``AbstractEvenNormalForm`` is the subfamily whose rates are even in $r$ and are
declared in $s = r^2$, which makes the cartesian field smooth at the origin; Hopf and
Bautin belong to it, Winfree's model does not.

The mathematics, with the symbols and derivations, is ``docs/design/normal-forms.md``;
this module implements its §9. ADR-0007, ADR-0013 and ADR-0016 record the decisions.
"""

import abc
from typing import Any, ClassVar

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import sympy as sp
from jaxtyping import Array, ArrayLike, Bool, Complex, Float

from ...utils.numerics import cartesian_to_polar, polar_to_cartesian
from ..base import AbstractODE, DEFAULT_SOLVER_CONFIG, SolverConfig
from ..symbolic import (
    compile_expressions,
    CompiledExpressions,
    defines_concretely,
    is_abstract,
    SymbolicODE,
)


ROOT_EXPAND_ITERS = 64
"""Bracket-expansion steps for ``radius_from_isostable`` (in ``ln r``, doubling)."""
ROOT_NEWTON_ITERS = 48
"""Bisection-safeguarded Newton iterations for ``radius_from_isostable``."""


class AbstractNormalForm(AbstractODE):
    r"""A planar oscillator $\dot r = r\rho(r)$, $\dot\theta = \omega(r)$ with its
    cycle at $r = 1$ and closed-form phase and isostable coordinates.

    A subclass supplies `_symbolic_rates`, `phase_shift` and `isostable`, overrides
    `basin_radii` when the basin is not the punctured plane, and takes ``integration``
    in its constructor, storing it through `_set_integration`.

    **Attributes:**

    - `dim`: 2.
    - `default_integration`: The integration that `flow` uses when none is given.
        ``"closed_form"`` is exact and cheaper than any ODE solve, so it is the default
        for every normal form; the numerical integrations in
        ``integration.INTEGRATIONS`` serve the comparison experiments and the analysis
        of a normal form as a black-box field, since ``analysis`` works through `flow`
        and the closed forms then stay the oracle.
    - `_compiled_rates`: $(\rho(r), \omega(r))$ compiled to JAX, set per concrete
        class by `__init_subclass__` from `_symbolic_rates` (ADR-0016).
    """

    dim: int = eqx.field(static=True, default=2, init=False)
    default_integration: str = eqx.field(static=True, default="closed_form", init=False)

    def _set_integration(self, integration: str) -> None:
        """Validate `integration` and store it as `default_integration`; called from
        every concrete constructor.

        **Raises:** `ValueError` for a name not in ``integration.INTEGRATIONS``.
        """
        from .integration import resolve_integration

        resolve_integration(integration)  # raises on an unknown name
        self.default_integration = integration

    # ------------------------------------------------ defining data (abstract) ----
    _compiled_rates: ClassVar[CompiledExpressions]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if defines_concretely(cls, "_symbolic_rates_sq"):
            # AbstractEvenNormalForm's hook; derive both the s- and the r-compiled forms
            # (the hook lives on the even subclass; reach it by name for the checker)
            rates_sq = getattr(cls, "_symbolic_rates_sq")  # noqa: B009
            s, rho_s, omega_s, params = rates_sq()
            _validate_rates(s, rho_s, omega_s, params)
            cls._compiled_rates_sq = compile_expressions(
                (rho_s, omega_s), (s,), tuple(params)
            )
            r = sp.Symbol("r", positive=True)
            cls._compiled_rates = compile_expressions(
                (rho_s.subs(s, r**2), omega_s.subs(s, r**2)), (r,), tuple(params)
            )
        elif defines_concretely(cls, "_symbolic_rates") and not is_abstract(
            getattr(cls, "_symbolic_rates_sq", None)
        ):
            # a non-even form's own hook (AbstractEvenNormalForm also defines a
            # derived `_symbolic_rates`, but its `_symbolic_rates_sq` is still abstract)
            r, rho, omega, params = cls._symbolic_rates()
            _validate_rates(r, rho, omega, params)
            cls._compiled_rates = compile_expressions((rho, omega), (r,), tuple(params))

    @classmethod
    @abc.abstractmethod
    def _symbolic_rates(
        cls,
    ) -> tuple[sp.Symbol, sp.Expr, sp.Expr, dict[str, sp.Symbol]]:
        r"""The defining data as SymPy: `(r, ρ(r), ω(r), params)`.

        `params` maps the name of the attribute (field or property) that holds each
        parameter's value to its symbol, which is named after it.
        `AbstractEvenNormalForm` derives this method from `_symbolic_rates_sq`.
        """

    def log_growth_rate(self, r: Float[Array, ""]) -> Float[Array, ""]:
        r"""$\rho(r) = \dot r / r$ at radius `r` in the basin; $\rho(1) = 0$ and
        $\rho'(1) < 0$. Derived from `_symbolic_rates`."""
        return type(self)._compiled_rates(self, r)[0]

    def angular_rate(self, r: Float[Array, ""]) -> Float[Array, ""]:
        r"""$\omega(r) = \dot\theta$ at radius `r` in the basin; $\omega(1) \ne 0$.
        Derived from `_symbolic_rates`."""
        return type(self)._compiled_rates(self, r)[1]

    def symbolic(self) -> SymbolicODE:
        r"""The cartesian field as SymPy, for the AUTO code generator
        (`SupportsSymbolic`): $\dot x = x\rho - y\omega$, $\dot y = y\rho + x\omega$
        with $r = \sqrt{x^2 + y^2}$ substituted."""
        r, rho, omega, params = self._symbolic_rates()
        x, y = sp.symbols("x y", real=True)
        return _cartesian(x, y, rho, omega, r, sp.sqrt(x**2 + y**2), params)

    @abc.abstractmethod
    def phase_shift(self, r: Float[Array, ""]) -> Float[Array, ""]:
        r"""The phase shift $h(r)$ at radius `r`, with the asymptotic phase
        $\Theta = \theta + h(r)$ and $h(1) = 0$ (design document §5.1). ``nan`` or
        diverging outside the basin."""

    @abc.abstractmethod
    def isostable(self, r: Float[Array, ""]) -> Float[Array, ""]:
        r"""The isostable coordinate $\Psi(r)$ at radius `r`: $\dot\Psi = \kappa\Psi$,
        $\Psi(1) = 0$, $\partial_r\Psi(1) = 1$, strictly increasing on the basin and
        $-\infty$ at its inner edge (design document §5.2); ``nan`` outside the
        basin."""

    # ---------------------------------------------------------------- basin -------
    def basin_radii(self) -> tuple[float, float]:
        r"""The radii $(r_{\rm in}, r_{\rm out})$ bounding the basin of the cycle, the
        open annulus between them.

        $r_{\rm in}$ is $0$ when the enclosed fixed point is the only phaseless point
        and the radius of the inner repelling cycle otherwise; $r_{\rm out}$ is
        $\infty$ or the radius of an outer repelling cycle. Both are Python floats, so
        they can drive a ``jnp.where`` or a sampler.
        """
        return (0.0, float("inf"))

    def in_basin(self, u: Float[Array, " 2"]) -> Bool[Array, ""]:
        """Whether ``r_in < |u| < r_out`` for the cartesian point `u`."""
        r_in, r_out = self.basin_radii()
        r = jnp.sqrt(jnp.sum(u * u))
        return (r > r_in) & (r < r_out)

    # ------------------------------------------------------- linear invariants ----
    def omega(self) -> Float[Array, ""]:
        r"""$\omega_1 = \omega(1)$, the angular frequency on the limit cycle."""
        return self.angular_rate(jnp.asarray(1.0))

    def period(self) -> Float[Array, ""]:
        r"""$T = 2\pi/\omega_1$, signed: negative for a clockwise cycle."""
        return 2 * jnp.pi / self.omega()

    def floquet_exponent(self) -> Float[Array, ""]:
        r"""$\kappa = \rho'(1) < 0$, by autodiff of `log_growth_rate`: the derivative
        in the $r$ chart, with no factor 2 (design document §4.2)."""
        return jax.grad(self.log_growth_rate)(jnp.asarray(1.0))

    def floquet_multiplier(self) -> Float[Array, ""]:
        r"""$\mu = e^{\kappa T}$."""
        return jnp.exp(self.floquet_exponent() * self.period())

    # ------------------------------------------------------------------ charts -----
    @staticmethod
    def to_polar(u: Float[Array, " 2"]) -> Float[Array, " 2"]:
        r"""The polar coordinates $(r, \theta)$ of cartesian `u`, with
        $\theta \in (-\pi, \pi]$."""
        return cartesian_to_polar(u)

    @staticmethod
    def from_polar(z: Float[Array, " 2"]) -> Float[Array, " 2"]:
        r"""The cartesian point of polar `z` $= (r, \theta)$."""
        return polar_to_cartesian(z)

    def to_phase_amplitude(self, u: Float[Array, " 2"]) -> Float[Array, " 2"]:
        r"""The phase–amplitude coordinates $(\Theta, \Psi)$ of cartesian `u`: the
        chart in which the flow is linear, $\dot\Theta = \omega_1$ and
        $\dot\Psi = \kappa\Psi$ (design document §7).

        **Returns:** ``(Θ, Ψ)`` with $\Theta \in (-\pi, \pi]$; both ``nan`` outside
        the basin.
        """
        r, theta = self.to_polar(u)
        r_in, r_out = self.basin_radii()
        inside = (r > r_in) & (r < r_out)
        r_safe = jnp.where(inside, r, 1.0)  # keep the closed forms off their poles
        Theta = _wrap(theta + self.phase_shift(r_safe))
        Psi = self.isostable(r_safe)
        nan = jnp.asarray(jnp.nan, dtype=r.dtype)
        return jnp.stack((jnp.where(inside, Theta, nan), jnp.where(inside, Psi, nan)))

    def from_phase_amplitude(self, z: Float[Array, " 2"]) -> Float[Array, " 2"]:
        r"""The cartesian point of `z` $= (\Theta, \Psi)$: $r$ by
        `radius_from_isostable` and $\theta = \Theta - h(r)$."""
        Theta, Psi = z
        r = self.radius_from_isostable(Psi)
        theta = Theta - self.phase_shift(r)
        return jnp.stack((r * jnp.cos(theta), r * jnp.sin(theta)))

    def radius_from_isostable(self, psi: Float[Array, ""]) -> Float[Array, ""]:
        r"""The radius $r$ with $\Psi(r) = $ `psi`.

        The solve is a bracketed, bisection-safeguarded Newton iteration in
        $\ell = \ln r$ (design document §6.1 and §7), differentiable in `psi` and in
        the parameters through the implicit function theorem. A subclass with an
        explicit inverse, such as Hopf, overrides it.

        **Returns:** The radius, or ``nan`` for a `psi` above the range of $\Psi$.
        """
        return _radius_from_isostable(self, psi)

    # ----------------------------------------------------------- vector fields -----
    def rhs(
        self, t: Float[ArrayLike, ""], u: Float[Array, " 2"], args: Any = None
    ) -> Float[Array, " 2"]:
        r"""The cartesian field
        $\rho(|u|)\thinspace u + \omega(|u|)\thinspace\mathsf J u$ at `u`, with $|u|$ by
        a safe square root whose derivative at exactly $u = 0$ is taken as $0$. Smooth
        wherever $\rho$ and $\omega$ are smooth in $r > 0$; `AbstractEvenNormalForm`
        overrides it with the square-root-free form."""
        del t, args
        s = jnp.sum(u * u)
        r = jnp.sqrt(jnp.where(s > 0.0, s, 1.0))
        r = jnp.where(s > 0.0, r, 0.0)
        rho, w = self.log_growth_rate(r), self.angular_rate(r)
        x, y = u
        return jnp.stack((rho * x - w * y, rho * y + w * x))

    def rhs_polar(
        self, t: Float[ArrayLike, ""], z: Float[Array, " 2"], args: Any = None
    ) -> Float[Array, " 2"]:
        r"""The field in the polar chart at `z` $= (r, \theta)$:
        $(r\rho(r), \omega(r))$."""
        del t, args
        r, _ = z
        return jnp.stack((r * self.log_growth_rate(r), self.angular_rate(r)))

    # ---------------------------------------------------------- phase/amplitude ----
    def phase(self, u: Float[Array, " 2"]) -> Float[Array, ""]:
        r"""The asymptotic phase $\Theta(u) \in (-\pi, \pi]$ of cartesian `u`: the
        isochron through `u`, labeled by the polar angle of the point on the cycle
        that it converges to. ``nan`` outside the basin."""
        return self.to_phase_amplitude(u)[0]

    def amplitude(self, u: Float[Array, " 2"]) -> Float[Array, ""]:
        r"""The isostable coordinate $\Psi(u)$ of cartesian `u`: zero on the cycle,
        negative inside it, and $\Psi(t) = \Psi_0 e^{\kappa t}$ along every
        trajectory. ``nan`` outside the basin."""
        return self.to_phase_amplitude(u)[1]

    def limit_cycle(self, Theta: Float[Array, " *n"]) -> Float[Array, " *n 2"]:
        r"""The points of the limit cycle at phases `Theta`, where $r = 1$ and
        $h = 0$; the output has one trailing axis of length 2."""
        return jnp.stack((jnp.cos(Theta), jnp.sin(Theta)), axis=-1)

    def isochron(
        self, Theta: Float[ArrayLike, ""], r: Float[Array, " n"]
    ) -> Float[Array, "n 2"]:
        r"""The isochron of phase `Theta` sampled at radii `r`.

        **Arguments:**

        - `Theta`: The phase, a scalar.
        - `r`: The radii, shape ``(n,)``.

        **Returns:** The cartesian points on $\theta = \Theta - h(r)$ (design
        document §5.1), shape ``(n, 2)``; ``nan`` at radii outside the basin.
        """
        r_in, r_out = self.basin_radii()
        inside = (r > r_in) & (r < r_out)
        theta = Theta - self.phase_shift(jnp.where(inside, r, 1.0))
        pts = jnp.stack((r * jnp.cos(theta), r * jnp.sin(theta)), axis=-1)
        return jnp.where(inside[..., None], pts, jnp.nan)

    # ------------------------------------------------------------------- flow ------
    def flow(
        self,
        ts: Float[Array, " time"],
        u0: Float[Array, " 2"],
        args: Any = None,
        *,
        config: SolverConfig = DEFAULT_SOLVER_CONFIG,
        integration=None,
    ) -> dfx.Solution:
        """The diffrax ``Solution`` through cartesian ``u0``, with ``.ys`` the
        **cartesian** trajectory whichever ``integration`` produced it. ``integration``
        is an ``AbstractFlowIntegration``, one of the names in
        ``integration.INTEGRATIONS`` (``"cartesian"``, ``"polar"``, ``"r_squared"``,
        ``"closed_form"``), or ``None`` for ``self.default_integration``; it is static,
        so each value traces separately under ``jit``/``vmap``. ``config`` is ignored by
        ``"closed_form"``."""
        from .integration import resolve_integration

        method = resolve_integration(integration or self.default_integration)
        return method(self, ts, u0, args, config)


class AbstractEvenNormalForm(AbstractNormalForm):
    r"""Normal forms whose $\rho$ and $\omega$ are even in $r$, declared as smooth
    functions of $s = r^2$ through `_symbolic_rates_sq`.

    Evenness makes the cartesian field
    $\tilde\rho(|u|^2)\thinspace u + \tilde\omega(|u|^2)\thinspace\mathsf J u$ smooth at
    the origin with no square root (a polynomial for Hopf and Bautin), so the origin
    is the enclosed fixed point of the basin, `basin_radii` starts at $0$, and the
    eigenvalues at the origin, $\rho(0) \pm i\omega(0)$, are smooth-conjugacy
    invariants (design document §1 and §9, Option B). The ``r_squared`` integration
    needs this family.

    **Attributes:**

    - `_compiled_rates_sq`: $(\tilde\rho(s), \tilde\omega(s))$ compiled to JAX, set
        per concrete class by `AbstractNormalForm.__init_subclass__`.
    """

    _compiled_rates_sq: ClassVar[CompiledExpressions]

    @classmethod
    @abc.abstractmethod
    def _symbolic_rates_sq(
        cls,
    ) -> tuple[sp.Symbol, sp.Expr, sp.Expr, dict[str, sp.Symbol]]:
        r"""The defining data as SymPy in $s = r^2$: `(s, ρ̃(s), ω̃(s), params)`, as
        `_symbolic_rates` but smooth in $s$."""

    @classmethod
    def _symbolic_rates(cls):
        r"""Derived from `_symbolic_rates_sq`: $\rho(r) = \tilde\rho(r^2)$ and
        $\omega(r) = \tilde\omega(r^2)$."""
        s, rho_s, omega_s, params = cls._symbolic_rates_sq()
        r = sp.Symbol("r", positive=True)
        return r, rho_s.subs(s, r**2), omega_s.subs(s, r**2), params

    def _log_growth_rate_sq(self, s: Float[Array, ""]) -> Float[Array, ""]:
        r"""$\tilde\rho(s) = \rho(\sqrt s)$ at `s` $= r^2$; $\tilde\rho(1) = 0$ and
        $\tilde\rho'(1) < 0$. Derived from `_symbolic_rates_sq`."""
        return type(self)._compiled_rates_sq(self, s)[0]

    def _angular_rate_sq(self, s: Float[Array, ""]) -> Float[Array, ""]:
        r"""$\tilde\omega(s) = \omega(\sqrt s)$ at `s` $= r^2$. Derived from
        `_symbolic_rates_sq`."""
        return type(self)._compiled_rates_sq(self, s)[1]

    def symbolic(self) -> SymbolicODE:
        r"""The cartesian field as SymPy with $s = x^2 + y^2$ substituted, so with no
        square root: polynomial for Hopf and Bautin."""
        s, rho_s, omega_s, params = self._symbolic_rates_sq()
        x, y = sp.symbols("x y", real=True)
        return _cartesian(x, y, rho_s, omega_s, s, x**2 + y**2, params)

    # the r-chart views are derived (final); autodiff through r * r gives ρ'(1) = 2ρ̃'(1)
    def log_growth_rate(self, r: Float[Array, ""]) -> Float[Array, ""]:
        r"""$\rho(r) = \tilde\rho(r^2)$ at radius `r`."""
        return self._log_growth_rate_sq(r * r)

    def angular_rate(self, r: Float[Array, ""]) -> Float[Array, ""]:
        r"""$\omega(r) = \tilde\omega(r^2)$ at radius `r`."""
        return self._angular_rate_sq(r * r)

    def eigenvalues_origin(self) -> Complex[Array, " 2"]:
        r"""The eigenvalues $\rho(0) \pm i\thinspace\omega(0)$ of the fixed point at
        the origin, a smooth-conjugacy invariant (design document §4.1)."""
        zero = jnp.asarray(0.0)
        re, im = self._log_growth_rate_sq(zero), self._angular_rate_sq(zero)
        return jax.lax.complex(re * jnp.ones(2), im * jnp.array([1.0, -1.0]))

    def rhs(
        self, t: Float[ArrayLike, ""], u: Float[Array, " 2"], args: Any = None
    ) -> Float[Array, " 2"]:
        r"""The cartesian field
        $\tilde\rho(|u|^2)\thinspace u + \tilde\omega(|u|^2)\thinspace\mathsf J u$ at
        `u`: smooth at the origin, with no square root (design document §2 and §9)."""
        del t, args
        s = jnp.sum(u * u)
        rho, w = self._log_growth_rate_sq(s), self._angular_rate_sq(s)
        x, y = u
        return jnp.stack((rho * x - w * y, rho * y + w * x))


def _wrap(angle):
    r"""`angle` wrapped to $(-\pi, \pi]$."""
    return jnp.arctan2(jnp.sin(angle), jnp.cos(angle))


def _validate_rates(var, rho, omega, params):
    """Raise `ValueError` unless the parameter symbols are named after their keys and
    `rho`, `omega` have no free symbol beyond `var` and the parameters."""
    for name, symbol in params.items():
        if symbol.name != name:
            raise ValueError(
                f"parameter symbol {symbol.name!r} must be named after its attribute "
                f"{name!r}."
            )
    allowed = {var} | set(params.values())
    for expr in (rho, omega):
        stray = expr.free_symbols - allowed
        if stray:
            raise ValueError(f"free symbols {stray} are neither {var} nor params.")


def _cartesian(x, y, rho, omega, var, radial, params) -> SymbolicODE:
    r"""The cartesian field from the polar rates, by the chain rule
    $\dot x = \dot r\cos\theta - r\dot\theta\sin\theta$ with $r\cos\theta = x$: that
    is $\dot x = x\rho - y\omega$ and $\dot y = y\rho + x\omega$, after `radial`
    (either $r$ or $s$ in terms of $x$ and $y$) is substituted for `var`."""
    rho_xy, omega_xy = (e.subs(var, radial) for e in (rho, omega))
    return SymbolicODE(
        states=(x, y),
        params=dict(params),
        rhs=(x * rho_xy - y * omega_xy, y * rho_xy + x * omega_xy),
    ).validate()


# ------------------------------------------------------------ the root solve ------
@eqx.filter_custom_jvp
def _radius_from_isostable(nf: AbstractNormalForm, psi):
    """The root of ``Ψ(e^ℓ) = psi`` in ``ℓ = ln r``, unique because ``Ψ`` is strictly
    increasing on the basin.

    One end of the bracket is ``ℓ = 0``, where ``Ψ(1) = 0``. The other is found by
    doubling a step away from it until ``Ψ`` passes `psi`; inside the cycle this
    always happens, since ``Ψ → -∞`` at the basin's inner edge, while outside ``Ψ``
    may saturate below `psi`, and the result is then ``nan``. A non-finite ``Ψ`` marks
    a probe beyond the basin's edge, and the step is halved. The root is then polished
    by bisection-safeguarded Newton with a fixed iteration count, which reverse mode
    can differentiate through, as in ``CubicBSpline`` (ADR-0006). The custom JVP below
    makes the result differentiable in `psi` and in the normal form's parameters
    through the implicit function theorem.
    """
    f = lambda ell: nf.isostable(jnp.exp(ell))  # noqa: E731
    inside = psi < 0.0
    step0 = jnp.where(inside, -1.0, 1.0)

    def expand(_, state):
        ell, step, found = state
        val = f(ell + step)
        finite = jnp.isfinite(val)
        # moving away from ℓ = 0: inside, go down until Ψ(e^ℓ) <= psi; outside, up
        # until Ψ(e^ℓ) >= psi. A non-finite Ψ means the step overshot a basin edge:
        # halve it and retry rather than moving.
        passed = finite & jnp.where(inside, val <= psi, val >= psi)
        stop = found | passed
        move = finite & ~stop
        new_ell = jnp.where(move, ell + step, ell)
        new_step = jnp.where(stop, step, jnp.where(finite, 2 * step, 0.5 * step))
        return new_ell, new_step, stop

    ell_in, step, found = jax.lax.fori_loop(
        0, ROOT_EXPAND_ITERS, expand, (jnp.asarray(0.0), step0, jnp.asarray(False))
    )
    far = ell_in + step  # Ψ(e^far) has passed psi when `found`
    lo, hi = jnp.minimum(far, 0.0), jnp.maximum(far, 0.0)

    def newton(_, state):
        lo, hi, ell = state
        res = f(ell) - psi
        lo = jnp.where(res < 0, ell, lo)
        hi = jnp.where(res < 0, hi, ell)
        ell_n = ell - res / jax.grad(f)(ell)
        ok = (ell_n >= lo) & (ell_n <= hi) & jnp.isfinite(ell_n)
        return lo, hi, jnp.where(ok, ell_n, 0.5 * (lo + hi))

    _, _, ell = jax.lax.fori_loop(
        0, ROOT_NEWTON_ITERS, newton, (lo, hi, 0.5 * (lo + hi))
    )
    return jnp.where(found | (psi == 0.0), jnp.exp(ell), jnp.nan)


@_radius_from_isostable.def_jvp
def _radius_from_isostable_jvp(primals, tangents):
    """The implicit function theorem on ``Ψ(r; nf) = psi``:
    ``dr = (dpsi - ∂Ψ/∂nf · dnf) / Ψ'(r)``."""
    nf, psi = primals
    nf_dot, psi_dot = tangents
    r = _radius_from_isostable(nf, psi)
    dpsi_dr = jax.grad(nf.isostable)(r)
    psi_dot = jnp.zeros_like(r) if psi_dot is None else psi_dot
    nf_arrays, nf_static = eqx.partition(nf, eqx.is_array)
    # tangents of the parameters, with `None` (not differentiated) read as zero
    dot_arrays = jax.tree.map(
        lambda primal, tangent: jnp.zeros_like(primal) if tangent is None else tangent,
        nf_arrays,
        eqx.filter(nf_dot, eqx.is_array) if nf_dot is not None else nf_arrays,
        is_leaf=lambda x: x is None,
    )
    if nf_dot is None:
        dot_arrays = jax.tree.map(jnp.zeros_like, nf_arrays)
    _, dpsi_dnf = jax.jvp(
        lambda a: eqx.combine(a, nf_static).isostable(r), (nf_arrays,), (dot_arrays,)
    )
    return r, (psi_dot - dpsi_dnf) / dpsi_dr
