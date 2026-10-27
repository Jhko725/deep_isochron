r"""Planar oscillators with closed-form phase–amplitude structure.

A normal form is a planar ODE that separates in polar coordinates $(r, \theta)$ as
$\dot r = r\thinspace\rho(r)$, $\dot\theta = \omega(r)$, with the stable limit cycle
at $r = 1$. A concrete class writes its field in polar coordinates as a SymPy
template (`_symbolic_template_polar`), from which the log growth rate
$\rho = \dot r / r$ and the angular rate $\omega = \dot\theta$ are read. It also
supplies two closed forms in $r$: the phase shift $h(r)$, with the asymptotic phase
$\Theta = \theta + h(r)$, and the isostable coordinate $\Psi(r)$, with
$\dot\Psi = \kappa\Psi$. The base class derives everything else: the cartesian
template and ``rhs``, the JAX rates, the Floquet exponent $\kappa = \rho'(1)$ by
autodiff, the vector fields in both charts, ``phase``, ``amplitude``,
``limit_cycle``, ``isochron``, the three charts (polar, cartesian, phase–amplitude),
and ``flow`` with a choice of integration.

Two invariants span the class. Everything public takes and returns cartesian
coordinates, with closed forms in $r$. The chart exists on the cycle's basin only, the
annulus ``basin_radii()``; outside it, ``phase``, ``amplitude`` and the closed-form flow
return ``nan`` (ADR-0013).

``AbstractEvenNormalForm`` is the subfamily whose rates are functions of $s = r^2$,
which makes the cartesian field smooth at the origin. A form joins it by declaring it
as its base class, and the class is checked when it is created; Hopf and Bautin belong
to it, Winfree's model does not.

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
from ..base import DEFAULT_SOLVER_CONFIG, SolverConfig
from ..symbolic import AbstractSymbolicODE, CompiledExpressions, SymbolicODE


ROOT_EXPAND_ITERS = 64
"""Bracket-expansion steps for ``radius_from_isostable`` (in ``ln r``, doubling)."""
ROOT_NEWTON_ITERS = 48
"""Bisection-safeguarded Newton iterations for ``radius_from_isostable``."""


class AbstractNormalForm(AbstractSymbolicODE):
    r"""A planar oscillator $\dot r = r\rho(r)$, $\dot\theta = \omega(r)$ with its
    cycle at $r = 1$ and closed-form phase and isostable coordinates.

    A subclass supplies `_symbolic_template_polar`, `phase_shift` and `isostable`,
    overrides
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
    - `_compiled_rates`: $(\rho(r), \omega(r))$ compiled to JAX, set on each
        concrete class by `_compile` (ADR-0016).
    """

    dim: int = eqx.field(static=True, default=2, init=False)
    default_integration: str = eqx.field(static=True, default="closed_form", init=False)

    def _set_integration(self, integration: str) -> None:
        """Validate `integration` and store it as `default_integration`; called from
        every concrete constructor.

        **Raises:**

        `ValueError` for a name not in ``integration.INTEGRATIONS``.
        """
        from .integration import resolve_integration

        resolve_integration(integration)  # raises on an unknown name
        self.default_integration = integration

    # ------------------------------------------------ defining data (abstract) ----
    _compiled_rates: ClassVar[CompiledExpressions]

    @classmethod
    @abc.abstractmethod
    def _symbolic_template_polar(cls) -> SymbolicODE:
        r"""The vector field in polar coordinates as SymPy: states $(r, \theta)$ and
        expressions $(\dot r, \dot\theta)$, with symbols only and no values.

        The field must not depend on $\theta$, and $\dot r$ must have the factor $r$
        so that $\rho = \dot r / r$ is regular at $r = 0$; `_polar_rates` checks both.
        """

    @classmethod
    def _polar_rates(cls) -> tuple[sp.Symbol, sp.Expr, sp.Expr, dict[str, sp.Symbol]]:
        r"""`(r, ρ, ω, params)` read off the polar template, with
        $\rho = \dot r / r$ canceled and $\omega = \dot\theta$.

        **Raises:**

        `ValueError` when the template does not have two states, when the field
        depends on $\theta$, or when $\dot r$ lacks the factor $r$; and from
        `SymbolicODE.validate`.
        """
        template = cls._symbolic_template_polar().validate()
        if template.dim != 2:
            raise ValueError(
                f"a polar template has two states (r, theta), not {template.dim}."
            )
        r, theta = template.states
        r_dot, theta_dot = template.rhs
        if r_dot.has(theta) or theta_dot.has(theta):
            raise ValueError(
                f"the polar field depends on {theta}, but a normal form is "
                "rotationally symmetric."
            )
        rho = sp.cancel(r_dot / r)
        if sp.fraction(sp.together(rho))[1].has(r):
            raise ValueError(
                f"r_dot = {r_dot} lacks the factor r, so rho = r_dot / r is singular "
                "at r = 0."
            )
        return r, rho, theta_dot, template.params

    @classmethod
    def _symbolic_template(cls) -> SymbolicODE:
        r"""The cartesian field, derived from the polar template:
        $\dot x = x\rho - y\omega$ and $\dot y = y\rho + x\omega$ with
        $r = \sqrt{x^2 + y^2}$. SymPy reduces $(\sqrt q)^2$ to $q$, so a form whose
        rates are functions of $r^2$ comes out with no square root."""
        r, rho, omega, params = cls._polar_rates()
        x, y = sp.symbols("x y", real=True)
        radius = sp.sqrt(x**2 + y**2)
        rho_xy, omega_xy = rho.subs(r, radius), omega.subs(r, radius)
        return SymbolicODE(
            states=(x, y),
            params=dict(params),
            rhs=(x * rho_xy - y * omega_xy, y * rho_xy + x * omega_xy),
        )

    @classmethod
    def _compile(cls) -> None:
        r"""Compile the cartesian template, then the rates $(\rho(r), \omega(r))$
        into `_compiled_rates`.

        **Raises:**

        `ValueError` from `_polar_rates`.
        """
        super()._compile()
        r, rho, omega, params = cls._polar_rates()
        cls._compiled_rates = CompiledExpressions.from_expressions(
            (rho, omega), (r,), tuple(params)
        )

    def log_growth_rate(self, r: Float[Array, ""]) -> Float[Array, ""]:
        r"""$\rho(r) = \dot r / r$ at radius `r` in the basin; $\rho(1) = 0$ and
        $\rho'(1) < 0$. Read off the polar template."""
        return type(self)._compiled_rates(self, r)[0]

    def angular_rate(self, r: Float[Array, ""]) -> Float[Array, ""]:
        r"""$\omega(r) = \dot\theta$ at radius `r` in the basin; $\omega(1) \ne 0$.
        Read off the polar template."""
        return type(self)._compiled_rates(self, r)[1]

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

        **Returns:**

        ``(Θ, Ψ)`` with $\Theta \in (-\pi, \pi]$; both ``nan`` outside the basin.
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

        **Returns:**

        The radius, or ``nan`` for a `psi` above the range of $\Psi$.
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

        **Returns:**

        The cartesian points on $\theta = \Theta - h(r)$ (design document §5.1), shape
        ``(n, 2)``; ``nan`` at radii outside the basin.
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
    r"""Normal forms whose $\rho$ and $\omega$ are functions of $s = r^2$.

    A subclass writes the same `_symbolic_template_polar` as any normal form and
    declares this class as its base; `_compile` checks that both rates are functions of
    $r^2$ and compiles them in $s$. Evenness makes the cartesian field
    $\tilde\rho(|u|^2)\thinspace u + \tilde\omega(|u|^2)\thinspace\mathsf J u$ smooth at
    the origin with no square root (a polynomial for Hopf and Bautin), so the origin
    is the enclosed fixed point of the basin, `basin_radii` starts at $0$, and the
    eigenvalues at the origin, $\rho(0) \pm i\omega(0)$, are smooth-conjugacy
    invariants (design document §1 and §9, Option B). The ``r_squared`` integration
    needs this family.

    **Attributes:**

    - `_compiled_rates_sq`: $(\tilde\rho(s), \tilde\omega(s))$ compiled to JAX, set
        on each concrete class by `_compile`.
    """

    _compiled_rates_sq: ClassVar[CompiledExpressions]

    @classmethod
    def _compile(cls) -> None:
        r"""Compile as `AbstractNormalForm` does, then the rates as functions of
        $s = r^2$ into `_compiled_rates_sq`.

        **Raises:**

        `ValueError` when a rate is not a function of $r^2$: after $r = \sqrt s$ is
        substituted, a power of $s$ with a non-integer exponent remains. This is
        stricter than evenness, and it is what the square-root-free field needs:
        $\cos r$ is even, but $\cos\sqrt s$ has no derivative at $s = 0$.
        """
        super()._compile()
        r, rho, omega, params = cls._polar_rates()
        s = sp.Symbol("s", positive=True)
        rho_s, omega_s = rho.subs(r, sp.sqrt(s)), omega.subs(r, sp.sqrt(s))
        for name, expr in (("rho", rho_s), ("omega", omega_s)):
            fractional = [
                power
                for power in expr.atoms(sp.Pow)
                if power.base == s and power.exp.is_integer is not True
            ]
            if fractional:
                raise ValueError(
                    f"{cls.__name__} is declared even, but {name} = {expr} is not a "
                    f"function of r**2 (it contains {fractional[0]})."
                )
        cls._compiled_rates_sq = CompiledExpressions.from_expressions(
            (rho_s, omega_s), (s,), tuple(params)
        )

    def _log_growth_rate_sq(self, s: Float[Array, ""]) -> Float[Array, ""]:
        r"""$\tilde\rho(s) = \rho(\sqrt s)$ at `s` $= r^2$; $\tilde\rho(1) = 0$ and
        $\tilde\rho'(1) < 0$. Read off the polar template."""
        return type(self)._compiled_rates_sq(self, s)[0]

    def _angular_rate_sq(self, s: Float[Array, ""]) -> Float[Array, ""]:
        r"""$\tilde\omega(s) = \omega(\sqrt s)$ at `s` $= r^2$. Read off the polar
        template."""
        return type(self)._compiled_rates_sq(self, s)[1]

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
