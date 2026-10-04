r"""``AbstractNormalForm``: planar oscillators with closed-form phase–amplitude
structure.

The mathematics — symbols, conventions, derivations — is ``docs/design/normal-forms.md``
(the design document); this module implements its §9. In short: a normal form is a
planar ODE that in polar coordinates $(r, \theta)$ separates as

$$\dot r = r\,\rho(r), \qquad \dot\theta = \omega(r),$$

with the stable limit cycle at $r = 1$ ($\rho(1) = 0$, $\rho'(1) < 0$). $\rho$ is the
**log growth rate** of the radius ($d\ln r/dt$) and $\omega$ the **angular rate**; both
must be *even* functions of $r$ (design document §1), which a subclass guarantees by
supplying them as smooth functions of $s = r^2$ through the ``*_sq`` hooks — the one
place the $s$-chart appears in the API besides the ``r_squared`` integration (§9,
Option B). Everything public is in $r$ and in **cartesian** coordinates.

Closed forms a subclass supplies, in $r$: the phase shift $h(r)$ with the asymptotic
phase $\Theta = \theta + h(r)$, $h(1) = 0$; and the isostable coordinate $\Psi(r)$ with
$\dot\Psi = \kappa\Psi$, normalised à la Wilson & Moehlis, $\Psi(1) = 0$,
$\partial_r\Psi(1) = 1$ (negative inside the cycle, §5.2). The base class derives the
Floquet exponent $\kappa = \rho'(1)$ by autodiff (§4.2, no factor 2), the multiplier,
the eigenvalues at the origin, the vector fields in both charts,
``phase``/``amplitude``,
``limit_cycle``, ``isochron``, the three charts (polar, cartesian, phase–amplitude; §7),
and ``flow`` with a choice of integration (§6).
"""

import abc
from typing import Any

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Complex, Float

from ...misc import cartesian_to_polar, polar_to_cartesian
from ..base import AbstractODE, DEFAULT_SOLVER_CONFIG, SolverConfig


ROOT_EXPAND_ITERS = 64
"""Bracket-expansion steps for ``radius_from_isostable`` (in ``ln r``, doubling)."""
ROOT_NEWTON_ITERS = 48
"""Bisection-safeguarded Newton iterations for ``radius_from_isostable``."""


class AbstractNormalForm(AbstractODE):
    dim: int = eqx.field(static=True, default=2, init=False)
    default_integration: str = eqx.field(static=True, default="closed_form", init=False)
    """The integration ``flow`` uses when none is given. ``"closed_form"`` is exact
    (phase and isostable evolved analytically, §6) and cheaper than any ODE solve, so
    it is the default for every normal form; the numerical integrations exist for the
    comparison experiments (``integration.INTEGRATIONS``)."""

    # ------------------------------------------------ defining data (abstract) ----
    @abc.abstractmethod
    def _log_growth_rate_sq(self, s: Float[Array, ""]) -> Float[Array, ""]:
        r"""$\tilde\rho(s) = \rho(\sqrt s)$, with $\dot r = r\,\tilde\rho(r^2)$;
        $\tilde\rho(1) = 0$, $\tilde\rho'(1) < 0$. Smooth in $s$."""

    @abc.abstractmethod
    def _angular_rate_sq(self, s: Float[Array, ""]) -> Float[Array, ""]:
        r"""$\tilde\omega(s) = \omega(\sqrt s)$, with $\dot\theta = \tilde\omega(r^2)$.
        Smooth in $s$."""

    @abc.abstractmethod
    def phase_shift(self, r: Float[Array, ""]) -> Float[Array, ""]:
        r"""$h(r)$: asymptotic phase $\Theta = \theta + h(r)$; $h(1) = 0$ (§5.1)."""

    @abc.abstractmethod
    def isostable(self, r: Float[Array, ""]) -> Float[Array, ""]:
        r"""$\Psi(r)$: $\dot\Psi = \kappa\Psi$; $\Psi(1) = 0$, $\partial_r\Psi(1) = 1$,
        strictly increasing on the basin (§5.2)."""

    # ----------------------------------------------------- public views in r -----
    def log_growth_rate(self, r: Float[Array, ""]) -> Float[Array, ""]:
        r"""$\rho(r) = \dot r / r$."""
        return self._log_growth_rate_sq(r * r)

    def angular_rate(self, r: Float[Array, ""]) -> Float[Array, ""]:
        r"""$\omega(r) = \dot\theta$."""
        return self._angular_rate_sq(r * r)

    # ------------------------------------------------------- linear invariants ----
    def omega(self) -> Float[Array, ""]:
        r"""$\omega_1 = \omega(1)$: angular frequency on the limit cycle."""
        return self.angular_rate(jnp.asarray(1.0))

    def period(self) -> Float[Array, ""]:
        return 2 * jnp.pi / self.omega()

    def floquet_exponent(self) -> Float[Array, ""]:
        r"""$\kappa = \rho'(1) < 0$ (design document §4.2; the $r$-chart derivative,
        no factor 2), by autodiff of ``log_growth_rate``."""
        return jax.grad(self.log_growth_rate)(jnp.asarray(1.0))

    def floquet_multiplier(self) -> Float[Array, ""]:
        r"""$\mu = e^{\kappa T}$."""
        return jnp.exp(self.floquet_exponent() * self.period())

    def eigenvalues_origin(self) -> Complex[Array, " 2"]:
        r"""$\rho(0) \pm i\,\omega(0)$ (design document §4.1)."""
        zero = jnp.asarray(0.0)
        re, im = self._log_growth_rate_sq(zero), self._angular_rate_sq(zero)
        return jax.lax.complex(re * jnp.ones(2), im * jnp.array([1.0, -1.0]))

    # ------------------------------------------------------------------ charts -----
    @staticmethod
    def to_polar(u: Float[Array, " 2"]) -> Float[Array, " 2"]:
        r"""Cartesian $(x, y)$ -> polar $(r, \theta)$, $\theta \in (-\pi, \pi]$."""
        return cartesian_to_polar(u)

    @staticmethod
    def from_polar(z: Float[Array, " 2"]) -> Float[Array, " 2"]:
        r"""Polar $(r, \theta)$ -> cartesian $(x, y)$."""
        return polar_to_cartesian(z)

    def to_phase_amplitude(self, u: Float[Array, " 2"]) -> Float[Array, " 2"]:
        r"""Cartesian -> $(\Theta, \Psi)$, the chart in which the flow is linear
        ($\dot\Theta = \omega_1$, $\dot\Psi = \kappa\Psi$; §7); $\Theta \in (-\pi,
        \pi]$."""
        r, theta = self.to_polar(u)
        return jnp.stack((_wrap(theta + self.phase_shift(r)), self.isostable(r)))

    def from_phase_amplitude(self, z: Float[Array, " 2"]) -> Float[Array, " 2"]:
        r"""$(\Theta, \Psi)$ -> cartesian: $r = \Psi^{-1}(\Psi)$ by
        ``radius_from_isostable``
        and $\theta = \Theta - h(r)$."""
        Theta, Psi = z
        r = self.radius_from_isostable(Psi)
        theta = Theta - self.phase_shift(r)
        return jnp.stack((r * jnp.cos(theta), r * jnp.sin(theta)))

    def radius_from_isostable(self, psi: Float[Array, ""]) -> Float[Array, ""]:
        r"""$r$ with $\Psi(r) = \psi$: a bracketed, bisection-safeguarded Newton solve
        in
        $\ell = \ln r$ (design document §6.1/§7), differentiable through the implicit
        function theorem. Subclasses with an explicit inverse (Hopf) override this.
        Returns ``nan`` for a $\psi$ above the range of $\Psi$ (unreachable)."""
        return _radius_from_isostable(self, psi)

    # ----------------------------------------------------------- vector fields -----
    def rhs(
        self, t: Float[ArrayLike, ""], u: Float[Array, " 2"], args: Any = None
    ) -> Float[Array, " 2"]:
        r"""Cartesian field $\tilde\rho(|u|^2)\,u + \tilde\omega(|u|^2)\,\mathsf J u$:
        smooth at the origin, no square root (§2, §9)."""
        del t, args
        s = jnp.sum(u * u)
        rho, w = self._log_growth_rate_sq(s), self._angular_rate_sq(s)
        x, y = u
        return jnp.stack((rho * x - w * y, rho * y + w * x))

    def rhs_polar(
        self, t: Float[ArrayLike, ""], z: Float[Array, " 2"], args: Any = None
    ) -> Float[Array, " 2"]:
        r"""Field in the polar chart: $(r\rho(r), \omega(r))$."""
        del t, args
        r, _ = z
        return jnp.stack((r * self.log_growth_rate(r), self.angular_rate(r)))

    # ---------------------------------------------------------- phase/amplitude ----
    def phase(self, u: Float[Array, " 2"]) -> Float[Array, ""]:
        r"""Asymptotic phase $\Theta(u) \in (-\pi, \pi]$: the isochron through $u$,
        labelled by the polar angle of the point it converges to on the cycle."""
        return self.to_phase_amplitude(u)[0]

    def amplitude(self, u: Float[Array, " 2"]) -> Float[Array, ""]:
        r"""Isostable coordinate $\Psi(u)$: zero on the cycle, negative inside,
        $\Psi(t) = \Psi_0 e^{\kappa t}$ along every trajectory."""
        return self.to_phase_amplitude(u)[1]

    def limit_cycle(self, Theta: Float[Array, " *n"]) -> Float[Array, " *n 2"]:
        r"""Point(s) of the limit cycle at phase $\Theta$ ($r = 1$, where $h = 0$)."""
        return jnp.stack((jnp.cos(Theta), jnp.sin(Theta)), axis=-1)

    def isochron(
        self, Theta: Float[ArrayLike, ""], r: Float[Array, " n"]
    ) -> Float[Array, "n 2"]:
        r"""The isochron $\Theta$ sampled at radii $r > 0$: cartesian points on
        $\theta = \Theta - h(r)$ (§5.1)."""
        theta = Theta - self.phase_shift(r)
        return jnp.stack((r * jnp.cos(theta), r * jnp.sin(theta)), axis=-1)

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


def _wrap(angle):
    return jnp.arctan2(jnp.sin(angle), jnp.cos(angle))


# ------------------------------------------------------------ the root solve ------
@eqx.filter_custom_jvp
def _radius_from_isostable(nf: AbstractNormalForm, psi):
    """Monotone root ``Ψ(e^ℓ) = psi`` in ``ℓ = ln r``.

    Bracket: ``ℓ = 0`` is one end (``Ψ(1) = 0``); the other is found by doubling away
    from it until ``Ψ`` passes ``psi`` (``Ψ → -∞`` as ``r → 0`` guarantees the inside;
    outside, ``Ψ`` may saturate below ``psi``, in which case the result is ``nan``).
    Then bisection-safeguarded Newton with a fixed iteration count (reverse-mode
    friendly), as in ``CubicBSpline`` (ADR-0006). ``Ψ`` is strictly increasing on the
    basin, so the root is unique. Differentiable in ``psi`` *and* in the normal form's
    parameters through the implicit function theorem (``def_jvp`` below).
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
    # Implicit function theorem on Ψ(r; nf) = psi:
    #   dr = (dpsi - ∂Ψ/∂nf · dnf) / Ψ'(r).
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
