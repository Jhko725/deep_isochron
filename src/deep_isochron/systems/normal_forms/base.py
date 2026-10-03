r"""``AbstractNormalForm``: planar oscillators with closed-form phase–amplitude
structure.

A normal form here is a planar ODE that, in polar coordinates $(r, \theta)$ about the
origin, separates as

$$\dot r = r\,\rho(r^2), \qquad \dot\theta = \omega(r^2),$$

with a unique stable limit cycle at $r = 1$ ($\rho(1) = 0$, $\rho'(1) < 0$). A subclass
supplies ``radial_rate`` ($\rho$) and ``angular_rate`` ($\omega$) as functions of $s =
r^2$ — polynomials for the Hopf and Bautin normal forms, so that the cartesian vector
field $\dot u = \rho(|u|^2)\,u + \omega(|u|^2)\,J u$ ($J$ the rotation by $\pi/2$) is
smooth at the origin — together with the closed forms the analysis needs:

* ``period`` $T = 2\pi/\omega(1)$ and the non-trivial ``floquet_exponent``
  $\kappa = 2\rho'(1)$ (the linearisation of $\dot s = 2 s \rho(s)$ at $s = 1$);
* ``phase_shift`` $h(r)$ with the **asymptotic phase** $\varphi(u) = \theta + h(r)$
  satisfying $\dot\varphi = \omega(1)$ everywhere, i.e. $h'(r) = (\omega(1) -
  \omega(r^2)) / (r\rho(r^2))$ and $h(1) = 0$;
* ``amplitude`` — the **isostable coordinate** $\psi(r)$ with $\dot\psi = \kappa\psi$,
  $\psi(1) = 0$, i.e. $\psi'/\psi = \kappa / (r\rho(r^2))$, normalised so that
  $\psi \to +\infty$ as $r \to 0$ and $\psi < 0$ outside the cycle.

Isochrons are the level sets of $\varphi$; ``isochron(phi, r)`` returns the curve
$\theta = \varphi - h(r)$ in cartesian coordinates. For the observed systems none of
this is available in closed form; ``deep_isochron.analysis`` computes the same
quantities numerically for any ``AbstractODE``.

Everything public is in **cartesian** coordinates; ``to_chart``/``from_chart`` convert.
The flow can be integrated in several ways (cartesian, polar, $r^2$), all returning
cartesian trajectories; see ``integration.py``.

The conventions and derivations are being consolidated in
``docs/design/normal-forms.md`` (roadmap B10); the public API will be expressed in $r$
once that document is agreed.
"""

import abc
from typing import Any

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float

from ...misc import cartesian_to_polar, polar_to_cartesian
from ..base import AbstractODE, DEFAULT_SOLVER_CONFIG, SolverConfig


class AbstractNormalForm(AbstractODE):
    dim: int = eqx.field(static=True, default=2, init=False)

    # ------------------------------------------------------------ the two rates ----
    @abc.abstractmethod
    def radial_rate(self, s: Float[Array, ""]) -> Float[Array, ""]:
        r"""$\rho(s)$ with $\dot r = r\rho(r^2)$; $\rho(1) = 0$."""

    @abc.abstractmethod
    def angular_rate(self, s: Float[Array, ""]) -> Float[Array, ""]:
        r"""$\omega(s)$ with $\dot\theta = \omega(r^2)$."""

    # ------------------------------------------------------- closed-form structure --
    def floquet_exponent(self) -> Float[Array, ""]:
        r"""Non-trivial Floquet exponent $\kappa = 2\rho'(1) < 0$ of the limit cycle
        (linearisation of $\dot s = 2s\rho(s)$ at $s = 1$); by autodiff of
        ``radial_rate``."""
        return 2 * jax.grad(self.radial_rate)(jnp.asarray(1.0))

    @abc.abstractmethod
    def phase_shift(self, r: Float[Array, ""]) -> Float[Array, ""]:
        r"""$h(r)$ with asymptotic phase $\varphi = \theta + h(r)$; $h(1) = 0$."""

    @abc.abstractmethod
    def isostable(self, r: Float[Array, ""]) -> Float[Array, ""]:
        r"""$\psi(r)$ with $\dot\psi = \kappa\psi$ along the flow; $\psi(1) = 0$."""

    def omega(self) -> Float[Array, ""]:
        """Angular frequency on the limit cycle, ``angular_rate(1)``."""
        return self.angular_rate(jnp.asarray(1.0))

    def period(self) -> Float[Array, ""]:
        return 2 * jnp.pi / self.omega()

    def floquet_multiplier(self) -> Float[Array, ""]:
        """``exp(kappa T)``: contraction of the amplitude over one period."""
        return jnp.exp(self.floquet_exponent() * self.period())

    # ------------------------------------------------------------------ charts -----
    @staticmethod
    def to_chart(u: Float[Array, " 2"]) -> Float[Array, " 2"]:
        """Cartesian ``(x, y)`` -> polar ``(r, theta)``, ``theta`` in ``(-pi, pi]``."""
        return cartesian_to_polar(u)

    @staticmethod
    def from_chart(z: Float[Array, " 2"]) -> Float[Array, " 2"]:
        """Polar ``(r, theta)`` -> cartesian ``(x, y)``."""
        return polar_to_cartesian(z)

    # ----------------------------------------------------------- vector fields -----
    def rhs(
        self, t: Float[ArrayLike, ""], u: Float[Array, " 2"], args: Any = None
    ) -> Float[Array, " 2"]:
        """Cartesian vector field, smooth at the origin."""
        del t, args
        s = jnp.sum(u**2)
        rho, w = self.radial_rate(s), self.angular_rate(s)
        x, y = u
        return jnp.stack((rho * x - w * y, rho * y + w * x))

    def rhs_polar(
        self, t: Float[ArrayLike, ""], z: Float[Array, " 2"], args: Any = None
    ) -> Float[Array, " 2"]:
        """Vector field in the polar chart ``(r, theta)``."""
        del t, args
        r, _ = z
        s = r**2
        return jnp.stack((r * self.radial_rate(s), self.angular_rate(s)))

    # ---------------------------------------------------------- phase/amplitude ----
    def phase(self, u: Float[Array, " 2"]) -> Float[Array, ""]:
        r"""Asymptotic phase $\varphi(u) \in (-\pi, \pi]$ (the isochron through ``u``,
        labelled by the phase of the point it converges to on the cycle)."""
        r, theta = self.to_chart(u)
        phi = theta + self.phase_shift(r)
        return jnp.arctan2(jnp.sin(phi), jnp.cos(phi))

    def amplitude(self, u: Float[Array, " 2"]) -> Float[Array, ""]:
        r"""Isostable coordinate $\psi(u)$: zero on the cycle, decays as
        $e^{\kappa t}$ along every trajectory."""
        r, _ = self.to_chart(u)
        return self.isostable(r)

    def limit_cycle(self, phi: Float[Array, " *n"]) -> Float[Array, " *n 2"]:
        """Point(s) on the limit cycle at asymptotic phase ``phi`` (``r = 1``, where
        the phase shift vanishes)."""
        return jnp.stack((jnp.cos(phi), jnp.sin(phi)), axis=-1)

    def isochron(
        self, phi: Float[ArrayLike, ""], r: Float[Array, " n"]
    ) -> Float[Array, "n 2"]:
        """The isochron of phase ``phi`` sampled at radii ``r`` (``r > 0``), as
        cartesian points ``theta = phi - h(r)``."""
        theta = phi - self.phase_shift(r)
        return jnp.stack((r * jnp.cos(theta), r * jnp.sin(theta)), axis=-1)

    # ------------------------------------------------------------------- flow ------
    def flow(
        self,
        ts: Float[Array, " time"],
        u0: Float[Array, " 2"],
        args: Any = None,
        *,
        config: SolverConfig = DEFAULT_SOLVER_CONFIG,
        integration="r_squared",
    ) -> dfx.Solution:
        """The diffrax ``Solution`` through cartesian ``u0``, with ``.ys`` the
        **cartesian** trajectory whichever ``integration`` produced it. ``integration``
        is an ``AbstractFlowIntegration`` or one of the names in
        ``integration.INTEGRATIONS`` (``"cartesian"``, ``"polar"``, ``"r_squared"``);
        it is static, so each value traces separately under ``jit``/``vmap``."""
        from .integration import resolve_integration

        return resolve_integration(integration)(self, ts, u0, args, config)
