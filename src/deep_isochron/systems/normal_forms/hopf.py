r"""The Hopf (Stuart–Landau) normal form (design document §3): $\rho(r) = a(1 - r^2)$,
$\omega(r) = \omega_0 + (\omega_1 - \omega_0) r^2$.

Limit cycle $r = 1$ with frequency $\omega_1$; unstable focus at the origin with
eigenvalues $a \pm i\omega_0$; $\omega_0 \ne \omega_1$ shears the isochrons into
logarithmic spirals (nonisochronicity $\omega_1 - \omega_0$). Closed forms, with
$c = (\omega_1 - \omega_0)/a$:

* $\kappa = \rho'(1) = -2a$, $\mu = e^{-4\pi a/\omega_1}$;
* $h(r) = c\ln r$;
* $\Psi(r) = (r^2 - 1)/(2r^2) = \tfrac12(1 - r^{-2})$ — Wilson–Moehlis normalized
  ($\partial_r\Psi(1) = 1$), range $(-\infty, \tfrac12)$, inverse
  $r = (1 - 2\Psi)^{-1/2}$;
* explicit trajectories (§6.1): $r(t)^{-2} = 1 + (r_0^{-2} - 1)e^{-2at}$.

Hopf is Bautin with $b = 0$. Parameters: $a > 0$ through
``A_CONSTRAINT = GreaterThan(0)`` (``raw = 0`` ↦ $a = 1$); $\omega_1$ (``w``),
$\omega_0$ (``w0``) free, ``w0`` defaults to
``w``. The constructor takes constrained values.
"""

import jax.numpy as jnp
from jaxtyping import Array, Float

from ...model.invertible.constraints import GreaterThan
from .base import AbstractNormalForm


A_CONSTRAINT = GreaterThan(0.0)
"""``a > 0``; ``raw = 0`` ↦ ``a = 1`` (the default ``at_zero = lower + 1``)."""


class HopfNormalForm(AbstractNormalForm):
    r"""$\dot r = a r (1 - r^2)$,
    $\dot\theta = \omega_0 + (\omega_1 - \omega_0) r^2$."""

    raw_a: Float[Array, ""]
    w: Float[Array, ""]
    w0: Float[Array, ""]

    def __init__(self, a: float = 1.0, w: float = 1.0, w0: float | None = None):
        if a <= 0:
            raise ValueError("a must be positive.")
        self.raw_a = A_CONSTRAINT.inverse(jnp.asarray(a, dtype=float))
        self.w = jnp.asarray(w, dtype=float)
        self.w0 = jnp.asarray(w if w0 is None else w0, dtype=float)

    @property
    def a(self) -> Float[Array, ""]:
        return A_CONSTRAINT(self.raw_a)

    def params(self):
        return {"a": float(self.a), "w": float(self.w), "w0": float(self.w0)}

    # defining data, in s = r²
    def _log_growth_rate_sq(self, s):
        return self.a * (1 - s)

    def _angular_rate_sq(self, s):
        return self.w0 + (self.w - self.w0) * s

    # closed forms, in r
    def phase_shift(self, r):
        return (self.w - self.w0) / self.a * jnp.log(r)

    def isostable(self, r):
        return 0.5 * (1 - r ** (-2))

    def radius_from_isostable(self, psi):
        """Explicit inverse: ``r = (1 - 2 Ψ)^(-1/2)`` for ``Ψ < 1/2``."""
        return (1 - 2 * psi) ** (-0.5)
