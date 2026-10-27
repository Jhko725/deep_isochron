r"""The Hopf (Stuart–Landau) normal form: $\rho(r) = a(1 - r^2)$ and
$\omega(r) = \omega_0 + (\omega_1 - \omega_0) r^2$.

The limit cycle $r = 1$ has frequency $\omega_1$; the origin is an unstable focus with
eigenvalues $a \pm i\omega_0$, and $\omega_0 \ne \omega_1$ shears the isochrons into
logarithmic spirals. Hopf is Bautin with $b = 0$. The closed forms, with
$c = (\omega_1 - \omega_0)/a$: $\kappa = -2a$, $h(r) = c\ln r$ and
$\Psi(r) = \tfrac12(1 - r^{-2})$, with the explicit inverse $r = (1 - 2\Psi)^{-1/2}$
(design document §3.1 and §6.1).

The parameter $a > 0$ is a constrained leaf through ``A_CONSTRAINT``; $\omega_1$
(``w``) and $\omega_0$ (``w0``) are free. The constructor takes constrained values.
"""

import jax.numpy as jnp
import sympy as sp
from jaxtyping import Array, Float

from ...model.invertible.constraints import GreaterThan
from .base import AbstractEvenNormalForm


A_CONSTRAINT = GreaterThan(0.0)
"""The constraint ``a > 0``, with ``raw = 0`` mapped to ``a = 1`` (the default
``at_zero = lower + 1``)."""


class HopfNormalForm(AbstractEvenNormalForm):
    r"""$\dot r = a r (1 - r^2)$, $\dot\theta = \omega_0 + (\omega_1 - \omega_0) r^2$.

    **Arguments:**

    (constrained values)

    - `a`: The radial rate, positive.
    - `w`: $\omega_1$, the frequency on the cycle.
    - `w0`: $\omega_0$, the frequency at the origin; `w` when omitted.
    - `integration`: The default integration of `flow`.

    **Attributes:**

    - `raw_a`: The unconstrained leaf of `a` (ADR-0001).
    - `w`, `w0`: As above, as scalar arrays.

    **Raises:**

    `ValueError` for ``a <= 0``.
    """

    raw_a: Float[Array, ""]
    w: Float[Array, ""]
    w0: Float[Array, ""]

    def __init__(
        self,
        a: float = 1.0,
        w: float = 1.0,
        w0: float | None = None,
        integration: str = "closed_form",
    ):
        self._set_integration(integration)
        if a <= 0:
            raise ValueError("a must be positive.")
        self.raw_a = A_CONSTRAINT.inverse(jnp.asarray(a, dtype=float))
        self.w = jnp.asarray(w, dtype=float)
        self.w0 = jnp.asarray(w if w0 is None else w0, dtype=float)

    @property
    def a(self) -> Float[Array, ""]:
        """The constrained radial rate, ``A_CONSTRAINT(raw_a)``."""
        return A_CONSTRAINT(self.raw_a)

    def params(self):
        """``{"a", "w", "w0"}`` as constrained floats."""
        return {"a": float(self.a), "w": float(self.w), "w0": float(self.w0)}

    @classmethod
    def _symbolic_rates_sq(cls):
        r"""$\tilde\rho(s) = a(1 - s)$,
        $\tilde\omega(s) = \omega_0 + (\omega_1 - \omega_0) s$."""
        s = sp.Symbol("s", positive=True)
        a, w, w0 = sp.symbols("a w w0", real=True)
        return s, a * (1 - s), w0 + (w - w0) * s, {"a": a, "w": w, "w0": w0}

    def phase_shift(self, r):
        r"""$h(r) = c\ln r$ with $c = (\omega_1 - \omega_0)/a$."""
        return (self.w - self.w0) / self.a * jnp.log(r)

    def isostable(self, r):
        r"""$\Psi(r) = \tfrac12(1 - r^{-2})$."""
        return 0.5 * (1 - r ** (-2))

    def radius_from_isostable(self, psi):
        r"""The explicit inverse $r = (1 - 2\Psi)^{-1/2}$, for $\Psi < \tfrac12$."""
        return (1 - 2 * psi) ** (-0.5)
