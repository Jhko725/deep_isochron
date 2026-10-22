r"""The Bautin (generalized Hopf) normal form (design document §3):
$\rho(r) = a(1 - r^2)(1 + b r^2)$, $\omega(r) = \omega_0 + (\omega_1 - \omega_0) r^2$.

The quintic factor makes the radial contraction depend on $r$, decoupling the cycle's
Floquet exponent from the focus's instability: with $c = (\omega_1 - \omega_0)/a$,

* $\kappa = \rho'(1) = -2a(1 + b)$, $\mu = e^{-4\pi a(1+b)/\omega_1}$;
* $h(r) = c\,[\ln r - \tfrac12\ln((1 + b r^2)/(1 + b))]$;
* $\Psi(r) = \dfrac{(r^2 - 1)(1 + b r^2)^b}{2(1 + b)^b\, r^{2(1+b)}}$ — Wilson–Moehlis
  normalized ($\partial_r\Psi(1) = 1$); $\Psi \to \tfrac12(b/(1+b))^b$ as $r \to \infty$
  for $b \ge 0$.

For $b \ge 0$ the cycle attracts $\mathbb R^2 \setminus \{0\}$. For $-1 < b < 0$ there
is
an unstable outer cycle at $r^2 = -1/b$ bounding the basin (the two-cycle regime of the
Bautin bifurcation); $\Psi \to +\infty$ there. As a conjugacy target use $b \ge 0$.

Parameters: $a > 0$ through ``A_CONSTRAINT = GreaterThan(0)`` (``raw = 0`` ↦ $a = 1$),
$b > -1$ through ``B_CONSTRAINT = GreaterThan(-1)`` (``raw = 0`` ↦ $b = 0$, the Hopf
form;
$b > -1$ is exactly $\rho'(1) < 0$); $\omega_1$ (``w``), $\omega_0$ (``w0``) free. The
constructor takes constrained values.
"""

import jax.numpy as jnp
import sympy as sp
from jaxtyping import Array, Float

from ...model.invertible.constraints import GreaterThan
from .base import AbstractEvenNormalForm
from .hopf import A_CONSTRAINT


B_CONSTRAINT = GreaterThan(-1.0)
"""``b > -1`` (stability of the cycle); ``raw = 0`` ↦ ``b = 0`` (the Hopf form)."""


class BautinNormalForm(AbstractEvenNormalForm):
    r"""$\dot r = a r (1 - r^2)(1 + b r^2)$, $\dot\theta = \omega_0 + (\omega_1 -
    \omega_0) r^2$."""

    raw_a: Float[Array, ""]
    raw_b: Float[Array, ""]
    w: Float[Array, ""]
    w0: Float[Array, ""]

    def __init__(
        self,
        a: float = 1.0,
        b: float = 0.1,
        w: float = 1.0,
        w0: float | None = None,
        integration: str = "closed_form",
    ):
        self._set_integration(integration)
        if a <= 0:
            raise ValueError("a must be positive.")
        if b <= -1:
            raise ValueError("b must exceed -1 (stability of the cycle).")
        self.raw_a = A_CONSTRAINT.inverse(jnp.asarray(a, dtype=float))
        self.raw_b = B_CONSTRAINT.inverse(jnp.asarray(b, dtype=float))
        self.w = jnp.asarray(w, dtype=float)
        self.w0 = jnp.asarray(w if w0 is None else w0, dtype=float)

    @property
    def a(self) -> Float[Array, ""]:
        return A_CONSTRAINT(self.raw_a)

    @property
    def b(self) -> Float[Array, ""]:
        return B_CONSTRAINT(self.raw_b)

    def params(self):
        return {
            "a": float(self.a),
            "b": float(self.b),
            "w": float(self.w),
            "w0": float(self.w0),
        }

    # defining data, in s = r² (symbolic; the JAX hooks are derived, ADR-0016)
    @classmethod
    def _symbolic_rates_sq(cls):
        s = sp.Symbol("s", positive=True)
        a, b, w, w0 = sp.symbols("a b w w0", real=True)
        return (
            s,
            a * (1 - s) * (1 + b * s),
            w0 + (w - w0) * s,
            {"a": a, "b": b, "w": w, "w0": w0},
        )

    # closed forms, in r — written in 1/s = r⁻² so that they stay finite for r up to
    # the float range: the textbook forms (design document §3) pair an overflowing
    # numerator with an overflowing denominator (inf/inf = nan) from r ≈ 1e150 on,
    # which a learned bijection can reach before anything else goes wrong
    def phase_shift(self, r):
        # c [ln r − ½(ln(1 + b r²) − ln(1 + b))] = ½ c [ln(1 + b) − ln(r⁻² + b)]
        b = self.b
        inv_s = 1.0 / (r * r)
        return 0.5 * (self.w - self.w0) / self.a * (jnp.log1p(b) - jnp.log(inv_s + b))

    def isostable(self, r):
        # (s − 1)(1 + b s)^b / (2 (1 + b)^b s^(1+b))
        #   = (1 − 1/s)(1/s + b)^b / (2 (1 + b)^b)
        b = self.b
        inv_s = 1.0 / (r * r)
        return (1.0 - inv_s) * (inv_s + b) ** b / (2 * (1 + b) ** b)
