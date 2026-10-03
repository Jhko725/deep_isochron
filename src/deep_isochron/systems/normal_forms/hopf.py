r"""The Hopf (Stuart–Landau) normal form: $\rho(s) = a(1 - s)$, $\omega$ linear in $s$.

Limit cycle $r = 1$ with frequency $w$; the origin is an unstable focus with
eigenvalues $a \pm i w_0$; $w_0 \ne w$ gives sheared (spiralling) isochrons. Closed
forms, derived by
separating $\dot\varphi = w$ and $\dot\psi = \kappa\psi$ and verified by autodiff in
``tests/test_normal_forms.py``: $\kappa = -2a$, $h(r) = \frac{w - w_0}{a}\ln r$,
$\psi(r) = (1 - r^2)/r^2$. The Bautin form with $b = 0$ coincides with this class.

Parameters are unconstrained leaves constrained on read (ADR-0001 in spirit): $a > 0$
through ``A_CONSTRAINT = GreaterThan(0.0)`` (``raw = 0`` ↦ $a = 1$); $w$, $w_0$ free.
The constructor takes constrained values.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

from ...model.invertible.constraints import GreaterThan
from .base import AbstractNormalForm


A_CONSTRAINT = GreaterThan(0.0)
"""``a > 0``; ``raw = 0`` ↦ ``a = 1`` (the default ``at_zero = lower + 1``)."""


class HopfNormalForm(AbstractNormalForm):
    r"""$\dot r = a r (1 - r^2)$, $\dot\theta = w_0 + (w - w_0) r^2$."""

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

    def radial_rate(self, s):
        return self.a * (1 - s)

    def angular_rate(self, s):
        return self.w0 + (self.w - self.w0) * s

    def floquet_exponent(self):
        return -2 * self.a

    def phase_shift(self, r):
        return (self.w - self.w0) / self.a * jnp.log(r)

    def isostable(self, r):
        return (1 - r**2) / r**2

    def eigenvalues_origin(self) -> Complex[Array, " 2"]:
        return jax.lax.complex(self.a * jnp.ones(2), self.w0 * jnp.array([1.0, -1.0]))
