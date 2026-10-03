r"""The Bautin normal form: $\rho(s) = a(1 - s)(1 + bs)$, $\omega(s) = w_0 + (w - w_0)s$

The extra factor $(1 + b r^2)$ makes the radial contraction rate depend on $r$ (the
amplitude dynamics are no longer those of a Hopf form), which is what lets the conjugacy
target match an observed oscillator's non-trivial Floquet exponent *and* its frequency
independently of $a$. Closed forms (derived by separating $\dot\varphi = w$ and
$\dot\psi = \kappa\psi$; verified by autodiff in ``tests/test_normal_forms.py``), with
$c = (w - w_0)/a$:

* $\kappa = -2a(1 + b)$
* $h(r) = c\,[\ln r - \tfrac12 \ln\tfrac{1 + b r^2}{1 + b}]$   (so that $h(1) = 0$)
* $\psi(r) = (1 - r^2)(1 + b r^2)^b / r^{2(1 + b)}$

For $b \ge 0$ the cycle $r = 1$ attracts all of $\mathbb R^2 \setminus \{0\}$. For
$-1 < b < 0$ there is a second, *unstable* cycle at $r^2 = -1/b$ (the two-cycle regime
of the Bautin bifurcation): the basin of $r = 1$ is $r < 1/\sqrt{-b}$, trajectories
outside escape in finite time, and the isostable is defined only inside. As a conjugacy
target use $b \ge 0$.

Parameters are unconstrained leaves constrained on read: $a > 0$ through
``A_CONSTRAINT = GreaterThan(0.0)`` (``raw = 0`` ↦ $a = 1$) and $b > -1$ through
``B_CONSTRAINT = GreaterThan(-1.0)`` (``raw = 0`` ↦ $b = 0$, the Hopf form; $b > -1$ is
exactly $\rho'(1) < 0$, the stability of the cycle). $w$, $w_0$ free. The constructor
takes constrained values.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

from ...model.invertible.constraints import GreaterThan
from .base import AbstractNormalForm
from .hopf import A_CONSTRAINT


B_CONSTRAINT = GreaterThan(-1.0)
"""``b > -1`` (stability of the cycle); ``raw = 0`` ↦ ``b = 0`` (the Hopf form)."""


class BautinNormalForm(AbstractNormalForm):
    r"""$\dot r = a r (1 - r^2)(1 + b r^2)$, $\dot\theta = w_0 + (w - w_0) r^2$."""

    raw_a: Float[Array, ""]
    raw_b: Float[Array, ""]
    w: Float[Array, ""]
    w0: Float[Array, ""]

    def __init__(
        self, a: float = 1.0, b: float = 0.1, w: float = 1.0, w0: float | None = None
    ):
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

    def radial_rate(self, s):
        return self.a * (1 - s) * (1 + self.b * s)

    def angular_rate(self, s):
        return self.w0 + (self.w - self.w0) * s

    def floquet_exponent(self):
        return -2 * self.a * (1 + self.b)

    def phase_shift(self, r):
        b = self.b
        return (
            (self.w - self.w0)
            / self.a
            * (jnp.log(r) - 0.5 * (jnp.log1p(b * r**2) - jnp.log1p(b)))
        )

    def isostable(self, r):
        b = self.b
        return (1 - r**2) * (1 + b * r**2) ** b / r ** (2 * (1 + b))

    def eigenvalues_origin(self) -> Complex[Array, " 2"]:
        return jax.lax.complex(self.a * jnp.ones(2), self.w0 * jnp.array([1.0, -1.0]))
