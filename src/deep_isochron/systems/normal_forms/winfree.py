r"""Winfree's planar model "with a hole" (Langfield et al. 2025, §4.1, Eqs. 10–13;
after Winfree 1980).

In polar coordinates, $\dot r = (1 - r)(r - a)\thinspace r$ and
$\dot\theta = -(1 + w\thinspace(1 - r))$, the paper's $(a, \omega) = (0.25, -0.5)$. The
unit circle is the attracting cycle, traversed clockwise with period $2\pi$, and for
$a > 0$ the circle $r = a$ is a repelling cycle bounding the basin, so the closed disk
$r \le a$ is the phaseless set. The rates are not even in $r$, so this is an
``AbstractNormalForm`` but not an ``AbstractEvenNormalForm``, and the cartesian field
carries $\sqrt{x^2 + y^2}$.

Three regimes in $a$ (design document §3.3): $a > 0$ is the paper's hole; $a = 0$
closes the hole to a non-hyperbolic point at the origin, which a hyperbolic latent focus
cannot be conjugate to; $a < 0$, not in the paper, makes the origin a hyperbolic
repelling focus with the same closed forms, and is the member to train conjugacies
against. The parameters are plain floats: an observed system, not a learnable one
(ADR-0013 §5).

The closed forms $h(r)$ and $\Psi(r)$, their $a = 0$ limits, and the equivalence with
the paper's Eq. 12 are derived and checked in the design document §3.3.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import sympy as sp
from jaxtyping import Array, Complex, Float

from .base import AbstractNormalForm


class WinfreeNormalForm(AbstractNormalForm):
    r"""$\dot r = (1 - r)(r - a)\thinspace r$, $\dot\theta = -(1 + w(1 - r))$.

    **Arguments:**

    - `a`: The hole radius, below 1: a hole of radius `a` for $a > 0$, a
        non-hyperbolic point for $a = 0$, a hyperbolic focus for $a < 0$ (the
        conjugacy target; the default).
    - `w`: The nonisochronicity, the paper's $\omega$.
    - `integration`: The default integration of `flow`.

    **Attributes:**

    - `a`, `w`: As above, static floats.

    **Raises:** `ValueError` for ``a >= 1``, where the cycle would not attract.
    """

    a: float = eqx.field(static=True)
    w: float = eqx.field(static=True)

    def __init__(
        self, a: float = -0.25, w: float = -0.5, integration: str = "closed_form"
    ):
        self._set_integration(integration)
        if not a < 1.0:
            raise ValueError("a must be < 1 (the cycle r = 1 must attract).")
        self.a = float(a)
        self.w = float(w)

    def params(self):
        """``{"a", "w"}``."""
        return {"a": self.a, "w": self.w}

    @classmethod
    def _symbolic_rates(cls):
        r"""$\rho(r) = (1 - r)(r - a)$, $\omega(r) = -(1 + w(1 - r))$; not even."""
        r = sp.Symbol("r", positive=True)
        a, w = sp.symbols("a w", real=True)
        return r, (1 - r) * (r - a), -(1 + w * (1 - r)), {"a": a, "w": w}

    def _on_basin(self, r, value):
        """`value` where `r` is in the basin, ``nan`` elsewhere.

        The ``nan`` is explicit because for an integer $1/a$ the power of a negative
        base would be finite garbage, and the root solve relies on ``nan`` to detect
        the basin's edge.
        """
        return jnp.where(r > self.basin_radii()[0], value, jnp.nan)

    def phase_shift(self, r):
        r"""$h(r) = \tfrac{w}{a}[\ln\tfrac{r - a}{r} - \ln(1 - a)]$, and its limit
        $w(1 - 1/r)$ at $a = 0$; $+\infty$ as $r \to a^+$."""
        a, w = self.a, self.w
        rs = jnp.where(r > self.basin_radii()[0], r, 1.0)  # off the formula's pole
        if a == 0.0:
            return self._on_basin(r, w * (1.0 - 1.0 / rs))
        return self._on_basin(r, (w / a) * (jnp.log((rs - a) / rs) - jnp.log(1.0 - a)))

    def isostable(self, r):
        r"""$\Psi(r) = \tfrac{r - 1}{r}\bigl(\tfrac{(1 - a) r}{r - a}\bigr)^{1/a}$, and
        its limit $\tfrac{r - 1}{r} e^{1/r - 1}$ at $a = 0$; $-\infty$ as $r \to a^+$,
        saturating at $(1 - a)^{1/a}$ outside."""
        a = self.a
        rs = jnp.where(r > self.basin_radii()[0], r, 1.0)
        if a == 0.0:
            return self._on_basin(r, (rs - 1.0) / rs * jnp.exp(1.0 / rs - 1.0))
        return self._on_basin(
            r, (rs - 1.0) / rs * ((1.0 - a) * rs / (rs - a)) ** (1.0 / a)
        )

    def basin_radii(self) -> tuple[float, float]:
        r"""$(\max(a, 0), \infty)$."""
        return (max(self.a, 0.0), float("inf"))

    def eigenvalues_origin(self) -> Complex[Array, " 2"]:
        r"""The eigenvalues at the origin, $-a \pm i(1 + w)$: a conjugacy invariant
        only for $a < 0$, when the origin is the basin's enclosed hyperbolic fixed
        point; for $a > 0$ it is the stable focus inside the hole, and for $a = 0$ it
        is non-hyperbolic."""
        re = jnp.asarray(-self.a)
        im = jnp.asarray(-(1.0 + self.w))
        return jax.lax.complex(re * jnp.ones(2), im * jnp.array([1.0, -1.0]))

    def inner_floquet_exponent(self) -> Float[Array, ""]:
        r"""$\kappa_u = a\thinspace\rho'(a) = a(1 - a)$, the exponent of the repelling
        cycle $r = a$ bounding the basin, by autodiff; $0$ when $a = 0$, where the
        origin is non-hyperbolic."""
        a = jnp.asarray(self.a)
        return a * jax.grad(self.log_growth_rate)(a)
