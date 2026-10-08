r"""Winfree's planar model "with a hole" (Langfield, Krauskopf, Lee & Osinga 2025,
Commun. Nonlinear Sci. Numer. Simul. 151, 109043, §4.1, Eqs. (10)–(13); after Winfree
1980): in polar coordinates

$$\dot r = (1 - r)(r - a)\,r, \qquad \dot\theta = -\bigl(1 + w\,(1 - r)\bigr),$$

the paper's $(a, \omega) = (0.25, -0.5)$. *Cited*: the unit circle is the attracting
periodic orbit with period $2\pi$; the circle $r = a$ is a repelling periodic orbit
bounding the basin of attraction, and the closed disk $r \le a$ is the phaseless set;
the origin is a stable equilibrium; the isochron of phase $\vartheta \in [0, 1)$ is
$\psi(r) = 2\pi\bigl(\tfrac{\omega}{2\pi a}[\ln\tfrac{r}{r-a} - \ln\tfrac{1}{1-a}] -
\vartheta\bigr)$, $r > a$ (their Eq. (12)).

In the design document's symbols (*deduced* from §5.1–5.2 and checked against Eq. (12)
numerically; ``tests/test_normal_forms.py``):

* $\rho(r) = (1 - r)(r - a)$ and $\omega(r) = -(1 + w(1 - r))$ are **not even** in $r$,
  so this is an ``AbstractNormalForm`` but not an ``AbstractEvenNormalForm``: the
  cartesian field (their Eq. (11)) carries $\sqrt{x^2 + y^2}$;
* $\omega_1 = -1$ (clockwise; the signed ``period()`` is $-2\pi$),
  $\kappa = \rho'(1) = a - 1$; the inner cycle's exponent is
  $\kappa_u = a\,\rho'(a) = a(1 - a) > 0$;
* $h(r) = \dfrac{w}{a}\Bigl[\ln\dfrac{r - a}{r} - \ln(1 - a)\Bigr]$, which is
  $\theta = \Theta_0 - h(r)$ ⇔ Eq. (12) with $\Theta_0 = -2\pi\vartheta$;
  $h \to +\infty$ as $r \to a^+$ (every isochron spirals into the hole, their
  Fig. 4(a));
* $\Psi(r) = \dfrac{r - 1}{r}\Bigl(\dfrac{(1 - a)\,r}{r - a}\Bigr)^{1/a}$ —
  Wilson–Moehlis normalized, $-\infty$ at $r \to a^+$, saturating at $(1 - a)^{1/a}$
  outside;
* $a = 0$ is allowed: the hole closes to the origin, which becomes a *non-hyperbolic*
  equilibrium ($\dot r = r^2(1 - r)$), and the limits are $h = w(1 - 1/r)$,
  $\Psi = \tfrac{r-1}{r}\,e^{1/r - 1}$ (an essential singularity at the origin
  instead of Bautin's power law).

``basin_radii() = (a, ∞)``. Parameters are plain floats (an observed system, not a
latent one): no constraints, no learning.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from .base import AbstractNormalForm


class WinfreeNormalForm(AbstractNormalForm):
    r"""$\dot r = (1 - r)(r - a)\,r$, $\dot\theta = -(1 + w(1 - r))$; $0 \le a < 1$."""

    a: float = eqx.field(static=True)
    w: float = eqx.field(static=True)

    def __init__(self, a: float = 0.25, w: float = -0.5):
        if not 0.0 <= a < 1.0:
            raise ValueError(
                "a must satisfy 0 <= a < 1 (the hole lies inside the cycle)."
            )
        self.a = float(a)
        self.w = float(w)

    def params(self):
        return {"a": self.a, "w": self.w}

    # ------------------------------------------------------------ defining data ----
    def log_growth_rate(self, r):
        return (1.0 - r) * (r - self.a)

    def angular_rate(self, r):
        return -(1.0 + self.w * (1.0 - r))

    def _on_basin(self, r, value):
        # explicit nan outside r > a: for 1/a integer-valued the power of a negative
        # base would be finite garbage, and the root solve relies on nan to detect the
        # basin's edge
        return jnp.where(r > self.a, value, jnp.nan)

    def phase_shift(self, r):
        a, w = self.a, self.w
        rs = jnp.where(r > a, r, 1.0)  # keep the formula off its pole
        if a == 0.0:
            return self._on_basin(r, w * (1.0 - 1.0 / rs))
        return self._on_basin(r, (w / a) * (jnp.log((rs - a) / rs) - jnp.log(1.0 - a)))

    def isostable(self, r):
        a = self.a
        rs = jnp.where(r > a, r, 1.0)
        if a == 0.0:
            return self._on_basin(r, (rs - 1.0) / rs * jnp.exp(1.0 / rs - 1.0))
        return self._on_basin(
            r, (rs - 1.0) / rs * ((1.0 - a) * rs / (rs - a)) ** (1.0 / a)
        )

    # ---------------------------------------------------------------- basin -------
    def basin_radii(self) -> tuple[float, float]:
        return (self.a, float("inf"))

    def inner_floquet_exponent(self) -> Float[Array, ""]:
        r"""$\kappa_u = a\,\rho'(a) = a(1 - a)$, the exponent of the repelling cycle
        $r = a$ bounding the basin (by autodiff; $0$ when $a = 0$, where the origin is
        non-hyperbolic)."""
        a = jnp.asarray(self.a)
        return a * jax.grad(self.log_growth_rate)(a)
