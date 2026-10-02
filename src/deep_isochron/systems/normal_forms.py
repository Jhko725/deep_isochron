r"""The Hopf (Stuart–Landau) and Bautin normal forms, as ``AbstractNormalForm``s.

Both are written through the two rates of ``normal_form.py``:

* Hopf:   $\rho(s) = a(1 - s)$,                 $\omega(s) = w_0 + (w - w_0)\,s$
* Bautin: $\rho(s) = a(1 - s)(1 + b\,s)$,       $\omega(s) = w_0 + (w - w_0)\,s$

with $s = r^2$, so the limit cycle is $r = 1$ with frequency $w$, the origin is an
unstable focus with eigenvalues $a \pm i w_0$, and $w_0 \ne w$ makes the isochrons
spiral (shear). Closed forms (derived by separating $\dot\varphi = w$ and $\dot\psi =
\kappa\psi$; verified by autodiff in ``tests/test_systems.py``):

| | Hopf | Bautin |
|---|---|---|
| $\kappa$ | $-2a$ | $-2a(1 + b)$ |
| $h(r)$ | $c\ln r$ | $c[\ln r - \tfrac12\ln\tfrac{1 + b r^2}{1 + b}]$ |

with $c = (w - w_0)/a$.
| $\psi(r)$ | $(1 - r^2)/r^2$ | $(1 - r^2)(1 + b r^2)^b / r^{2(1+b)}$ |

Parameters are stored as unconstrained leaves and constrained on read (ADR-0001 in
spirit): $a > 0$ through ``Positive()`` (``raw = 0`` ↦ $a = 1$); Bautin's $b > -1$
through ``Positive(eps=-1, at_zero=0)`` (``raw = 0`` ↦ $b = 0$, which *is* the Hopf
form; $b > -1$ is exactly $\rho'(1) < 0$, the cycle's stability). $w$, $w_0$ are free.
The constructor takes constrained values.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

from ..model.invertible.constraints import Positive
from .normal_form import AbstractNormalForm


_positive_a = Positive(0.0, at_zero=1.0)
_shifted_b = Positive(-1.0, at_zero=0.0)


class HopfNormalForm(AbstractNormalForm):
    r"""$\dot r = a r (1 - r^2)$, $\dot\theta = w_0 + (w - w_0) r^2$."""

    raw_a: Float[Array, ""]
    w: Float[Array, ""]
    w0: Float[Array, ""]

    def __init__(self, a: float = 1.0, w: float = 1.0, w0: float | None = None):
        if a <= 0:
            raise ValueError("a must be positive.")
        self.raw_a = _positive_a.inverse(jnp.asarray(a, dtype=float))
        self.w = jnp.asarray(w, dtype=float)
        self.w0 = jnp.asarray(w if w0 is None else w0, dtype=float)

    @property
    def a(self) -> Float[Array, ""]:
        return _positive_a(self.raw_a)

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


class BautinNormalForm(AbstractNormalForm):
    r"""$\dot r = a r (1 - r^2)(1 + b r^2)$, $\dot\theta = w_0 + (w - w_0) r^2$.

    The extra factor makes the radial contraction rate depend on $r$ (the amplitude
    dynamics are no longer those of a Hopf form), which is what lets the conjugacy
    target match an observed oscillator's non-trivial Floquet exponent *and* its
    frequency independently of $a$.

    For $b \ge 0$ the cycle $r = 1$ attracts all of $\mathbb R^2 \setminus \{0\}$. For
    $-1 < b < 0$ there is a second, *unstable* cycle at $r^2 = -1/b$ (the two-cycle
    configuration of the Bautin bifurcation): the basin of $r = 1$ is $r < 1/\sqrt{-b}$,
    trajectories outside it escape to infinity in finite time, and the isostable is
    only defined inside. As a conjugacy target use $b \ge 0$."""

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
        self.raw_a = _positive_a.inverse(jnp.asarray(a, dtype=float))
        self.raw_b = _shifted_b.inverse(jnp.asarray(b, dtype=float))
        self.w = jnp.asarray(w, dtype=float)
        self.w0 = jnp.asarray(w if w0 is None else w0, dtype=float)

    @property
    def a(self) -> Float[Array, ""]:
        return _positive_a(self.raw_a)

    @property
    def b(self) -> Float[Array, ""]:
        return _shifted_b(self.raw_b)

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
