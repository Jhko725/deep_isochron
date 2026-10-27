r"""Planar systems whose limit cycle is a prescribed level set — exact test orbits.

For a smooth $g : \mathbb R^2 \to \mathbb R$ with a closed regular level set
$\lbrace g = 0\rbrace$, the field

$$
f = s\thinspace\bigl(R\nabla g - g\thinspace\nabla g\bigr),
\qquad R = \text{rotation by } +90°,
$$

has $\dot g = \nabla g \cdot f = -s\thinspace g\thinspace\Vert\nabla g\Vert^2$, so the
level set is invariant and attracting wherever $\nabla g \neq 0$, and is traversed
counterclockwise at speed $s\Vert\nabla g\Vert$ (*deduced*). Guckenheimer & Meloon
(2000, §5.1) use exactly this construction with $g = x^2 - y^2 + \tfrac23 y^3 + c$ and
$s = 1/2$ (their field $\dot x = y - y^2 - xg$, $\dot y = x + (y - y^2)g$) as a test
orbit whose location is known in closed form: $\max\vert g\vert$ along a computed orbit
is its error, the measure their Fig. 5.1 reports. A non-convex level set (a Cassini
oval) gives a cycle that a straight section cuts four times — the case a Poincaré
map's minimum-period logic must handle.

The period is *not* in closed form (it is $\oint ds / (s\Vert\nabla g\Vert)$ along the
curve); for Guckenheimer & Meloon's parameters it is "approximately 7.7" (*cited*, p.
970), with $(0, 0.2952161257895192)$ on the orbit to round-off.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float

from .base import AbstractODE


class LevelSetCycle(AbstractODE):
    r"""$f = s\thinspace(R\nabla g - g\nabla g)$: the zero level set of `g` is the
    attracting cycle.

    **Attributes:**

    - `g`: The level function, a Python function of the state. It is a static field,
        so distinct `g` compile separately.
    - `speed`: The factor $s > 0$.
    """

    dim: int = eqx.field(static=True, default=2, init=False)
    g: Callable[[Array], Array] = eqx.field(static=True)
    speed: float = eqx.field(static=True, default=1.0)

    def __init__(self, g: Callable[[Array], Array], speed: float = 1.0):
        if not speed > 0:
            raise ValueError("speed must be positive.")
        self.g = g
        self.speed = float(speed)

    def level(self, u: Float[ArrayLike, " 2"]) -> Float[Array, ""]:
        """``g(u)``: zero on the cycle; its magnitude is the distance-like error of a
        point off the cycle."""
        return self.g(jnp.asarray(u))

    def rhs(self, t, u: Float[Array, " 2"], args=None) -> Float[Array, " 2"]:
        r"""The field $s\thinspace(R\nabla g - g\nabla g)$ at `u`, with $\nabla g$ by
        autodiff of `level`."""
        del t, args
        grad = jax.grad(self.level)(u)
        rot = jnp.array([-grad[1], grad[0]])
        return self.speed * (rot - self.level(u) * grad)

    def params(self):
        """``{"speed": s}``; `g` is a function and is not recorded."""
        return {"speed": self.speed}


def guckenheimer_meloon(c: float = 0.07) -> LevelSetCycle:
    """The test orbit of Guckenheimer & Meloon 2000, §5.1: the level set
    ``x² − y² + 2y³/3 + c = 0`` at their time scale, ``s = 1/2``.

    **Arguments:**

    - `c`: The level, in ``(0, 1/3)`` so that the level set is a closed orbit.

    **Raises:**

    `ValueError` for `c` outside that interval.
    """
    if not 0 < c < 1 / 3:
        raise ValueError(
            "c must be in (0, 1/3) for the level set to be a closed orbit."
        )
    return LevelSetCycle(lambda u: u[0] ** 2 - u[1] ** 2 + 2 * u[1] ** 3 / 3 + c, 0.5)


def cassini(a: float = 1.0, b: float = 1.1) -> LevelSetCycle:
    """A Cassini oval ``(x² + y²)² − 2a²(x² − y²) = b⁴ − a⁴`` as the cycle.

    For ``a < b < a√2`` the oval is a single non-convex curve with its waist at
    ``y = ±√(b² − a²)``, so a horizontal section above the waist cuts it four times.

    **Arguments:**

    - `a`, `b`: The oval's parameters.

    **Raises:**

    `ValueError` unless ``a < b < a√2``.
    """
    if not a < b < a * 2**0.5:
        raise ValueError("a < b < a√2 is needed for a single non-convex oval.")
    return LevelSetCycle(
        lambda u: (
            (u[0] ** 2 + u[1] ** 2) ** 2
            - 2 * a**2 * (u[0] ** 2 - u[1] ** 2)
            - (b**4 - a**4)
        )
    )


__all__ = ["LevelSetCycle", "cassini", "guckenheimer_meloon"]
