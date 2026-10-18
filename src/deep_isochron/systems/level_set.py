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
    r"""$f = s\thinspace(R\nabla g - g\nabla g)$: the zero level set of ``g`` is the
    attracting cycle. ``g`` is a static field (a Python function), so distinct ``g``
    compile separately."""

    dim: int = eqx.field(static=True, default=2, init=False)
    g: Callable[[Array], Array] = eqx.field(static=True)
    speed: float = eqx.field(static=True, default=1.0)

    def __init__(self, g: Callable[[Array], Array], speed: float = 1.0):
        if not speed > 0:
            raise ValueError("speed must be positive.")
        self.g = g
        self.speed = float(speed)

    def level(self, u: Float[ArrayLike, " 2"]) -> Float[Array, ""]:
        """``g(u)``: zero on the cycle, its magnitude the distance-like error of a
        point off it."""
        return self.g(jnp.asarray(u))

    def rhs(self, t, u: Float[Array, " 2"], args=None) -> Float[Array, " 2"]:
        del t, args
        grad = jax.grad(self.level)(u)
        rot = jnp.array([-grad[1], grad[0]])
        return self.speed * (rot - self.level(u) * grad)

    def params(self):
        return {"speed": self.speed}


def guckenheimer_meloon(c: float = 0.07) -> LevelSetCycle:
    """Guckenheimer & Meloon (2000) §5.1: the orbit on ``x² − y² + 2y³/3 + c = 0``,
    ``c ∈ (0, 1/3)``, with their time scale (``s = 1/2``)."""
    if not 0 < c < 1 / 3:
        raise ValueError(
            "c must be in (0, 1/3) for the level set to be a closed orbit."
        )
    return LevelSetCycle(lambda u: u[0] ** 2 - u[1] ** 2 + 2 * u[1] ** 3 / 3 + c, 0.5)


def cassini(a: float = 1.0, b: float = 1.1) -> LevelSetCycle:
    """A Cassini oval ``(x² + y²)² − 2a²(x² − y²) = b⁴ − a⁴``: for ``a < b < a√2`` a
    single non-convex ("peanut") curve with waist at ``y = ±√(b² − a²)``; a horizontal
    line above the waist cuts it four times."""
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
