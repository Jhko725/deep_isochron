r"""Cross-sections: the hyperplanes a Poincaré map is defined on.

A cross-section is a hyperplane
$\Sigma = \lbrace x : \langle n, x - p\rangle = 0\rbrace$ given by a point $p$ and a
*unit* normal $n$. The one used throughout ``analysis.ode``
is the hyperplane through a point of (or near) the orbit, **normal to the flow there**,
$n = f(p)/\Vert f(p)\Vert$: it is transversal by construction, with the largest
possible margin, and needs nothing but the vector field. Parker & Chua 1989, §2.1.2
for cross-sections and the direction of a crossing; §5.5, "Choosing Σ": "let the
program automatically set $h = f(x^{(0)})$ and $x_\Sigma = x^{(0)}$"); Guckenheimer &
Meloon (2000, §2: cross-sections "defined in the algorithms as hyperplanes
perpendicular to the vector field at a point"); BifurcationKit's ``SectionSS`` default
(normal ``F(x₁)/‖F(x₁)‖``, center ``x₁``).

The unit normal matters for the shooting phase condition
$\langle n, x_1 - p\rangle = 0$: it keeps that residual in state units, commensurate
with the matching residuals.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float

from ...systems import AbstractODE


class CrossSection(eqx.Module):
    r"""The hyperplane $\Sigma = \lbrace x : \langle n, x - p\rangle = 0\rbrace$ with
    unit normal $n$.

    ``signed_distance(x)`` is $\langle n, x - p\rangle$: positive on the side the normal
    points to. The "positive direction" of a crossing (Parker & Chua §2.1.2; §5.5,
    footnote 4) is the one along which this distance increases.
    """

    point: Float[Array, " dim"]
    normal: Float[Array, " dim"]

    def __init__(self, point: ArrayLike, normal: ArrayLike):
        point = jnp.asarray(point, dtype=float)
        normal = jnp.asarray(normal, dtype=float)
        if point.shape != normal.shape or point.ndim != 1:
            raise ValueError("point and normal must be vectors of the same dimension.")
        norm = jnp.linalg.norm(normal)
        if not bool(norm > 0):
            raise ValueError("the normal must be nonzero.")
        self.point = point
        self.normal = normal / norm

    @classmethod
    def normal_to_flow(
        cls, ode: AbstractODE, point: ArrayLike, t: float = 0.0
    ) -> CrossSection:
        r"""The section through ``point`` whose normal is the flow direction there,
        $n = f(p)/\Vert f(p)\Vert$. Raises ``ValueError`` at an equilibrium (no flow
        direction to be normal to).

        This is one of the methods to choose a Poincare section for finding limit
        cycles, as discussed in Parker & Chua §5.5.2 (Choosing $\Sigma$).
        """
        point = jnp.asarray(point, dtype=float)
        f = ode.rhs(t, point)
        if not bool(jnp.linalg.norm(f) > 0):
            raise ValueError(
                "the vector field vanishes at point: no flow-normal section."
            )
        return cls(point, f)

    @property
    def dim(self) -> int:
        return self.normal.shape[0]

    def signed_distance(self, x: Float[Array, " dim"]) -> Float[Array, ""]:
        return jnp.dot(self.normal, x - self.point)

    def flip(self) -> CrossSection:
        """The same hyperplane with the opposite orientation."""
        return CrossSection(self.point, -self.normal)


__all__ = ["CrossSection"]
