r"""Cross-sections: the hyperplanes a Poincaré map is defined on.

A cross-section is a hyperplane
$\Sigma = \lbrace x : \langle n, x - p\rangle = 0\rbrace$ given by a point $p$ and a
unit normal $n$. The one used throughout ``analysis.ode``
passes through a point of the orbit, or near it, and is normal to the flow there,
$n = f(p)/\Vert f(p)\Vert$: it is transversal by construction, with the largest
possible margin, and needs nothing but the vector field (Parker & Chua 1989, §5.5,
"Choosing Σ"; Guckenheimer & Meloon 2000, §2; BifurcationKit's ``SectionSS``
default). The unit normal keeps the shooting phase condition
$\langle n, x_1 - p\rangle = 0$ in state units, commensurate with the matching
residuals.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float

from ...systems import AbstractODE


class CrossSection(eqx.Module):
    r"""The hyperplane $\Sigma = \lbrace x : \langle n, x - p\rangle = 0\rbrace$ with
    unit normal $n$.

    The positive direction of a crossing (Parker & Chua 1989, §2.1.2 and §5.5,
    footnote 4) is the one along which `signed_distance` increases.

    **Arguments:**

    - `point`: $p$, shape ``(dim,)``.
    - `normal`: Any nonzero vector along $n$; it is normalized.

    **Attributes:**

    - `point`, `normal`: As above, with `normal` a unit vector.

    **Raises:**

    `ValueError` unless both are vectors of one dimension and the normal is nonzero.
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
        r"""The section through `point` whose normal is the flow direction of `ode`
        there, $n = f(p)/\Vert f(p)\Vert$: one of the ways to choose a section for a
        limit-cycle search (Parker & Chua 1989, §5.5, "Choosing Σ").

        **Arguments:**

        - `ode`: The system.
        - `point`: $p$, shape ``(dim,)``.
        - `t`: The time at which the field is evaluated; irrelevant for an
            autonomous system.

        **Raises:**

        `ValueError` at an equilibrium, where there is no flow direction to be normal
        to.
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
        """The state dimension."""
        return self.normal.shape[0]

    def signed_distance(self, x: Float[Array, " dim"]) -> Float[Array, ""]:
        r"""$\langle n, x - p\rangle$: positive on the side the normal points to."""
        return jnp.dot(self.normal, x - self.point)

    def flip(self) -> CrossSection:
        """The same hyperplane with the opposite orientation."""
        return CrossSection(self.point, -self.normal)


__all__ = ["CrossSection"]
