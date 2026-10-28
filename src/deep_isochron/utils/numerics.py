"""Numerical helpers shared by the invertible constraints, the normal forms and the
analysis: the bounded positive scale transforms, the polar-to-cartesian change of
coordinates and the wrapping of a phase."""

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float


def inv_softplus(x: Float[ArrayLike, "*shape"]) -> Float[Array, "*shape"]:
    """The inverse of ``softplus``, ``log(exp(x) - 1)``, for ``x > 0``."""
    return jnp.log(jnp.expm1(x))


def squashed_exp(x, a: float = 2.0):
    """The bounded positive scale transform ``exp(a * tanh(x / a))``.

    It is strictly increasing and bounded away from both 0 and infinity, with range
    ``(exp(-a), exp(a))``, about ``(0.135, 7.39)`` for ``a = 2``. Unlike ``softplus``,
    bounded below but not above, a large positive conditioner output cannot drive the
    scale to infinity, which removes the overflow mechanism for the cubic generator's
    ``a`` and ``b``.
    """
    return jnp.exp(a * jnp.tanh(x / a))


def inv_squashed_exp(y, a: float = 2.0):
    """The ``x`` with ``squashed_exp(x, a) == y``, for `y` in ``(exp(-a), exp(a))``."""
    return a * jnp.arctanh(jnp.log(y) / a)


def cartesian_to_polar(xy: Float[Array, " 2 *rest"]) -> Float[Array, " 2 *rest"]:
    """``(r, θ)`` of ``(x, y)``, with ``θ`` in ``(-π, π]``."""
    x, y = xy
    return jnp.stack((jnp.hypot(x, y), jnp.arctan2(y, x)))


def polar_to_cartesian(rtheta: Float[Array, " 2 *rest"]) -> Float[Array, " 2 *rest"]:
    """``(x, y)`` of ``(r, θ)``."""
    r, theta = rtheta
    return jnp.stack((r * jnp.cos(theta), r * jnp.sin(theta)))


def wrap_phase(angle: Float[ArrayLike, "*shape"]) -> Float[Array, "*shape"]:
    r"""`angle` wrapped to $(-\pi, \pi]$, with $-\pi$ sent to $\pi$.

    The truncated remainder by $2\pi$ is exact, and so is the one correction by
    $\pm 2\pi$ that follows (Sterbenz's lemma), so the result equals
    ``math.remainder(angle, 2π)`` bit for bit away from the ties at $\pm\pi$. The
    derivative is 1.
    """
    angle = jnp.asarray(angle, dtype=float)
    two_pi = jnp.asarray(2 * jnp.pi, dtype=angle.dtype)
    rem = jax.lax.rem(angle, two_pi)
    rem = jnp.where(rem > jnp.pi, rem - two_pi, rem)
    return jnp.where(rem <= -jnp.pi, rem + two_pi, rem)


def arclength(points: Float[ArrayLike, "n dim"]) -> Float[Array, " n"]:
    """The cumulative arclength along the polyline through `points`, 0 at the first;
    shape ``(n,)``."""
    points = jnp.asarray(points, dtype=float)
    steps = jnp.linalg.norm(jnp.diff(points, axis=0), axis=1)
    return jnp.concatenate([jnp.zeros((1,), dtype=points.dtype), jnp.cumsum(steps)])
