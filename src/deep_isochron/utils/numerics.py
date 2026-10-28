"""Numerical helpers shared by the invertible constraints, the normal forms and the
analysis: the bounded positive scale transforms, the polar-to-cartesian change of
coordinates and the wrapping of a phase, which is written against the array API so that
it serves NumPy arrays on the host as well as JAX arrays under ``jit`` and ``vmap``."""

import math
from typing import Any

import jax.numpy as jnp
import numpy as np
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


def wrap_phase(
    angle: Float[ArrayLike, "*shape"],
) -> Float[Array, "*shape"] | Float[np.ndarray, "*shape"]:
    r"""`angle` wrapped to $(-\pi, \pi]$, with $-\pi$ sent to $\pi$.

    The truncated remainder by $2\pi$ is exact, and so is the one correction by
    $\pm 2\pi$ that follows (Sterbenz's lemma), so the result equals
    ``math.remainder(angle, 2π)`` bit for bit away from the ties at $\pm\pi$. The
    derivative is 1.

    Written against the array API (``__array_namespace__``): a JAX array or tracer is
    wrapped with ``jax.numpy``, under ``jit``, ``vmap`` and ``grad``, and a NumPy
    array with ``numpy``, in its own dtype whatever ``jax_enable_x64`` is; a Python
    scalar goes through NumPy. The truncated remainder is ``fmod`` in both
    namespaces, which the standard does not name but both provide.
    """
    array: Array | np.ndarray = (
        angle
        if isinstance(angle, Array | np.ndarray)
        else np.asarray(angle, dtype=float)
    )
    xp: Any = array.__array_namespace__()
    two_pi = xp.asarray(2 * math.pi, dtype=array.dtype)
    rem = xp.fmod(array, two_pi)
    rem = xp.where(rem > math.pi, rem - two_pi, rem)
    return xp.where(rem <= -math.pi, rem + two_pi, rem)


def arclength(points: Float[ArrayLike, "n dim"]) -> Float[Array, " n"]:
    """The cumulative arclength along the polyline through `points`, 0 at the first;
    shape ``(n,)``."""
    points = jnp.asarray(points, dtype=float)
    steps = jnp.linalg.norm(jnp.diff(points, axis=0), axis=1)
    return jnp.cumulative_sum(steps, include_initial=True)
