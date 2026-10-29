"""Numerical helpers shared by the invertible constraints, the normal forms and the
analysis: the bounded positive scale transforms, the polar-to-cartesian change of
coordinates, and the two phase helpers `wrap_phase` and `phase_difference`, which are
written against the array API so that they serve NumPy arrays on the host as well as
JAX arrays under ``jit`` and ``vmap``."""

from typing import Any, overload

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


def _array_and_namespace(
    x: Float[ArrayLike, "*shape"],
) -> tuple[Array | np.ndarray, Any]:
    """`x` as an array, with its array-API namespace (``__array_namespace__``).

    This prologue is shared by the array-API helpers of this module. A JAX array or
    tracer, and a NumPy array, pass through unchanged, so a JAX input is computed with
    ``jax.numpy`` under ``jit``, ``vmap`` and ``grad``, and a NumPy input with
    ``numpy`` in its own dtype whatever ``jax_enable_x64`` is. Anything else, such as
    a Python scalar, becomes ``np.asarray(x, dtype=float)`` and goes through NumPy.
    """
    array: Array | np.ndarray = (
        x if isinstance(x, Array | np.ndarray) else np.asarray(x, dtype=float)
    )
    return array, array.__array_namespace__()


@overload
def wrap_phase(phase: Float[Array, "*shape"]) -> Float[Array, "*shape"]: ...


@overload
def wrap_phase(
    phase: Float[np.ndarray, "*shape"] | float,
) -> Float[np.ndarray, "*shape"]: ...


def wrap_phase(
    phase: Float[ArrayLike, "*shape"],
) -> Float[Array, "*shape"] | Float[np.ndarray, "*shape"]:
    r"""`phase` wrapped to $[0, 1)$, as ``phase - floor(phase)``.

    The phase is a fraction of the period, so one turn is 1 (ADR-0020, decision 1).
    The floor is exact, and the subtraction that follows rounds once; a negative `phase`
    within half an ulp of 0, smaller in magnitude than about ``1.1e-16``, rounds to 1
    and is sent to 0, so that the range holds. The derivative is 1.

    Written against the array API through `_array_and_namespace`: a JAX array in is a
    JAX array out, and a NumPy array or a Python scalar in is a NumPy array out.
    """
    array, xp = _array_and_namespace(phase)
    r = array - xp.floor(array)
    return xp.asarray(xp.where(r >= 1, xp.zeros_like(r), r))


@overload
def phase_difference(
    a: Float[Array, "*#shape"], b: Float[ArrayLike, "*#shape"]
) -> Float[Array, "*shape"]: ...


@overload
def phase_difference(
    a: Float[ArrayLike, "*#shape"], b: Float[Array, "*#shape"]
) -> Float[Array, "*shape"]: ...


@overload
def phase_difference(
    a: Float[np.ndarray, "*#shape"] | float,
    b: Float[np.ndarray, "*#shape"] | float,
) -> Float[np.ndarray, "*shape"]: ...


def phase_difference(
    a: Float[ArrayLike, "*#shape"],
    b: Float[ArrayLike, "*#shape"],
) -> Float[Array, "*shape"] | Float[np.ndarray, "*shape"]:
    r"""The signed shortest arc from phase `b` to phase `a`, in $(-\tfrac12, \tfrac12]$.

    The result is positive when `a` is ahead of `b`, and a difference of exactly half a
    turn maps to $+\tfrac12$. This helper is the one way to compare two phases
    (ADR-0020, decision 3). The arguments `a` and `b` broadcast, as a phase against an
    offset. The derivative with respect to `a` is 1 away from the seam
    at half a turn.

    Each argument goes through `_array_and_namespace` before the difference ``a - b``
    is formed, so a JAX array in either argument gives a JAX array, under ``jit``,
    ``vmap`` and ``grad``, and NumPy arrays or Python scalars give a NumPy array.
    """
    a_array, _ = _array_and_namespace(a)
    b_array, _ = _array_and_namespace(b)
    wrapped, xp = _array_and_namespace(wrap_phase(a_array - b_array))
    return xp.where(wrapped > 0.5, wrapped - 1, wrapped)


def arclength(points: Float[ArrayLike, "n dim"]) -> Float[Array, " n"]:
    """The cumulative arclength along the polyline through `points`, 0 at the first;
    shape ``(n,)``."""
    points = jnp.asarray(points, dtype=float)
    steps = jnp.linalg.norm(jnp.diff(points, axis=0), axis=1)
    return jnp.cumulative_sum(steps, include_initial=True)
