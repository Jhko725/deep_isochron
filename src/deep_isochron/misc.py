import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float


def inv_softplus(x: Float[ArrayLike, "*shape"]) -> Float[Array, "*shape"]:
    return jnp.log(jnp.expm1(x))


def squashed_exp(x, a: float = 2.0):
    """Bounded positive scale transform: ``exp(a * tanh(x / a))``.

    Strictly increasing and bounded away from both 0 and infinity, with range
    ``(exp(-a), exp(a)) ~ (0.135, 7.39)`` for a=2.  Unlike ``softplus`` (bounded below
    but unbounded above), a large positive conditioner output cannot drive the
    scale to infinity, which removes the overflow/NaN mechanism for the cubic
    generator's ``a, b`` coefficients.
    """
    return jnp.exp(2.0 * jnp.tanh(x / 2.0))


def inv_squashed_exp(y, a: float = 2.0):
    """Raw x such that ``squashed_exp(x) == y`` (for y in (exp(-a), exp(a)))."""
    return a * jnp.arctanh(jnp.log(y) / a)


def cartesian_to_polar(xy: Float[Array, " 2 *rest"]) -> Float[Array, " 2 *rest"]:
    x, y = xy
    return jnp.stack((jnp.hypot(x, y), jnp.arctan2(y, x)))
