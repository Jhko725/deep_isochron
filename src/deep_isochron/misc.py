import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float


def inv_softplus(x: Float[ArrayLike, "*shape"]) -> Float[Array, "*shape"]:
    return jnp.log(jnp.expm1(x))
