import math

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox._misc import default_floating_dtype
from equinox.nn._misc import default_init
from jaxtyping import Array, Float, PRNGKeyArray

from .base import AbstractInvertibleTransform


class InvertibleLinear(AbstractInvertibleTransform):
    weight: Float[Array, "{self.dim} {self.dim}"]
    bias: Float[Array, " {self.dim}"] | None

    dim: int = eqx.field(static=True)

    # TODO: implement orthogonal initialization (or identity initialization?)
    def __init__(
        self, dim: int, dtype=None, use_bias: bool = True, *, key: PRNGKeyArray
    ):
        dtype = default_floating_dtype() if dtype is None else dtype

        key_w, key_b = jax.random.split(key)
        lim = 1 / math.sqrt(dim)

        weight = default_init(key_w, (dim, dim), dtype, lim)
        self.weight = jnp.linalg.qr(weight)[0]
        self.bias = default_init(key_b, (dim,), dtype, lim) if use_bias else None
        self.dim = dim

    def __call__(self, x: Float[Array, " {self.dim}"]) -> Float[Array, " {self.dim}"]:
        y = self.weight @ x
        if self.bias is not None:
            y = y + self.bias
        return y

    def inverse(self, y: Float[Array, " {self.dim}"]) -> Float[Array, " {self.dim}"]:
        if self.bias is not None:
            y = y - self.bias
        return jnp.linalg.solve(self.weight, y)
