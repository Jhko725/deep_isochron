import math

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox._misc import default_floating_dtype
from equinox.nn._misc import default_init
from jaxtyping import Array, Float, PRNGKeyArray

from .base import AbstractBijection


class InvertibleLinear(AbstractBijection):
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
        self.bias = jnp.zeros((dim,), dtype=dtype) if use_bias else None
        # self.bias = default_init(key_b, (dim,), dtype, lim) if use_bias else None
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


class BiLipschitzLinear(AbstractBijection):
    _U: Float[Array, "{self.dim} {self.dim}"]
    _V: Float[Array, "{self.dim} {self.dim}"]
    _s: Float[Array, " {self.dim}"]
    bias: Float[Array, " {self.dim}"] | None

    dim: int = eqx.field(static=True)
    max_lipschitz: float = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        max_lipschitz: float,
        dtype=None,
        use_bias: bool = True,
        *,
        key: PRNGKeyArray,
    ):
        dtype = default_floating_dtype() if dtype is None else dtype

        key_u, key_v, key_b = jax.random.split(key, 3)
        lim = 1 / math.sqrt(dim)

        _U = default_init(key_u, (dim, dim), dtype, lim)
        _V = default_init(key_v, (dim, dim), dtype, lim)
        self._U = jnp.triu(_U, k=1)
        self._V = jnp.triu(_V, k=1)

        if max_lipschitz < 1:
            raise ValueError("Maximum Lipschitz constant cannot be smaller than 1.")
        self.max_lipschitz = L = max_lipschitz
        self._s = jax.scipy.special.logit(jnp.asarray(1 / (1 + L), dtype=dtype))

        self.bias = jnp.zeros((dim,), dtype=dtype) if use_bias else None
        # self.bias = default_init(key_b, (dim,), dtype, lim) if use_bias else None
        self.dim = dim

    @property
    def U(self) -> Float[Array, "{self.dim} {self.dim}"]:
        return jax.scipy.linalg.expm(self._U - self._U.T)

    @property
    def V(self) -> Float[Array, "{self.dim} {self.dim}"]:
        return jax.scipy.linalg.expm(self._V - self._V.T)

    @property
    def s(self) -> Float[Array, " {self.dim}"]:
        L = self.max_lipschitz
        return jax.nn.sigmoid(self._s) * (L - 1 / L) + 1 / L

    @property
    def weight(self) -> Float[Array, "{self.dim} {self.dim}"]:
        return (self.U * self.s) @ self.V.T

    def __call__(self, x: Float[Array, " {self.dim}"]) -> Float[Array, " {self.dim}"]:
        y = self.weight @ x
        if self.bias is not None:
            y = y + self.bias
        return y

    def inverse(self, y: Float[Array, " {self.dim}"]) -> Float[Array, " {self.dim}"]:
        if self.bias is not None:
            y = y - self.bias
        weight_inv = (self.V * (1 / self.s)) @ self.U.T
        return weight_inv @ y
