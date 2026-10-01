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
    smoothness: int | None = eqx.field(static=True, default=None, init=False)

    def __init__(
        self, dim: int, dtype=None, use_bias: bool = True, *, key: PRNGKeyArray
    ):
        """Initialised as a random *rotation* (Haar-distributed on SO(dim)) with zero
        bias. The QR factor of a Gaussian matrix is only orthogonal: Householder QR
        returns a reflection (det = -1) essentially always, so the signs of R's diagonal
        are folded into Q and the last column is flipped if det is still negative; an
        INN should start orientation-preserving unless asked otherwise."""
        dtype = default_floating_dtype() if dtype is None else dtype

        lim = 1 / math.sqrt(dim)
        q, r = jnp.linalg.qr(default_init(key, (dim, dim), dtype, lim))
        q = q * jnp.sign(jnp.diag(r))  # Haar-distributed on O(dim)
        q = q.at[:, -1].multiply(jnp.sign(jnp.linalg.det(q)))  # ... and on SO(dim)
        self.weight = q
        self.bias = jnp.zeros((dim,), dtype=dtype) if use_bias else None
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
    """BiLipschitz linear layer, as introduced in [1].

    [1]: D. A. Serino et al. Fast-slow neural networks for learning singularly perturbed
     dynamical systms. J. Comput. Phys. 537, 114090 (2025)."""

    _U: Float[Array, "{self.dim} {self.dim}"]
    _V: Float[Array, "{self.dim} {self.dim}"]
    _s: Float[Array, " {self.dim}"]
    """Unconstrained singular values; ``s = sigmoid(_s) * (L - 1/L) + 1/L`` in (1/L, L)."""
    bias: Float[Array, " {self.dim}"] | None

    dim: int = eqx.field(static=True)
    smoothness: int | None = eqx.field(static=True, default=None, init=False)
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

        key_u, key_v = jax.random.split(key)
        lim = 1 / math.sqrt(dim)

        _U = default_init(key_u, (dim, dim), dtype, lim)
        _V = default_init(key_v, (dim, dim), dtype, lim)
        self._U = jnp.triu(_U, k=1)
        self._V = jnp.triu(_V, k=1)

        if max_lipschitz < 1:
            raise ValueError("Maximum Lipschitz constant cannot be smaller than 1.")
        self.max_lipschitz = L = max_lipschitz
        # s = 1 at init: sigmoid(_s) * (L - 1/L) + 1/L == 1  <=>  sigmoid(_s) = 1/(1+L)
        self._s = jnp.full(
            (dim,), math.log((1 / (1 + L)) / (1 - 1 / (1 + L))), dtype=dtype
        )

        self.bias = jnp.zeros((dim,), dtype=dtype) if use_bias else None
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
