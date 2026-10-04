r"""The linear layer of the INN vocabulary.

``BiLipschitzLinear`` is the *only* linear bijection: ``W = U diag(s) Vᵀ`` with ``U, V``
in ``SO(dim)`` (matrix exponentials of skew-symmetric generators) and singular values
``s`` in ``(1/L, L)``. Every point of the parameter space is therefore an
orientation-preserving linear map with condition number below ``L²``, and the
unconstrained leaves can be optimized freely. An unconstrained matrix ``W`` (the former
``InvertibleLinear``) can cross ``det W = 0`` during training, and with it both
invertibility and orientation; nothing an INN of conjugacies needs is lost by removing
it, since ``L`` can be made as large as wanted. See the change document
``docs/changes/2026-10-02-invertible-cleanup.md`` (A4).

The parametrization follows ADR-0001 in spirit: the leaves ``raw_U``, ``raw_V``,
``raw_s`` are unconstrained, and the constrained ``LinearParams(U, V, s)`` are computed
on read by ``params``. All-zero leaves give ``U = V = I``, ``s = 1``: the identity map
(ADR-0002), which is the default initialization.
"""

import math
from typing import Literal, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox._misc import default_floating_dtype
from equinox.nn._misc import default_init
from jaxtyping import Array, Float, PRNGKeyArray

from .base import AbstractBijection
from .constraints import Interval


class LinearParams(NamedTuple):
    U: Float[Array, "dim dim"]
    V: Float[Array, "dim dim"]
    s: Float[Array, " dim"]


class BiLipschitzLinear(AbstractBijection):
    r"""Bi-Lipschitz linear layer ``x -> U diag(s) Vᵀ x + bias``, as introduced in [1].

    ``U = expm(A - Aᵀ)`` and ``V = expm(B - Bᵀ)`` are rotations; ``s`` lies in
    ``(1/L, L)`` (``Interval`` with ``at_zero = 1``), so the layer and its inverse are
    both ``L``-Lipschitz and ``det W > 0`` everywhere in parameter space.

    **Arguments:**

    - ``dim``: dimension.
    - ``max_lipschitz``: ``L > 1``.
    - ``init``: ``"identity"`` (default; all raw leaves zero, so the layer is the
      identity map) or ``"rotation"`` (Gaussian skew generators of scale
      ``1/sqrt(dim)``, giving random rotations ``U``, ``V``; ``s = 1``). The rotation is
      *not* Haar-distributed; it is the former default initialization, kept for
      experiments that want a random orthogonal mixing at init.
    - ``dtype``, ``use_bias``, ``key``: as in ``eqx.nn.Linear``. ``key`` is only drawn
      from for ``init="rotation"`` but is always required, for a uniform constructor
      signature across layers.

    [1] D. A. Serino et al. Fast-slow neural networks for learning singularly perturbed
        dynamical systems. J. Comput. Phys. 537, 114090 (2025).
    """

    raw_U: Float[Array, "{self.dim} {self.dim}"]
    raw_V: Float[Array, "{self.dim} {self.dim}"]
    raw_s: Float[Array, " {self.dim}"]
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
        init: Literal["identity", "rotation"] = "identity",
        *,
        key: PRNGKeyArray,
    ):
        dtype = default_floating_dtype() if dtype is None else dtype
        if not max_lipschitz > 1:
            # L = 1 would leave no room for s: Interval(1/L, L) needs 1/L < 1 < L.
            raise ValueError("Maximum Lipschitz constant must be greater than 1.")
        self.max_lipschitz = max_lipschitz
        self.dim = dim

        if init == "identity":
            self.raw_U = jnp.zeros((dim, dim), dtype=dtype)
            self.raw_V = jnp.zeros((dim, dim), dtype=dtype)
        elif init == "rotation":
            key_u, key_v = jax.random.split(key)
            lim = 1 / math.sqrt(dim)
            self.raw_U = jnp.triu(default_init(key_u, (dim, dim), dtype, lim), k=1)
            self.raw_V = jnp.triu(default_init(key_v, (dim, dim), dtype, lim), k=1)
        else:
            raise ValueError(f"init must be 'identity' or 'rotation', got {init!r}.")
        self.raw_s = jnp.zeros((dim,), dtype=dtype)
        self.bias = jnp.zeros((dim,), dtype=dtype) if use_bias else None

    def constrain(self, raw_U, raw_V, raw_s) -> LinearParams:
        """Unconstrained leaves -> ``(U, V, s)``; all zeros give the identity."""
        L = self.max_lipschitz
        return LinearParams(
            U=jax.scipy.linalg.expm(raw_U - raw_U.T),
            V=jax.scipy.linalg.expm(raw_V - raw_V.T),
            s=Interval(1 / L, L, at_zero=1.0)(raw_s),
        )

    @property
    def params(self) -> LinearParams:
        return self.constrain(self.raw_U, self.raw_V, self.raw_s)

    @property
    def weight(self) -> Float[Array, "{self.dim} {self.dim}"]:
        p = self.params
        return (p.U * p.s) @ p.V.T

    def __call__(self, x: Float[Array, " {self.dim}"]) -> Float[Array, " {self.dim}"]:
        y = self.weight @ x
        if self.bias is not None:
            y = y + self.bias
        return y

    def inverse(self, y: Float[Array, " {self.dim}"]) -> Float[Array, " {self.dim}"]:
        if self.bias is not None:
            y = y - self.bias
        p = self.params
        weight_inv = (p.V * (1 / p.s)) @ p.U.T
        return weight_inv @ y
