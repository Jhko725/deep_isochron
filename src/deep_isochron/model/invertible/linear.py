r"""The linear layer of the INN vocabulary.

``BiLipschitzLinear`` is the only linear bijection: ``W = U diag(s) Vᵀ`` with ``U`` and
``V`` in ``SO(dim)``, Cayley transforms of skew-symmetric generators (ADR-0010), and
singular values ``s`` in ``(1/L, L)``. Every point of the parameter space is therefore
an orientation-preserving linear map with condition number below ``L²``, and the
unconstrained leaves can be optimized freely. An unconstrained matrix, the former
``InvertibleLinear``, can cross ``det W = 0`` during training and lose invertibility and
orientation with it; nothing an INN of conjugacies needs is lost by removing it, since
``L`` can be made as large as wanted (change document 2026-10-02, A4). The leaves
``raw_U``, ``raw_V`` and ``raw_s`` are unconstrained and the constrained
``LinearParams`` are computed on read, in the spirit of ADR-0001; all-zero leaves give
the identity map (ADR-0002), the default initialization.
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
    """The constrained parameters of `BiLipschitzLinear`: the rotations `U` and `V`
    and the singular values `s`."""

    U: Float[Array, "dim dim"]
    V: Float[Array, "dim dim"]
    s: Float[Array, " dim"]


def cayley(skew: Float[Array, "dim dim"]) -> Float[Array, "dim dim"]:
    r"""The Cayley transform ``Q = (I - A)(I + A)⁻¹`` of the skew-symmetric `skew`.

    ``Q`` is orthogonal with ``det Q = +1`` for every skew ``A``, since the eigenvalues
    of ``I + A`` are ``1 + iλ`` and the solve is always well conditioned; ``Q = I`` at
    ``A = 0`` (ADR-0002); and the map is smooth and loop-free. It reaches every
    rotation except those with an eigenvalue ``-1``, a half turn in a plane, which no
    training trajectory from the identity crosses. It replaces
    ``jax.scipy.linalg.expm`` (ADR-0010), a 16-step scan of ``lax.cond`` whose
    predicates the GPU backend reads back from the device one round trip each, 0.65 ms
    per 2×2 call on a V100 against 0.05 ms for a closed form. For ``dim = 2`` the solve
    is written out: with ``a = A[1, 0]``, ``Q`` is the rotation by ``-2 arctan a``.
    """
    n = skew.shape[-1]
    if n == 2:
        a = skew[1, 0]
        den = 1 + a * a
        c, s = (1 - a * a) / den, 2 * a / den
        return jnp.array([[c, s], [-s, c]], dtype=skew.dtype)
    eye = jnp.eye(n, dtype=skew.dtype)
    return jnp.linalg.solve(eye + skew, eye - skew)


class BiLipschitzLinear(AbstractBijection):
    r"""The bi-Lipschitz linear layer ``x -> U diag(s) Vᵀ x + bias`` of Serino et al.
    2025.

    ``U = cayley(A - Aᵀ)`` and ``V = cayley(B - Bᵀ)`` are rotations and ``s`` lies in
    ``(1/L, L)``, an `Interval` with ``at_zero = 1``, so the layer and its inverse are
    both ``L``-Lipschitz and ``det W > 0`` everywhere in parameter space.

    **Arguments:**

    - `dim`: The dimension.
    - `max_lipschitz`: ``L > 1``.
    - `dtype`, `use_bias`: As in ``eqx.nn.Linear``.
    - `init`: ``"identity"``, every raw leaf zero so that the layer is the identity
        map, or ``"rotation"``, Gaussian skew generators of scale ``1/sqrt(dim)``
        giving random rotations ``U`` and ``V`` with ``s = 1``. The rotation is not
        Haar-distributed; it is the former default initialization, kept for
        experiments that want a random orthogonal mixing at init.
    - `key`: Drawn from only for ``init="rotation"``, but always required, for a
        uniform constructor signature across layers.

    **Attributes:**

    - `raw_U`, `raw_V`: The skew generators' raw leaves, shape ``(dim, dim)``.
    - `raw_s`: The singular values' raw leaf, shape ``(dim,)``.
    - `bias`: The bias, or ``None``.
    - `max_lipschitz`: ``L`` (static).

    **Raises:**

    `ValueError` for ``max_lipschitz <= 1``, which leaves no room for ``s``, or an
    unknown `init`."""

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
        """The constrained ``(U, V, s)`` of the raw leaves; all zeros give the
        identity."""
        L = self.max_lipschitz
        return LinearParams(
            U=cayley(raw_U - raw_U.T),
            V=cayley(raw_V - raw_V.T),
            s=Interval(1 / L, L, at_zero=1.0)(raw_s),
        )

    @property
    def params(self) -> LinearParams:
        """The constrained parameters of this instance."""
        return self.constrain(self.raw_U, self.raw_V, self.raw_s)

    @property
    def weight(self) -> Float[Array, "{self.dim} {self.dim}"]:
        """The matrix ``W = U diag(s) Vᵀ``."""
        p = self.params
        return (p.U * p.s) @ p.V.T

    def __call__(self, x: Float[Array, " {self.dim}"]) -> Float[Array, " {self.dim}"]:
        """``W x + bias``."""
        y = self.weight @ x
        if self.bias is not None:
            y = y + self.bias
        return y

    def inverse(self, y: Float[Array, " {self.dim}"]) -> Float[Array, " {self.dim}"]:
        """``W⁻¹ (y - bias)``, with ``W⁻¹ = V diag(1/s) Uᵀ``."""
        if self.bias is not None:
            y = y - self.bias
        p = self.params
        weight_inv = (p.V * (1 / p.s)) @ p.U.T
        return weight_inv @ y
