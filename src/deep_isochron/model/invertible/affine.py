r"""Affine scalar bijections and the coupling layers built from them.

``Shift`` and ``Affine`` are the two simplest scalar bijections in the
``raw``/``constrain`` shape (ADR-0001). As ``CouplingFlow`` templates they give the
additive coupling layer of Dinh et al. 2015 (NICE) and the affine coupling layer of
Dinh et al. 2017 (RealNVP); ``ResidualCoupling`` and ``AffineCoupling`` are thin
constructors for exactly that. Because ``constrain(0)`` is the identity
(ADR-0002) and the default conditioner is zero-initialized, both coupling layers are
the identity map at init.
"""

from collections.abc import Callable
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from .base import AbstractScalarBijection
from .constraints import BoundedPositive, free
from .coupling import CouplingFlow


class ShiftParams(NamedTuple):
    """The constrained parameters of `Shift`: the translation `loc`."""

    loc: Float[Array, ""]


class Shift(AbstractScalarBijection[ShiftParams]):
    """The translation ``x -> x + loc``, with the raw parameter ``(loc,)``
    unconstrained; ``raw = 0`` is the identity."""

    raw: Float[Array, " 1"] | None
    num_params: int = eqx.field(static=True, default=1, init=False)
    smoothness: int | None = eqx.field(static=True, default=None, init=False)

    def __init__(self, *, raw: Float[Array, " 1"] | None = None):
        self.raw = self._init_raw(jnp.zeros(1) if raw is None else raw)

    def constrain(self, raw) -> ShiftParams:
        """``loc = raw[0]``."""
        (loc,) = raw
        return ShiftParams(loc=free(loc))

    @classmethod
    def from_constrained(cls, loc) -> "Shift":
        """The shift by `loc`."""
        return cls(raw=jnp.stack([jnp.asarray(loc)]))

    def __call__(self, x: Float[Array, ""]) -> Float[Array, ""]:
        """``x + loc``."""
        return x + self.params.loc

    def inverse(self, y: Float[Array, ""]) -> Float[Array, ""]:
        """``y - loc``."""
        return y - self.params.loc


class AffineParams(NamedTuple):
    """The constrained parameters of `Affine`: the translation `loc` and the
    positive `scale`."""

    loc: Float[Array, ""]
    scale: Float[Array, ""]


class Affine(AbstractScalarBijection[AffineParams]):
    r"""The affine map ``x -> scale * x + loc`` with a bounded positive scale.

    The raw parameters are ``(loc, log_scale)``. The scale is
    ``exp(clamp * tanh(log_scale / clamp))``, `BoundedPositive` with ``a = clamp``, so
    it lies in ``(exp(-clamp), exp(clamp))``: the soft clamping of RealNVP-style
    flows, which stops a large conditioner output from driving the scale to 0 or
    infinity. ``raw = 0`` is the identity, ``loc = 0`` and ``scale = 1``.

    **Arguments:**

    - `raw`: The raw parameters; the identity when omitted.
    - `clamp`: The log-scale bound ``a``, positive; 2 by default, a scale in
        ``(0.135, 7.39)``.

    **Raises:** `ValueError` for a non-positive `clamp`.
    """

    raw: Float[Array, " 2"] | None
    clamp: float = eqx.field(static=True)
    num_params: int = eqx.field(static=True, default=2, init=False)
    smoothness: int | None = eqx.field(static=True, default=None, init=False)

    def __init__(self, *, raw: Float[Array, " 2"] | None = None, clamp: float = 2.0):
        if not clamp > 0:
            raise ValueError("clamp must be positive.")
        self.clamp = clamp
        self.raw = self._init_raw(jnp.zeros(2) if raw is None else raw)

    def _scale(self) -> BoundedPositive:
        """The scale's primitive, ``BoundedPositive(0, at_zero=1, a=clamp)``."""
        return BoundedPositive(0.0, at_zero=1.0, a=self.clamp)

    def constrain(self, raw) -> AffineParams:
        """``loc`` free and ``scale`` through `_scale`."""
        loc, log_scale = raw
        return AffineParams(loc=free(loc), scale=self._scale()(log_scale))

    @classmethod
    def from_constrained(cls, loc, scale, *, clamp: float = 2.0) -> "Affine":
        """The map with the constrained `loc` and `scale`; `scale` must lie in the
        image of the clamp."""
        raw = jnp.stack(
            [
                jnp.asarray(loc),
                BoundedPositive(0.0, at_zero=1.0, a=clamp).inverse(scale),
            ]
        )
        return cls(raw=raw, clamp=clamp)

    def __call__(self, x: Float[Array, ""]) -> Float[Array, ""]:
        """``scale * x + loc``."""
        p = self.params
        return p.scale * x + p.loc

    def inverse(self, y: Float[Array, ""]) -> Float[Array, ""]:
        """``(y - loc) / scale``."""
        p = self.params
        return (y - p.loc) / p.scale


def ResidualCoupling(  # noqa: N802 — constructor-like factory, keeps the class name
    dim: int,
    split_idx: int | None = None,
    flip: bool = False,
    mlp_width: int = 10,
    mlp_depth: int = 1,
    activation: Callable = jax.nn.gelu,
    dtype=None,
    *,
    key: PRNGKeyArray,
) -> CouplingFlow[Shift]:
    """The additive coupling layer ``y_coupled = x_coupled + t(x_const)`` of Dinh et
    al. 2015: a `CouplingFlow` with a `Shift` template, the identity at init. The
    arguments are those of `CouplingFlow`."""
    return CouplingFlow(
        dim,
        Shift(),
        split_idx=split_idx,
        flip=flip,
        mlp_width=mlp_width,
        mlp_depth=mlp_depth,
        activation=activation,
        dtype=dtype,
        key=key,
    )


def AffineCoupling(  # noqa: N802 — constructor-like factory, keeps the class name
    dim: int,
    split_idx: int | None = None,
    flip: bool = False,
    clamp: float = 2.0,
    mlp_width: int = 10,
    mlp_depth: int = 1,
    activation: Callable = jax.nn.gelu,
    dtype=None,
    *,
    key: PRNGKeyArray,
) -> CouplingFlow[Affine]:
    """The affine coupling layer ``y_coupled = s(x_const) * x_coupled + t(x_const)``
    of Dinh et al. 2017, with ``log s`` soft-clamped: a `CouplingFlow` with an
    ``Affine(clamp)`` template, the identity at init. The other arguments are those of
    `CouplingFlow`."""
    return CouplingFlow(
        dim,
        Affine(clamp=clamp),
        split_idx=split_idx,
        flip=flip,
        mlp_width=mlp_width,
        mlp_depth=mlp_depth,
        activation=activation,
        dtype=dtype,
        key=key,
    )
