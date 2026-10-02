r"""Affine scalar bijections and the coupling layers built from them.

``Shift`` and ``Affine`` are the two simplest scalar bijections in the ``raw`` /
``constrain`` shape (ADR-0001). Used as ``CouplingFlow`` templates they give the classic
additive (NICE, [1]) and affine (RealNVP, [2]) coupling layers; ``ResidualCoupling`` and
``AffineCoupling`` below are thin constructors for exactly that, replacing the former
standalone classes that duplicated ``CouplingFlow``'s split/flip/MLP logic.

Because ``constrain(0)`` is the identity (ADR-0002) and the default conditioner is
zero-initialised, both coupling layers are the identity map at init. (The former
``ResidualCoupling`` was not: its MLP had no zeroed final layer.)

[1] L. Dinh, D. Krueger, Y. Bengio. NICE: Non-linear Independent Components Estimation.
    ICLR workshop (2015).
[2] L. Dinh, J. Sohl-Dickstein, S. Bengio. Density estimation using Real NVP. ICLR
    (2017).
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
    loc: Float[Array, ""]


class Shift(AbstractScalarBijection[ShiftParams]):
    """Translation ``x -> x + loc``.

    Raw parameters: ``(loc,)``, unconstrained; ``raw = 0`` is the identity.
    """

    raw: Float[Array, " 1"] | None
    num_params: int = eqx.field(static=True, default=1, init=False)
    smoothness: int | None = eqx.field(static=True, default=None, init=False)

    def __init__(self, *, raw: Float[Array, " 1"] | None = None):
        self.raw = self._init_raw(jnp.zeros(1) if raw is None else raw)

    def constrain(self, raw) -> ShiftParams:
        (loc,) = raw
        return ShiftParams(loc=free(loc))

    @classmethod
    def from_constrained(cls, loc) -> "Shift":
        return cls(raw=jnp.stack([jnp.asarray(loc)]))

    def __call__(self, x: Float[Array, ""]) -> Float[Array, ""]:
        return x + self.params.loc

    def inverse(self, y: Float[Array, ""]) -> Float[Array, ""]:
        return y - self.params.loc


class AffineParams(NamedTuple):
    loc: Float[Array, ""]
    scale: Float[Array, ""]


class Affine(AbstractScalarBijection[AffineParams]):
    r"""Affine map ``x -> scale * x + loc`` with a bounded positive scale.

    Raw parameters: ``(loc, log_scale)``. The scale is ``exp(clamp * tanh(log_scale /
    clamp))`` (``BoundedPositive`` with ``a = clamp``), so it lies in ``(exp(-clamp),
    exp(clamp))``: the soft clamping of RealNVP-style flows, which stops a large
    conditioner output from driving the scale to ``0`` or infinity. ``raw = 0`` is the
    identity (``loc = 0``, ``scale = 1``).

    **Arguments:**

    - ``clamp``: log-scale bound ``a``; the scale is bounded by ``exp(±a)``. Default
      ``2`` (scale in ``(0.135, 7.39)``).
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
        return BoundedPositive(0.0, at_zero=1.0, a=self.clamp)

    def constrain(self, raw) -> AffineParams:
        loc, log_scale = raw
        return AffineParams(loc=free(loc), scale=self._scale()(log_scale))

    @classmethod
    def from_constrained(cls, loc, scale, *, clamp: float = 2.0) -> "Affine":
        raw = jnp.stack(
            [
                jnp.asarray(loc),
                BoundedPositive(0.0, at_zero=1.0, a=clamp).inverse(scale),
            ]
        )
        return cls(raw=raw, clamp=clamp)

    def __call__(self, x: Float[Array, ""]) -> Float[Array, ""]:
        p = self.params
        return p.scale * x + p.loc

    def inverse(self, y: Float[Array, ""]) -> Float[Array, ""]:
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
    """Additive coupling layer ``y_coupled = x_coupled + t(x_const)`` (NICE):
    ``CouplingFlow`` with a ``Shift`` template. Identity at init."""
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
    """Affine coupling layer ``y_coupled = s(x_const) * x_coupled + t(x_const)``
    (RealNVP, with soft-clamped ``log s``): ``CouplingFlow`` with an ``Affine(clamp)``
    template. Identity at init."""
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
