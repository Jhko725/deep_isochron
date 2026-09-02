from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from .base import AbstractInvertibleTransform


class AffineCoupling(AbstractInvertibleTransform):
    dim: int = eqx.field(static=True)

    split_idx: int
    flip: bool

    s: eqx.nn.MLP
    t: eqx.nn.MLP

    def __init__(
        self,
        dim: int,
        split_idx: int | None = None,
        width_hidden: int = 10,
        depth: int = 1,
        flip: bool = False,
        activation: Callable = jax.nn.gelu,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        self.dim = dim
        self.split_idx = dim // 2 if split_idx is None else split_idx
        self.flip = flip

        in_size, out_size = (
            self.split_sizes if not self.flip else self.split_sizes[::-1]
        )
        key_s, key_t = jax.random.split(key)
        self.s = eqx.nn.MLP(
            in_size=in_size,
            out_size=out_size,
            width_size=width_hidden,
            depth=depth,
            activation=activation,
            dtype=dtype,
            key=key_s,
        )
        self.t = eqx.nn.MLP(
            in_size=in_size,
            out_size=out_size,
            width_size=width_hidden,
            depth=depth,
            activation=activation,
            dtype=dtype,
            key=key_t,
        )

    @property
    def split_sizes(self) -> tuple[int, int]:
        out_sizes = self.split_idx, self.dim - self.split_idx
        return out_sizes

    def __call__(self, x: Float[Array, " dim"]) -> Float[Array, " dim"]:
        x_up, x_down = jnp.split(x, [self.split_idx])

        if not self.flip:
            y_up = x_up
            y_down = (x_down + self.t(x_up)) * self.s(x_up)
        else:
            y_up = (x_up + self.t(x_down)) * self.s(x_down)
            y_down = x_down

        return jnp.concatenate((y_up, y_down))

    def inverse(self, y: Float[Array, " dim"]) -> Float[Array, " dim"]:
        y_up, y_down = jnp.split(y, [self.split_idx])

        if not self.flip:
            x_up = y_up
            x_down = y_down / self.s(y_up) - self.t(y_up)
        else:
            x_up = y_up / self.s(y_down) - self.t(y_down)
            x_down = y_down

        return jnp.concatenate((x_up, x_down))


class ResidualCoupling(
    AbstractInvertibleTransform
):  # TODO: support custom s, t in AffineCouplingTransform, and make this a subclass or so.
    dim: int = eqx.field(static=True)
    split_idx: int
    flip: bool

    t: eqx.nn.MLP

    def __init__(
        self,
        dim: int,
        split_idx: int | None = None,
        width_hidden: int = 10,
        depth: int = 1,
        flip: bool = False,
        activation: Callable = jax.nn.gelu,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        self.dim = dim
        self.split_idx = dim // 2 if split_idx is None else split_idx
        self.flip = flip

        in_size, out_size = (
            self.split_sizes if not self.flip else self.split_sizes[::-1]
        )
        self.t = eqx.nn.MLP(
            in_size=in_size,
            out_size=out_size,
            width_size=width_hidden,
            depth=depth,
            activation=activation,
            dtype=dtype,
            key=key,
        )

    @property
    def split_sizes(self) -> tuple[int, int]:
        out_sizes = self.split_idx, self.dim - self.split_idx
        return out_sizes

    def __call__(self, x: Float[Array, " dim"]) -> Float[Array, " dim"]:
        x_up, x_down = jnp.split(x, [self.split_idx])

        if not self.flip:
            y_up = x_up
            y_down = x_down + self.t(x_up)
        else:
            y_up = x_up + self.t(x_down)
            y_down = x_down

        return jnp.concatenate((y_up, y_down))

    def inverse(self, y: Float[Array, " dim"]) -> Float[Array, " dim"]:
        y_up, y_down = jnp.split(y, [self.split_idx])

        if not self.flip:
            x_up = y_up
            x_down = y_down - self.t(y_up)
        else:
            x_up = y_up - self.t(y_down)
            x_down = y_down

        return jnp.concatenate((x_up, x_down))
