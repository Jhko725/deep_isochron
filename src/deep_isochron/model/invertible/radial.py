from typing import ClassVar

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from deep_isochron.model.invertible import (
    AbstractBijection,
)


class OffsetedBijection(AbstractBijection):
    bijection: AbstractBijection

    def __init__(self, bijection: AbstractBijection):
        self.bijection = bijection

    @property
    def dim(self) -> int:
        return self.bijection.dim

    def __call__(self, x: Float[Array, " dim"]) -> Float[Array, " dim"]:
        return self.bijection(x) - self.bijection(jnp.zeros_like(x))

    def inverse(self, y: Float[Array, " dim"]) -> Float[Array, " dim"]:
        offset = self.bijection(jnp.zeros_like(y))
        return self.bijection.inverse(y + offset)


class RadialBijection(AbstractBijection):
    dim: ClassVar[int] = 2 #ty: ignore

    center: Float[Array, " dim"]
    log_scale: Float[Array, " dim"]

    radial_bijection: OffsetedBijection

    eps_r: float = eqx.field(static=True)

    def __init__(self, bijection, center, log_scale, eps_r: float = 1e-7):
        self.center = center
        self.log_scale = log_scale
        self.radial_bijection = OffsetedBijection(bijection)
        self.eps_r = eps_r

    @property
    def scale(self) -> Float[Array, " dim"]:
        return jnp.exp(self.log_scale)

    def __call__(self, x: Float[Array, " 2"]) -> Float[Array, " 2"]:
        x_scaled = (x - self.center) * self.scale
        r_in = jnp.sqrt(jnp.sum(x_scaled**2) + self.eps_r**2)
        r_out = self.radial_bijection(r_in)

        y_scaled = (r_out / r_in) * x_scaled
        return (y_scaled / self.scale) + self.center

    def inverse(self, y: Float[Array, " 2"]) -> Float[Array, " 2"]:
        y_scaled = (y - self.center) * self.scale
        r_out = jnp.sqrt(jnp.sum(y_scaled**2) + self.eps_r**2)
        r_in = self.radial_bijection.inverse(r_out)

        x_scaled = (r_in / r_out) * y_scaled
        return (x_scaled / self.scale) + self.center