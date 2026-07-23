from typing import ClassVar

import jax.numpy as jnp
from jaxtyping import Array, Float

from .base import AbstractODE


class HodgekinHuxley(AbstractODE):
    dim: ClassVar[int] = 4

    I: float = 30.0
    C: float = 1.0
    G_Na: float = 120.0
    G_K: float = 36.0
    G_L: float = 0.3
    E_Na: float = 50.0
    E_K: float = -77.0
    E_L: float = -54.4

    # The equations in the paper have major typos.
    # Below are from the original source code at: https://github.com/kyoukuntaro/PhaseAutoencoder/blob/main/utils/limitcycle.py
    def alpha_m(self, V):
        return -0.1 * (V + 40) / jnp.expm1(-(V + 40) / 10)

    def beta_m(self, V):
        return 4.0 * jnp.exp(-(V + 65) / 18)

    def alpha_h(self, V):
        return 0.07 * jnp.exp(-(V + 65) / 20)

    def beta_h(self, V):
        return 1 / (1 + jnp.exp(-(V + 35) / 10))

    def alpha_n(self, V):
        return -0.01 * (V + 55) / jnp.expm1(-(V + 55) / 10)

    def beta_n(self, V):
        return 0.125 * jnp.exp(-(V + 65) / 80.0)

    def rhs(
        self, t: Float[Array, ""], u: Float[Array, " dim"], args=None
    ) -> Float[Array, " dim"]:
        del t, args
        V, m, h, n = u
        dV = (
            self.G_Na * m**3 * h * (self.E_Na - V)
            + self.G_K * n**4 * (self.E_K - V)
            + self.G_L * (self.E_L - V)
            + self.I
        ) / self.C
        dm = self.alpha_m(V) * (1 - m) - self.beta_m(V) * m
        dh = self.alpha_h(V) * (1 - h) - self.beta_h(V) * h
        dn = self.alpha_n(V) * (1 - n) - self.beta_n(V) * n
        return jnp.stack([dV, dm, dh, dn], axis=0)
