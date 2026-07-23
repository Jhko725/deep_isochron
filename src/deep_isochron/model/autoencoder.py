from collections.abc import Callable

import equinox as eqx
import jax
from jaxtyping import Array, Float, PRNGKeyArray

from .latent_dynamics import AbstractLatentDynamics


# TODO: Need to add pre-latent hook fn / post-latent hook fn
class PhaseAmplitudeAutoencoder(eqx.Module):
    encoder: eqx.nn.MLP
    decoder: eqx.nn.MLP
    latent_dynamics: AbstractLatentDynamics

    def __init__(
        self,
        obs_dim: int,
        latent_dynamics: AbstractLatentDynamics,
        mlp_depth: int = 1,
        mlp_width: int = 32,
        activation: Callable = jax.nn.gelu,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        key_e, key_d = jax.random.split(key)
        self.latent_dynamics = latent_dynamics

        self.encoder = eqx.nn.MLP(
            in_size=obs_dim,
            out_size=self.latent_dim,
            width_size=mlp_width,
            depth=mlp_depth,
            activation=activation,
            dtype=dtype,
            key=key_e,
        )
        self.decoder = eqx.nn.MLP(
            in_size=self.latent_dim,
            out_size=obs_dim,
            width_size=mlp_width,
            depth=mlp_depth,
            activation=activation,
            dtype=dtype,
            key=key_d,
        )

    @property
    def latent_dim(self) -> int:
        return self.latent_dynamics.dim

    def __call__(
        self,
        ts: Float[Array, " time"],
        x0: Float[Array, " obs_dim"],
        return_latent_trajectory: bool = True,
    ) -> tuple[Float[Array, "time obs_dim"], Float[Array, "time latent_dim"] | None]:
        y0: Float[Array, " latent_dim"] = self.encoder(x0)
        yt: Float[Array, "time latent_dim"] = self.latent_dynamics(ts, y0)
        xt: Float[Array, "time obs_dim"] = eqx.filter_vmap(self.decoder)(yt)

        if return_latent_trajectory:
            return xt, yt
        else:
            return xt, None
