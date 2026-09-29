import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ..model.conjugacy import ConjugateLatentDynamics


class ConjugacyTrajectoryLoss(eqx.Module):
    weight_latent: float = eqx.field(static=True, default=0.0)

    def __call__(self, model: ConjugateLatentDynamics, batch, args=None):
        t_batch: Float[Array, "batch time"]
        x_batch: Float[Array, "batch time dim"]
        t_batch, x_batch = batch
        x_pred, y_pred = eqx.filter_vmap(model)(t_batch, x_batch[:, 0])

        mse = jnp.mean((x_batch - x_pred) ** 2)

        y_batch = jax.vmap(jax.vmap(model.bijection))(x_batch)
        mse_latent = jnp.mean((y_batch - y_pred) ** 2)

        # Weighting the latent space mse by 0
        return mse + self.weight_latent * mse_latent, {
            "mse": mse,
            "mse_latent": mse_latent,
        }
