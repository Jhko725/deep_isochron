import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ..model.autoencoder import PhaseAmplitudeAutoencoder
from ..model.conjugacy import ConjugateLatentDynamics
from ..model.latent_dynamics import PhaseAmplitudeLatentDynamics


class ConjugacyTrajectoryLoss(eqx.Module):
    weight_latent: float = eqx.field(static=True, default=0.0)

    def __call__(self, model: ConjugateLatentDynamics, batch, args=None):
        t_batch: Float[Array, "batch time"]
        x_batch: Float[Array, "batch time dim"]
        t_batch, x_batch = batch["t"], batch["u"]
        x_pred, y_pred = eqx.filter_vmap(model)(t_batch, x_batch[:, 0])

        mse = jnp.mean((x_batch - x_pred) ** 2)

        y_batch = jax.vmap(jax.vmap(model.bijection))(x_batch)
        mse_latent = jnp.mean((y_batch - y_pred) ** 2)

        # Weighting the latent space mse by 0
        return mse + self.weight_latent * mse_latent, {
            "mse": mse,
            "mse_latent": mse_latent,
        }


class PhaseAutoencoderLoss(eqx.Module):
    r"""Loss of Yawata et al. (Chaos 34, 063111, 2024), Eqs. (21)–(26), on a window
    ``batch = {"t": (B, K+1), "u": (B, K+1, d)}`` of ``K`` prediction steps:

    - ``recon`` (21): $\mathbb E\,\|X - f_{\rm dec}(f_{\rm enc}(X))\|^2$ over all window
      points;
    - ``pha`` (22): $\mathbb E\sum_{k=1}^K \alpha_k \sum_{i=1,2} |f_{\rm enc}(X_{t_k})_i
      - \Phi_{t_k - t_0}(f_{\rm enc}(X_{t_0}))_i|^2$, the latent flow replacing the
      $k$-fold step map $f_{\rm step}^k$;
    - ``dev`` (23): the same for $i = 3$;
    - ``aux`` (25): $\|\tfrac1B\sum_b (Y_1^b, Y_2^b)\|^2$ at $t_0$ — the centre of mass
      of the batch on the latent unit circle, which breaks the $\omega = 0$ solution;
    - $\alpha_k = k^{-\min(1, L_{\rm pha})}$ (24) with $L_{\rm pha}$ taken from the
      current batch under ``stop_gradient`` (the paper updates it once per epoch);
    - total (26): $w_{\rm recon}L_{\rm recon} + w_{\rm pha}L_{\rm pha}
      + w_{\rm amp}L_{\rm dev} + w_{\rm aux}L_{\rm aux}$ with ``weights``, switched to
      ``switched_weights`` once $L_{\rm pha} < $ ``switch_at[0]`` and $L_{\rm aux} < $
      ``switch_at[1]``; the paper switches once and for all, this module decides per
      batch (set ``switched_weights = weights`` to disable the switch, or drive the
      schedule from the trainer).

    Defaults are the paper's: weights ``(1, 0.5, 0.5, 2)`` → ``(1, 5, 0.5, 0)`` at
    ``(0.01, 0.05)``.
    """

    weights: tuple[float, float, float, float] = eqx.field(
        static=True, default=(1.0, 0.5, 0.5, 2.0)
    )
    switched_weights: tuple[float, float, float, float] = eqx.field(
        static=True, default=(1.0, 5.0, 0.5, 0.0)
    )
    switch_at: tuple[float, float] = eqx.field(static=True, default=(0.01, 0.05))

    def __call__(self, model: PhaseAmplitudeAutoencoder, batch, args=None):
        t_batch: Float[Array, "batch time"] = batch["t"]
        x_batch: Float[Array, "batch time dim"] = batch["u"]
        encode = jax.vmap(jax.vmap(model.encode))
        decode = jax.vmap(jax.vmap(model.decode))

        y_batch: Float[Array, "batch time 3"] = encode(x_batch)
        recon = jnp.mean(jnp.sum((decode(y_batch) - x_batch) ** 2, axis=-1))

        y_pred: Float[Array, "batch time 3"] = jax.vmap(model.latent_dynamics)(
            t_batch, y_batch[:, 0]
        )
        err = (y_batch - y_pred)[:, 1:] ** 2  # k = 1..K
        pha_k = jnp.mean(err[..., 0] + err[..., 1], axis=0)  # (K,)
        dev_k = jnp.mean(err[..., 2], axis=0)
        k = jnp.arange(1, err.shape[1] + 1, dtype=err.dtype)
        alpha = k ** (-jnp.minimum(1.0, jax.lax.stop_gradient(jnp.sum(pha_k))))
        pha = jnp.sum(alpha * pha_k)
        dev = jnp.sum(alpha * dev_k)

        centre = jnp.mean(y_batch[:, 0, :2], axis=0)
        aux = jnp.sum(centre**2)

        switched = (jax.lax.stop_gradient(pha) < self.switch_at[0]) & (
            jax.lax.stop_gradient(aux) < self.switch_at[1]
        )
        w = jnp.where(
            switched, jnp.asarray(self.switched_weights), jnp.asarray(self.weights)
        )
        total = w[0] * recon + w[1] * pha + w[2] * dev + w[3] * aux
        aux_out = {
            "recon": recon,
            "pha": pha,
            "dev": dev,
            "aux": aux,
            "switched": switched,
        }
        if isinstance(model.latent_dynamics, PhaseAmplitudeLatentDynamics):
            aux_out["omega"] = model.latent_dynamics.omega
            aux_out["kappa"] = model.latent_dynamics.kappa
        return total, aux_out
