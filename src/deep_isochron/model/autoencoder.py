r"""The phase autoencoder of Yawata et al. (Chaos 34, 063111, 2024) — the non-invertible
baseline (``docs/design/normal-forms.md`` §5.3, roadmap B12).

Encoder $f_{\rm enc}: \mathbb R^{d} \to \mathbb R^3$ whose first two outputs are
normalised to the unit circle (Eqs. (15)–(16)), decoder $f_{\rm dec}: \mathbb R^3 \to
\mathbb R^d$ (Eq. (17)), and a latent flow (``PhaseAmplitudeLatentDynamics``). The
learned phase is $\Theta(X) = \mathrm{atan2}(Y_2, Y_1)$ (Eq. (19)); the phase
sensitivity function $Z(\theta) = \nabla_X \Theta$ at $X = f_{\rm dec}(\cos\theta,
\sin\theta, 0)$ (Eq. (20)) comes from ``AbstractPhaseAmplitudeModel``, by autodiff.

Differences from the paper, deliberate: no batch normalisation in the MLPs (their
encoder: 2×100 ReLU + BN; decoder 3×100); the latent flow is continuous in $t$; input
standardisation is left to the data pipeline.
"""

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from .base import AbstractPhaseAmplitudeModel
from .latent_dynamics import AbstractLatentDynamics, PhaseAmplitudeLatentDynamics


def normalise_phase_plane(y: Float[Array, "3"]) -> Float[Array, "3"]:
    """Eqs. (15)–(16): ``(Y1, Y2) / sqrt(Y1² + Y2²)``, ``Y3`` untouched."""
    radius = jnp.sqrt(y[0] ** 2 + y[1] ** 2)
    return jnp.stack((y[0] / radius, y[1] / radius, y[2]))


class PhaseAmplitudeAutoencoder(AbstractPhaseAmplitudeModel):
    """``encoder``/``decoder`` are MLPs by default; any callables ``R^d -> R^3`` and
    ``R^3 -> R^d`` may be substituted with ``eqx.tree_at`` (the tests plug in the exact
    chart of a normal form)."""

    encoder: Callable[[Float[Array, " obs_dim"]], Float[Array, " latent_dim"]]
    decoder: Callable[[Float[Array, " latent_dim"]], Float[Array, " obs_dim"]]
    latent_dynamics: AbstractLatentDynamics

    def __init__(
        self,
        obs_dim: int,
        latent_dynamics: AbstractLatentDynamics | None = None,
        mlp_depth: int = 2,
        mlp_width: int = 100,
        activation: Callable = jax.nn.relu,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        key_e, key_d = jax.random.split(key)
        self.latent_dynamics = (
            PhaseAmplitudeLatentDynamics()
            if latent_dynamics is None
            else latent_dynamics
        )
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
            depth=mlp_depth + 1,
            activation=activation,
            dtype=dtype,
            key=key_d,
        )

    @property
    def latent_dim(self) -> int:
        return self.latent_dynamics.dim

    def encode(self, x: Float[Array, " obs_dim"]) -> Float[Array, " latent_dim"]:
        """``f_enc``: the MLP followed by the unit-circle normalisation (Eq. (16))."""
        return normalise_phase_plane(self.encoder(x))

    def decode(self, y: Float[Array, " latent_dim"]) -> Float[Array, " obs_dim"]:
        return self.decoder(y)

    def phase(self, x: Float[Array, " obs_dim"]) -> Float[Array, ""]:
        """Learned asymptotic phase ``Θ(X) = atan2(Y2, Y1)`` in ``(-π, π]`` (Eq.
        (19))."""
        y = self.encode(x)
        return jnp.arctan2(y[1], y[0])

    def amplitude(self, x: Float[Array, " obs_dim"]) -> Float[Array, ""]:
        """Learned amplitude-like latent ``Y3`` (the paper's deviation variable)."""
        return self.encode(x)[2]

    def cycle_point(self, theta: Float[Array, ""]) -> Float[Array, " obs_dim"]:
        """``f_dec(cos θ, sin θ, 0)``: the learned limit cycle at phase ``θ``."""
        return self.decode(jnp.stack((jnp.cos(theta), jnp.sin(theta), 0.0)))

    def __call__(
        self,
        ts: Float[Array, " time"],
        x0: Float[Array, " obs_dim"],
    ) -> tuple[Float[Array, "time obs_dim"], Float[Array, "time latent_dim"]]:
        """Encode ``x0``, evolve the latent in closed form over ``ts``, decode."""
        y0 = self.encode(x0)
        yt = self.latent_dynamics(ts, y0)
        xt = eqx.filter_vmap(self.decode)(yt)
        return xt, yt
