r"""The phase autoencoder of Yawata et al. 2024, the non-invertible baseline
(``docs/design/normal-forms.md`` §5.3; roadmap B12).

An encoder $f_{\rm enc}: \mathbb R^{d} \to \mathbb R^3$ whose first two outputs are
normalized to the unit circle (Eqs. 15–16), a decoder
$f_{\rm dec}: \mathbb R^3 \to \mathbb R^d$ (Eq. 17), and the latent flow
``PhaseAmplitudeLatentDynamics``. The learned phase is
$\Theta(X) = \mathrm{atan2}(Y_2, Y_1) / 2\pi$ wrapped to $[0, 1)$, in turns (Eq. 19 in
the angle form; ADR-0020); the phase sensitivity function at
$X = f_{\rm dec}(\cos 2\pi\Theta, \sin 2\pi\Theta, 0)$ (Eq. 20) comes from
``AbstractPhaseAmplitudeModel`` by autodiff. The latent plane's angle stays an angle:
the factor $2\pi$ is written in `phase`, `cycle_point` and `isochron` only. Three
differences from the paper are deliberate: no batch normalization in the MLPs, whose
encoder is 2×100 ReLU with BN and decoder 3×100; a latent flow continuous in $t$; and
input standardization left to the data pipeline.
"""

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from ..utils.numerics import wrap_phase
from .base import AbstractPhaseAmplitudeModel
from .latent_dynamics import AbstractLatentDynamics, PhaseAmplitudeLatentDynamics


def normalize_phase_plane(y: Float[Array, "3"]) -> Float[Array, "3"]:
    """`y` with ``(Y1, Y2)`` normalized to the unit circle and ``Y3`` untouched
    (Yawata et al. 2024, Eqs. 15–16)."""
    radius = jnp.sqrt(y[0] ** 2 + y[1] ** 2)
    return jnp.stack((y[0] / radius, y[1] / radius, y[2]))


class PhaseAmplitudeAutoencoder(AbstractPhaseAmplitudeModel):
    """The phase autoencoder: an encoder, a decoder and a latent flow.

    **Arguments:**

    - `obs_dim`: The observation dimension ``d``.
    - `latent_dynamics`: The latent flow; `PhaseAmplitudeLatentDynamics` at its
        defaults when ``None``.
    - `mlp_depth`, `mlp_width`, `activation`, `dtype`: The MLPs; the decoder is one
        layer deeper than the encoder.
    - `key`: The PRNG key, split between the two MLPs.

    **Attributes:**

    - `encoder`, `decoder`: MLPs by default; any callables ``R^d -> R^3`` and
        ``R^3 -> R^d`` may be substituted with ``eqx.tree_at``, as the tests do with
        the exact chart of a normal form.
    - `latent_dynamics`: As above.
    """

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
        """The latent flow's dimension."""
        return self.latent_dynamics.dim

    def encode(self, x: Float[Array, " obs_dim"]) -> Float[Array, " latent_dim"]:
        """``f_enc``: the MLP followed by the unit-circle normalization (Eq. 16)."""
        return normalize_phase_plane(self.encoder(x))

    def decode(self, y: Float[Array, " latent_dim"]) -> Float[Array, " obs_dim"]:
        """``f_dec``: the decoder MLP."""
        return self.decoder(y)

    def phase(self, x: Float[Array, " obs_dim"]) -> Float[Array, ""]:
        """The learned asymptotic phase ``Θ(X) = atan2(Y2, Y1) / 2π`` wrapped to
        ``[0, 1)``, in turns (Eq. 19 in the angle form)."""
        y = self.encode(x)
        return wrap_phase(jnp.arctan2(y[1], y[0]) / (2 * jnp.pi))

    def amplitude(self, x: Float[Array, " obs_dim"]) -> Float[Array, ""]:
        """The learned amplitude-like latent ``Y3``, the paper's deviation variable."""
        return self.encode(x)[2]

    def cycle_point(self, theta: Float[Array, ""]) -> Float[Array, " obs_dim"]:
        """``f_dec(cos 2πθ, sin 2πθ, 0)``: the learned limit cycle at phase `theta`, in
        turns."""
        angle = 2 * jnp.pi * theta
        return self.decode(jnp.stack((jnp.cos(angle), jnp.sin(angle), 0.0)))

    def isochron_coordinate(self, x: Float[Array, " obs_dim"]) -> Float[Array, ""]:
        """The deviation variable ``Y3``, the same as `amplitude`; 0 on the cycle."""
        return self.encode(x)[2]

    def isochron(
        self, theta: Float[Array, ""], r: Float[Array, " n"]
    ) -> Float[Array, "n obs_dim"]:
        """``f_dec(cos 2πθ, sin 2πθ, Y3)`` at each ``Y3`` in `r`, for the phase `theta`
        in turns."""
        angle = 2 * jnp.pi * theta
        latent = jnp.stack(
            (jnp.full_like(r, jnp.cos(angle)), jnp.full_like(r, jnp.sin(angle)), r),
            axis=-1,
        )
        return eqx.filter_vmap(self.decode)(latent)

    def __call__(
        self,
        ts: Float[Array, " time"],
        x0: Float[Array, " obs_dim"],
    ) -> tuple[Float[Array, "time obs_dim"], Float[Array, "time latent_dim"]]:
        """Encode `x0`, evolve the latent in closed form over `ts`, decode."""
        y0 = self.encode(x0)
        yt = self.latent_dynamics(ts, y0)
        xt = eqx.filter_vmap(self.decode)(yt)
        return xt, yt
