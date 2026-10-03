r"""Latent dynamics of the non-invertible phase-autoencoder baseline.

``PhaseAmplitudeLatentDynamics`` is the latent flow of Yawata et al., *Phase autoencoder
for limit-cycle oscillators*, Chaos 34, 063111 (2024), Eqs. (12)–(14): the first two
latent variables rotate on the unit circle at a learned constant frequency $\omega$, the
third decays as $e^{\kappa\tau}$ with a learned $\kappa < 0$ (their $\lambda$; "we
assume that $\lambda$ is negative"). Written as a flow continuous in $t$ rather than
their one-step map $f_{\rm step}$ (Eq. (18)), which the paper itself intends to hold
"for all $\tau > 0$". In the notation of ``docs/design/normal-forms.md`` §5.3 the latent
space is the phase–amplitude chart $\mathbf Y = (\cos\Theta, \sin\Theta, \Psi)$ and the
flow is $\dot\Theta = \omega_1$, $\dot\Psi = \kappa\Psi$.
"""

import abc

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from .invertible.constraints import GreaterThan


class AbstractLatentDynamics(eqx.Module):
    """Latent dynamics of the non-invertible autoencoder baseline
    (``PhaseAmplitudeAutoencoder``): a closed-form flow ``(ts, y0) -> y[t]`` with no
    vector field. Distinct from ``systems.AbstractNormalForm``, which is an ODE with
    closed-form phase–amplitude structure and is what the conjugacy models use."""

    dim: eqx.AbstractVar[int]

    @abc.abstractmethod
    def __call__(
        self,
        ts: Float[Array, " time"],
        y0: Float[Array, " dim"],
    ) -> Float[Array, "time dim"]: ...


DECAY_CONSTRAINT = GreaterThan(0.0)
"""``-kappa > 0``; ``raw = 0`` gives ``kappa = -1``."""


class PhaseAmplitudeLatentDynamics(AbstractLatentDynamics):
    r"""Rotation ⊕ decay on $\mathbb R^3$ (Yawata et al. 2024, Eqs. (12)–(14)):

    $$(Y_1, Y_2)(t) = R(\omega (t - t_0))\,(Y_1, Y_2)(t_0), \qquad
    Y_3(t) = e^{\kappa (t - t_0)}\, Y_3(t_0),$$

    with $R$ the rotation matrix. ``omega`` is free (its sign is the orientation of the
    cycle in the latent plane); ``kappa < 0`` through ``DECAY_CONSTRAINT``. The rotation
    is linear, so a non-normalised ``y0`` keeps its radius; the encoder normalises.
    """

    omega: Float[Array, ""]
    raw_decay: Float[Array, ""]
    dim: int = eqx.field(static=True, init=False, default=3)

    def __init__(self, omega: float = 1.0, kappa: float = -1.0):
        if kappa >= 0:
            raise ValueError(f"kappa must be negative (stable cycle); got {kappa}")
        self.omega = jnp.asarray(omega, dtype=float)
        self.raw_decay = DECAY_CONSTRAINT.inverse(jnp.asarray(-kappa, dtype=float))

    @property
    def kappa(self) -> Float[Array, ""]:
        return -DECAY_CONSTRAINT(self.raw_decay)

    def params(self) -> dict[str, float]:
        return {"omega": float(self.omega), "kappa": float(self.kappa)}

    def __call__(
        self,
        ts: Float[Array, " time"],
        y0: Float[Array, "3"],
    ) -> Float[Array, "time 3"]:
        dt = ts - ts[0]
        c, s = jnp.cos(self.omega * dt), jnp.sin(self.omega * dt)
        y1 = c * y0[0] - s * y0[1]
        y2 = s * y0[0] + c * y0[1]
        y3 = jnp.exp(self.kappa * dt) * y0[2]
        return jnp.stack((y1, y2, y3), axis=-1)
