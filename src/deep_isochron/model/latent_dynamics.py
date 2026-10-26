r"""The latent dynamics of the non-invertible phase-autoencoder baseline.

``PhaseAmplitudeLatentDynamics`` is the latent flow of Yawata et al. 2024, Eqs. 12–14:
the first two latent variables rotate on the unit circle at a learned constant
frequency $\omega$, and the third decays as $e^{\kappa\tau}$ with a learned
$\kappa < 0$, their $\lambda$, which they assume negative. It is written as a flow
continuous in $t$ rather than their one-step map $f_{\rm step}$ (Eq. 18), which the
paper itself intends to hold for all $\tau > 0$. In the notation of
``docs/design/normal-forms.md`` §5.3 the latent space is the phase–amplitude chart
$\mathbf Y = (\cos\Theta, \sin\Theta, \Psi)$ and the flow is $\dot\Theta = \omega_1$,
$\dot\Psi = \kappa\Psi$.
"""

import abc

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from .invertible.constraints import GreaterThan


class AbstractLatentDynamics(eqx.Module):
    """The latent dynamics of the autoencoder baseline: a closed-form flow
    ``(ts, y0) -> y[t]`` with no vector field, distinct from
    ``systems.AbstractNormalForm``, an ODE with closed-form phase–amplitude structure,
    which the conjugacy models use.

    **Attributes:**

    - `dim`: The latent dimension.
    """

    dim: eqx.AbstractVar[int]

    @abc.abstractmethod
    def __call__(
        self,
        ts: Float[Array, " time"],
        y0: Float[Array, " dim"],
    ) -> Float[Array, "time dim"]:
        """The latent trajectory from `y0` at `ts`, with ``ts[0]`` the initial time."""


DECAY_CONSTRAINT = GreaterThan(0.0)
"""The constraint ``-kappa > 0``; ``raw = 0`` gives ``kappa = -1``."""


class PhaseAmplitudeLatentDynamics(AbstractLatentDynamics):
    r"""Rotation and decay on $\mathbb R^3$ (Yawata et al. 2024, Eqs. 12–14):
    $(Y_1, Y_2)(t) = R(\omega (t - t_0))\thinspace(Y_1, Y_2)(t_0)$ and
    $Y_3(t) = e^{\kappa (t - t_0)}\thinspace Y_3(t_0)$, with $R$ the rotation matrix.

    The rotation is linear, so a non-normalized ``y0`` keeps its radius; the encoder
    normalizes.

    **Arguments:**

    - `omega`: The frequency, free; its sign is the orientation of the cycle in the
        latent plane.
    - `kappa`: The decay rate, negative, stored through ``DECAY_CONSTRAINT``.

    **Attributes:**

    - `omega`: As above.
    - `raw_decay`: The unconstrained leaf of ``-kappa``.

    **Raises:**

    `ValueError` for ``kappa >= 0``."""

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
        """The constrained decay rate, ``-DECAY_CONSTRAINT(raw_decay)``."""
        return -DECAY_CONSTRAINT(self.raw_decay)

    def params(self) -> dict[str, float]:
        """``{"omega", "kappa"}`` as floats."""
        return {"omega": float(self.omega), "kappa": float(self.kappa)}

    def __call__(
        self,
        ts: Float[Array, " time"],
        y0: Float[Array, "3"],
    ) -> Float[Array, "time 3"]:
        """The rotated and decayed latent trajectory from `y0` at `ts`."""
        dt = ts - ts[0]
        c, s = jnp.cos(self.omega * dt), jnp.sin(self.omega * dt)
        y1 = c * y0[0] - s * y0[1]
        y2 = s * y0[0] + c * y0[1]
        y3 = jnp.exp(self.kappa * dt) * y0[2]
        return jnp.stack((y1, y2, y3), axis=-1)
