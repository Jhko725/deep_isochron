r"""``AbstractPhaseAmplitudeModel``: the contract shared by the two approaches to the
same problem — the invertible conjugacy (``ConjugateLatentDynamics``) and the
non-invertible phase autoencoder (``PhaseAmplitudeAutoencoder``) — so that training and
evaluation code treats them on the same footing.

The contract is deliberately minimal and will grow as the science finds its rough edges.
What it promises: a **phase** ``Θ(x)`` in ``(-π, π]``; an **amplitude** ``A(x)`` that is
*isostable-like* — zero on the limit cycle, with the sign of the deviation and decaying
along the flow — but with **no fixed normalization** (the conjugacy model's is the
Wilson–Moehlis ``Ψ`` of ``docs/design/normal-forms.md`` §5.2; the autoencoder's ``Y₃``
is determined only up to scale, §5.4), so evaluations compare amplitudes up to a scale
fitted on the data; a point of the learned **limit cycle** at a given phase; and the
**prediction** ``x(t)`` from ``x(t₀)`` through the model's latent dynamics. Derived from
these, once for all models: the phase sensitivity function ``Z(θ) = ∇ₓΘ`` on the cycle
(Yawata et al. 2024, Eq. (20)) and the phase gradient anywhere.
"""

import abc

import equinox as eqx
import jax
from jaxtyping import Array, Float


class AbstractPhaseAmplitudeModel(eqx.Module):
    @abc.abstractmethod
    def phase(self, x: Float[Array, " obs_dim"]) -> Float[Array, ""]:
        """Asymptotic phase ``Θ(x)`` in ``(-π, π]``, up to the model's phase origin."""

    @abc.abstractmethod
    def amplitude(self, x: Float[Array, " obs_dim"]) -> Float[Array, ""]:
        """Isostable-like coordinate: zero on the cycle, decaying along the flow;
        normalization is the model's own."""

    @abc.abstractmethod
    def cycle_point(self, theta: Float[Array, ""]) -> Float[Array, " obs_dim"]:
        """The learned limit cycle at phase ``theta`` (``Θ(cycle_point(θ)) == θ``)."""

    @abc.abstractmethod
    def __call__(
        self, ts: Float[Array, " time"], x0: Float[Array, " obs_dim"]
    ) -> tuple[Float[Array, "time obs_dim"], Float[Array, "time latent_dim"]]:
        """Predicted trajectory through ``x0`` at ``ts`` (``ts[0]`` is the initial
        time), with the latent trajectory it was decoded from."""

    # ------------------------------------------------------------- derived (final) --
    def phase_gradient(self, x: Float[Array, " obs_dim"]) -> Float[Array, " obs_dim"]:
        """``∇ₓΘ(x)`` by autodiff — the infinitesimal phase response at ``x``."""
        return jax.grad(self.phase)(x)

    def phase_sensitivity(self, theta: Float[Array, ""]) -> Float[Array, " obs_dim"]:
        """Phase sensitivity function ``Z(θ) = ∇ₓΘ`` at the cycle point of phase
        ``θ`` (Yawata et al. 2024, Eq. (20))."""
        return self.phase_gradient(self.cycle_point(theta))
