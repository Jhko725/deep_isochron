r"""``AbstractPhaseAmplitudeModel``: the contract shared by the two approaches to the
problem, the invertible conjugacy ``ConjugateLatentDynamics`` and the non-invertible
phase autoencoder ``PhaseAmplitudeAutoencoder``, so that training and evaluation treat
them on one footing.

The contract is minimal on purpose and will grow as the science finds its rough edges.
It promises a phase ``Θ(x)`` in ``(-π, π]``; an amplitude ``A(x)`` that is
isostable-like, zero on the limit cycle, with the sign of the deviation and decaying
along the flow, but with no fixed normalization, since the conjugacy model's is the
Wilson–Moehlis ``Ψ`` (``docs/design/normal-forms.md`` §5.2) while the autoencoder's
``Y₃`` is determined only up to scale (§5.4), so evaluations compare amplitudes up to
a scale fitted on the data; a point of the learned limit cycle at a given phase; and
the prediction ``x(t)`` from ``x(t₀)`` through the model's latent dynamics; the
learned isochron of a phase, sampled along the model's own latent radial coordinate,
and that coordinate at a point, so that a learned isochron can be matched to a reference
one point by point. Derived from these once for all models are the phase gradient
anywhere and the phase sensitivity function ``Z(θ) = ∇ₓΘ`` on the cycle (Yawata et al.
2024, Eq. 20).
"""

import abc

import equinox as eqx
import jax
from jaxtyping import Array, Float


class AbstractPhaseAmplitudeModel(eqx.Module):
    """A model of a limit-cycle oscillator with a learned phase, amplitude, cycle and
    prediction."""

    @abc.abstractmethod
    def phase(self, x: Float[Array, " obs_dim"]) -> Float[Array, ""]:
        """The asymptotic phase ``Θ(x)`` in ``(-π, π]``, up to the model's phase
        origin."""

    @abc.abstractmethod
    def amplitude(self, x: Float[Array, " obs_dim"]) -> Float[Array, ""]:
        """The isostable-like coordinate of `x`: zero on the cycle, decaying along the
        flow, with the model's own normalization."""

    @abc.abstractmethod
    def cycle_point(self, theta: Float[Array, ""]) -> Float[Array, " obs_dim"]:
        """The point of the learned limit cycle at phase `theta`, with
        ``Θ(cycle_point(θ)) == θ``."""

    @abc.abstractmethod
    def latent_radius(self, x: Float[Array, " obs_dim"]) -> Float[Array, ""]:
        """The model's latent radial coordinate of `x`: the coordinate along which
        `isochron` is sampled, in the model's own normalization (the normal form's
        radius for the conjugacy model, the deviation variable for the autoencoder)."""

    @abc.abstractmethod
    def isochron(
        self, theta: Float[Array, ""], r: Float[Array, " n"]
    ) -> Float[Array, "n obs_dim"]:
        """The points of the learned isochron of phase `theta` at the latent radii `r`,
        with ``Θ(isochron(θ, r)) == θ`` and ``latent_radius(isochron(θ, r)) == r``."""

    @abc.abstractmethod
    def __call__(
        self, ts: Float[Array, " time"], x0: Float[Array, " obs_dim"]
    ) -> tuple[Float[Array, "time obs_dim"], Float[Array, "time latent_dim"]]:
        """The predicted trajectory through `x0` at `ts`, with ``ts[0]`` the initial
        time, and the latent trajectory it was decoded from."""

    # ------------------------------------------------------------- derived (final) --
    def phase_gradient(self, x: Float[Array, " obs_dim"]) -> Float[Array, " obs_dim"]:
        """``∇ₓΘ(x)`` by autodiff: the infinitesimal phase response at `x`."""
        return jax.grad(self.phase)(x)

    def phase_sensitivity(self, theta: Float[Array, ""]) -> Float[Array, " obs_dim"]:
        """The phase sensitivity function ``Z(θ) = ∇ₓΘ`` at the cycle point of phase
        `theta` (Yawata et al. 2024, Eq. 20)."""
        return self.phase_gradient(self.cycle_point(theta))
