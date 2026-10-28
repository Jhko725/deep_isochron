"""The invertible conjugacy model: a bijection to a normal form's coordinates."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ..systems.base import DEFAULT_SOLVER_CONFIG, SolverConfig
from ..systems.normal_forms import AbstractNormalForm
from .base import AbstractPhaseAmplitudeModel
from .invertible import AbstractBijection


class ConjugateLatentDynamics(AbstractPhaseAmplitudeModel):
    """A bijection ``H`` and a normal form whose flow it conjugates,
    ``x(t) = H⁻¹(φ_t(H(x0)))``.

    The normal form is integrated through its own ``flow`` in its own cartesian
    coordinates; chart choices are the normal form's integrations, not this class's.
    Phase and amplitude are the normal form's closed forms transported by ``H``
    (``docs/design/normal-forms.md`` §5.2), ``Θ = Θ_NF ∘ H`` and ``Ψ = Ψ_NF ∘ H``, and
    the learned cycle is ``H⁻¹`` of the unit circle, a learned isochron ``H⁻¹`` of the
    normal form's, and the learned equilibrium ``H⁻¹(0)``.

    **Attributes:**

    - `latent_dynamics`: The normal form.
    - `bijection`: ``H``.
    - `solver_config`: The solver settings of the latent flow (static).
    """

    latent_dynamics: AbstractNormalForm
    bijection: AbstractBijection
    solver_config: SolverConfig = eqx.field(static=True)

    def __init__(
        self,
        latent_dynamics: AbstractNormalForm,
        bijection: AbstractBijection,
        solver_config: SolverConfig = DEFAULT_SOLVER_CONFIG,
    ):
        self.latent_dynamics = latent_dynamics
        self.bijection = bijection
        self.solver_config = solver_config

    @property
    def dim(self) -> int:
        """The normal form's dimension."""
        return self.latent_dynamics.dim

    def phase(self, x):
        """``Θ_NF(H(x))``."""
        return self.latent_dynamics.phase(self.bijection(x))

    def amplitude(self, x):
        """``Ψ_NF(H(x))``."""
        return self.latent_dynamics.amplitude(self.bijection(x))

    def cycle_point(self, theta):
        """``H⁻¹`` of the normal form's cycle point at `theta`."""
        return self.bijection.inverse(self.latent_dynamics.limit_cycle(theta))

    def latent_radius(self, x):
        """``|H(x)|``, the normal form's radius; 1 on the learned cycle."""
        return jnp.linalg.norm(self.bijection(x))

    def isochron(self, theta, r):
        """``H⁻¹`` of the normal form's isochron of phase `theta` at radii `r`."""
        return jax.vmap(self.bijection.inverse)(self.latent_dynamics.isochron(theta, r))

    def equilibrium(self) -> Float[Array, " obs_dim"]:
        """``H⁻¹(0)``, the learned equilibrium: the image of the normal form's
        origin, the center of its phaseless set."""
        return self.bijection.inverse(jnp.zeros(self.dim))

    def __call__(
        self,
        ts: Float[Array, " time"],
        x0: Float[Array, " obs_dim"],
    ) -> tuple[Float[Array, "time obs_dim"], Float[Array, "time latent_dim"]]:
        """Map `x0` through ``H``, flow the normal form over `ts`, map back."""
        y0: Float[Array, " latent_dim"] = self.bijection(x0)
        sol = self.latent_dynamics.flow(ts, y0, config=self.solver_config)
        assert sol.ys is not None
        yt: Float[Array, "time latent_dim"] = sol.ys
        xt: Float[Array, "time obs_dim"] = jax.vmap(self.bijection.inverse)(yt)
        return xt, yt
