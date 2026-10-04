import equinox as eqx
import jax
from jaxtyping import Array, Float

from ..systems.base import DEFAULT_SOLVER_CONFIG, SolverConfig
from ..systems.normal_forms import AbstractNormalForm
from .base import AbstractPhaseAmplitudeModel
from .invertible import AbstractBijection


class ConjugateLatentDynamics(AbstractPhaseAmplitudeModel):
    """A bijection ``H`` and a normal form whose flow it conjugates:
    ``x(t) = H⁻¹(φ_t(H(x0)))``.

    The normal form is integrated through its own ``flow`` in its own (cartesian)
    coordinates; chart choices are the normal form's business (its integrations), not
    this class's. Solver settings live in ``solver_config`` (static). Phase and
    amplitude are the normal form's closed forms transported by ``H``
    (``docs/design/normal-forms.md`` §5.2): ``Θ = Θ_NF ∘ H``, ``Ψ = Ψ_NF ∘ H``, and the
    learned cycle is ``H⁻¹`` of the unit circle.
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
        return self.latent_dynamics.dim

    def phase(self, x):
        return self.latent_dynamics.phase(self.bijection(x))

    def amplitude(self, x):
        return self.latent_dynamics.amplitude(self.bijection(x))

    def cycle_point(self, theta):
        return self.bijection.inverse(self.latent_dynamics.limit_cycle(theta))

    def __call__(
        self,
        ts: Float[Array, " time"],
        x0: Float[Array, " obs_dim"],
    ) -> tuple[Float[Array, "time obs_dim"], Float[Array, "time latent_dim"]]:
        y0: Float[Array, " latent_dim"] = self.bijection(x0)
        sol = self.latent_dynamics.flow(ts, y0, config=self.solver_config)
        assert sol.ys is not None
        yt: Float[Array, "time latent_dim"] = sol.ys
        xt: Float[Array, "time obs_dim"] = jax.vmap(self.bijection.inverse)(yt)
        return xt, yt
