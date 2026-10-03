import equinox as eqx
import jax
from jaxtyping import Array, Float

from ..systems.base import AbstractODE, DEFAULT_SOLVER_CONFIG, SolverConfig
from .invertible import AbstractBijection


class ConjugateLatentDynamics(eqx.Module):
    """A bijection ``Φ`` and a latent ODE whose flow it conjugates:
    ``x(t) = Φ⁻¹(φ_t(Φ(x0)))``.

    The latent ODE is integrated through its own ``flow`` in its own (cartesian)
    coordinates; chart choices are the ODE's business (``AbstractNormalForm``
    integrations), not this class's. Solver settings live in ``solver_config`` (static).
    """

    latent_dynamics: AbstractODE
    bijection: AbstractBijection
    solver_config: SolverConfig = eqx.field(static=True)

    def __init__(
        self,
        latent_dynamics: AbstractODE,
        bijection: AbstractBijection,
        solver_config: SolverConfig = DEFAULT_SOLVER_CONFIG,
    ):
        self.latent_dynamics = latent_dynamics
        self.bijection = bijection
        self.solver_config = solver_config

    @property
    def dim(self) -> int:
        return self.latent_dynamics.dim

    def __call__(
        self,
        ts: Float[Array, " time"],
        x0: Float[Array, " obs_dim"],
        return_latent_trajectory: bool = True,
    ) -> tuple[Float[Array, "time obs_dim"], Float[Array, "time latent_dim"] | None]:
        y0: Float[Array, " latent_dim"] = self.bijection(x0)
        sol = self.latent_dynamics.flow(ts, y0, config=self.solver_config)
        assert sol.ys is not None
        yt: Float[Array, "time latent_dim"] = sol.ys
        xt: Float[Array, "time obs_dim"] = jax.vmap(self.bijection.inverse)(yt)
        return (xt, yt) if return_latent_trajectory else (xt, None)
