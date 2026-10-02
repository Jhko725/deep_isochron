import diffrax as dfx
import equinox as eqx
import jax
from jaxtyping import Array, Float

from ..misc import cartesian_to_polar, polar_to_cartesian
from ..systems.base import AbstractODE
from .invertible import AbstractBijection


class ConjugateLatentDynamics(eqx.Module):
    """A bijection and a latent ODE (in the polar chart) whose flow it conjugates."""

    latent_dynamics: AbstractODE
    bijection: AbstractBijection

    def __init__(self, latent_dynamics: AbstractODE, bijection: AbstractBijection):
        self.latent_dynamics = latent_dynamics
        self.bijection = bijection

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
        y0_polar = cartesian_to_polar(y0)
        yt_polar: Float[Array, "time latent_dim"] = self.latent_dynamics.solve(
            ts, y0_polar, max_steps=8192, atol=1e-6, rtol=1e-4, solver=dfx.Kvaerno5()
        )
        yt = jax.vmap(polar_to_cartesian)(yt_polar)
        xt: Float[Array, "time obs_dim"] = eqx.filter_vmap(self.bijection.inverse)(yt)

        if return_latent_trajectory:
            return xt, yt
        else:
            return xt, None
