from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from ...misc import cartesian_to_polar
from ..fourier import TruncatedFourier
from .analytic import SinhConjugation
from .base import AbstractBijection, SequentialINN
from .splines import MonotonicRQSpline


class OffsetedBijection(AbstractBijection):
    bijection: AbstractBijection

    def __init__(self, bijection: AbstractBijection):
        self.bijection = bijection

    @property
    def dim(self) -> int:
        return self.bijection.dim

    # Shapes follow the wrapped bijection's (0-d for scalar bijections).
    def __call__(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        return self.bijection(x) - self.bijection(jnp.zeros_like(x))

    def inverse(self, y: Float[Array, "..."]) -> Float[Array, "..."]:
        offset = self.bijection(jnp.zeros_like(y))
        return self.bijection.inverse(y + offset)


class RadialBijection(AbstractBijection):
    dim: ClassVar[int] = 2  # ty: ignore

    center: Float[Array, " dim"]
    log_scale: Float[Array, " dim"]

    radial_bijection: OffsetedBijection

    eps_r: float = eqx.field(static=True)

    def __init__(self, bijection, center, log_scale, eps_r: float = 1e-7):
        self.center = center
        self.log_scale = log_scale
        self.radial_bijection = OffsetedBijection(bijection)
        self.eps_r = eps_r

    @property
    def scale(self) -> Float[Array, " dim"]:
        return jnp.exp(self.log_scale)

    def __call__(self, x: Float[Array, " 2"]) -> Float[Array, " 2"]:
        x_scaled = (x - self.center) * self.scale
        r_in = jnp.sqrt(jnp.sum(x_scaled**2) + self.eps_r**2)
        r_out = self.radial_bijection(r_in)

        y_scaled = (r_out / r_in) * x_scaled
        return (y_scaled / self.scale) + self.center

    def inverse(self, y: Float[Array, " 2"]) -> Float[Array, " 2"]:
        y_scaled = (y - self.center) * self.scale
        r_out = jnp.sqrt(jnp.sum(y_scaled**2) + self.eps_r**2)
        r_in = self.radial_bijection.inverse(r_out)

        x_scaled = (r_in / r_out) * y_scaled
        return (x_scaled / self.scale) + self.center


class CircularMonotonicRQCoupling(AbstractBijection):
    """Circular variant of monotonic rational quadratic spline coupling transform, as
    introduced in [1, 2].

    [1] C. Durkan et al. Neural Spline Flows. NeurIPS (2019).
    [2] https://github.com/bayesiains/nflows/blob/master/nflows/transforms/splines/rational_quadratic.py
    """

    theta_offset: Float[Array, ""]
    _dxs: Float[Array, " knots-1"]
    _dys: Float[Array, " knots-1"]
    _derivs: Float[Array, " knots-1"]

    dim: ClassVar[int] = 2  # ty: ignore
    num_knots: int = eqx.field(static=True)
    min_derivative: float = eqx.field(static=True)
    min_rel_x_bin_width: float = eqx.field(static=True)
    min_rel_y_bin_width: float = eqx.field(static=True)

    def __init__(
        self,
        num_knots: int = 10,
        min_rel_x_bin_width: float = 1e-3,
        min_rel_y_bin_width: float = 1e-3,
        min_derivative: float = 1e-3,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        """**Arguments:**

        - `dim`: Dimension of the inputs/outputs of the coupling transform.
        - `min_derivative`: Minimum value of the derivative at the knot points.


        """
        key_x, key_y, key_d = jax.random.split(key, 3)
        self._dxs = jnp.ones(
            (num_knots - 1,)
        )  # jax.random.normal(key_x, (num_knots-1,))
        self._dys = jnp.ones(
            (num_knots - 1,)
        )  # jax.random.normal(key_y, (num_knots-1,))
        self._derivs = jnp.ones(
            (num_knots - 1,)
        )  # jax.random.normal(key_d, (num_knots-1,))
        self.theta_offset = jnp.zeros(())
        self.num_knots = num_knots
        self.min_rel_x_bin_width = min_rel_x_bin_width
        self.min_rel_y_bin_width = min_rel_y_bin_width
        self.min_derivative = min_derivative

    def _make_knots(
        self,
        unnormalized_bin_widths: Float[Array, " {self.num_knots}-1"],
        min_bin_width: float,
    ) -> Float[Array, " {self.num_knots}"]:
        bin_widths = (
            jax.nn.softmax(unnormalized_bin_widths)
            * (1 - self.num_knots * min_bin_width)
            + min_bin_width
        )
        knots = jnp.pad(jnp.cumsum(bin_widths), pad_width=(1, 0))
        return -jnp.pi + 2 * jnp.pi * knots

    def make_spline(self) -> MonotonicRQSpline:
        # TODO: MonotonicRQSpline now fixes the boundary derivatives to 1, so the
        # circular derivative-matching (previously `pad(..., mode="wrap")`) is lost
        # and the map is only C^0 at theta = +-pi. Support free boundary derivatives
        # in MonotonicRQSpline to restore it.
        xs = self._make_knots(self._dxs, self.min_rel_x_bin_width)
        ys = self._make_knots(self._dys, self.min_rel_y_bin_width)
        ds = jax.nn.softplus(self._derivs[:-1]) + self.min_derivative
        return MonotonicRQSpline(
            jnp.diff(xs), jnp.diff(ys), ds, xy_range=(-float(jnp.pi), float(jnp.pi))
        )

    def __call__(self, x: Float[Array, " 2"]) -> Float[Array, " 2"]:
        r, theta = cartesian_to_polar(x)
        # theta = jnp.mod(theta-self.theta_offset, 2*jnp.pi)-jnp.pi
        spl = self.make_spline()
        theta_out = jnp.piecewise(
            theta,
            [theta < -jnp.pi, theta > jnp.pi],
            [lambda x: x, lambda x: x, lambda x: spl(x)],
        )
        return jnp.stack([r * jnp.cos(theta_out), r * jnp.sin(theta_out)])

    def inverse(self, y: Float[Array, " dim"]) -> Float[Array, " dim"]:
        r, theta = cartesian_to_polar(y)
        # theta = jnp.mod(theta, 2*jnp.pi)-jnp.pi
        spl = self.make_spline()
        theta_in = jnp.piecewise(
            theta,
            [theta < -jnp.pi, theta > jnp.pi],
            [lambda x: x, lambda x: x, lambda x: spl.inverse(x)],
        )
        # theta_in = theta_in+self.theta_offset
        return jnp.stack([r * jnp.cos(theta_in), r * jnp.sin(theta_in)])


class PolarConditionalBijection(AbstractBijection):
    dim: ClassVar[int] = 2

    center: Float[Array, " dim"]
    log_scale: Float[Array, " dim"]

    fourier: TruncatedFourier

    eps_r: float = eqx.field(static=True)

    def __init__(
        self,
        n_radial_blocks: int,
        fourier_order: int = 3,
        init_scale: float = 0.01,
        eps_r: float = 1e-7,
        *,
        key: PRNGKeyArray,
    ):
        key_c, key_s, key_f = jax.random.split(key, 3)
        self.center = jax.random.normal(key_c, (2,)) * init_scale
        self.log_scale = jax.random.normal(key_s, (2,)) * init_scale
        self.fourier = TruncatedFourier(
            dim=n_radial_blocks * 5,
            order=fourier_order,
            init_scale=init_scale,
            key=key_f,
        )
        self.eps_r = eps_r

    @property
    def scale(self) -> Float[Array, " dim"]:
        return jnp.exp(self.log_scale)

    def __call__(self, x: Float[Array, " 2"]) -> Float[Array, " 2"]:
        x_scaled = (x - self.center) * self.scale
        r_in = jnp.sqrt(jnp.sum(x_scaled**2) + self.eps_r**2)

        theta = jnp.arctan2(x_scaled[1], x_scaled[0])
        params = jnp.reshape(self.fourier(theta), (-1, 5))
        radial_bijection = OffsetedBijection(
            SequentialINN(
                [SinhConjugation.from_unnormalized_params(*p) for p in params]
            )
        )

        r_out = radial_bijection(r_in)

        y_scaled = (r_out / r_in) * x_scaled
        return (y_scaled / self.scale) + self.center

    def inverse(self, y: Float[Array, " 2"]) -> Float[Array, " 2"]:
        y_scaled = (y - self.center) * self.scale
        r_out = jnp.sqrt(jnp.sum(y_scaled**2) + self.eps_r**2)

        theta = jnp.arctan2(y_scaled[1], y_scaled[0])
        params = jnp.reshape(self.fourier(theta), (-1, 5))
        radial_bijection = OffsetedBijection(
            SequentialINN(
                [SinhConjugation.from_unnormalized_params(*p) for p in params]
            )
        )

        r_in = radial_bijection.inverse(r_out)

        x_scaled = (r_in / r_out) * y_scaled
        return (x_scaled / self.scale) + self.center
