import math

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from ..fourier import TruncatedFourier
from .analytic import SinhConjugation
from .base import AbstractBijection, SequentialINN
from .splines import MonotonicRQSpline


class OffsetedBijection(AbstractBijection):
    bijection: AbstractBijection
    dim: int = eqx.field(static=True, init=False)
    smoothness: int | None = eqx.field(static=True, init=False)

    def __init__(self, bijection: AbstractBijection):
        self.bijection = bijection
        self.dim = bijection.dim
        self.smoothness = bijection.smoothness

    # Shapes follow the wrapped bijection's (0-d for scalar bijections).
    def __call__(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        return self.bijection(x) - self.bijection(jnp.zeros_like(x))

    def inverse(self, y: Float[Array, "..."]) -> Float[Array, "..."]:
        offset = self.bijection(jnp.zeros_like(y))
        return self.bijection.inverse(y + offset)


class RadialBijection(AbstractBijection):
    dim: int = eqx.field(static=True, default=2, init=False)
    smoothness: int | None = eqx.field(static=True, init=False)

    center: Float[Array, " dim"]
    log_scale: Float[Array, " dim"]

    radial_bijection: OffsetedBijection

    eps_r: float = eqx.field(static=True)

    def __init__(self, bijection, center, log_scale, eps_r: float = 1e-7):
        self.center = center
        self.log_scale = log_scale
        self.radial_bijection = OffsetedBijection(bijection)
        self.eps_r = eps_r
        self.smoothness = bijection.smoothness

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
    """Angular rational-quadratic spline in polar coordinates: ``(r, θ) -> (r, s(θ))``
    with ``s`` a ``MonotonicRQSpline`` on ``(-π, π)``, after [1, 2].

    The spline is held as a standalone trainable bijection (its ``raw`` leaf is
    unconstrained, so training cannot break monotonicity).

    **Implementation.** The map is applied as a rotation of the input by
    ``Δ(θ) = s(θ) - θ``, which preserves ``|x|`` exactly and is exactly invertible
    (rotate back by ``s⁻¹(θ') - θ'``). The angle is read through a ``where``-safe
    ``arctan2`` that inside the disc ``|x| < eps_r`` returns a fixed fill angle: ``0``
    for the forward map and ``s(0)`` for the inverse. The origin then maps to itself,
    the Jacobians there are the finite rotations ``R(s(0))`` and ``R(-s(0))`` (mutually
    inverse, ``det = 1``), and away from the origin ``det J = s'(θ) > 0``.

    **Regularity.** As a map of the punctured plane it is as smooth as ``s``; at the
    origin it is continuous but not differentiable (a θ-dependent rotation), and across
    ``θ = ±π`` it is only C^0 because the spline's boundary derivatives are pinned to 1.
    Hence ``smoothness = 0``. Restoring circular derivative matching needs an
    endpoint-handling option on ``AbstractSpline`` (periodic / free boundary
    derivatives); deliberately deferred.

    [1] C. Durkan et al. Neural Spline Flows. NeurIPS (2019).
    [2] https://github.com/bayesiains/nflows/blob/master/nflows/transforms/splines/rational_quadratic.py
    """

    spline: MonotonicRQSpline
    dim: int = eqx.field(static=True, default=2, init=False)
    smoothness: int | None = eqx.field(static=True, default=0, init=False)
    eps_r: float = eqx.field(static=True)

    def __init__(
        self,
        num_bins: int = 10,
        *,
        raw: Float[Array, " num_params"] | None = None,
        min_rel_width: float = 1e-3,
        min_derivative: float = 1e-3,
        eps_r: float = 1e-12,
    ):
        self.spline = MonotonicRQSpline(
            num_bins,
            xy_range=(-math.pi, math.pi),
            raw=raw,
            min_rel_width=min_rel_width,
            min_derivative=min_derivative,
        )
        self.eps_r = eps_r

    def _safe_angle(
        self, xy: Float[Array, " 2"], fill: Float[Array, ""]
    ) -> Float[Array, ""]:
        """``arctan2`` with a double-``where`` guard: inputs inside the ``eps_r`` disc
        are replaced by ``(1, 0)`` *before* ``arctan2`` so neither branch produces NaN
        gradients (jax FAQ, "gradients contain NaN where using where"); the angle
        returned there is ``fill``."""
        x, y = xy
        small = x**2 + y**2 < self.eps_r**2
        x_s = jnp.where(small, 1.0, x)
        y_s = jnp.where(small, 0.0, y)
        return jnp.where(small, fill, jnp.arctan2(y_s, x_s))

    @staticmethod
    def _rotate(xy: Float[Array, " 2"], delta: Float[Array, ""]) -> Float[Array, " 2"]:
        c, s = jnp.cos(delta), jnp.sin(delta)
        x, y = xy
        return jnp.stack([c * x - s * y, s * x + c * y])

    def __call__(self, x: Float[Array, " 2"]) -> Float[Array, " 2"]:
        theta = self._safe_angle(x, jnp.zeros(()))
        return self._rotate(x, self.spline(theta) - theta)

    def inverse(self, y: Float[Array, " 2"]) -> Float[Array, " 2"]:
        theta = self._safe_angle(y, self.spline(jnp.zeros(())))
        return self._rotate(y, self.spline.inverse(theta) - theta)


class PolarConditionalBijection(AbstractBijection):
    dim: int = eqx.field(static=True, default=2, init=False)
    smoothness: int | None = eqx.field(static=True, default=None, init=False)

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
