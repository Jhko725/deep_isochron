"""Bijections in polar coordinates: a radial rescaling, an angular spline, and the
coupling layer that conditions a radial bijection on the angle."""

import math

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from ..fourier import TruncatedFourier
from .base import AbstractBijection, AbstractScalarBijection
from .splines import MonotonicRQSpline


class OffsetedBijection(AbstractScalarBijection[tuple]):
    """``g(r) = f(r) - f(0)`` for a scalar bijection ``f``: the increasing map that
    fixes the origin, so it sends ``R_+`` onto ``R_+`` and can act on a radius.

    A scalar bijection in its own right, delegating `num_params`, `constrain` and
    `smoothness` to ``f``; usable standalone, as a chain member, or as a coupling
    template. ``raw`` lives in the wrapped bijection.

    **Raises:** `TypeError` for a wrapped object that is not a scalar bijection.
    """

    bijection: AbstractScalarBijection
    raw: Float[Array, " num_params"] | None = eqx.field(
        static=True, default=None, init=False
    )
    num_params: int = eqx.field(static=True, init=False)
    smoothness: int | None = eqx.field(static=True, init=False)

    def __init__(self, bijection: AbstractScalarBijection):
        if not isinstance(bijection, AbstractScalarBijection):
            raise TypeError("OffsetedBijection wraps an AbstractScalarBijection.")
        self.bijection = bijection
        self.num_params = bijection.num_params
        self.smoothness = bijection.smoothness

    def constrain(self, raw):
        """The wrapped bijection's `constrain`."""
        return self.bijection.constrain(raw)

    @property
    def params(self):
        """The wrapped bijection's `params`."""
        return self.bijection.params

    def from_unconstrained(self, params_raw):
        """A copy wrapping ``f.from_unconstrained(params_raw)``."""
        return eqx.tree_at(
            lambda m: m.bijection, self, self.bijection.from_unconstrained(params_raw)
        )

    def __call__(self, x: Float[Array, ""]) -> Float[Array, ""]:
        """``f(x) - f(0)``."""
        return self.bijection(x) - self.bijection(jnp.zeros_like(x))

    def inverse(self, y: Float[Array, ""]) -> Float[Array, ""]:
        """``f⁻¹(y + f(0))``."""
        offset = self.bijection(jnp.zeros_like(y))
        return self.bijection.inverse(y + offset)


def _scaled_polar(
    xy: Float[Array, " 2"], center, scale, eps_r: float
) -> tuple[Float[Array, " 2"], Float[Array, ""]]:
    """``(x - center) * scale`` and its regularized radius ``sqrt(|·|² + eps_r²)``."""
    xy_s = (xy - center) * scale
    return xy_s, jnp.sqrt(jnp.sum(xy_s**2) + eps_r**2)


class RadialBijection(AbstractBijection):
    """``x -> center + (g(r) / r) (x - center)`` in the metric `scale`: a radial
    rescaling by the origin-fixing scalar bijection ``g = OffsetedBijection(f)``.

    The radius is regularized as ``r = sqrt(|x_s|² + eps_r²)`` only inside ``g`` and
    the ratio ``g(r) / r``, so the origin maps to the origin exactly and the Jacobian
    there is finite.

    **Arguments:**

    - `bijection`: ``f``, a standalone trainable scalar bijection.
    - `center`, `log_scale`: The chart's origin and log-metric, shape ``(2,)``.
    - `eps_r`: The radius regularization.
    """

    center: Float[Array, " 2"]
    log_scale: Float[Array, " 2"]
    radial_bijection: OffsetedBijection
    dim: int = eqx.field(static=True, default=2, init=False)
    smoothness: int | None = eqx.field(static=True, init=False)
    eps_r: float = eqx.field(static=True)

    def __init__(
        self,
        bijection: AbstractScalarBijection,
        center: Float[Array, " 2"],
        log_scale: Float[Array, " 2"],
        eps_r: float = 1e-7,
    ):
        self.center = center
        self.log_scale = log_scale
        self.radial_bijection = OffsetedBijection(bijection)
        self.eps_r = eps_r
        self.smoothness = bijection.smoothness

    @property
    def scale(self) -> Float[Array, " 2"]:
        """``exp(log_scale)``."""
        return jnp.exp(self.log_scale)

    def __call__(self, x: Float[Array, " 2"]) -> Float[Array, " 2"]:
        """The radial rescaling by ``g(r) / r``."""
        x_s, r_in = _scaled_polar(x, self.center, self.scale, self.eps_r)
        r_out = self.radial_bijection(r_in)
        return (r_out / r_in) * x_s / self.scale + self.center

    def inverse(self, y: Float[Array, " 2"]) -> Float[Array, " 2"]:
        """The radial rescaling by ``g⁻¹(r) / r``."""
        y_s, r_out = _scaled_polar(y, self.center, self.scale, self.eps_r)
        r_in = self.radial_bijection.inverse(r_out)
        return (r_in / r_out) * y_s / self.scale + self.center


class CircularMonotonicRQCoupling(AbstractBijection):
    """An angular rational-quadratic spline in polar coordinates,
    ``(r, θ) -> (r, s(θ))`` with ``s`` a `MonotonicRQSpline` on ``(-π, π)``, after
    Durkan et al. 2019 and the nflows implementation.

    The spline is held as a standalone trainable bijection, whose ``raw`` leaf is
    unconstrained, so training cannot break monotonicity. The map is applied as a
    rotation of the input by ``Δ(θ) = s(θ) - θ``, which preserves ``|x|`` exactly and
    is exactly invertible by rotating back through ``s⁻¹(θ') - θ'``. The angle is
    read through a ``where``-safe ``arctan2`` that inside the disc ``|x| < eps_r``
    returns a fill angle, 0 for the forward map and ``s(0)`` for the inverse, so the
    origin maps to itself, the Jacobians there are the finite rotations ``R(s(0))``
    and ``R(-s(0))``, and away from the origin ``det J = s'(θ) > 0``.

    As a map of the punctured plane it is as smooth as ``s``; at the origin it is
    continuous but not differentiable, being a ``θ``-dependent rotation, and across
    ``θ = ±π`` it is only $C^0$ because the spline's boundary derivatives are pinned
    to 1, hence ``smoothness = 0``. Restoring circular derivative matching needs an
    endpoint option on ``AbstractSpline``, deferred on purpose.

    **Arguments:**

    - `num_bins`: The spline's bins.
    - `raw`: The spline's raw parameters; the identity when omitted.
    - `min_rel_width`, `min_derivative`: The spline's floors.
    - `eps_r`: The radius of the fill disc.
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
        """The angle of `xy` by ``arctan2`` with a double-``where`` guard: inputs
        inside the ``eps_r`` disc are replaced by ``(1, 0)`` before ``arctan2``, so
        neither branch produces ``nan`` gradients (the JAX FAQ on ``where``), and the
        angle returned there is `fill`."""
        x, y = xy
        small = x**2 + y**2 < self.eps_r**2
        x_s = jnp.where(small, 1.0, x)
        y_s = jnp.where(small, 0.0, y)
        return jnp.where(small, fill, jnp.arctan2(y_s, x_s))

    @staticmethod
    def _rotate(xy: Float[Array, " 2"], delta: Float[Array, ""]) -> Float[Array, " 2"]:
        """`xy` rotated by `delta`."""
        c, s = jnp.cos(delta), jnp.sin(delta)
        x, y = xy
        return jnp.stack([c * x - s * y, s * x + c * y])

    def __call__(self, x: Float[Array, " 2"]) -> Float[Array, " 2"]:
        """`x` rotated by ``s(θ) - θ``."""
        theta = self._safe_angle(x, jnp.zeros(()))
        return self._rotate(x, self.spline(theta) - theta)

    def inverse(self, y: Float[Array, " 2"]) -> Float[Array, " 2"]:
        """`y` rotated by ``s⁻¹(θ') - θ'``."""
        theta = self._safe_angle(y, self.spline(jnp.zeros(())))
        return self._rotate(y, self.spline.inverse(theta) - theta)


class PolarCouplingFlow(AbstractBijection):
    r"""A coupling layer in polar coordinates, ``(r, θ) -> (g_θ(r), θ)`` with
    ``g_θ = template.from_unconstrained(conditioner(θ))``: the radius is transformed
    by a scalar bijection whose parameters are a function of the angle.

    This is `CouplingFlow` with the chart ``(r, θ)`` in place of
    ``(x_coupled, x_const)`` and a ``TruncatedFourier`` conditioner in place of the
    MLP, since the angle is periodic. The template is wrapped in `OffsetedBijection`
    so that ``g_θ(0) = 0`` and ``R_+ -> R_+``; the Fourier coefficients are
    zero-initialized, so the layer is the identity at init. The radius is regularized
    as in `RadialBijection`, and the angle is read through a ``where``-safe
    ``arctan2`` with fill 0 inside the ``eps_r`` disc, so the origin is a fixed point.
    `smoothness` is the template's: the Fourier conditioner is $C^\infty$ and the
    composition in ``(r, θ)`` is as smooth as ``g``, while at the origin the map is
    continuous only, as for any angle-conditioned map.

    **Arguments:**

    - `bijection`: The scalar template, wrapped in `OffsetedBijection`.
    - `fourier_order`: The highest harmonic of the angular conditioner.
    - `center`, `log_scale`: The chart's origin and log-metric; 0 by default.
    - `eps_r`: The radius regularization.
    - `key`: Unused, kept for builder compatibility; the layer is zero-initialized.

    **Raises:** `TypeError` for a template that is not a scalar bijection.
    """

    conditioner: TruncatedFourier
    center: Float[Array, " 2"]
    log_scale: Float[Array, " 2"]
    template: AbstractScalarBijection = eqx.field(static=True)
    dim: int = eqx.field(static=True, default=2, init=False)
    smoothness: int | None = eqx.field(static=True, init=False)
    eps_r: float = eqx.field(static=True)

    def __init__(
        self,
        bijection: AbstractScalarBijection,
        fourier_order: int = 3,
        *,
        center: Float[Array, " 2"] | None = None,
        log_scale: Float[Array, " 2"] | None = None,
        eps_r: float = 1e-7,
        key: PRNGKeyArray | None = None,
    ):
        del key
        if not isinstance(bijection, AbstractScalarBijection):
            raise TypeError(
                "PolarCouplingFlow needs an AbstractScalarBijection template."
            )
        template = OffsetedBijection(bijection)
        self.template = eqx.partition(template, eqx.is_array)[1]
        self.smoothness = template.smoothness
        self.conditioner = TruncatedFourier(
            dim=template.num_params,
            order=fourier_order,
            init_scale=0.0,
            key=jax.random.key(0),
        )
        self.center = jnp.zeros(2) if center is None else center
        self.log_scale = jnp.zeros(2) if log_scale is None else log_scale
        self.eps_r = eps_r

    @property
    def scale(self) -> Float[Array, " 2"]:
        """``exp(log_scale)``."""
        return jnp.exp(self.log_scale)

    def _angle(self, xy_s: Float[Array, " 2"]) -> Float[Array, ""]:
        """The angle of `xy_s` by a double-``where`` guarded ``arctan2``, 0 inside the
        ``eps_r`` disc."""
        x, y = xy_s
        small = x**2 + y**2 < self.eps_r**2
        return jnp.where(
            small, 0.0, jnp.arctan2(jnp.where(small, 0.0, y), jnp.where(small, 1.0, x))
        )

    def radial_bijection(self, theta: Float[Array, ""]) -> AbstractScalarBijection:
        """``g_θ``, the template filled from the conditioner at `theta`."""
        return self.template.from_unconstrained(self.conditioner(theta))

    def __call__(self, x: Float[Array, " 2"]) -> Float[Array, " 2"]:
        """The radial rescaling by ``g_θ(r) / r``."""
        x_s, r_in = _scaled_polar(x, self.center, self.scale, self.eps_r)
        r_out = self.radial_bijection(self._angle(x_s))(r_in)
        return (r_out / r_in) * x_s / self.scale + self.center

    def inverse(self, y: Float[Array, " 2"]) -> Float[Array, " 2"]:
        """The radial rescaling by ``g_θ⁻¹(r) / r``, with the same ``θ``."""
        y_s, r_out = _scaled_polar(y, self.center, self.scale, self.eps_r)
        r_in = self.radial_bijection(self._angle(y_s)).inverse(r_out)
        return (r_in / r_out) * y_s / self.scale + self.center
