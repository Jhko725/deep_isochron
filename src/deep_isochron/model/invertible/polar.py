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

    A scalar bijection in its own right (``R -> R``), delegating ``num_params``,
    ``constrain`` and ``smoothness`` to ``f``; usable standalone, as a chain member, or
    as a coupling template. ``raw`` lives in the wrapped bijection.
    """

    bijection: AbstractScalarBijection
    raw: None = eqx.field(static=True, default=None, init=False)
    smoothness: int | None = eqx.field(static=True, init=False)

    def __init__(self, bijection: AbstractScalarBijection):
        if not isinstance(bijection, AbstractScalarBijection):
            raise TypeError("OffsetedBijection wraps an AbstractScalarBijection.")
        self.bijection = bijection
        self.smoothness = bijection.smoothness

    @property
    def num_params(self) -> int:
        return self.bijection.num_params

    def constrain(self, raw):
        return self.bijection.constrain(raw)

    @property
    def params(self):
        return self.bijection.params

    def from_unconstrained(self, params_raw):
        return eqx.tree_at(
            lambda m: m.bijection, self, self.bijection.from_unconstrained(params_raw)
        )

    def __call__(self, x: Float[Array, ""]) -> Float[Array, ""]:
        return self.bijection(x) - self.bijection(jnp.zeros_like(x))

    def inverse(self, y: Float[Array, ""]) -> Float[Array, ""]:
        offset = self.bijection(jnp.zeros_like(y))
        return self.bijection.inverse(y + offset)


def _scaled_polar(
    xy: Float[Array, " 2"], center, scale, eps_r: float
) -> tuple[Float[Array, " 2"], Float[Array, ""]]:
    """``(x - center) * scale`` and its regularised radius ``sqrt(|.|² + eps_r²)``."""
    xy_s = (xy - center) * scale
    return xy_s, jnp.sqrt(jnp.sum(xy_s**2) + eps_r**2)


class RadialBijection(AbstractBijection):
    """``x -> center + (g(r) / r) (x - center)`` in the metric ``scale``: a radial
    rescaling by the origin-fixing scalar bijection ``g = OffsetedBijection(f)``.

    The radius is regularised as ``r = sqrt(|x_s|² + eps_r²)`` only *inside* ``g`` and
    the ratio ``g(r) / r``, so the origin maps to the origin exactly and the Jacobian
    there is finite. ``f`` is a standalone trainable scalar bijection.
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
        return jnp.exp(self.log_scale)

    def __call__(self, x: Float[Array, " 2"]) -> Float[Array, " 2"]:
        x_s, r_in = _scaled_polar(x, self.center, self.scale, self.eps_r)
        r_out = self.radial_bijection(r_in)
        return (r_out / r_in) * x_s / self.scale + self.center

    def inverse(self, y: Float[Array, " 2"]) -> Float[Array, " 2"]:
        y_s, r_out = _scaled_polar(y, self.center, self.scale, self.eps_r)
        r_in = self.radial_bijection.inverse(r_out)
        return (r_in / r_out) * y_s / self.scale + self.center


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


class PolarCouplingFlow(AbstractBijection):
    """A coupling layer in polar coordinates: the radius is transformed by a scalar
    bijection whose parameters are a function of the angle,
    ``(r, θ) -> (g_θ(r), θ)`` with ``g_θ =
    template.from_unconstrained(conditioner(θ))``.

    This is ``CouplingFlow`` with the chart ``(r, θ)`` in place of ``(x_coupled,
    x_const)`` and a ``TruncatedFourier`` conditioner in place of the MLP (the angle is
    periodic, so a Fourier series is the natural conditioner). The template is wrapped
    in ``OffsetedBijection`` so that ``g_θ(0) = 0`` and ``R_+ -> R_+``. The Fourier
    coefficients are zero-initialised, so the layer is the identity at init.

    ``center`` and ``log_scale`` set the chart's origin and metric; the radius is
    regularised as in ``RadialBijection``. The angle is read through a ``where``-safe
    ``arctan2`` (fill ``0`` inside the ``eps_r`` disc), so the origin is a fixed point.

    **Regularity.** ``smoothness`` is the template's: the Fourier conditioner is C^∞ and
    the composition in ``(r, θ)`` is as smooth as ``g``; at the origin the map is
    continuous (the angular dependence of ``g_θ`` is not differentiable there), as for
    any angle-conditioned map.
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
        """**Arguments:**

        - ``bijection``: scalar template (``R -> R``); wrapped in ``OffsetedBijection``.
        - ``fourier_order``: highest harmonic of the angular conditioner.
        - ``center``, ``log_scale``: chart origin and log-metric (default: 0).
        - ``eps_r``: radius regularisation.
        - ``key``: unused (kept for builder compatibility; the layer is
        zero-initialised).
        """
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
        return jnp.exp(self.log_scale)

    def _angle(self, xy_s: Float[Array, " 2"]) -> Float[Array, ""]:
        x, y = xy_s
        small = x**2 + y**2 < self.eps_r**2
        return jnp.where(
            small, 0.0, jnp.arctan2(jnp.where(small, 0.0, y), jnp.where(small, 1.0, x))
        )

    def radial_bijection(self, theta: Float[Array, ""]) -> AbstractScalarBijection:
        return self.template.from_unconstrained(self.conditioner(theta))

    def __call__(self, x: Float[Array, " 2"]) -> Float[Array, " 2"]:
        x_s, r_in = _scaled_polar(x, self.center, self.scale, self.eps_r)
        r_out = self.radial_bijection(self._angle(x_s))(r_in)
        return (r_out / r_in) * x_s / self.scale + self.center

    def inverse(self, y: Float[Array, " 2"]) -> Float[Array, " 2"]:
        y_s, r_out = _scaled_polar(y, self.center, self.scale, self.eps_r)
        r_in = self.radial_bijection(self._angle(y_s)).inverse(r_out)
        return (r_in / r_out) * y_s / self.scale + self.center
