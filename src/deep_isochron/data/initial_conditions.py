from __future__ import annotations

import abc
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from ..systems.normal_forms import (
    AbstractNormalForm,
)


# ------------------------------------------------------------------ IC samplers ---
class AbstractICSampler(eqx.Module):
    @abc.abstractmethod
    def __call__(self, key: PRNGKeyArray, n: int) -> Float[Array, "n dim"]: ...

    def params(self) -> dict[str, Any]:
        """JSON-able configuration for the metadata."""
        return {
            k: (v.tolist() if hasattr(v, "tolist") else v)
            for k, v in vars(self).items()
        }


class UniformBox(AbstractICSampler):
    """Uniform on the box ``[lo, hi]`` (per-coordinate bounds)."""

    lo: Float[Array, " dim"]
    hi: Float[Array, " dim"]

    def __init__(self, lo, hi):
        self.lo, self.hi = jnp.asarray(lo, dtype=float), jnp.asarray(hi, dtype=float)
        if self.lo.shape != self.hi.shape or not bool(jnp.all(self.lo < self.hi)):
            raise ValueError("lo and hi must have the same shape with lo < hi.")

    def __call__(self, key, n):
        return jax.random.uniform(
            key, (n, self.lo.shape[0]), minval=self.lo, maxval=self.hi
        )


class UniformAnnulus(AbstractICSampler):
    """Uniform in angle and in radius on ``[r_min, r_max]`` about ``center`` (planar);
    avoids the origin of a normal form."""

    r_min: float
    r_max: float
    center: Float[Array, " 2"]

    def __init__(self, r_min: float, r_max: float, center=(0.0, 0.0)):
        if not 0 <= r_min < r_max:
            raise ValueError("need 0 <= r_min < r_max.")
        self.r_min, self.r_max = float(r_min), float(r_max)
        self.center = jnp.asarray(center, dtype=float)

    def __call__(self, key, n):
        kr, kt = jax.random.split(key)
        r = jax.random.uniform(kr, (n,), minval=self.r_min, maxval=self.r_max)
        theta = jax.random.uniform(kt, (n,), minval=-jnp.pi, maxval=jnp.pi)
        return self.center + jnp.stack((r * jnp.cos(theta), r * jnp.sin(theta)), -1)


class OnCycleGaussian(AbstractICSampler):
    r"""Yawata et al. (Chaos 34, 063111, 2024), Eqs. (27)–(28): a point of the limit
    cycle, drawn uniformly from ``cycle_points``, plus Gaussian noise
    $\gamma_2\,\sigma \odot \xi$ with $\sigma$ the per-coordinate standard deviation of
    the cycle points and $\xi \sim \mathcal N(0, I)$ (their $\gamma_2 = 0.5$). The
    paper evolves each initial state for $\gamma_1 T = 3T$; that is the ``ts`` passed
    to ``generate``.

    ``cycle_points`` come from ``AbstractNormalForm.limit_cycle`` (``from_normal_form``)
    or, for an observed system, from a long integration of one orbit.
    """

    cycle_points: Float[Array, "points dim"]
    gamma2: float

    def __init__(self, cycle_points, gamma2: float = 0.5):
        self.cycle_points = jnp.asarray(cycle_points, dtype=float)
        if self.cycle_points.ndim != 2 or self.cycle_points.shape[0] < 2:
            raise ValueError("cycle_points must be (points, dim) with >= 2 points.")
        if gamma2 < 0:
            raise ValueError("gamma2 must be non-negative.")
        self.gamma2 = float(gamma2)

    @classmethod
    def from_normal_form(
        cls,
        normal_form: AbstractNormalForm,
        num_points: int = 1000,
        gamma2: float = 0.5,
    ) -> "OnCycleGaussian":
        theta = jnp.linspace(-jnp.pi, jnp.pi, num_points, endpoint=False)
        return cls(jax.vmap(normal_form.limit_cycle)(theta), gamma2)

    @property
    def sigma(self) -> Float[Array, " dim"]:
        return jnp.std(self.cycle_points, axis=0)

    def params(self) -> dict[str, Any]:
        return {
            "gamma2": self.gamma2,
            "num_cycle_points": int(self.cycle_points.shape[0]),
            "sigma": self.sigma.tolist(),
        }

    def __call__(self, key, n):
        k_idx, k_xi = jax.random.split(key)
        idx = jax.random.randint(k_idx, (n,), 0, self.cycle_points.shape[0])
        xi = jax.random.normal(k_xi, (n, self.cycle_points.shape[1]))
        return self.cycle_points[idx] + self.gamma2 * self.sigma * xi
