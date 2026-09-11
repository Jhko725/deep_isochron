from typing import Any, ClassVar

import diffrax as dfx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

from ..misc import inv_softplus
from .base import AbstractODE


class HopfNormalForm(AbstractODE):
    dim: ClassVar[int] = 2  # ty: ignore

    _a: Float[Array, ""]
    w: Float[Array, ""]
    w0: Float[Array, ""] | None

    def __init__(self, a: float = 1.0, w: float = 1.0, w0: float | None = None):
        if a <= 0:
            raise ValueError("a must be positive.")

        self._a = inv_softplus(jnp.asarray(a))
        self.w = jnp.asarray(w)
        self.w0 = jnp.asarray(w0) if w0 is not None else None

    @property
    def a(self) -> Float[Array, ""]:
        return jax.nn.softplus(self._a)

    def rhs(
        self, t: Float[Array, ""], u: Float[Array, " {self.dim}"], args=None
    ) -> Float[Array, " {self.dim}"]:
        del t
        r, _ = u
        dr = self.a * r * (1 - r**2)
        if self.w0 is None:
            dtheta = self.w
        else:
            dtheta = self.w0 + (self.w - self.w0) * r**2
        return jnp.stack((dr, dtheta))

    def period(self) -> Float[Array, ""]:
        return 2 * jnp.pi / self.w

    def eigenvalues_origin(self) -> Complex[Array, "2"]:
        real = self.a * jnp.asarray([1, 1])
        if self.w0 is None:
            imag = jnp.zeros(2)
        else:
            imag = self.w0 * jnp.asarray([1, -1])
        return jax.lax.complex(real, imag)

    def floquet_nontrivial(self) -> Float[Array, ""]:
        return jnp.exp(-4 * jnp.pi * self.a / self.w)


class BautinNormalForm(AbstractODE):
    dim: ClassVar[int] = 2  # ty: ignore

    _a: Float[Array, ""]
    _b: Float[Array, ""]
    w: Float[Array, ""]
    w0: Float[Array, ""] | None

    def __init__(
        self, a: float = 1.0, b: float = 0.1, w: float = 1.0, w0: float | None = None
    ):
        if a <= 0:
            raise ValueError("a must be positive.")
        # self._a = inv_softplus(jnp.asarray(a))
        self._a = jnp.log(jnp.asarray(a))

        if b < 0:
            raise ValueError("b must be nonnegative")
        # self._b = inv_softplus(jnp.asarray(b))
        self._b = jnp.log(jnp.asarray(b))

        self.w = jnp.asarray(w)
        self.w0 = jnp.asarray(w0) if w0 is not None else None

    @property
    def a(self) -> Float[Array, ""]:
        # return jax.nn.softplus(self._a)
        return jnp.exp(self._a)

    @property
    def b(self) -> Float[Array, ""]:
        # return jax.nn.softplus(self._b)
        return jnp.exp(self._b)

    def rhs(
        self, t: Float[Array, ""], u: Float[Array, " {self.dim}"], args=None
    ) -> Float[Array, " {self.dim}"]:
        del t
        r, _ = u
        dr = self.a * r * (1 - r**2) * (1 + self.b * r**2)
        if self.w0 is None:
            dtheta = self.w
        else:
            dtheta = self.w0 + (self.w - self.w0) * r**2
        return jnp.stack((dr, dtheta))

    def period(self) -> Float[Array, ""]:
        return 2 * jnp.pi / self.w

    def eigenvalues_origin(self) -> Complex[Array, "2"]:
        real = self.a * jnp.asarray([1, 1])
        if self.w0 is None:
            imag = jnp.zeros(2)
        else:
            imag = self.w0 * jnp.asarray([1, -1])
        return jax.lax.complex(real, imag)

    def floquet_nontrivial(self) -> Float[Array, ""]:
        return jnp.exp(-4 * jnp.pi * self.a * (1 + self.b) / self.w)

    def solve(
        self,
        ts: Float[Array, " time"],
        u0: Float[Array, " {self.dim}"],
        args: Any = None,
        *,
        solver: dfx.AbstractAdaptiveSolver = dfx.Tsit5(),
        rtol: float = 1e-6,
        atol: float = 1e-8,
        max_steps=4096,
        **kwargs,
    ) -> Float[Array, "time {self.dim}"]:
        def _ode_r_squared(t, u, args=None):
            r_sq, _ = u
            dr_sq = 2 * self.a * r_sq * (1 - r_sq) * (1 + self.b * r_sq)
            return jnp.stack((dr_sq, r_sq), axis=0)

        r_sq0, theta0 = u0[0] ** 2, u0[1]
        sol = dfx.diffeqsolve(
            dfx.ODETerm(_ode_r_squared),  # ty: ignore
            solver=solver,
            t0=ts[0],
            t1=ts[-1],
            dt0=None,
            y0=jnp.stack((r_sq0, jnp.zeros_like(r_sq0))),
            args=args,
            saveat=dfx.SaveAt(ts=ts),
            stepsize_controller=dfx.PIDController(atol=atol, rtol=rtol),
            max_steps=max_steps,
        )
        r_sq = sol.ys[:, 0]
        if self.w0 is None:
            theta = theta0 + self.w * (ts - ts[0])
        else:
            theta = theta0 + self.w0 * (ts - ts[0]) + (self.w - self.w0) * sol.ys[:, 1]
        return jnp.stack((jnp.sqrt(r_sq), theta), axis=-1)
