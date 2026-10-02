import abc

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array, Float, PRNGKeyArray


class AbstractLatentDynamics(eqx.Module):
    """Latent dynamics of the non-invertible autoencoder baseline
    (``PhaseAmplitudeAutoencoder``): a closed-form flow ``(ts, y0) -> y[t]`` with no
    vector field. Distinct from ``systems.AbstractNormalForm``, which is an ODE with
    closed-form phase–amplitude structure and is what the conjugacy models use."""

    dim: eqx.AbstractVar[int]

    @abc.abstractmethod
    def __call__(
        self,
        ts: Float[Array, " time"],
        y0: Float[Array, " dim"],
    ) -> Float[Array, "time dim"]: ...


def solve_2x2_const_coeff_ode(
    eig: Float[Array, "2"],
    ts: Float[Array, " time"],
    y0: Float[Array, "2"],
) -> Float[Array, "time 2"]:
    at, bt = eig[0] * ts, eig[1] * ts
    exp_at, cos_bt, sin_bt = jnp.exp(at), jnp.cos(bt), jnp.sin(bt)
    y_t0: Float[Array, " time"] = exp_at * (y0[0] * cos_bt - y0[1] * sin_bt)
    y_t1: Float[Array, " time"] = exp_at * (y0[0] * sin_bt + y0[1] * cos_bt)
    return jnp.stack((y_t0, y_t1), axis=-1)


def solve_1x1_const_coeff_ode(
    c: Float[Array, ""],
    ts: Float[Array, " time"],
    y0: Float[Array, ""],
) -> Float[Array, " time"]:
    return y0 * jnp.exp(c * ts)


class LinearLatentDynamics(AbstractLatentDynamics):
    r"""Class representing linear latent dynamics of the form $\dot{z}=Az.

    $z$ and $A$ are both real, so the eigenvalues of $A$ are either real or complex
    conjugates.

    The trainable parameters `_eig_Re` correspond to the nontrivial real parts of the
    eigenvalues of $A$ - that is, the real part of the complex eigenvalues as well as
    the real eigenvalues.

    Likewise, `_eig_Im` denote the nontrivial imaginary parts, which are the pure
    imaginary eigenvalues and the imaginary part of the complex eigenvalues.

    Note that when `positive_real_eigenvalues=False`, -softplus(_eig_Re) corresponds to
    the actual non-positive real parts of the eigenvalues."""

    _eig_Re: Float[Array, " num_eig_comp+num_eig_real"]
    _eig_Im: Float[Array, " num_eig_imag+num_eig_comp"]

    dim: int = eqx.field(static=True)
    num_eig_imag: int = eqx.field(static=True)
    num_eig_comp: int = eqx.field(static=True)
    positive_real_eigs_allowed: bool = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        num_eig_imag: int = 1,
        num_eig_comp: int = 0,
        positive_real_eigs_allowed: bool = False,
        *,
        key: PRNGKeyArray,
    ):
        r"""
        **Arguments:**

        - dim: Dimension of the latent dynamics: i.e., $\dim(z)$.
        - num_eig_imag: Number of pure imaginary eigenvalue pairs of the coefficient
            matrix $A$.
        - num_eig_comp: Number of complex eigenvalue
        """
        self.dim = dim
        self.num_eig_imag = num_eig_imag
        self.num_eig_comp = num_eig_comp
        self.positive_real_eigs_allowed = positive_real_eigs_allowed

        if self.num_eig_real < 0:
            raise ValueError("""2*(number of pure imaginary or complex eigenvalue pairs)
             cannot be larger dimension of the dynamics (2*(num_eig_imag+num_eig_comp)<=
             dim)""")

        # TODO: How do I initialize these values?
        key_r, key_i = jax.random.split(key)
        self._eig_Re = jax.random.uniform(
            key_r, (self.num_eig_comp + self.num_eig_real,)
        )
        self._eig_Im = jax.random.uniform(
            key_i, (self.num_eig_imag + self.num_eig_comp,)
        )

    @property
    def num_eig_real(self) -> int:
        return self.dim - 2 * (self.num_eig_imag + self.num_eig_comp)

    @property
    def eigenvalues(
        self,
    ) -> tuple[
        Float[Array, "num_eig_imag+num_eig_comp 2"], Float[Array, " num_eig_real"]
    ]:
        r"""Return the complex and real eigenvalues of the
        $\text{dim}\times\text{dim}$ matrix $A$.

        Note that for the complex eigenvalues, only one of the conjugate pairs are
        returned. These are also ordered in the second dimension as pure imaginary /
        complex.
        """
        if not self.positive_real_eigs_allowed:
            _eig_Re = -jax.nn.softplus(self._eig_Re)
        eig_comp_re, eig_real = jnp.split(_eig_Re, [self.num_eig_comp])

        eig_comp_re = jnp.concatenate(
            [jnp.zeros_like(eig_comp_re, shape=(self.num_eig_imag,)), eig_comp_re]
        )
        eig_comp = jnp.stack((eig_comp_re, self._eig_Im), axis=-1)
        return eig_comp, eig_real

    def __call__(
        self,
        ts: Float[Array, " time"],
        y0: Float[Array, " dim"],
    ) -> Float[Array, "time dim"]:
        t = ts - ts[0]
        y0_comp, y0_real = jnp.split(y0, [2 * (self.num_eig_imag + self.num_eig_comp)])
        y0_comp = rearrange(y0_comp, "(n d)-> n d", d=2)
        eig_comp, eig_real = self.eigenvalues

        y_t_comp: Float[Array, "num_eig_comp time 2"] = jax.vmap(
            solve_2x2_const_coeff_ode, in_axes=(0, None, 0)
        )(eig_comp, t, y0_comp)
        y_t_comp: Float[Array, "time 2*num_eig_comp"] = rearrange(
            y_t_comp, "n t d -> t (n d)"
        )

        y_t_real: Float[Array, "time num_eig_real"] = jax.vmap(
            solve_1x1_const_coeff_ode, in_axes=(0, None, 0), out_axes=-1
        )(eig_real, t, y0_real)

        y_t: Float[Array, "dim time"] = jnp.concatenate((y_t_comp, y_t_real), axis=-1)
        return y_t
