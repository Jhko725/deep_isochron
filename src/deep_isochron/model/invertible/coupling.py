from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
from einops import rearrange
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, Float, PRNGKeyArray

from ..utils import zero_final_layer
from .base import AbstractBijection


class BijectionFactory[B: AbstractBijection](eqx.Module):
    _bijection_static: B = eqx.field(static=True)
    unflatten_fn: Callable[[Float[Array, " params"]], B] = eqx.field(static=True)
    num_params: int = eqx.field(static=True)

    def __init__(self, bijection: B):
        params, static = eqx.partition(bijection, eqx.is_inexact_array)
        params_flat, unflatten_fn = ravel_pytree(params)

        self._bijection_static = static
        self.unflatten_fn = unflatten_fn
        self.num_params = len(params_flat)

    def __call__(self, params: Float[Array, " params"]) -> B:
        params = self.unflatten_fn(params)
        return eqx.combine(params, self._bijection_static)


class CouplingFlow[B: AbstractBijection](AbstractBijection):
    """An invertible coupling flow layer, as described in [1].

    Inputs are split into a constant and coupling part. The constant part is passed to a
    multilayer perceptron, and mapped into parameters of a bijection. This bijection is
    then used to transform the coupling part. Output is obtained by concatenating the
    two pieces.

    [1] G. Papamakarios et al. Normalizing Flows for Probabilistic Modeling and
    Inference. JMLR 22 (2021)."""

    mlp: eqx.nn.MLP
    bijection_factory: BijectionFactory[B]

    dim: int = eqx.field(static=True)
    split_idx: int = eqx.field(static=True)
    flip: bool = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        bijection: AbstractBijection,
        split_idx: int | None = None,
        flip: bool = False,
        mlp_width: int = 10,
        mlp_depth: int = 1,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        if dim < 2:
            raise ValueError("Dimension of a coupling flow cannot be less than 2.")
        self.dim = dim

        if split_idx is None:
            self.split_idx = dim // 2
        else:
            if (0 < split_idx) and (split_idx < self.dim):
                raise ValueError("split_idx must be between 0 and dim.")
            self.split_idx = split_idx

        self.flip = flip

        if bijection.dim > 1:
            raise NotImplementedError("""Support for higher dimensional bijections not 
            implemented yet.""")
        self.bijection_factory = BijectionFactory(bijection)

        mlp = eqx.nn.MLP(
            in_size=self.split_idx,
            out_size=self.bijection_factory.num_params * self.dim_coupled,
            width_size=mlp_width,
            depth=mlp_depth,
            dtype=dtype,
            key=key,
        )
        self.mlp = zero_final_layer(mlp)

    @property
    def dim_coupled(self) -> int:
        return self.dim - self.split_idx

    def split(
        self, x: Float[Array, " dim"]
    ) -> tuple[Float[Array, " dim-dim_cond"], Float[Array, " dim_cond"]]:
        """Split the input array into two parts."""
        x = jnp.flip(x) if self.flip else x
        return jnp.split(x, [self.split_idx])

    def combine(
        self, y_const: Float[Array, " dim-dim_cond"], y_cond: Float[Array, " dim_cond"]
    ) -> Float[Array, " dim"]:
        y = jnp.concatenate((y_const, y_cond))
        y = jnp.flip(y) if self.flip else y
        return y

    def make_bijection(self, x_const: Float[Array, " dim_const"]) -> B:
        """Returns a vmapped bijection, corresponding to a dim=1 bijection for each
        dimension of x(y)_coupled.

        This vmapped bijection must be called under vmap block, as outlined in equinox
        docs (https://docs.kidger.site/equinox/tricks/; see Ensembling section.)"""
        params = rearrange(self.mlp(x_const), "(D C) -> D C", D=self.dim_coupled)
        return eqx.filter_vmap(self.bijection_factory)(params)

    def __call__(self, x: Float[Array, " dim"]) -> Float[Array, " dim"]:
        x_const, x_coupled = self.split(x)
        bijection: B = self.make_bijection(x_const)
        y_const, y_coupled = (
            x_const,
            eqx.filter_vmap(lambda bij, x_: bij(x_))(bijection, x_coupled),
        )
        return self.combine(y_const, y_coupled)

    def inverse(self, y: Float[Array, " dim"]) -> Float[Array, " dim"]:
        y_const, y_coupled = self.split(y)
        bijection: B = self.make_bijection(y_const)
        x_const, x_coupled = (
            y_const,
            eqx.filter_vmap(lambda bij, y_: bij.inverse(y_))(bijection, y_coupled),
        )
        return self.combine(x_const, x_coupled)
