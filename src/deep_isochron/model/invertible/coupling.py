from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array, Float, PRNGKeyArray

from ..utils import zero_final_layer
from .base import AbstractBijection


class CouplingFlow[B: AbstractBijection](AbstractBijection):
    """An invertible coupling flow layer, as described in [1].

    Inputs are split into a constant and coupling part. The constant part is passed to a
    multilayer perceptron whose output is mapped, through
    ``template.from_unconstrained``, into one scalar bijection per coupled dimension.
    Output is obtained by concatenating the two pieces.

    The conditioner's final layer is zero-initialised, and ``from_unconstrained(0)`` is
    the identity for every scalar bijection, so a fresh layer is the identity map.

    [1] G. Papamakarios et al. Normalizing Flows for Probabilistic Modeling and
    Inference. JMLR 22 (2021)."""

    mlp: eqx.nn.MLP
    # The template's array leaves are dropped (replaced by None) so that it is a
    # hashable static field and its values are not counted as trainable state;
    # `from_unconstrained` only needs its static configuration (bin count, range, ...).
    template: B = eqx.field(static=True)

    dim: int = eqx.field(static=True)
    split_idx: int = eqx.field(static=True)
    flip: bool = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        bijection: B,
        split_idx: int | None = None,
        flip: bool = False,
        mlp_width: int = 10,
        mlp_depth: int = 1,
        activation: Callable = jax.nn.gelu,
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
            if not (0 < split_idx < dim):
                raise ValueError("split_idx must be strictly between 0 and dim.")
            self.split_idx = split_idx

        self.flip = flip

        if bijection.dim != 1 or not hasattr(bijection, "from_unconstrained"):
            raise NotImplementedError(
                "Only scalar (dim=1) bijections with `from_unconstrained` are "
                "supported as coupling templates."
            )
        self.template = eqx.partition(bijection, eqx.is_array)[1]

        mlp = eqx.nn.MLP(
            in_size=self.split_idx,
            out_size=self.template.num_params * self.dim_coupled,
            width_size=mlp_width,
            depth=mlp_depth,
            activation=activation,
            dtype=dtype,
            key=key,
        )
        self.mlp = zero_final_layer(mlp)

    @property
    def dim_coupled(self) -> int:
        return self.dim - self.split_idx

    def split(
        self, x: Float[Array, " dim"]
    ) -> tuple[Float[Array, " dim_const"], Float[Array, " dim-dim_const"]]:
        """Split the input array into the constant and the coupled part."""
        x = jnp.flip(x) if self.flip else x
        x_const, x_coupled = jnp.split(x, [self.split_idx])
        return x_const, x_coupled

    def combine(
        self,
        y_const: Float[Array, " dim_const"],
        y_coupled: Float[Array, " dim_coupled"],
    ) -> Float[Array, " dim_const+dim_coupled"]:
        y = jnp.concatenate((y_const, y_coupled))
        return jnp.flip(y) if self.flip else y

    def make_bijection(self, x_const: Float[Array, " dim_const"]) -> B:
        """Returns a vmapped bijection, corresponding to a dim=1 bijection for each
        dimension of x(y)_coupled.

        This vmapped bijection must be called under vmap block, as outlined in equinox
        docs (https://docs.kidger.site/equinox/tricks/; see Ensembling section.)"""
        params = rearrange(self.mlp(x_const), "(D C) -> D C", D=self.dim_coupled)
        return eqx.filter_vmap(self.template.from_unconstrained)(params)

    def __call__(self, x: Float[Array, " dim"]) -> Float[Array, " dim"]:
        x_const, x_coupled = self.split(x)
        bijection = self.make_bijection(x_const)
        y_coupled = eqx.filter_vmap(lambda bij, x_: bij(x_))(bijection, x_coupled)
        return self.combine(x_const, y_coupled)

    def inverse(self, y: Float[Array, " dim"]) -> Float[Array, " dim"]:
        y_const, y_coupled = self.split(y)
        bijection = self.make_bijection(y_const)
        x_coupled = eqx.filter_vmap(lambda bij, y_: bij.inverse(y_))(
            bijection, y_coupled
        )
        return self.combine(y_const, x_coupled)
