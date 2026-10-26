"""The coupling layer, which lifts a scalar bijection to the plane by conditioning its
parameters on the other coordinates (Papamakarios et al. 2021)."""

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array, Float, PRNGKeyArray

from ..utils import zero_final_layer
from .base import AbstractBijection, AbstractScalarBijection


# jax.nn activations that are not C^1. A coupling layer is only as smooth as its
# conditioner, so these are rejected when the template is C^1 or better.
_NON_SMOOTH_ACTIVATIONS = frozenset(
    {
        jax.nn.relu,
        jax.nn.relu6,
        jax.nn.leaky_relu,
        jax.nn.hard_tanh,
        jax.nn.hard_sigmoid,
        jax.nn.hard_swish,
        jax.nn.hard_silu,
    }
)


def _as_template[B: AbstractScalarBijection](bijection: B) -> B:
    """`bijection` with every ``raw`` leaf set to ``None``, recursively for chains: a
    hashable static configuration with zero trainable size. By contract ``raw`` is
    the only array leaf of a scalar bijection, so dropping every array leaf gives the
    template."""
    return eqx.partition(bijection, eqx.is_array)[1]


class CouplingFlow[B: AbstractScalarBijection](AbstractBijection):
    r"""A coupling layer (Papamakarios et al. 2021).

    The input is split into a constant part and a coupled part. The constant part is
    passed to a conditioner, any module ``x_const -> raw`` with `raw` of length
    ``template.num_params * dim_coupled``, whose output is mapped through
    ``template.from_unconstrained`` into one scalar bijection per coupled dimension;
    the output concatenates the two pieces. The default conditioner is an MLP with a
    zero-initialized final layer, and since ``from_unconstrained(0)`` is the identity
    for every scalar bijection, a fresh layer is the identity map. A custom
    conditioner must output zeros at init for the same to hold, as
    ``PolarCouplingFlow``'s zero-initialized ``TruncatedFourier`` does.

    The layer is jointly $C^k$ in its input when the template is $C^k$ and the
    conditioner is at least $C^k$; `smoothness` reports the template's value, and the
    constructor rejects the non-smooth ``jax.nn`` activations of the default MLP when
    the template is $C^1$ or better. Any other activation, and any custom
    conditioner, is assumed to be $C^\infty$.

    **Arguments:**

    - `dim`: The input and output dimension, at least 2.
    - `bijection`: The scalar template; its ``raw`` leaves are dropped.
    - `split_idx`: The size of the constant part; ``dim // 2`` by default.
    - `flip`: Whether to reverse the input before splitting, which swaps the roles
        of the halves.
    - `mlp_width`, `mlp_depth`, `activation`, `dtype`, `key`: The default MLP
        conditioner; `key` is required unless `conditioner` is given.
    - `conditioner`: A custom module ``x_const -> raw``, which overrides the MLP
        options.

    **Attributes:**

    - `conditioner`: The conditioner.
    - `template`: The bijection with ``raw=None``: a hashable static field with zero
        trainable size, since `from_unconstrained` only needs its static
        configuration.
    - `dim`, `smoothness`, `split_idx`, `flip`: As above.

    **Raises:** `ValueError` for ``dim < 2``, a `split_idx` outside ``(0, dim)``, a
    missing `key`, or a non-smooth activation with a $C^1$ or smoother template;
    `TypeError` for a template that is not an ``AbstractScalarBijection``.
    """

    conditioner: Callable[[Float[Array, " dim_const"]], Float[Array, " raw"]]
    template: B = eqx.field(static=True)

    dim: int = eqx.field(static=True)
    smoothness: int | None = eqx.field(static=True, init=False)
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
        conditioner: Callable[[Array], Array] | None = None,
        *,
        key: PRNGKeyArray | None = None,
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

        if not isinstance(bijection, AbstractScalarBijection):
            raise TypeError(
                "Coupling templates must be AbstractScalarBijection instances; got "
                f"{type(bijection).__name__}."
            )
        self.template = _as_template(bijection)
        self.smoothness = bijection.smoothness

        if conditioner is not None:
            self.conditioner = conditioner
        else:
            if key is None:
                raise ValueError("`key` is required for the default MLP conditioner.")
            if activation in _NON_SMOOTH_ACTIVATIONS and bijection.smoothness != 0:
                k = bijection.smoothness
                raise ValueError(
                    f"{getattr(activation, '__name__', activation)} is not C^1, but "
                    f"the template is C^{'∞' if k is None else k}; use a smooth "
                    "activation."
                )
            mlp = eqx.nn.MLP(
                in_size=self.split_idx,
                out_size=self.template.num_params * self.dim_coupled,
                width_size=mlp_width,
                depth=mlp_depth,
                activation=activation,
                dtype=dtype,
                key=key,
            )
            self.conditioner = zero_final_layer(mlp)

    @property
    def dim_coupled(self) -> int:
        """The size of the coupled part, ``dim - split_idx``."""
        return self.dim - self.split_idx

    def split(
        self, x: Float[Array, " dim"]
    ) -> tuple[Float[Array, " dim_const"], Float[Array, " dim-dim_const"]]:
        """``(x_const, x_coupled)``, after the flip when `flip` is set."""
        x = jnp.flip(x) if self.flip else x
        x_const, x_coupled = jnp.split(x, [self.split_idx])
        return x_const, x_coupled

    def combine(
        self,
        y_const: Float[Array, " dim_const"],
        y_coupled: Float[Array, " dim_coupled"],
    ) -> Float[Array, " dim_const+dim_coupled"]:
        """The concatenation of the two parts, flipped back when `flip` is set."""
        y = jnp.concatenate((y_const, y_coupled))
        return jnp.flip(y) if self.flip else y

    def make_bijection(self, x_const: Float[Array, " dim_const"]) -> B:
        """The scalar bijections of the coupled dimensions for `x_const`, stacked
        along a leading axis: one template filled per dimension from the
        conditioner's output. The stacked bijection must be called under ``vmap``, as
        in equinox's ensembling trick (https://docs.kidger.site/equinox/tricks/)."""
        raw = self.conditioner(x_const)
        params = rearrange(raw, "(D C) -> D C", D=self.dim_coupled)
        return eqx.filter_vmap(self.template.from_unconstrained)(params)

    def __call__(self, x: Float[Array, " dim"]) -> Float[Array, " dim"]:
        """The coupled part transformed by the bijections the constant part
        conditions; the constant part passes through."""
        x_const, x_coupled = self.split(x)
        bijection = self.make_bijection(x_const)
        y_coupled = eqx.filter_vmap(lambda bij, x_: bij(x_))(bijection, x_coupled)
        return self.combine(x_const, y_coupled)

    def inverse(self, y: Float[Array, " dim"]) -> Float[Array, " dim"]:
        """The inverse, by the same bijections, since the constant part is
        unchanged."""
        y_const, y_coupled = self.split(y)
        bijection = self.make_bijection(y_const)
        x_coupled = eqx.filter_vmap(lambda bij, y_: bij.inverse(y_))(
            bijection, y_coupled
        )
        return self.combine(y_const, x_coupled)
