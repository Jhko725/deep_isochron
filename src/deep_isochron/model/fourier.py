
import equinox as eqx
import jax
import jax.numpy as jnp
from equinox._misc import default_floating_dtype
from jaxtyping import Array, Float, PRNGKeyArray


class TruncatedFourier(eqx.Module):
    a: Float[Array, "dim order+1"]
    b: Float[Array, "dim order"]

    dim: int = eqx.field(static=True)
    order: int = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        order: int,
        init_scale: float = 0.1,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        key_a, key_b = jax.random.split(key)
        dtype = default_floating_dtype() if dtype is None else dtype

        self.a = jax.random.normal(shape=(
                dim,
                order + 1,
            ), dtype=dtype, key=key_a)*init_scale
        self.b = jax.random.normal(shape=(
                dim,
                order,
            ), dtype=dtype, key=key_b)*init_scale
        self.dim = dim
        self.order = order

    def __call__(self, theta: Float[Array, ""]) -> Float[Array, " dim"]:
        ktheta: Float[Array, " order+1"] = theta * jnp.arange(self.order + 1)
        acos_ktheta = jnp.sum(jnp.cos(ktheta) * self.a, axis=-1)
        bsin_ktheta = jnp.sum(jnp.sin(ktheta[1:]) * self.b, axis=-1)
        return acos_ktheta + bsin_ktheta
