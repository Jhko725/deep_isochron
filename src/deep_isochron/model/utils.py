import equinox as eqx
import jax.numpy as jnp


def zero_final_layer(mlp: eqx.nn.MLP) -> eqx.nn.MLP:
    mlp = eqx.tree_at(
        lambda x: (x.layers[-1].weight, x.layers[-1].bias),
        mlp,
        replace_fn=jnp.zeros_like,
    )
    return mlp
