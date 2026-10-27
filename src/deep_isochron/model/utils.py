"""A helper for the conditioners: zeroing an MLP's final layer so that a coupling
layer is the identity at init."""

import equinox as eqx
import jax.numpy as jnp


def zero_final_layer(mlp: eqx.nn.MLP) -> eqx.nn.MLP:
    """`mlp` with the weight and bias of its final layer set to zero."""
    mlp = eqx.tree_at(
        lambda x: (x.layers[-1].weight, x.layers[-1].bias),
        mlp,
        replace_fn=jnp.zeros_like,
    )
    return mlp
