"""Optimizer helpers: which parameters a regularizer may touch.

``optax.adamw``, like any decoupled weight decay, pulls every parameter toward zero.
For the learned bijection that is a sensible prior, since ``raw = 0`` is the identity
map (ADR-0002) and decay pulls the conjugacy toward the identity. For the latent
dynamics it is a bias on physical quantities under estimation: Bautin's ``raw_a`` and
``raw_b`` at zero are $a = 1$, $b = 0$, the Hopf form, ``w`` and ``w0`` at zero are a
frozen oscillator, and the phase autoencoder's ``omega`` and ``raw_decay`` likewise,
so the estimates of the period and the Floquet exponent would be systematically off.
``decay_mask`` builds the ``mask`` argument that keeps decay off those leaves;
``configs/optimizer/adamw.yaml`` uses it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import jax


def decay_mask(
    exclude: Sequence[str] = ("latent_dynamics",),
) -> Callable[[Any], Any]:
    """An ``optax`` mask: a function of the parameter pytree returning ``True`` for
    every leaf except those under an attribute named in `exclude`, anywhere on the
    leaf's path.

    Both project models keep their dynamics under ``latent_dynamics``, so the default
    masks exactly the normal-form and latent-dynamics parameters. Pass the result as
    ``optax.adamw(..., mask=decay_mask())`` or to ``optax.add_decayed_weights``.
    """
    names = tuple(exclude)

    def mask(params):
        def keep(path, _):
            return not any(getattr(key, "name", None) in names for key in path)

        return jax.tree_util.tree_map_with_path(keep, params)

    return mask
