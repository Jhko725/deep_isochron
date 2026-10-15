"""Optimizer helpers: which parameters a regularizer may touch.

``optax.adamw`` (and any decoupled weight decay) pulls every parameter toward zero. For
the learned bijection that is a sensible prior — ``raw = 0`` is the identity map
(ADR-0002), so decay pulls the conjugacy toward the identity — but for the latent
dynamics it is a bias on physical quantities under estimation: Bautin's ``raw_a``,
``raw_b`` at zero are $a = 1$, $b = 0$ (the Hopf form), and ``w``, ``w0`` at zero are a
frozen oscillator; the phase autoencoder's ``omega`` and ``raw_decay`` likewise. The
estimates of the conjugacy invariants (period, Floquet exponent) would then be
systematically off. ``decay_mask`` builds the ``mask`` argument that keeps decay off
those leaves; the config ``configs/optimizer/adamw.yaml`` uses it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import jax


def decay_mask(
    exclude: Sequence[str] = ("latent_dynamics",),
) -> Callable[[Any], Any]:
    """An ``optax`` mask: a function of the parameter pytree returning ``True`` for
    every leaf except those under an attribute named in ``exclude`` (anywhere on the
    leaf's path). Both project models keep their dynamics under ``latent_dynamics``, so
    the default masks exactly the normal-form / latent-dynamics parameters. Pass it as
    ``optax.adamw(..., mask=decay_mask())`` (or ``optax.add_decayed_weights``)."""
    names = tuple(exclude)

    def mask(params):
        def keep(path, _):
            return not any(getattr(key, "name", None) in names for key in path)

        return jax.tree_util.tree_map_with_path(keep, params)

    return mask
