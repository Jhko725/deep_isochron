"""From the ``model`` config group to an ``AbstractPhaseAmplitudeModel``.

Leaf objects are Hydra ``_target_`` entries instantiated as written; what a builder adds
is the part a config cannot express — PRNG keys split per layer and the alternating
``flip`` of the coupling layers. ``model.kind`` selects the builder.
"""

from __future__ import annotations

import jax
from jaxtyping import PRNGKeyArray
from omegaconf import DictConfig

from ..model import (
    AbstractPhaseAmplitudeModel,
    ConjugateLatentDynamics,
    PhaseAmplitudeAutoencoder,
)
from ..model.invertible import AbstractBijection, SequentialINN
from .instantiate import instantiate


def build_inn(inn_cfg: DictConfig, key: PRNGKeyArray) -> SequentialINN:
    """``blocks`` × (``linear`` + ``coupling``) with per-layer keys and alternating
    ``flip``; both layer configs are ``_target_`` templates instantiated per block."""
    layers: list[AbstractBijection] = []
    for i, k in enumerate(jax.random.split(key, inn_cfg.blocks)):
        k_lin, k_cpl = jax.random.split(k)
        layers.append(instantiate(inn_cfg.linear, key=k_lin))
        layers.append(instantiate(inn_cfg.coupling, flip=(i % 2 == 0), key=k_cpl))
    return SequentialINN(layers)


def build_model(
    model_cfg: DictConfig, key: PRNGKeyArray
) -> AbstractPhaseAmplitudeModel:
    kind = model_cfg.kind
    if kind == "conjugacy":
        latent = instantiate(model_cfg.latent_dynamics)
        return ConjugateLatentDynamics(latent, build_inn(model_cfg.inn, key))
    if kind == "autoencoder":
        latent = instantiate(model_cfg.latent_dynamics)
        return PhaseAmplitudeAutoencoder(
            model_cfg.obs_dim,
            latent,
            mlp_depth=model_cfg.mlp_depth,
            mlp_width=model_cfg.mlp_width,
            key=key,
        )
    raise ValueError(f"unknown model.kind {kind!r}.")
