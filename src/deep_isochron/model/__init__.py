"""The models: the shared ``AbstractPhaseAmplitudeModel`` contract, the invertible
conjugacy ``ConjugateLatentDynamics`` and the phase-autoencoder baseline
``PhaseAmplitudeAutoencoder`` with its latent dynamics."""

from .autoencoder import PhaseAmplitudeAutoencoder as PhaseAmplitudeAutoencoder
from .base import AbstractPhaseAmplitudeModel as AbstractPhaseAmplitudeModel
from .conjugacy import ConjugateLatentDynamics as ConjugateLatentDynamics
from .latent_dynamics import (
    AbstractLatentDynamics as AbstractLatentDynamics,
    PhaseAmplitudeLatentDynamics as PhaseAmplitudeLatentDynamics,
)
