r"""Losses as weighted sums of named terms (roadmap C2, ADR-0009).

A loss is an ``AbstractLoss``: it computes a dictionary of **terms** from a model and a
batch, names the terms that are weighted (``weight_names``), and ``__call__`` returns
``(Σ_i w_i · term_i, terms)``. The weights are an argument — a vector in the order of
``weight_names`` — so that the *trainer* owns their schedule (constant, step-dependent
curriculum, or a threshold switch; ``training/schedules.py``), the loss stays a pure
function, and the weights can be logged next to the terms. The building blocks the two
losses are composed of are plain functions on arrays at the top of this module, reusable
by future losses (Phase E).

Batches are the data layer's dicts ``{"t": (B, L), "u": (B, L, dim)}``; a model is an
``AbstractPhaseAmplitudeModel`` (``model(ts, x0) -> (xt, yt)``)."""

import abc

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, PyTree

from ..model.autoencoder import PhaseAmplitudeAutoencoder
from ..model.conjugacy import ConjugateLatentDynamics
from ..model.latent_dynamics import PhaseAmplitudeLatentDynamics


Batch = dict[str, Array | np.ndarray]


# ------------------------------------------------------------------ building blocks --
def trajectory_mse(
    pred: Float[Array, "*batch time dim"], target: Float[Array, "*batch time dim"]
) -> Float[Array, ""]:
    """Mean over batch and time of the squared Euclidean error ``‖pred − target‖²``."""
    return jnp.mean(jnp.sum((pred - target) ** 2, axis=-1))


def final_mse(
    pred: Float[Array, "*batch time dim"], target: Float[Array, "*batch time dim"]
) -> Float[Array, ""]:
    """The same at the last time of the window only (rollout error)."""
    return jnp.mean(jnp.sum((pred[..., -1, :] - target[..., -1, :]) ** 2, axis=-1))


def step_weighted_consistency(
    y: Float[Array, "batch time dim"],
    y_pred: Float[Array, "batch time dim"],
    step_weights: Float[Array, " steps"],
) -> Float[Array, " dim"]:
    """Per-component ``Σ_k α_k · mean_b (y − y_pred)²[:, k]`` over the prediction steps
    ``k = 1..K`` (the window's first point is the initial condition, not a prediction).
    Yawata et al. (2024), Eqs. (22)–(23) before the split into phase and deviation."""
    err = jnp.mean((y - y_pred)[:, 1:] ** 2, axis=0)  # (K, dim)
    return jnp.sum(step_weights[:, None] * err, axis=0)


def alpha_schedule(num_steps: int, phase_loss: Float[Array, ""]) -> Float[Array, " K"]:
    """``α_k = k^{−min(1, L_pha)}``, ``k = 1..K`` (Yawata et al. 2024, Eq. (24)): early
    in training (``L_pha > 1``) short-range predictions dominate; as ``L_pha → 0`` all
    steps count equally. ``phase_loss`` enters under ``stop_gradient``."""
    k = jnp.arange(1, num_steps + 1, dtype=phase_loss.dtype)
    return k ** (-jnp.minimum(1.0, jax.lax.stop_gradient(phase_loss)))


def batch_centre_of_mass(y: Float[Array, "batch 2"]) -> Float[Array, ""]:
    """``‖mean_b y‖²`` — the squared centre of mass of a batch of points on the latent
    unit circle (Yawata et al. 2024, Eq. (25)); small when the phases spread evenly."""
    return jnp.sum(jnp.mean(y, axis=0) ** 2)


# ------------------------------------------------------------------------ contract --
class AbstractLoss(eqx.Module):
    """``terms(model, batch)`` → named scalars; ``__call__`` weights the ones named in
    ``weight_names`` (in that order) by ``weights`` (``None`` → ``default_weights``,
    which an instance may override: ``ConjugacyTrajectoryLoss(default_weights=(1,
    0.1))``; it is an ``init=True`` static field because equinox warns about float
    leaves in ``init=False`` fields)."""

    weight_names: eqx.AbstractVar[tuple[str, ...]]
    default_weights: eqx.AbstractVar[tuple[float, ...]]

    @abc.abstractmethod
    def terms(self, model: PyTree, batch: Batch) -> dict[str, Float[Array, ""]]: ...

    @property
    def num_weights(self) -> int:
        return len(self.weight_names)

    def __call__(
        self,
        model: PyTree,
        batch: Batch,
        weights: Float[Array, " {self.num_weights}"] | None = None,
    ) -> tuple[Float[Array, ""], dict[str, Float[Array, ""]]]:
        w = jnp.asarray(self.default_weights if weights is None else weights)
        batch = {k: jnp.asarray(v) for k, v in batch.items()}  # NumPy batches welcome
        terms = self.terms(model, batch)
        total = sum(
            (w[i] * terms[name] for i, name in enumerate(self.weight_names)),
            start=jnp.asarray(0.0, dtype=w.dtype),
        )
        return total, terms


class ConjugacyTrajectoryLoss(AbstractLoss):
    """Trajectory MSE in data space plus, weighted by ``latent``, the MSE between the
    encoded data and the latent prediction (``w_latent = 0`` reproduces the former
    behaviour)."""

    weight_names: tuple[str, ...] = eqx.field(
        static=True, init=False, default=("data", "latent")
    )
    default_weights: tuple[float, ...] = eqx.field(static=True, default=(1.0, 0.0))

    def terms(self, model: ConjugateLatentDynamics, batch):
        t, x = batch["t"], batch["u"]
        x_pred, y_pred = eqx.filter_vmap(model)(t, x[:, 0])
        y = jax.vmap(jax.vmap(model.bijection))(x)
        return {
            "data": trajectory_mse(x_pred, x),
            "latent": trajectory_mse(y_pred, y),
            "final": final_mse(x_pred, x),
        }


class PhaseAutoencoderLoss(AbstractLoss):
    r"""Yawata et al. (Chaos 34, 063111, 2024), Eqs. (21)–(26): ``recon`` (21), ``pha``
    (22, components 1–2) and ``amp`` (23, component 3) of the ``α_k``-weighted latent
    consistency over the window's ``K`` prediction steps, and ``aux`` (25), the centre
    of mass of the batch's ``(Y₁, Y₂)`` at ``t₀``. ``α_k`` (24) uses the current
    batch's ``L_pha`` under ``stop_gradient`` (the paper updates it per epoch). Default
    weights are the paper's first stage ``(1, 0.5, 0.5, 2)``; its switch to
    ``(1, 5, 0.5, 0)`` once ``pha < 0.01`` and ``aux < 0.05`` is
    ``schedules.ThresholdSwitch(YAWATA_STAGE_1, YAWATA_STAGE_2, {"pha": 0.01,
    "aux": 0.05})`` in the trainer. ``omega`` and ``kappa`` are reported for logging.
    """

    weight_names: tuple[str, ...] = eqx.field(
        static=True, init=False, default=("recon", "pha", "amp", "aux")
    )
    default_weights: tuple[float, ...] = eqx.field(
        static=True, default=(1.0, 0.5, 0.5, 2.0)
    )

    def terms(self, model: PhaseAmplitudeAutoencoder, batch):
        t, x = batch["t"], batch["u"]
        encode = jax.vmap(jax.vmap(model.encode))
        decode = jax.vmap(jax.vmap(model.decode))
        y = encode(x)
        y_pred = jax.vmap(model.latent_dynamics)(t, y[:, 0])
        num_steps = x.shape[1] - 1
        # α_k needs the unweighted phase error first (Eq. (24))
        pha_raw = jnp.sum(
            step_weighted_consistency(y, y_pred, jnp.ones(num_steps, x.dtype))[:2]
        )
        per_component = step_weighted_consistency(
            y, y_pred, alpha_schedule(num_steps, pha_raw)
        )
        terms = {
            "recon": trajectory_mse(decode(y), x),
            "pha": per_component[0] + per_component[1],
            "amp": per_component[2],
            "aux": batch_centre_of_mass(y[:, 0, :2]),
        }
        if isinstance(model.latent_dynamics, PhaseAmplitudeLatentDynamics):
            terms["omega"] = model.latent_dynamics.omega
            terms["kappa"] = model.latent_dynamics.kappa
        return terms


YAWATA_STAGE_1 = (1.0, 0.5, 0.5, 2.0)
"""Loss weights of Yawata et al. (2024), Sec. IV.A, initial stage."""
YAWATA_STAGE_2 = (1.0, 5.0, 0.5, 0.0)
"""… after ``L_pha < 0.01`` and ``L_aux < 0.05``."""
YAWATA_SWITCH_AT = {"pha": 0.01, "aux": 0.05}
