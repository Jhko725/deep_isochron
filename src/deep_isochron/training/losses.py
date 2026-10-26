r"""Losses as weighted sums of named terms (roadmap C2, ADR-0009).

A loss is an ``AbstractLoss``: it computes the weighted terms, exactly the ones named
by ``weight_names``, and unweighted auxiliary diagnostics from a model and a batch, and
``__call__`` returns ``(Σ_i w_i · term_i, terms | aux)``. The weights are an argument,
a vector in the order of ``weight_names``, so that the trainer owns their schedule
(``training/schedules.py``), the loss stays a pure function, and the weights can be
logged next to the terms. The building blocks the two losses are composed of are plain
functions on arrays at the top of this module, for future losses to reuse.

Batches are the data layer's dicts ``{"t": (B, L), "u": (B, L, dim)}``; a model is an
``AbstractPhaseAmplitudeModel`` with ``model(ts, x0) -> (xt, yt)``.
"""

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
"""A batch of windows, ``{"t": (B, L), "u": (B, L, dim)}``, as JAX or NumPy arrays."""


# ------------------------------------------------------------------ building blocks --
def trajectory_mse(
    pred: Float[Array, "*batch time dim"], target: Float[Array, "*batch time dim"]
) -> Float[Array, ""]:
    """The mean over batch and time of the squared Euclidean error
    ``‖pred − target‖²``."""
    return jnp.mean(jnp.sum((pred - target) ** 2, axis=-1))


def final_mse(
    pred: Float[Array, "*batch time dim"], target: Float[Array, "*batch time dim"]
) -> Float[Array, ""]:
    """The mean squared Euclidean error at the last time of the window only: the
    rollout error."""
    return jnp.mean(jnp.sum((pred[..., -1, :] - target[..., -1, :]) ** 2, axis=-1))


def step_weighted_consistency(
    y: Float[Array, "batch time dim"],
    y_pred: Float[Array, "batch time dim"],
    step_weights: Float[Array, " steps"],
) -> Float[Array, " dim"]:
    """The per-component weighted consistency
    ``Σ_k α_k · mean_b (y − y_pred)²[:, k]`` over the prediction steps ``k = 1..K``,
    shape ``(dim,)``: Yawata et al. 2024, Eqs. 22–23, before the split into phase and
    deviation. The window's first point is the initial condition, not a prediction.

    **Arguments:**

    - `y`, `y_pred`: The encoded data and the latent prediction, shape
        ``(batch, time, dim)``.
    - `step_weights`: ``α_k``, shape ``(K,)`` with ``K = time - 1``.
    """
    err = jnp.mean((y - y_pred)[:, 1:] ** 2, axis=0)  # (K, dim)
    return jnp.sum(step_weights[:, None] * err, axis=0)


def alpha_schedule(num_steps: int, phase_loss: Float[Array, ""]) -> Float[Array, " K"]:
    """The step weights ``α_k = k^{−min(1, L_pha)}``, ``k = 1..num_steps``, of Yawata
    et al. 2024, Eq. 24: early in training, with ``L_pha > 1``, short-range
    predictions dominate, and as ``L_pha → 0`` all steps count equally. `phase_loss`
    enters under ``stop_gradient``."""
    k = jnp.arange(1, num_steps + 1, dtype=phase_loss.dtype)
    return k ** (-jnp.minimum(1.0, jax.lax.stop_gradient(phase_loss)))


def batch_center_of_mass(y: Float[Array, "batch 2"]) -> Float[Array, ""]:
    """``‖mean_b y‖²``, the squared center of mass of a batch of points `y` on the
    latent unit circle (Yawata et al. 2024, Eq. 25); small when the phases spread
    evenly."""
    return jnp.sum(jnp.mean(y, axis=0) ** 2)


# ------------------------------------------------------------------------ contract --
class AbstractLoss(eqx.Module):
    """A loss as a weighted sum of named terms: `terms` computes them, `__call__`
    weights them.

    **Attributes:**

    - `weight_names`: The names of the weighted terms, in weight order.
    - `default_weights`: The weights used when `__call__` gets none; an instance may
        override them, ``ConjugacyTrajectoryLoss(default_weights=(1, 0.1))``. An
        ``init=True`` static field, because equinox warns about float leaves in
        ``init=False`` fields.

    **Raises:** `ValueError` at construction when the two differ in length.
    """

    weight_names: eqx.AbstractVar[tuple[str, ...]]
    default_weights: eqx.AbstractVar[tuple[float, ...]]

    @abc.abstractmethod
    def terms(
        self, model: PyTree, batch: Batch
    ) -> tuple[dict[str, Float[Array, ""]], dict[str, Float[Array, ""]]]:
        """``(terms, aux)`` for `model` on `batch`: `terms` holds exactly the weighted
        scalars, keyed by `weight_names`; `aux` holds unweighted diagnostics logged
        alongside, such as the `final` error or the learned `omega` and `kappa`."""

    @property
    def num_weights(self) -> int:
        """The number of weighted terms."""
        return len(self.weight_names)

    def __check_init__(self) -> None:
        """Check that `default_weights` and `weight_names` have one length."""
        if len(self.default_weights) != len(self.weight_names):
            raise ValueError(
                f"{type(self).__name__}: {len(self.default_weights)} default weights "
                f"for {len(self.weight_names)} weighted terms {self.weight_names}."
            )

    def __call__(
        self,
        model: PyTree,
        batch: Batch,
        weights: Float[Array, " {self.num_weights}"] | None = None,
    ) -> tuple[Float[Array, ""], dict[str, Float[Array, ""]]]:
        """The weighted loss and every term.

        **Arguments:**

        - `model`: The model.
        - `batch`: The batch, as JAX or NumPy arrays.
        - `weights`: The weights, shape ``(num_weights,)``, in the order of
            `weight_names`; `default_weights` when ``None``.

        **Returns:** ``(Σ_i weights[i] · terms[weight_names[i]], terms | aux)``.

        **Raises:** `ValueError` when `terms` returns other keys than `weight_names`,
        or an `aux` key clashes with a term.
        """
        w = jnp.asarray(self.default_weights if weights is None else weights)
        batch = {k: jnp.asarray(v) for k, v in batch.items()}  # NumPy batches welcome
        terms, aux = self.terms(model, batch)
        if set(terms) != set(self.weight_names):
            raise ValueError(
                f"{type(self).__name__}.terms returned {sorted(terms)}; expected "
                f"exactly {sorted(self.weight_names)} (put diagnostics in aux)."
            )
        if set(aux) & set(terms):
            raise ValueError(
                f"aux keys {sorted(set(aux) & set(terms))} clash with terms."
            )
        total = sum(
            (w[i] * terms[name] for i, name in enumerate(self.weight_names)),
            start=jnp.asarray(0.0, dtype=w.dtype),
        )
        return total, {**terms, **aux}


class ConjugacyTrajectoryLoss(AbstractLoss):
    """The trajectory MSE in data space (``data``) plus the MSE between the encoded
    data and the latent prediction (``latent``); ``w_latent = 0`` reproduces the
    former behavior. The auxiliary ``final`` is the error at the window's last time,
    since prediction error grows along the window and the mean hides a frequency
    drift."""

    weight_names: tuple[str, ...] = eqx.field(
        static=True, init=False, default=("data", "latent")
    )
    default_weights: tuple[float, ...] = eqx.field(static=True, default=(1.0, 0.0))

    def terms(self, model: ConjugateLatentDynamics, batch):
        """``data``, ``latent``; aux ``final``."""
        t, x = batch["t"], batch["u"]
        x_pred, y_pred = eqx.filter_vmap(model)(t, x[:, 0])
        y = jax.vmap(jax.vmap(model.bijection))(x)
        terms = {"data": trajectory_mse(x_pred, x), "latent": trajectory_mse(y_pred, y)}
        return terms, {"final": final_mse(x_pred, x)}


class PhaseAutoencoderLoss(AbstractLoss):
    r"""The loss of Yawata et al. 2024, Eqs. 21–26: ``recon`` (Eq. 21), ``pha``
    (Eq. 22, components 1–2) and ``amp`` (Eq. 23, component 3) of the
    ``α_k``-weighted latent consistency over the window's ``K`` prediction steps, and
    ``aux`` (Eq. 25), the center of mass of the batch's ``(Y₁, Y₂)`` at ``t₀``.

    ``α_k`` (Eq. 24) uses the current batch's ``L_pha`` under ``stop_gradient``,
    where the paper updates it per epoch. The default weights are the paper's first
    stage, ``YAWATA_STAGE_1``; its switch to ``YAWATA_STAGE_2`` once ``pha < 0.01``
    and ``aux < 0.05`` is ``schedules.ThresholdSwitch(YAWATA_STAGE_1,
    YAWATA_STAGE_2, YAWATA_SWITCH_AT)`` in the trainer. The auxiliaries are ``omega``
    and ``kappa`` of the latent dynamics.
    """

    weight_names: tuple[str, ...] = eqx.field(
        static=True, init=False, default=("recon", "pha", "amp", "aux")
    )
    default_weights: tuple[float, ...] = eqx.field(
        static=True, default=(1.0, 0.5, 0.5, 2.0)
    )

    def terms(self, model: PhaseAmplitudeAutoencoder, batch):
        """``recon``, ``pha``, ``amp``, ``aux``; aux ``omega``, ``kappa``."""
        t, x = batch["t"], batch["u"]
        encode = jax.vmap(jax.vmap(model.encode))
        decode = jax.vmap(jax.vmap(model.decode))
        y = encode(x)
        y_pred = jax.vmap(model.latent_dynamics)(t, y[:, 0])
        num_steps = x.shape[1] - 1
        # α_k needs the unweighted phase error first (Eq. 24)
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
            "aux": batch_center_of_mass(y[:, 0, :2]),
        }
        aux: dict[str, Float[Array, ""]] = {}
        if isinstance(model.latent_dynamics, PhaseAmplitudeLatentDynamics):
            aux["omega"] = model.latent_dynamics.omega
            aux["kappa"] = model.latent_dynamics.kappa
        return terms, aux


YAWATA_STAGE_1 = (1.0, 0.5, 0.5, 2.0)
"""The loss weights of Yawata et al. 2024, §IV.A, in the initial stage."""
YAWATA_STAGE_2 = (1.0, 5.0, 0.5, 0.0)
"""The loss weights after the switch."""
YAWATA_SWITCH_AT = {"pha": 0.01, "aux": 0.05}
"""The thresholds of the switch: ``L_pha < 0.01`` and ``L_aux < 0.05``."""
