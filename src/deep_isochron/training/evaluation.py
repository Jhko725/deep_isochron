"""Held-out evaluation against the model contract (roadmap C3, ADR-0009).

``Evaluator(val_batches, reference=…)`` is a callable ``model -> {name: float}`` for
``Trainer.train(evaluate=…)``; it works on any ``AbstractPhaseAmplitudeModel`` and adds
what each model kind can report:

- **prediction**: ``val/mse`` (trajectory MSE over the validation windows — the quantity
  the checkpointer selects on) and ``val/final_mse`` (error at the window's last time);
- **learned physics**, both kinds: ``period`` and ``kappa`` (the normal form's closed
  forms for the conjugacy model; ``2π/|ω|`` and ``κ`` of
  ``PhaseAmplitudeLatentDynamics`` for the autoencoder); the conjugacy model also
  reports its normal form's parameters (``nf/a``, ``nf/b``, …) and, on a grid over the
  validation data's bounding box, the bijection's max round-trip error ``H⁻¹(H(x)) − x``
  and the min/max singular value of its Jacobian — per layer for a ``SequentialINN``
  (``jac/sv_min/<i>``);
- **against a reference** normal form, when the data were generated from one
  (``reference=nf``, the design document's §7 chart as ground truth):
  ``phase/circ_std``, the circular standard deviation of ``Θ_model − Θ_exact`` over the
  validation points, minimised over the two orientations (a phase is learned up to a
  constant and a sign), and ``amplitude/corr``, ``|corr(A_model, Ψ_exact)|`` (an
  amplitude is learned up to scale, so correlation is the scale-free comparison).

For FitzHugh–Nagumo there is no reference until Phase E: prediction error and the
learned period are what is reported."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from ..model.autoencoder import PhaseAmplitudeAutoencoder
from ..model.base import AbstractPhaseAmplitudeModel
from ..model.conjugacy import ConjugateLatentDynamics
from ..model.invertible import AbstractBijection, SequentialINN
from ..model.latent_dynamics import PhaseAmplitudeLatentDynamics
from ..systems.normal_forms import AbstractNormalForm
from .losses import Batch, final_mse, trajectory_mse


def bounding_box_grid(
    points: Float[Array, "n dim"], num: int = 32, margin: float = 0.05
) -> Float[Array, "m dim"]:
    """A regular grid over the axis-aligned bounding box of ``points`` (``num`` per axis
    for planar data, ``8`` beyond), widened by ``margin`` of each side's extent."""
    lo, hi = jnp.min(points, axis=0), jnp.max(points, axis=0)
    pad = margin * (hi - lo)
    lo, hi = lo - pad, hi + pad
    per_axis = num if points.shape[1] <= 2 else 8
    axes = [jnp.linspace(lo[i], hi[i], per_axis) for i in range(points.shape[1])]
    return jnp.stack(jnp.meshgrid(*axes, indexing="ij"), axis=-1).reshape(
        -1, points.shape[1]
    )


def circular_std(angles: Float[Array, " n"]) -> Float[Array, ""]:
    """``sqrt(-2 ln R)`` with ``R`` the mean resultant length."""
    r = jnp.abs(jnp.mean(jnp.exp(1j * angles)))
    return jnp.sqrt(-2.0 * jnp.log(jnp.clip(r, 1e-300, 1.0)))


@eqx.filter_jit
def prediction_errors(model, batch: Batch) -> tuple[Float[Array, ""], Float[Array, ""]]:
    t, x = jnp.asarray(batch["t"]), jnp.asarray(batch["u"])
    x_pred, _ = eqx.filter_vmap(model)(t, x[:, 0])
    return trajectory_mse(x_pred, x), final_mse(x_pred, x)


@eqx.filter_jit
def phase_amplitude_agreement(
    model: AbstractPhaseAmplitudeModel,
    reference: AbstractNormalForm,
    points: Float[Array, "n dim"],
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """``(circ_std of Θ_model − Θ_ref, |corr(A_model, Ψ_ref)|)``; the phase difference
    is taken for both orientations and the smaller spread kept."""
    theta = jax.vmap(model.phase)(points)
    amp = jax.vmap(model.amplitude)(points)
    exact = jax.vmap(reference.to_phase_amplitude)(points)
    spread = jnp.minimum(
        circular_std(theta - exact[:, 0]), circular_std(-theta - exact[:, 0])
    )
    corr = jnp.corrcoef(amp, exact[:, 1])[0, 1]
    return spread, jnp.abs(corr)


@eqx.filter_jit
def bijection_grid_metrics(
    bijection: AbstractBijection, grid: Float[Array, "m dim"]
) -> dict[str, Float[Array, ""]]:
    """Max round-trip error and min/max Jacobian singular value on ``grid``; per layer
    too for a ``SequentialINN`` (the Jacobian of layer ``i`` at that layer's input)."""

    def layer_metrics(f, x):
        svs = jnp.linalg.svd(jax.vmap(f.jacobian)(x), compute_uv=False)
        return jnp.min(svs), jnp.max(svs)

    out: dict[str, Float[Array, ""]] = {}
    y = jax.vmap(bijection)(grid)
    out["roundtrip/max"] = jnp.max(jnp.abs(jax.vmap(bijection.inverse)(y) - grid))
    out["jac/sv_min"], out["jac/sv_max"] = layer_metrics(bijection, grid)
    if isinstance(bijection, SequentialINN):
        x = grid
        for i, layer in enumerate(bijection.transforms):
            out[f"jac/sv_min/{i}"], out[f"jac/sv_max/{i}"] = layer_metrics(layer, x)
            x = jax.vmap(layer)(x)
    return out


def learned_physics(model: AbstractPhaseAmplitudeModel) -> dict[str, Array]:
    """``period``, ``kappa`` and, for the conjugacy model, the normal form's
    parameters."""
    out: dict[str, Array] = {}
    if isinstance(model, ConjugateLatentDynamics):
        nf = model.latent_dynamics
        out["period"] = nf.period()
        out["kappa"] = nf.floquet_exponent()
        for name, value in nf.params().items():
            out[f"nf/{name}"] = jnp.asarray(value)
    elif isinstance(model, PhaseAmplitudeAutoencoder) and isinstance(
        model.latent_dynamics, PhaseAmplitudeLatentDynamics
    ):
        dyn = model.latent_dynamics
        out["period"] = 2 * math.pi / jnp.abs(dyn.omega)
        out["kappa"] = dyn.kappa
    return out


@dataclass
class Evaluator:
    """``evaluate(model)`` over a fixed, finite collection of validation batches
    (build it once with ``collect_batches``, so every evaluation sees the same
    windows)."""

    val_batches: Sequence[dict[str, Array | np.ndarray]]
    reference: AbstractNormalForm | None = None
    grid_points: int = 32
    _batches: list[Batch] = field(default_factory=list, init=False, repr=False)
    _grid: Array | None = field(default=None, init=False, repr=False)
    _points: Array | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._batches = [
            {k: jnp.asarray(v) for k, v in b.items()} for b in self.val_batches
        ]
        if not self._batches:
            raise ValueError("val_batches is empty.")

    @property
    def points(self) -> Float[Array, "n dim"]:
        """Every state in the validation windows, flattened."""
        if self._points is None:
            self._points = jnp.concatenate(
                [b["u"].reshape(-1, b["u"].shape[-1]) for b in self._batches]
            )
        return self._points

    @property
    def grid(self) -> Float[Array, "m dim"]:
        if self._grid is None:
            self._grid = bounding_box_grid(self.points, self.grid_points)
        return self._grid

    def __call__(self, model: AbstractPhaseAmplitudeModel) -> dict[str, float]:
        errors = [prediction_errors(model, b) for b in self._batches]
        out: dict[str, Array] = {
            "val/mse": jnp.mean(jnp.stack([e[0] for e in errors])),
            "val/final_mse": jnp.mean(jnp.stack([e[1] for e in errors])),
        }
        out.update(learned_physics(model))
        if isinstance(model, ConjugateLatentDynamics):
            out.update(bijection_grid_metrics(model.bijection, self.grid))
        if self.reference is not None:
            spread, corr = phase_amplitude_agreement(model, self.reference, self.points)
            out["phase/circ_std"], out["amplitude/corr"] = spread, corr
        return {k: float(v) for k, v in out.items()}


def collect_batches(loader: Iterable[Batch], num: int) -> list[dict[str, np.ndarray]]:
    """The first ``num`` batches of a loader, as NumPy on the host."""
    out: list[dict[str, np.ndarray]] = []
    for batch in loader:
        out.append({k: np.asarray(v) for k, v in batch.items()})
        if len(out) == num:
            break
    return out
