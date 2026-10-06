"""Held-out evaluation against the model contract (roadmap C3, ADR-0009).

``Evaluator(val_data, reference=…)`` is a callable ``model -> {name: float}`` for
``Trainer.train(evaluate=…)``; it works on any ``AbstractPhaseAmplitudeModel`` and adds
what each model kind can report:

- **prediction**: ``val/mse`` (trajectory MSE over the validation windows — the default
  checkpoint-selection metric) and ``val/final_mse`` (error at the window's last time);
  with ``t_split``, also ``val/mse_early`` / ``val/mse_late`` — the same error over the
  windows that start before / at-or-after ``t_split``, so the transient's fit is not
  drowned by the many near-periodic windows (``docs/design/validation-split.md``); the
  checkpointer can select on ``val/mse_early``;
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
  validation points, minimized over the two orientations (a phase is learned up to a
  constant and a sign), and ``amplitude/corr``, ``|corr(A_model, Ψ_exact)|`` (an
  amplitude is learned up to scale, so correlation is the scale-free comparison).

For FitzHugh–Nagumo there is no reference until Phase E: prediction error and the
learned period are what is reported."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field

import equinox as eqx
import grain
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from ..data.device import single_threaded
from ..model.autoencoder import PhaseAmplitudeAutoencoder
from ..model.base import AbstractPhaseAmplitudeModel
from ..model.conjugacy import ConjugateLatentDynamics
from ..model.invertible import AbstractBijection, SequentialINN
from ..model.latent_dynamics import PhaseAmplitudeLatentDynamics
from ..systems.normal_forms import AbstractNormalForm
from .losses import Batch, final_mse, trajectory_mse


def bounding_box_grid(
    bounds: Float[Array, "2 dim"], num: int = 32, margin: float = 0.05
) -> Float[Array, "m dim"]:
    """A regular grid over the axis-aligned box ``bounds = (lo, hi)`` (``num`` per axis
    for planar data, ``8`` beyond), widened by ``margin`` of each side's extent."""
    lo, hi = bounds[0], bounds[1]
    pad = margin * (hi - lo)
    lo, hi = lo - pad, hi + pad
    dim = bounds.shape[1]
    per_axis = num if dim <= 2 else 8
    axes = [jnp.linspace(lo[i], hi[i], per_axis) for i in range(dim)]
    return jnp.stack(jnp.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, dim)


def circular_std(angles: Float[Array, " n"]) -> Float[Array, ""]:
    """``sqrt(-2 ln R̄)`` with ``R̄`` the mean resultant length (see
    ``phase_amplitude_agreement`` for why this is the right spread of angles)."""
    return circular_std_from_resultant(jnp.abs(jnp.mean(jnp.exp(1j * angles))))


@eqx.filter_jit
def prediction_errors(model, batch: Batch) -> tuple[Float[Array, ""], Float[Array, ""]]:
    t, x = jnp.asarray(batch["t"]), jnp.asarray(batch["u"])
    x_pred, _ = eqx.filter_vmap(model)(t, x[:, 0])
    return trajectory_mse(x_pred, x), final_mse(x_pred, x)


@eqx.filter_jit
def prediction_sums(
    model, batch: Batch, t_split: Float[Array, ""]
) -> dict[str, Float[Array, ""]]:
    """Sufficient statistics of one batch for the prediction metrics: sums over windows
    of the per-window trajectory MSE and final error, the window count, and the same
    sums restricted to windows that **start before** ``t_split`` (the transient bucket;
    design document ``validation-split.md``). Summed over batches by ``Evaluator``."""
    t, x = jnp.asarray(batch["t"]), jnp.asarray(batch["u"])
    x_pred, _ = eqx.filter_vmap(model)(t, x[:, 0])
    per_window = eqx.filter_vmap(trajectory_mse)(x_pred, x)
    per_window_final = eqx.filter_vmap(final_mse)(x_pred, x)
    early = t[:, 0] < t_split
    return {
        "n": jnp.asarray(x.shape[0], dtype=per_window.dtype),
        "mse": jnp.sum(per_window),
        "final": jnp.sum(per_window_final),
        "n_early": jnp.sum(early).astype(per_window.dtype),
        "mse_early": jnp.sum(jnp.where(early, per_window, 0.0)),
    }


@eqx.filter_jit
def phase_amplitude_sums(
    model: AbstractPhaseAmplitudeModel,
    reference: AbstractNormalForm,
    points: Float[Array, "n dim"],
) -> dict[str, Array]:
    """Sufficient statistics of one batch for the phase and amplitude agreement:
    ``Σ exp(i(±Θ_model − Θ_ref))`` for both orientations and the moments of
    ``(A_model, Ψ_ref)``; summed over batches by ``Evaluator``."""
    theta = jax.vmap(model.phase)(points)
    amp = jax.vmap(model.amplitude)(points)
    exact = jax.vmap(reference.to_phase_amplitude)(points)
    psi = exact[:, 1]
    return {
        "n": jnp.asarray(points.shape[0], dtype=theta.dtype),
        "z_plus": jnp.sum(jnp.exp(1j * (theta - exact[:, 0]))),
        "z_minus": jnp.sum(jnp.exp(1j * (-theta - exact[:, 0]))),
        "a": jnp.sum(amp),
        "p": jnp.sum(psi),
        "aa": jnp.sum(amp * amp),
        "pp": jnp.sum(psi * psi),
        "ap": jnp.sum(amp * psi),
    }


def phase_amplitude_agreement(sums: dict[str, Array]) -> tuple[Array, Array]:
    """``(circ_std, |corr|)`` from summed ``phase_amplitude_sums``.

    Why the circular standard deviation rather than an MSE of the angle difference: the
    learned phase is determined only up to a constant (design document §5.4), so the
    difference ``Δ = Θ_model − Θ_ref`` has an unknown mean, and ``Δ`` lives on the
    circle, so a model right to 0.01 rad can show ``|Δ| ≈ 2π`` near ``±π``. The circular
    standard deviation ``sqrt(−2 ln R̄)``, ``R̄ = |mean exp(iΔ)|`` (Mardia & Jupp,
    *Directional Statistics*, 1999; SciPy's ``circstd``), is invariant to the offset and
    to wrapping, and for small spread equals the ordinary standard deviation of ``Δ``
    after removing its mean — the RMS phase error in radians. The better of the two
    orientations is kept (a phase is learned up to a sign too). The amplitude is learned
    up to scale, so ``|corr(A_model, Ψ_ref)|`` is its scale-free comparison."""
    n = sums["n"]
    spread = jnp.minimum(
        circular_std_from_resultant(jnp.abs(sums["z_plus"]) / n),
        circular_std_from_resultant(jnp.abs(sums["z_minus"]) / n),
    )
    cov = sums["ap"] / n - (sums["a"] / n) * (sums["p"] / n)
    var_a = sums["aa"] / n - (sums["a"] / n) ** 2
    var_p = sums["pp"] / n - (sums["p"] / n) ** 2
    corr = cov / jnp.sqrt(jnp.maximum(var_a * var_p, 1e-300))
    return spread, jnp.abs(corr)


def circular_std_from_resultant(r: Array) -> Array:
    return jnp.sqrt(-2.0 * jnp.log(jnp.clip(r, 1e-300, 1.0)))


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
    """``evaluate(model)`` over a **finite, re-iterable** collection of validation
    batches — a ``validation_windows(...).batch(B)`` dataset (the intended source), or
    a list from ``collect_batches``. It is iterated to exhaustion on every call, so each
    evaluation sees the same windows and the metric is comparable across steps
    (levanter's eval-loop shape; ADR-0009). The bounding-box grid for the bijection
    diagnostics is taken from the data on the first pass."""

    val_data: Iterable[dict[str, Array | np.ndarray]]
    reference: AbstractNormalForm | None = None
    grid_points: int = 32
    t_split: float | None = None
    """Split the prediction error by window start: windows starting before ``t_split``
    (the transient) give ``val/mse_early``, the rest ``val/mse_late``; ``val/mse`` stays
    the overall mean. ``None``: no split. See ``docs/design/validation-split.md``."""
    _grid: Array | None = field(default=None, init=False, repr=False)

    def _batches(self) -> Iterable[dict[str, Array | np.ndarray]]:
        """One pass over the validation data; a grain ``MapDataset`` is read with a
        single thread so evaluation does not start 16 reader threads next to JAX."""
        if isinstance(self.val_data, grain.MapDataset | grain.IterDataset):
            return single_threaded(self.val_data)
        return self.val_data

    @property
    def grid(self) -> Float[Array, "m dim"]:
        if self._grid is None:
            self._grid = bounding_box_grid(self._bounds(), self.grid_points)
        return self._grid

    def _bounds(self) -> Float[Array, "2 dim"]:
        lo = hi = None
        for batch in self._batches():
            u = np.asarray(batch["u"]).reshape(-1, np.shape(batch["u"])[-1])
            lo = u.min(0) if lo is None else np.minimum(lo, u.min(0))
            hi = u.max(0) if hi is None else np.maximum(hi, u.max(0))
        if lo is None:
            raise ValueError("val_data is empty.")
        return jnp.stack((jnp.asarray(lo), jnp.asarray(hi)))

    def __call__(self, model: AbstractPhaseAmplitudeModel) -> dict[str, float]:
        t_split = jnp.asarray(-jnp.inf if self.t_split is None else self.t_split)
        pred: dict[str, Array] | None = None
        sums: dict[str, Array] | None = None
        for batch in self._batches():
            part_p = prediction_sums(model, batch, t_split)
            pred = part_p if pred is None else {k: pred[k] + part_p[k] for k in part_p}
            if self.reference is not None:
                u = jnp.asarray(batch["u"])
                part = phase_amplitude_sums(
                    model, self.reference, u.reshape(-1, u.shape[-1])
                )
                sums = part if sums is None else {k: sums[k] + part[k] for k in part}
        if pred is None:
            raise ValueError("val_data is empty.")
        n = pred["n"]
        out: dict[str, Array] = {
            "val/mse": pred["mse"] / n,
            "val/final_mse": pred["final"] / n,
        }
        if self.t_split is not None:
            n_early, n_late = pred["n_early"], n - pred["n_early"]
            # an empty bucket is reported as NaN rather than hidden or faked
            out["val/mse_early"] = jnp.where(
                n_early > 0, pred["mse_early"] / jnp.maximum(n_early, 1), jnp.nan
            )
            out["val/mse_late"] = jnp.where(
                n_late > 0,
                (pred["mse"] - pred["mse_early"]) / jnp.maximum(n_late, 1),
                jnp.nan,
            )
            out["val/n_early"] = n_early
            out["val/n_late"] = n_late
        out.update(learned_physics(model))
        if isinstance(model, ConjugateLatentDynamics):
            out.update(bijection_grid_metrics(model.bijection, self.grid))
        if sums is not None:
            out["phase/circ_std"], out["amplitude/corr"] = phase_amplitude_agreement(
                sums
            )
        return {k: float(v) for k, v in out.items()}


def collect_batches(loader: Iterable[Batch], num: int) -> list[dict[str, np.ndarray]]:
    """The first ``num`` batches of a loader, as NumPy on the host."""
    out: list[dict[str, np.ndarray]] = []
    for batch in loader:
        out.append({k: np.asarray(v) for k, v in batch.items()})
        if len(out) == num:
            break
    return out
