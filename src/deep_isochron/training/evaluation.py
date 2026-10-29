"""Held-out evaluation against the model contract (roadmap C3, ADR-0009).

``Evaluator`` is a callable ``model -> {name: float}`` for
``Trainer.train(evaluate=…)``. It works on any ``AbstractPhaseAmplitudeModel`` and
reports five groups. The prediction metrics are ``val/mse``, the trajectory MSE over
the validation windows and the default checkpoint-selection metric, and
``val/final_mse``, the error at the window's last time; with ``t_split``, also
``val/mse_early`` and ``val/mse_late`` over the windows that start before and at or
after it, so that the transient's fit is not drowned by the many near-periodic windows
(``docs/design/validation-split.md``). The learned physics are ``period`` and
``kappa`` for both model kinds, and for the conjugacy model its normal form's
parameters as ``nf/<name>``. The latent extent metrics detect a map that flings points
toward the latent origin or far outside the cycle. The bijection diagnostics, for the
conjugacy model, are the round-trip error and the Jacobian singular values on a grid
over the data's bounding box, per layer for a ``SequentialINN``. And against a
reference normal form, when the data were generated from one, ``phase/circ_std`` and
``amplitude/corr`` measure the agreement of the learned phase and amplitude with the
exact chart; ``phase_amplitude_agreement`` says why those two statistics. For
FitzHugh–Nagumo the reference is Phase E's phase field, compared through
``align_phases``, the explicit form of the same comparison: the orientation, the offset
and the residual field, shared with the plots.
"""

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
from ..utils.numerics import phase_difference, wrap_phase
from .losses import Batch, final_mse, trajectory_mse


def bounding_box_grid(
    bounds: Float[Array, "2 dim"], num: int = 32, margin: float = 0.05
) -> Float[Array, "m dim"]:
    """A regular grid over the axis-aligned box ``bounds = (lo, hi)``, shape
    ``(m, dim)``: `num` points per axis for planar data and 8 beyond, widened by
    `margin` of each side's extent."""
    lo, hi = bounds[0], bounds[1]
    pad = margin * (hi - lo)
    lo, hi = lo - pad, hi + pad
    dim = bounds.shape[1]
    per_axis = num if dim <= 2 else 8
    axes = [jnp.linspace(lo[i], hi[i], per_axis) for i in range(dim)]
    return jnp.stack(jnp.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, dim)


@dataclass(frozen=True)
class PhaseAlignment:
    r"""How a model's phase maps onto a reference phase, ``model ≈ sense · reference +
    offset``, and the residual after the alignment.

    **Attributes:**

    - `sense`: ``+1`` or ``−1``, the orientation of the model's phase.
    - `offset`: The model's phase at reference phase 0, in turns, in $[0, 1)$.
    - `residual`: The signed shortest arc `phase_difference(model − sense · reference,
        offset)` in $(-\tfrac12, \tfrac12]$ turns, the shape of the inputs; ``nan``
        where either input is.
    - `rms`: The root mean square of `residual` over the finite entries: the aligned
        phase error in turns.
    """

    sense: float
    offset: float
    residual: Float[np.ndarray, "..."]
    rms: float


def align_phases(
    model_phase: Float[Array, "..."],
    reference: Float[Array, "..."],
    offset: float | None = None,
) -> PhaseAlignment:
    """Align a model's phases with reference phases at the same points.

    The orientation is the one with the smaller residual, and the offset is the
    circular mean, in turns, of ``model − sense · reference`` unless `offset` is given,
    as from ``analysis.phase_offset``, in which case only the sense is chosen (roadmap
    item M: the aligned phase-field error, shared by the metric and the plot).

    **Arguments:**

    - `model_phase`, `reference`: Phases in turns at the same points, any common
        shape; ``nan`` marks a point outside the basin and is ignored.
    - `offset`: The model's phase at reference phase 0, in turns, when known.

    **Returns:**

    The `PhaseAlignment`.
    """
    best: PhaseAlignment | None = None
    for sense in (1.0, -1.0):
        shift = model_phase - sense * reference
        theta0 = (
            jnp.angle(jnp.nanmean(jnp.exp(2j * jnp.pi * shift))) / (2 * jnp.pi)
            if offset is None
            else offset
        )
        residual = phase_difference(shift, theta0)
        rms = jnp.sqrt(jnp.nanmean(residual**2))
        if best is None or float(rms) < best.rms:
            best = PhaseAlignment(
                sense, float(wrap_phase(theta0)), np.asarray(residual), float(rms)
            )
    assert best is not None
    return best


def circular_std(phases: Float[Array, " n"]) -> Float[Array, ""]:
    """The circular standard deviation, in turns, of `phases` given in turns: the
    ``sqrt(-2 ln R̄)`` of the angle ``2πΔ``, divided by ``2π``, with
    ``R̄ = |mean exp(2πiΔ)|`` the mean resultant length; `phase_amplitude_agreement`
    says why this is the right spread of phase differences."""
    resultant = jnp.abs(jnp.mean(jnp.exp(2j * jnp.pi * phases)))
    return circular_std_from_resultant(resultant) / (2 * jnp.pi)


@eqx.filter_jit
def prediction_errors(model, batch: Batch) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """The trajectory MSE and the final error of `model` on one `batch`."""
    t, x = jnp.asarray(batch["t"]), jnp.asarray(batch["u"])
    x_pred, _ = eqx.filter_vmap(model)(t, x[:, 0])
    return trajectory_mse(x_pred, x), final_mse(x_pred, x)


@eqx.filter_jit
def prediction_sums(
    model, batch: Batch, t_split: Float[Array, ""]
) -> dict[str, Float[Array, ""]]:
    """The sufficient statistics of one `batch` for the prediction metrics: the sums
    over windows of the per-window trajectory MSE and final error, the window count,
    and the same sums restricted to the windows that start before `t_split`, the
    transient bucket (``docs/design/validation-split.md``). `Evaluator` sums them over
    batches."""
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
def latent_extent(
    model: AbstractPhaseAmplitudeModel, points: Float[Array, "n dim"]
) -> dict[str, Array]:
    """Where `model` sends `points`: the extremes of the amplitude ``A(x)`` and, for a
    conjugacy model, of the latent radius ``|H(x)|``, with the counts of points whose
    amplitude or image is not finite. `Evaluator` combines them across batches.

    This is the detector for a learned map that flings points toward the latent
    origin, where ``r_min → 0`` and ``amp_min → −∞`` and the normal form's closed
    forms overflow, or far outside the cycle, where ``r_max`` is large, ``Ψ``
    saturates and the chart's inverse loses its conditioning.
    """
    amp = jax.vmap(model.amplitude)(points)
    finite = jnp.isfinite(amp)
    safe = jnp.where(finite, amp, 0.0)
    out = {
        "n": jnp.asarray(points.shape[0], dtype=amp.dtype),
        "nonfinite": jnp.sum(~finite).astype(amp.dtype),
        "amp_min": jnp.min(jnp.where(finite, safe, jnp.inf)),
        "amp_max": jnp.max(jnp.where(finite, safe, -jnp.inf)),
    }
    if isinstance(model, ConjugateLatentDynamics):
        y = jax.vmap(model.bijection)(points)
        r = jnp.linalg.norm(y, axis=-1)
        r_finite = jnp.isfinite(r)
        out["r_min"] = jnp.min(jnp.where(r_finite, r, jnp.inf))
        out["r_max"] = jnp.max(jnp.where(r_finite, r, -jnp.inf))
        out["nonfinite_image"] = jnp.sum(~r_finite).astype(amp.dtype)
    return out


def _combine_extent(acc: dict[str, Array] | None, part: dict[str, Array]):
    """`acc` combined with `part`: minima, maxima and sums by key suffix."""
    if acc is None:
        return dict(part)
    out = {}
    for k, v in part.items():
        if k.endswith("_min"):
            out[k] = jnp.minimum(acc[k], v)
        elif k.endswith("_max"):
            out[k] = jnp.maximum(acc[k], v)
        else:
            out[k] = acc[k] + v
    return out


@eqx.filter_jit
def phase_amplitude_sums(
    model: AbstractPhaseAmplitudeModel,
    reference: AbstractNormalForm,
    points: Float[Array, "n dim"],
) -> dict[str, Array]:
    """The sufficient statistics of one batch of `points` for the phase and amplitude
    agreement of `model` with `reference`: ``Σ exp(2πi(±Θ_model − Θ_ref))`` for both
    orientations and the moments of ``(A_model, Ψ_ref)``. `Evaluator` sums them over
    batches."""
    theta = jax.vmap(model.phase)(points)
    amp = jax.vmap(model.amplitude)(points)
    exact = jax.vmap(reference.to_phase_amplitude)(points)
    psi = exact[:, 1]
    return {
        "n": jnp.asarray(points.shape[0], dtype=theta.dtype),
        "z_plus": jnp.sum(jnp.exp(2j * jnp.pi * (theta - exact[:, 0]))),
        "z_minus": jnp.sum(jnp.exp(2j * jnp.pi * (-theta - exact[:, 0]))),
        "a": jnp.sum(amp),
        "p": jnp.sum(psi),
        "aa": jnp.sum(amp * amp),
        "pp": jnp.sum(psi * psi),
        "ap": jnp.sum(amp * psi),
    }


def phase_amplitude_agreement(sums: dict[str, Array]) -> tuple[Array, Array]:
    """``(circ_std, |corr|)`` from the summed `phase_amplitude_sums`, with ``circ_std``
    in turns.

    The circular standard deviation rather than an MSE of the phase difference,
    because the learned phase is determined only up to a constant (design document
    §5.4), so the difference ``Δ = Θ_model − Θ_ref`` has an unknown mean, and ``Δ``
    lives on the circle, so a model right to 0.002 turns can show ``|Δ| ≈ 1`` near
    the seam at 0 and 1. The circular standard deviation ``sqrt(−2 ln R̄) / 2π`` with
    ``R̄ = |mean exp(2πiΔ)|`` (Mardia & Jupp 1999; SciPy's ``circstd``) is invariant to
    the offset and to wrapping, and for small spread equals the ordinary standard
    deviation of ``Δ`` after removing its mean: the RMS phase error in turns. Both
    orientations are compared and the better kept, because the model may learn either
    way round while the reference has one (ADR-0020).
    The amplitude is learned up to scale, so ``|corr(A_model, Ψ_ref)|`` is its
    scale-free comparison.
    """
    n = sums["n"]
    spread = jnp.minimum(
        circular_std_from_resultant(jnp.abs(sums["z_plus"]) / n),
        circular_std_from_resultant(jnp.abs(sums["z_minus"]) / n),
    ) / (2 * jnp.pi)
    cov = sums["ap"] / n - (sums["a"] / n) * (sums["p"] / n)
    var_a = sums["aa"] / n - (sums["a"] / n) ** 2
    var_p = sums["pp"] / n - (sums["p"] / n) ** 2
    corr = cov / jnp.sqrt(jnp.maximum(var_a * var_p, 1e-300))
    return spread, jnp.abs(corr)


def circular_std_from_resultant(r: Array) -> Array:
    """``sqrt(-2 ln r)`` for a mean resultant length `r`, clipped to ``(0, 1]``."""
    return jnp.sqrt(-2.0 * jnp.log(jnp.clip(r, 1e-300, 1.0)))


@eqx.filter_jit
def bijection_grid_metrics(
    bijection: AbstractBijection, grid: Float[Array, "m dim"]
) -> dict[str, Float[Array, ""]]:
    """The maximum round-trip error ``H⁻¹(H(x)) − x`` and the minimum and maximum
    Jacobian singular value of `bijection` on `grid`, as ``roundtrip/max``,
    ``jac/sv_min`` and ``jac/sv_max``; per layer too for a ``SequentialINN``, as
    ``jac/sv_min/<i>``, with the Jacobian of layer ``i`` at that layer's input."""

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
    """``period`` and ``kappa`` of `model`: the normal form's closed forms for the
    conjugacy model, with its parameters as ``nf/<name>``; ``2π/|ω|`` and ``κ`` of
    ``PhaseAmplitudeLatentDynamics`` for the autoencoder; empty otherwise."""
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
    """The validation metrics of a model over a finite, re-iterable collection of
    batches, iterated to exhaustion on every call so that each evaluation sees the
    same windows and the metric is comparable across steps (ADR-0009).

    **Attributes:**

    - `val_data`: The batches: a ``validation_windows(...).batch(B)`` dataset, the
        intended source, or a list from `collect_batches`.
    - `reference`: The normal form the data were generated from, when there is one;
        the ground truth for phase and amplitude.
    - `grid_points`: Points per axis of the bounding-box grid for the bijection
        diagnostics; the grid is taken from the data on the first pass.
    - `t_split`: Splits the prediction error by window start: windows starting before
        it, the transient, give ``val/mse_early``, the rest ``val/mse_late``, and
        ``val/mse`` stays the overall mean. ``None`` for no split
        (``docs/design/validation-split.md``).
    """

    val_data: Iterable[dict[str, Array | np.ndarray]]
    reference: AbstractNormalForm | None = None
    grid_points: int = 32
    t_split: float | None = None
    _grid: Array | None = field(default=None, init=False, repr=False)

    def _batches(self) -> Iterable[dict[str, Array | np.ndarray]]:
        """One pass over the validation data; a grain dataset is read with a single
        thread so that evaluation does not start 16 reader threads next to JAX."""
        if isinstance(self.val_data, grain.MapDataset | grain.IterDataset):
            return single_threaded(self.val_data)
        return self.val_data

    @property
    def grid(self) -> Float[Array, "m dim"]:
        """The bounding-box grid, built from the data on first use."""
        if self._grid is None:
            self._grid = bounding_box_grid(self._bounds(), self.grid_points)
        return self._grid

    def _bounds(self) -> Float[Array, "2 dim"]:
        """The per-coordinate minimum and maximum over the validation data.

        **Raises:**

        `ValueError` when the data are empty."""
        lo = hi = None
        for batch in self._batches():
            u = np.asarray(batch["u"]).reshape(-1, np.shape(batch["u"])[-1])
            lo = u.min(0) if lo is None else np.minimum(lo, u.min(0))
            hi = u.max(0) if hi is None else np.maximum(hi, u.max(0))
        if lo is None:
            raise ValueError("val_data is empty.")
        return jnp.stack((jnp.asarray(lo), jnp.asarray(hi)))

    def __call__(self, model: AbstractPhaseAmplitudeModel) -> dict[str, float]:
        """The metrics of `model` over one pass of the validation data, as floats;
        an empty bucket of the split is reported as ``nan`` rather than hidden.

        **Raises:**

        `ValueError` when the data are empty."""
        t_split = jnp.asarray(-jnp.inf if self.t_split is None else self.t_split)
        pred: dict[str, Array] | None = None
        sums: dict[str, Array] | None = None
        extent: dict[str, Array] | None = None
        for batch in self._batches():
            part_p = prediction_sums(model, batch, t_split)
            pred = part_p if pred is None else {k: pred[k] + part_p[k] for k in part_p}
            u = jnp.asarray(batch["u"])
            extent = _combine_extent(
                extent, latent_extent(model, u.reshape(-1, u.shape[-1]))
            )
            if self.reference is not None:
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
        assert extent is not None
        out["latent/nonfinite"] = extent["nonfinite"] / extent["n"]
        for k in ("amp_min", "amp_max", "r_min", "r_max"):
            if k in extent:
                out[f"latent/{k}"] = extent[k]
        if "nonfinite_image" in extent:
            out["latent/nonfinite_image"] = extent["nonfinite_image"] / extent["n"]
        if isinstance(model, ConjugateLatentDynamics):
            out.update(bijection_grid_metrics(model.bijection, self.grid))
        if sums is not None:
            out["phase/circ_std"], out["amplitude/corr"] = phase_amplitude_agreement(
                sums
            )
        return {k: float(v) for k, v in out.items()}


def collect_batches(loader: Iterable[Batch], num: int) -> list[dict[str, np.ndarray]]:
    """The first `num` batches of `loader`, as NumPy arrays on the host."""
    out: list[dict[str, np.ndarray]] = []
    for batch in loader:
        out.append({k: np.asarray(v) for k, v in batch.items()})
        if len(out) == num:
            break
    return out
