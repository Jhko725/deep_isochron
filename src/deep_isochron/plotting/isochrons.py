r"""The phase-plane figures of a reference phase field, its isochrons, and a trained
model against them (roadmap O3, O7 and M).

Four figures, each the shape of the notebook cell it grew from
(``prototype_analysis.ipynb``, 2026-10-29): `plot_phase_field` draws one asymptotic
phase field as a panel, with every $k$-th reference isochron over it as a thin white
curve, so the curves read as its contour lines; `plot_phase_comparison` sets the
reference field, a model's field and their aligned difference side by side, the model's
phase aligned by `analysis.phase_origin` and `training.align_phases`;
`plot_isochron_match` draws the reference isochrons colored by phase on the cyclic map,
and, when a model is given, the learned isochron of each phase dashed in the same
color; `plot_isochron_hairpins` follows one reference isochron into the phaseless
point in the plane and in log-polar coordinates about the equilibrium, with the
model's matched points beside it. The functions return the figure and the numbers a
caller would print; they print nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from string import ascii_uppercase

import colorcet as cc
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Float
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, Normalize

from ..analysis.model import phase_origin
from ..analysis.ode import find_equilibrium, Isochron, IsochronFoliation, LimitCycle
from ..model.base import AbstractPhaseAmplitudeModel
from ..model.conjugacy import ConjugateLatentDynamics
from ..training.evaluation import align_phases, PhaseAlignment
from ..utils.numerics import wrap_phase
from .style import colored_curve


PHASE_CMAP: Colormap = cc.cm.CET_C6
r"""A cyclic map: phases $-\pi$ and $\pi$ share a color."""
PHASE_NORM = Normalize(-np.pi, np.pi)
MASKED_COLOR = "0.6"
"""The background of a field panel: the color of a masked point."""


def _cycle_curve(cycle: LimitCycle, n_points: int = 512) -> Float[np.ndarray, "n 2"]:
    """The sampled cycle, closed."""
    _, points = cycle.sample(n_points)
    points = np.asarray(points)
    return np.vstack([points, points[:1]])


def _learned_curve(
    model: AbstractPhaseAmplitudeModel, n_points: int = 512
) -> Float[np.ndarray, "n 2"]:
    """The learned cycle at `n_points` phases, closed."""
    thetas = jnp.linspace(-jnp.pi, jnp.pi, n_points, endpoint=False)
    points = np.asarray(jax.vmap(model.cycle_point)(thetas))
    return np.vstack([points, points[:1]])


def plot_phase_field(
    ax: Axes,
    xs: Float[np.ndarray, " nx"],
    ys: Float[np.ndarray, " ny"],
    field: Float[np.ndarray, "ny nx"],
    cycle: LimitCycle,
    isochrons: tuple[Isochron, ...] = (),
    *,
    every: int = 10,
    cmap: Colormap = PHASE_CMAP,
    limit: float = np.pi,
):
    """Draw a phase field on `ax`: the field as an image on a gray background (``nan``
    is masked), every `every`-th of `isochrons` as a thin white curve over it, the
    cycle in black and its phase origin as a dot.

    **Arguments:**

    - `ax`: The axes to draw on.
    - `xs`, `ys`: The grid, as for `analysis.ode.phase_grid`.
    - `field`: The phases on the grid, ``nan`` where masked.
    - `cycle`: The reference cycle.
    - `isochrons`: The reference isochrons; none draws the field alone.
    - `every`: Which isochrons to draw.
    - `cmap`, `limit`: The colormap and its symmetric range: the cyclic map over
        $\\pm\\pi$ for a phase, a diverging map over a smaller range for a residual.

    **Returns:**

    The image, for a colorbar.
    """
    ax.set_facecolor(MASKED_COLOR)
    image = ax.pcolormesh(
        xs, ys, field, cmap=cmap, vmin=-limit, vmax=limit, shading="nearest"
    )
    for iso in isochrons[::every]:
        ax.plot(*iso.points.T, color="w", lw=0.6)
    ax.plot(*_cycle_curve(cycle).T, color="k", lw=1.2)
    ax.plot(*np.asarray(cycle.x0s[0]), "ko")
    ax.set(xlabel="$x$", ylabel="$y$", aspect="equal")
    ax.set(xlim=(xs[0], xs[-1]), ylim=(ys[0], ys[-1]))
    return image


@dataclass(frozen=True)
class PhaseComparison:
    """What `plot_phase_comparison` found beside its figure.

    **Attributes:**

    - `alignment`: The model's phase aligned with the reference field.
    - `isochron_gaps`: Per reference isochron, the largest aligned difference between
        the model's phase on its points and its phase, in radians: how far each AUTO
        curve is from a level set of the model's phase. Empty without isochrons.
    """

    alignment: PhaseAlignment
    isochron_gaps: Float[np.ndarray, " k"]


def plot_phase_comparison(
    xs: Float[np.ndarray, " nx"],
    ys: Float[np.ndarray, " ny"],
    field: Float[np.ndarray, "ny nx"],
    cycle: LimitCycle,
    model: AbstractPhaseAmplitudeModel,
    isochrons: tuple[Isochron, ...] = (),
    *,
    name: str = "",
) -> tuple[plt.Figure, PhaseComparison]:
    r"""The reference phase field, the model's phase field and their aligned difference,
    side by side.

    The model's phase is aligned to the reference by `align_phases`, seeded with the
    offset that `phase_origin` finds at the reference's phase 0, so the left panel
    shows the reference field mapped into the model's phase, the middle one the
    model's own field over the reference's mask, and the right one the residual on a
    diverging map whose range is the 99.5th percentile of its magnitude; both cycles
    are drawn on every panel.

    **Arguments:**

    - `xs`, `ys`, `field`: The reference phase field, as for `plot_phase_field`.
    - `cycle`: The reference cycle.
    - `model`: The trained model.
    - `isochrons`: The reference isochrons, for the per-isochron gaps.
    - `name`: A prefix for the panel titles.

    **Returns:**

    The figure and the `PhaseComparison`.
    """
    theta0, sense = phase_origin(model, cycle)
    grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    model_field = np.asarray(jax.vmap(model.phase)(jnp.asarray(grid)))
    model_field = model_field.reshape(field.shape)
    mask = np.isfinite(field)
    alignment = align_phases(
        jnp.asarray(model_field), jnp.asarray(field), offset=float(theta0)
    )
    residual = np.asarray(alignment.residual)
    sense, theta0 = alignment.sense, alignment.offset

    fig, axes = plt.subplots(
        1, 3, figsize=(12, 4), constrained_layout=True, sharey=True
    )
    prefix = f"{name}: " if name else ""
    panels = (
        (
            "Reference phase",
            np.asarray(wrap_phase(sense * field + theta0)),
            PHASE_CMAP,
            np.pi,
        ),
        ("Learned phase", np.where(mask, model_field, np.nan), PHASE_CMAP, np.pi),
        (
            "Learned − Reference",
            residual,
            cc.cm.CET_D1A_r,
            float(np.nanpercentile(np.abs(residual), 99.5)),
        ),
    )
    images = []
    for ax, (title, data, cmap, limit) in zip(axes, panels, strict=True):
        images.append(plot_phase_field(ax, xs, ys, data, cycle, cmap=cmap, limit=limit))
        ax.plot(*_learned_curve(model).T, color="lightgrey", lw=1.0, ls="--")
        ax.set(title=f"{prefix}{title}")
    fig.colorbar(images[0], ax=axes[:2], shrink=0.8, label="Phase (rad)")
    fig.colorbar(images[2], ax=axes[2], shrink=0.8, label="Phase error (rad)")
    fig.supxlabel("$x$")
    fig.supylabel("$y$")

    gaps = []
    for iso in isochrons:
        phase = np.asarray(jax.vmap(model.phase)(jnp.asarray(iso.points)))
        gaps.append(float(np.abs(wrap_phase(phase - sense * iso.phase - theta0)).max()))
    return fig, PhaseComparison(alignment, np.asarray(gaps))


def plot_isochron_match(
    foliation: IsochronFoliation,
    model: AbstractPhaseAmplitudeModel | None = None,
    *,
    n_radii: int = 400,
    name: str = "",
) -> tuple[plt.Figure, Float[np.ndarray, " k"]]:
    r"""The reference isochrons in the plane, colored by phase on the cyclic map, and
    the model's isochrons of the same phases dashed in the same colors.

    Each learned isochron is sampled along the model's latent radial coordinate from
    $10^{-4}$ (the spiral onto the learned equilibrium) out past the latent image of
    the reference foliation's bounding box, and clipped to that box. Without a model
    the figure is the foliation alone: the cycle, the isochrons and the phase origin.

    **Arguments:**

    - `foliation`: The reference isochrons.
    - `model`: The trained model, or `None`.
    - `n_radii`: The samples of each learned isochron on each side of the cycle.
    - `name`: A prefix for the title.

    **Returns:**

    The figure, and per isochron the largest aligned difference between the model's
    phase on its points and its phase (empty without a model), as
    `plot_phase_comparison` reports it.
    """
    cycle = foliation.cycle
    fig, ax = plt.subplots(figsize=(5.5, 5), constrained_layout=True)
    ax.plot(*_cycle_curve(cycle).T, color="k", lw=1.5)
    points = np.concatenate([iso.points for iso in foliation.isochrons])
    lo, hi = points.min(axis=0), points.max(axis=0)
    gaps: list[float] = []
    if model is not None:
        theta0, sense = phase_origin(model, cycle)
        ax.plot(*_learned_curve(model).T, color="0.5", lw=1.0, ls="--")
        corners = np.array([[x, y] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])])
        r_max = 1.1 * float(
            jnp.max(jax.vmap(model.latent_radius)(jnp.asarray(corners)))
        )
        radii = jnp.concatenate(
            (
                jnp.geomspace(1e-4, 1.0, n_radii, endpoint=False),
                jnp.geomspace(1.0, r_max, n_radii),
            )
        )
    for iso in foliation.isochrons:
        color = PHASE_CMAP(PHASE_NORM(iso.phase))
        ax.plot(*iso.points.T, color=color, lw=1.2)
        if model is not None:
            theta = jnp.asarray(wrap_phase(sense * iso.phase + theta0))
            learned = np.array(model.isochron(theta, radii))
            outside = ((learned < lo) | (learned > hi)).any(axis=1)
            learned[outside] = np.nan
            ax.plot(*learned.T, color=color, lw=1.2, ls="--")
            phase = np.asarray(jax.vmap(model.phase)(jnp.asarray(iso.points)))
            gaps.append(float(np.abs(wrap_phase(phase - theta)).max()))
    ax.plot(*np.asarray(cycle.x0s[0]), "ko")  # phase 0, on the reference section
    prefix = f"{name}: " if name else ""
    title = "AUTO (solid), model (dashed)" if model is not None else "AUTO isochrons"
    ax.set(
        title=prefix + title,
        xlabel="$x$",
        ylabel="$y$",
        xlim=(lo[0], hi[0]),
        ylim=(lo[1], hi[1]),
        aspect="equal",
    )
    fig.colorbar(
        plt.cm.ScalarMappable(PHASE_NORM, PHASE_CMAP), ax=ax, label="phase $\\theta$"
    )
    return fig, np.asarray(gaps)


@dataclass(frozen=True)
class HairpinMatch:
    """What `plot_isochron_hairpins` found beside its figure.

    **Attributes:**

    - `index`: Which isochron of the foliation was drawn.
    - `equilibrium`: The reference equilibrium, found by `find_equilibrium`.
    - `model_equilibrium`: The model's, ``H⁻¹(0)``.
    - `hairpins`: The indices of the isochron's hairpins about the equilibrium, labeled
        A, B, … in the figure.
    - `matched_distance`: Per traced point, the distance to the model's matched point
        (the learned isochron of the same phase at the same latent radius), shape
        ``(n,)``: the model's phase error at that point as a displacement along its
        isostable curve.
    """

    index: int
    equilibrium: Float[np.ndarray, "2"]
    model_equilibrium: Float[np.ndarray, "2"]
    hairpins: np.ndarray
    matched_distance: Float[np.ndarray, " n"]


def plot_isochron_hairpins(
    foliation: IsochronFoliation,
    model: ConjugateLatentDynamics,
    *,
    index: int | None = None,
    min_excursion: float = 0.05,
    zoom_half_width: float | None = None,
    name: str = "",
) -> tuple[plt.Figure, HairpinMatch]:
    r"""One reference isochron followed into the phaseless point, against the model.

    Left, the plane: the isochron colored by arclength from its base point, the model's
    matched points dashed in the same colors, the equilibrium $q$, the model's
    $H^{-1}(0)$, the base point and the hairpins labeled; below it, the same about $q$
    at `zoom_half_width`. Right, against arclength: $\log_{10}\vert x - q\vert$ and the
    angle about $q$ in turns for both curves, and the matched distance on a log scale;
    the hairpins are marked on every panel.

    **Arguments:**

    - `foliation`: The reference isochrons.
    - `model`: The conjugacy model; the figure needs its equilibrium.
    - `index`: Which isochron; `None` takes the one that comes closest to $q$.
    - `min_excursion`: As for `Isochron.hairpins`.
    - `zoom_half_width`: The half-width of the zoom about $q$; `None` is 1.5 times the
        farthest hairpin's distance, or 0.2 without hairpins.
    - `name`: A prefix for the title.

    **Returns:**

    The figure and the `HairpinMatch`.

    **Raises:**

    `ValueError` when Newton does not converge to the equilibrium from the cycle's
    centroid.
    """
    cycle = foliation.cycle
    centroid = _cycle_curve(cycle)[:-1].mean(axis=0)
    q_array, converged = find_equilibrium(cycle.ode, jnp.asarray(centroid))
    if not bool(converged):
        raise ValueError(f"no equilibrium by Newton from the centroid {centroid}.")
    q = np.asarray(q_array)
    q_model = np.asarray(model.equilibrium())
    theta0, sense = phase_origin(model, cycle)
    isos = foliation.isochrons
    k = (
        int(np.argmin([np.linalg.norm(iso.points - q, axis=1).min() for iso in isos]))
        if index is None
        else index
    )
    iso = isos[k]
    theta = jnp.asarray(wrap_phase(sense * iso.phase + theta0))
    arclength = iso.arclength
    radii = jax.vmap(model.latent_radius)(jnp.asarray(iso.points))
    matched = np.array(model.isochron(theta, radii))
    error = np.linalg.norm(iso.points - matched, axis=1)
    rho_ref, phi_ref = iso.log_polar(q)
    d = matched - q
    rho_m = np.log10(np.hypot(d[:, 0], d[:, 1]))
    phi_m = np.unwrap(np.arctan2(d[:, 1], d[:, 0])) / (2 * np.pi)
    base = int(np.argmin(np.abs(arclength)))
    phi_m += np.round(phi_ref[base] - phi_m[base])
    tips = iso.hairpins(q, min_excursion)
    labels = ascii_uppercase[: len(tips)]

    norm = Normalize(float(arclength.min()), float(arclength.max()))
    cmap = cc.cm.CET_R3
    fig = plt.figure(figsize=(13, 7), constrained_layout=True)
    grid = fig.add_gridspec(3, 2, width_ratios=[1.1, 1])
    ax_full = fig.add_subplot(grid[:2, 0])
    ax_zoom = fig.add_subplot(grid[2, 0])
    ax_rho = fig.add_subplot(grid[0, 1])
    ax_phi = fig.add_subplot(grid[1, 1], sharex=ax_rho)
    ax_err = fig.add_subplot(grid[2, 1], sharex=ax_rho)

    curve = _cycle_curve(cycle)
    span = max(np.ptp(iso.points[:, 0]), np.ptp(iso.points[:, 1])) / 2
    full_center = iso.points.mean(axis=0)
    half = zoom_half_width or (
        1.5 * float(np.max(np.linalg.norm(iso.points[tips] - q, axis=1)))
        if len(tips)
        else 0.2
    )
    for ax, center, extent in ((ax_full, full_center, span), (ax_zoom, q, half)):
        ax.plot(*curve.T, color="k", lw=1.0)
        colored_curve(ax, iso.points, arclength, cmap, norm, linewidth=1.8)
        colored_curve(ax, matched, arclength, cmap, norm, linestyle="dashed")
        ax.plot(*q, "k+", ms=9, mew=1.5, label="$q$")
        ax.plot(*q_model, "kx", ms=7, mew=1.5, label="$H^{-1}(0)$")
        ax.plot(*iso.base_point, "ko", ms=5, label="base point")
        for label, t in zip(labels, tips, strict=True):
            ax.annotate(
                label,
                iso.points[t],
                xytext=(3, 3),
                textcoords="offset points",
                fontsize="small",
            )
        ax.set(
            xlim=(center[0] - extent, center[0] + extent),
            ylim=(center[1] - extent, center[1] + extent),
            aspect="equal",
            xlabel="$x$",
            ylabel="$y$",
        )
    ax_full.legend(loc="upper left", fontsize="small")
    prefix = f"{name}: " if name else ""
    ax_full.set_title(
        f"{prefix}isochron at phase {iso.phase:+.3f} (index {k}); "
        "AUTO solid, model dashed"
    )
    ax_zoom.set_title(f"about $q$, ±{half:.2g}", fontsize="medium")
    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=[ax_full, ax_zoom],
        label="arclength $L$",
        shrink=0.6,
    )

    for ax, (a, b) in zip(
        (ax_rho, ax_phi), ((rho_ref, rho_m), (phi_ref, phi_m)), strict=True
    ):
        ax.plot(arclength, a, color="C3", lw=1.4, label="AUTO")
        ax.plot(arclength, b, color="k", lw=0.8, ls="--", label="model")
    ax_rho.axhline(np.log10(np.linalg.norm(q_model - q)), color="0.6", lw=0.6, ls=":")
    ax_rho.set(ylabel=r"$\log_{10}|x - q|$")
    ax_rho.legend(loc="best", fontsize="small")
    ax_phi.set(ylabel="angle about $q$ (turns)")
    ax_err.semilogy(arclength, error, color="k", lw=0.8)
    ax_err.set(ylabel="matched distance", xlabel="arclength $L$ from the base point")
    for ax in (ax_rho, ax_phi, ax_err):
        ax.axvline(0.0, color="0.6", lw=0.6)
        for t in tips:
            ax.axvline(arclength[t], color="0.8", lw=0.6)
    for label, t in zip(labels, tips, strict=True):
        ax_rho.annotate(
            label,
            (arclength[t], 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(2, -10),
            textcoords="offset points",
            fontsize="small",
        )
    return fig, HairpinMatch(k, q, q_model, tips, error)


__all__ = [
    "HairpinMatch",
    "MASKED_COLOR",
    "PHASE_CMAP",
    "PHASE_NORM",
    "PhaseComparison",
    "plot_isochron_hairpins",
    "plot_isochron_match",
    "plot_phase_comparison",
    "plot_phase_field",
]
