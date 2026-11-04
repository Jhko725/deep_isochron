r"""The coupling-warp figure of slide 9b: a spline coupling layer drawn as a monotone
warp of one coordinate, beside the identity and an affine coupling layer as the foil.

A coupling layer on the plane keeps $x_1$ and maps $x_2$ through a scalar bijection
$s(\,\cdot\,; x_1)$ whose parameters the conditioner computes from $x_1$. The figure
has a column per layer (no coupling, affine, spline) and two rows: the warp
$s(x_2; x_1)$ at the column's middle $x_1$, and a square sheet, unmapped in the first
column and its image under the column's layer in the others. The sheet holds a regular
grid, whose lines of constant $x_1$ stay where they are and whose lines of constant
$x_2$ bend, and a few trajectories of the Bautin normal form.
`schematic_affine_coupling` and `schematic_spline_coupling` build the two layers as
real `CouplingFlow`s whose conditioners are hand-set functions of $x_1$, so that the
spline's steep band is legible by design rather than by training;
`schematic_trajectories` and `schematic_sheet` build the sheet. `WarpLabels` holds the
strings of the figure and `WarpStyle` its drawing constants; `draw_identity_warp`,
`draw_warp_curve` and `draw_grid_sheet` draw one panel each, and `plot_coupling_warps`
assembles the figure.

The design and its decisions are in `docs/changes/2026-11-04-inn-schematics.md`.
"""

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Array, Float
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from ..model.invertible import Affine, CouplingFlow, MonotonicRQSpline
from ..model.invertible.splines import AbstractSpline
from ..systems.normal_forms import BautinNormalForm
from .deformation import draw_trajectories, Sheet, SheetConfig, SheetStyle


def schematic_affine_coupling(
    log_slope: float = 0.5, x1_slope: float = 0.3
) -> CouplingFlow[Affine]:
    r"""The affine coupling layer of the figure: $x_2 \mapsto a(x_1)\, x_2$, a slope
    that varies mildly with $x_1$.

    The conditioner returns the raw `Affine` parameters `(loc, log_scale)` $= (0,\,
    \lambda)\,(1 + c\, x_1)$, with $\lambda$ = `log_slope` and $c$ = `x1_slope`, so
    the slope at $x_1 = 0$ is the scale of `Affine` at raw `log_slope`.

    **Arguments:**

    - `log_slope`: The raw log-scale at $x_1 = 0$.
    - `x1_slope`: The relative change of the raw parameters per unit of $x_1$.

    **Returns:**

    A `CouplingFlow` of dimension 2 that keeps $x_1$ and maps $x_2$.
    """
    raw = np.array([0.0, log_slope])

    def conditioner(x_const: Float[Array, " 1"]) -> Float[Array, " 2"]:
        """The raw parameters for `x_const`, scaled linearly in $x_1$."""
        return jnp.asarray(raw) * (1 + x1_slope * x_const[0])

    return CouplingFlow(dim=2, bijection=Affine(), conditioner=conditioner)


def schematic_spline_coupling(
    num_bins: int = 9,
    xy_range: tuple[float, float] = (-1.0, 1.0),
    x_width_raw: float = 0.26,
    y_width_raw: float = 2.6,
    derivative_raws: tuple[float, float] = (0.5, 0.9),
    x1_slope: float = 0.3,
) -> CouplingFlow[MonotonicRQSpline]:
    r"""The spline coupling layer of the figure: a `MonotonicRQSpline` warp of $x_2$
    that is gentle everywhere and steep in one band, which narrows as $x_1$ grows.

    The conditioner returns a raw vector that is zero except at four entries of the
    middle bin. Its x-width raw `x_width_raw` sets the band's width, and its y-width
    raw `y_width_raw`, much larger, makes the bin take most of the range's height
    from the other bins, so that the band is steep. The raws of the interior
    derivatives at its two ends, `derivative_raws`, are positive so that the slope
    there exceeds 1 and the band joins its neighbors smoothly. The other bins share
    the remaining width and height, so their slope is below 1. The vector
    is scaled by $1 + c\, x_1$, $c$ = `x1_slope`. Since the nonzero widths are those
    of the middle bin, the band stays centered at the range's midpoint; unequal
    derivative raws move its steepest point off center by an amount that changes with
    $x_1$.

    **Arguments:**

    - `num_bins`: The number of bins, odd so that there is a middle bin, at least 3.
    - `xy_range`: The range of the spline; the identity outside it.
    - `x_width_raw`: The raw of the middle bin's x-width.
    - `y_width_raw`: The raw of the middle bin's y-width.
    - `derivative_raws`: The raws of the derivatives at the middle bin's lower and
        upper knot.
    - `x1_slope`: The relative change of the raw vector per unit of $x_1$.

    **Returns:**

    A `CouplingFlow` of dimension 2 that keeps $x_1$ and maps $x_2$.

    **Raises:**

    `ValueError` if `num_bins` is even or less than 3.
    """
    if num_bins < 3 or num_bins % 2 == 0:
        raise ValueError(f"num_bins must be odd and at least 3, got {num_bins}.")
    template = MonotonicRQSpline(num_bins, xy_range)
    middle = num_bins // 2
    raw = np.zeros(template.num_params)
    # The layout of `MonotonicRQSpline.raw`: K x-widths, K y-widths, K - 1 interior
    # derivatives, of which entry j is the derivative at knot j + 1.
    raw[middle] = x_width_raw
    raw[num_bins + middle] = y_width_raw
    raw[2 * num_bins + middle - 1 : 2 * num_bins + middle + 1] = derivative_raws

    def conditioner(x_const: Float[Array, " 1"]) -> Float[Array, " raw"]:
        """The raw vector for `x_const`, scaled linearly in $x_1$."""
        return jnp.asarray(raw) * (1 + x1_slope * x_const[0])

    return CouplingFlow(dim=2, bijection=template, conditioner=conditioner)


def schematic_trajectories(
    box: tuple[float, float], n: int = 4, samples: int = 200, scale: float = 0.7
) -> Float[np.ndarray, "n samples 2"]:
    r"""Trajectories of the talk's Bautin normal form, fitted into the square `box`,
    for the sheet of the figure.

    The system is `BautinNormalForm(a=1/3, b=0.5, w=2π, w0=2π − 2/3)`, the one of
    `ksiam_animations.ipynb`, whose limit cycle is the unit circle and whose period is
    1. The `n` trajectories start at radii equally spaced from 0.35 to 1.4, inside and
    outside the cycle, at equally spaced angles, and run for 1.2 periods. They are then
    scaled by `scale` times half the box's width and shifted to its center, so that
    the cycle is a circle of radius `scale` times that half-width. Since no trajectory
    leaves the radius 1.4, every sample lies inside the box when `scale` is below
    1 / 1.4.

    **Arguments:**

    - `box`: The box `(lo, hi)` on both axes.
    - `n`: The number of trajectories.
    - `samples`: The number of samples of each trajectory, equally spaced in time.
    - `scale`: The radius of the cycle as a fraction of the box's half-width.

    **Returns:**

    The trajectories in the box's coordinates, as NumPy.
    """
    lo, hi = box
    system = BautinNormalForm(a=1 / 3, b=0.5, w=2 * jnp.pi, w0=2 * jnp.pi - 2 / 3)
    radii = np.linspace(0.35, 1.4, n)
    angles = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    u0s = jnp.asarray(radii[:, None] * np.stack([np.cos(angles), np.sin(angles)], -1))
    ts = jnp.linspace(0.0, 1.2 * system.period(), samples)
    ys = jax.vmap(lambda u0: system.flow(ts, u0).ys)(u0s)
    return 0.5 * (lo + hi) + scale * 0.5 * (hi - lo) * np.asarray(ys)


def schematic_sheet(
    box: tuple[float, float],
    trajectories: Float[np.ndarray, "t q 2"],
    cells: int = 16,
    line_samples: int = 200,
) -> Sheet:
    """The sheet of the figure: a regular grid of `cells` cells per axis on the square
    `box`, carrying `trajectories`.

    **Arguments:**

    - `box`: The box `(lo, hi)` on both axes.
    - `trajectories`: The trajectories on the sheet, such as `schematic_trajectories`.
    - `cells`: The number of grid cells along each axis, so `cells + 1` lines of each
        kind.
    - `line_samples`: The number of samples of each grid line.

    **Returns:**

    The `Sheet` on `(lo, lo, hi, hi)`.
    """
    lo, hi = box
    config = SheetConfig(cells=cells, line_samples=line_samples)
    return Sheet((lo, lo, hi, hi), trajectories, config=config)


@dataclass(frozen=True)
class WarpLabels:
    r"""The strings of the coupling-warp figure, as raw mathtext; `copy.replace`
    changes one.

    **Attributes:**

    - `identity_header`: The header of the identity column.
    - `affine_header`: The header of the affine column.
    - `spline_header`: The header of the spline column.
    - `warp_row`: The label of the row of warp curves.
    - `grid_row`: The label of the row of grid sheets.
    """

    identity_header: str = "no coupling (identity)"
    affine_header: str = "affine coupling (Real NVP, Glow)"
    spline_header: str = "spline coupling (ours)"
    warp_row: str = r"warp $s(x_2;\,x_1)$"
    grid_row: str = "image of the grid"


@dataclass(frozen=True)
class WarpStyle:
    """The drawing constants of the coupling-warp figure; `copy.replace` changes one.

    **Attributes:**

    - `labels`: The strings of the figure.
    - `moved_color`: The color of what the layer moves: the warp curve, its knots and
        the grid lines of constant $x_2$; Okabe–Ito bluish green, the color of the
        coupling block in `inn_stack`.
    - `fixed_color`: The color of what the layer keeps: the box, the identity diagonal
        and the grid lines of constant $x_1$.
    - `sheet`: The style of the trajectories on the grid sheets; its `curve_colors`,
        `curve_width`, `halo_width` and `marker_size` are used. The default colors are
        the Okabe–Ito blue, reddish purple, orange and sky blue, cycled, so that no
        trajectory takes the green of `moved_color` or the vermilion accent.
    - `curve_width`: The width of the warp curve, in points; the box and the diagonal
        are drawn at half of it.
    - `grid_width`: The width of the grid lines, in points.
    - `knot_size`: The diameter of the knot markers, in points.
    - `header_size`: The font size of the column headers, in points.
    - `label_size`: The font size of the row labels, in points.
    - `margin`: The fraction of the box's width added on each side of every panel, so
        that a warp leaving the box stays in view.
    - `figsize`: The size of the figure, in inches.
    - `fixed_every`: The spacing of the drawn lines of constant $x_1$, in sheet lines:
        the lines `0, fixed_every, …` of the sheet's `cells + 1` are drawn.
    - `line_samples`: The number of samples of the warp curve.

    **Raises:**

    `ValueError` naming the field if `margin` is negative, a width, size or `figsize`
    entry is not positive, `fixed_every` is less than 1, or `line_samples` is less
    than 2.
    """

    labels: WarpLabels = WarpLabels()
    moved_color: str = "#009E73"
    fixed_color: str = "0.65"
    sheet: SheetStyle = SheetStyle(
        curve_colors=("#0072B2", "#CC79A7", "#E69F00", "#56B4E9"),
        curve_width=1.6,
        halo_width=3.0,
        marker_size=6.0,
    )
    curve_width: float = 2.2
    grid_width: float = 1.0
    knot_size: float = 5.0
    header_size: float = 12.0
    label_size: float = 11.0
    margin: float = 0.15
    figsize: tuple[float, float] = (10.0, 6.0)
    fixed_every: int = 2
    line_samples: int = 200

    def __post_init__(self) -> None:
        """Check every validated field's range."""
        if self.margin < 0:
            raise ValueError(f"margin must not be negative, got {self.margin}.")
        for name in (
            "curve_width",
            "grid_width",
            "knot_size",
            "header_size",
            "label_size",
            "figsize",
        ):
            value = getattr(self, name)
            if min(value if isinstance(value, tuple) else (value,)) <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")
        if self.fixed_every < 1:
            raise ValueError(f"fixed_every must be at least 1, got {self.fixed_every}.")
        if self.line_samples < 2:
            raise ValueError(
                f"line_samples must be at least 2, got {self.line_samples}."
            )


def _panel_limits(box: tuple[float, float], style: WarpStyle) -> tuple[float, float]:
    """The box `(lo, hi)` widened by `style.margin` of its width on each side."""
    lo, hi = box
    pad = style.margin * (hi - lo)
    return lo - pad, hi + pad


def _set_panel_limits(ax: Axes, box: tuple[float, float], style: WarpStyle) -> None:
    """Set the widened box as the limits of both axes, at equal aspect; every panel of
    the figure calls it, so all six share one set of limits."""
    limits = _panel_limits(box, style)
    ax.set_xlim(*limits)
    ax.set_ylim(*limits)
    ax.set_aspect("equal")


def _format_warp_panel(ax: Axes, box: tuple[float, float], style: WarpStyle) -> None:
    """Set the panel limits and ticks only at the box's ends, leaving the spines and
    the grid to the rcParams: under `configure_rcparams` the left and bottom spines and
    a dashed grid at the ticks."""
    _set_panel_limits(ax, box, style)
    ax.set_xticks(box)
    ax.set_yticks(box)


def _draw_warp_frame(ax: Axes, box: tuple[float, float], style: WarpStyle) -> None:
    """Draw the box as a square outline and the identity as a dashed diagonal, both in
    `style.fixed_color` at half the curve width."""
    lo, hi = box
    frame_width = 0.5 * style.curve_width
    ax.add_patch(
        Rectangle(
            (lo, lo),
            hi - lo,
            hi - lo,
            fill=False,
            edgecolor=style.fixed_color,
            linewidth=frame_width,
        )
    )
    ax.plot(
        [lo, hi],
        [lo, hi],
        color=style.fixed_color,
        linewidth=frame_width,
        linestyle="--",
    )


def draw_identity_warp(
    ax: Axes, box: tuple[float, float], style: WarpStyle = WarpStyle()
) -> None:
    r"""Draw the warp of no coupling, $x_2 \mapsto x_2$, as the warp panels of
    `draw_warp_curve` draw a layer's: inside the box and over the identity diagonal.

    The identity is drawn across the widened box in `style.moved_color` at
    `style.curve_width`, so it covers the dashed diagonal inside the box.

    **Arguments:**

    - `ax`: The axes to draw on; its limits, aspect and ticks are set here.
    - `box`: The box `(lo, hi)` on both axes.
    - `style`: The drawing constants.
    """
    _draw_warp_frame(ax, box, style)
    limits = _panel_limits(box, style)
    ax.plot(limits, limits, color=style.moved_color, linewidth=style.curve_width)
    _format_warp_panel(ax, box, style)


def draw_warp_curve(
    ax: Axes,
    layer: CouplingFlow,
    box: tuple[float, float],
    x1: float,
    style: WarpStyle = WarpStyle(),
) -> None:
    r"""Draw the warp $x_2 \mapsto s(x_2; x_1)$ of `layer` at one $x_1$, inside the box
    and over the identity diagonal.

    The box is a square outline and the identity a dashed diagonal, both in
    `style.fixed_color`. The warp is evaluated on `style.line_samples` points across
    the widened box and drawn in `style.moved_color`, so a warp that leaves the box,
    as an affine one does, is seen leaving it. When the layer's template is a spline,
    its knots are marked.

    **Arguments:**

    - `ax`: The axes to draw on; its limits, aspect and ticks are set here.
    - `layer`: A coupling layer of dimension 2 with one coupled coordinate.
    - `box`: The box `(lo, hi)` on both axes.
    - `x1`: The value of the kept coordinate at which the warp is drawn.
    - `style`: The drawing constants.
    """
    _draw_warp_frame(ax, box, style)
    bijection = jax.tree.map(lambda a: a[0], layer.make_bijection(jnp.array([x1])))
    x2s = jnp.linspace(*_panel_limits(box, style), style.line_samples)
    ax.plot(
        np.asarray(x2s),
        np.asarray(jax.vmap(bijection)(x2s)),
        color=style.moved_color,
        linewidth=style.curve_width,
    )
    if isinstance(layer.template, AbstractSpline):
        ax.plot(
            np.asarray(bijection.xs),
            np.asarray(bijection.ys),
            linestyle="none",
            marker="o",
            markersize=style.knot_size,
            color=style.moved_color,
        )
    _format_warp_panel(ax, box, style)


def draw_grid_sheet(
    ax: Axes,
    sheet: Sheet,
    box: tuple[float, float],
    style: WarpStyle = WarpStyle(),
) -> None:
    r"""Draw the grid and the trajectories of `sheet`, a sheet of `schematic_sheet` or
    its image under a coupling layer.

    The caller maps the sheet, as `sheet.map(eqx.filter_jit(layer))`. Every
    `style.fixed_every`-th line of constant $x_1$ is drawn in `style.fixed_color`, since
    a coupling layer keeps it where it is, and every line of constant $x_2$ in
    `style.moved_color` above them. The trajectories come last, by `draw_trajectories`
    at `style.sheet`. The panel has no ticks, spines or grid.

    **Arguments:**

    - `ax`: The axes to draw on; its limits, aspect, ticks, spines and grid are set
        here.
    - `sheet`: The sheet to draw.
    - `box`: The box `(lo, hi)` of the unmapped sheet, which sets the limits.
    - `style`: The drawing constants.
    """
    cells = sheet.config.cells
    lines = np.asarray(sheet.lines)
    ax.add_collection(
        LineCollection(
            list(lines[: cells + 1 : style.fixed_every]),
            colors=style.fixed_color,
            linewidths=style.grid_width,
            zorder=1,
        )
    )
    ax.add_collection(
        LineCollection(
            list(lines[cells + 1 :]),
            colors=style.moved_color,
            linewidths=style.grid_width,
            zorder=2,
        )
    )
    draw_trajectories(ax, np.asarray(sheet.trajectories), style.sheet)
    _set_panel_limits(ax, box, style)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_coupling_warps(
    affine: CouplingFlow,
    spline: CouplingFlow[MonotonicRQSpline],
    trajectories: Float[np.ndarray, "t q 2"],
    box: tuple[float, float] | None = None,
    style: WarpStyle = WarpStyle(),
) -> Figure:
    """The coupling-warp figure: columns no coupling, affine and spline, rows the warp
    at the column's middle $x_1$ and the grid sheet before and after the layer.

    The sheet is `schematic_sheet(box, trajectories)`: the first column draws it
    unmapped, the others its images under the jitted layers. All six panels share
    the limits of the box widened by `style.margin`.

    **Arguments:**

    - `affine`: The affine coupling layer, the middle column.
    - `spline`: The spline coupling layer, the right column.
    - `trajectories`: The trajectories on the sheet, such as
        `schematic_trajectories(box)`.
    - `box`: The box `(lo, hi)` on both axes; the spline's range when omitted.
    - `style`: The drawing constants.

    **Returns:**

    The figure, at `style.figsize` with constrained layout.
    """
    box = spline.template.xy_range if box is None else box
    x1 = 0.5 * (box[0] + box[1])
    sheet = schematic_sheet(box, trajectories)
    fig, axes = plt.subplots(2, 3, figsize=style.figsize, layout="constrained")
    labels = style.labels
    draw_identity_warp(axes[0, 0], box, style)
    draw_grid_sheet(axes[1, 0], sheet, box, style)
    for column, layer in [(1, affine), (2, spline)]:
        draw_warp_curve(axes[0, column], layer, box, x1, style)
        draw_grid_sheet(axes[1, column], sheet.map(eqx.filter_jit(layer)), box, style)
    headers = [labels.identity_header, labels.affine_header, labels.spline_header]
    for ax, header in zip(axes[0], headers, strict=True):
        ax.set_title(header, fontsize=style.header_size)
    for row, label in enumerate([labels.warp_row, labels.grid_row]):
        axes[row, 0].set_ylabel(label, fontsize=style.label_size)
    return fig
