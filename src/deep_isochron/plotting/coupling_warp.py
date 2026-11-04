r"""The coupling-warp figure of slide 9b: a spline coupling layer drawn as a monotone
warp of one coordinate, beside an affine coupling layer as the foil.

A coupling layer on the plane keeps $x_1$ and maps $x_2$ through a scalar bijection
$s(\,\cdot\,; x_1)$ whose parameters the conditioner computes from $x_1$. The figure
has a column per layer and two rows: the warp $s(x_2; x_1)$ at the column's middle
$x_1$, and the image of a regular grid under the layer, whose lines of constant $x_1$
stay where they are and whose lines of constant $x_2$ bend. `schematic_affine_coupling`
and `schematic_spline_coupling` build the two layers as real `CouplingFlow`s whose
conditioners are hand-set functions of $x_1$, so that the spline's steep band is
legible by design rather than by training. `WarpLabels` holds the four strings of the
figure and `WarpStyle` its drawing constants; `draw_warp_curve` and `draw_warped_grid`
draw one panel each, and `plot_coupling_warps` assembles the figure.

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
from .deformation import Sheet, SheetConfig


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


@dataclass(frozen=True)
class WarpLabels:
    r"""The strings of the coupling-warp figure, as raw mathtext; `copy.replace`
    changes one.

    **Attributes:**

    - `affine_header`: The header of the affine column.
    - `spline_header`: The header of the spline column.
    - `warp_row`: The label of the row of warp curves.
    - `grid_row`: The label of the row of grid images.
    """

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
        the grid lines of constant $x_2$. A placeholder until the deck's palette is
        fixed.
    - `fixed_color`: The color of what the layer keeps: the box, the identity diagonal
        and the grid lines of constant $x_1$.
    - `curve_width`: The width of the warp curve, in points; the box and the diagonal
        are drawn at half of it.
    - `grid_width`: The width of the grid lines, in points.
    - `knot_size`: The diameter of the knot markers, in points.
    - `header_size`: The font size of the column headers, in points.
    - `label_size`: The font size of the row labels, in points.
    - `margin`: The fraction of the box's width added on each side of every panel, so
        that a warp leaving the box stays in view.
    - `figsize`: The size of the figure, in inches; about 60 percent of a 13.33 × 7.5
        inch slide.
    - `cells`: The number of grid cells along $x_1$, so `cells + 1` fixed lines.
    - `rows`: The number of grid cells along $x_2$, so `rows + 1` moved lines.
    - `line_samples`: The number of samples of the warp curve and of each grid line.

    **Raises:**

    `ValueError` naming the field if `margin` is negative, a width, size or `figsize`
    entry is not positive, `cells` or `rows` is less than 1, or `line_samples` is less
    than 2.
    """

    labels: WarpLabels = WarpLabels()
    moved_color: str = "#2b6cb0"
    fixed_color: str = "0.65"
    curve_width: float = 2.2
    grid_width: float = 1.0
    knot_size: float = 5.0
    header_size: float = 12.0
    label_size: float = 11.0
    margin: float = 0.15
    figsize: tuple[float, float] = (8.0, 5.6)
    cells: int = 9
    rows: int = 16
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
        for name in ("cells", "rows"):
            if getattr(self, name) < 1:
                raise ValueError(
                    f"{name} must be at least 1, got {getattr(self, name)}."
                )
        if self.line_samples < 2:
            raise ValueError(
                f"line_samples must be at least 2, got {self.line_samples}."
            )


def _panel_limits(box: tuple[float, float], style: WarpStyle) -> tuple[float, float]:
    """The box `(lo, hi)` widened by `style.margin` of its width on each side."""
    lo, hi = box
    pad = style.margin * (hi - lo)
    return lo - pad, hi + pad


def _format_panel(ax: Axes, box: tuple[float, float], style: WarpStyle) -> None:
    """Set the widened box as the limits of both axes at equal aspect, and remove the
    ticks, the spines and the grid, so that only what the drawer adds is seen."""
    limits = _panel_limits(box, style)
    ax.set_xlim(*limits)
    ax.set_ylim(*limits)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_warp_curve(
    ax: Axes,
    layer: CouplingFlow,
    box: tuple[float, float],
    x1: float,
    style: WarpStyle = WarpStyle(),
) -> None:
    r"""Draw the warp $x_2 \mapsto s(x_2; x_1)$ of `layer` at one $x_1$, inside the box
    and the identity diagonal.

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
    _format_panel(ax, box, style)


def draw_warped_grid(
    ax: Axes,
    layer: CouplingFlow,
    box: tuple[float, float],
    style: WarpStyle = WarpStyle(),
) -> None:
    r"""Draw the image under `layer` of a regular grid on the square `box`.

    The lines of constant $x_1$ are `style.cells + 1` and stay where they are, so they
    are drawn in `style.fixed_color`; the lines of constant $x_2$ are `style.rows + 1`
    and bend, so they are drawn in `style.moved_color` above the fixed ones. Each set
    comes from its own `Sheet`, at `cells` and at `rows` cells, mapped through the
    jitted layer.

    **Arguments:**

    - `ax`: The axes to draw on; its limits, aspect and ticks are set here.
    - `layer`: A coupling layer of dimension 2 with one coupled coordinate.
    - `box`: The box `(lo, hi)` on both axes.
    - `style`: The drawing constants.
    """
    lo, hi = box
    mapped = eqx.filter_jit(layer)
    no_trajectories = np.zeros((0, 2, 2))

    def image_lines(cells: int) -> Float[np.ndarray, "l s 2"]:
        """The mapped mesh lines of a sheet of `cells` cells per axis."""
        config = SheetConfig(cells=cells, line_samples=style.line_samples)
        sheet = Sheet((lo, lo, hi, hi), no_trajectories, config=config)
        return np.asarray(sheet.map(mapped).lines)

    fixed = image_lines(style.cells)[: style.cells + 1]
    moved = image_lines(style.rows)[style.rows + 1 :]
    ax.add_collection(
        LineCollection(
            list(fixed), colors=style.fixed_color, linewidths=style.grid_width, zorder=1
        )
    )
    ax.add_collection(
        LineCollection(
            list(moved), colors=style.moved_color, linewidths=style.grid_width, zorder=2
        )
    )
    _format_panel(ax, box, style)


def plot_coupling_warps(
    affine: CouplingFlow,
    spline: CouplingFlow[MonotonicRQSpline],
    box: tuple[float, float] | None = None,
    style: WarpStyle = WarpStyle(),
) -> Figure:
    """The coupling-warp figure: columns affine and spline, rows the warp at the
    column's middle $x_1$ and the image of the grid.

    **Arguments:**

    - `affine`: The affine coupling layer, the left column.
    - `spline`: The spline coupling layer, the right column.
    - `box`: The box `(lo, hi)` on both axes; the spline's range when omitted.
    - `style`: The drawing constants.

    **Returns:**

    The figure, at `style.figsize` with constrained layout.
    """
    box = spline.template.xy_range if box is None else box
    x1 = 0.5 * (box[0] + box[1])
    fig, axes = plt.subplots(2, 2, figsize=style.figsize, layout="constrained")
    labels = style.labels
    for column, (layer, header) in enumerate(
        [(affine, labels.affine_header), (spline, labels.spline_header)]
    ):
        draw_warp_curve(axes[0, column], layer, box, x1, style)
        draw_warped_grid(axes[1, column], layer, box, style)
        axes[0, column].set_title(header, fontsize=style.header_size)
    for row, label in enumerate([labels.warp_row, labels.grid_row]):
        axes[row, 0].set_ylabel(label, fontsize=style.label_size)
    return fig
