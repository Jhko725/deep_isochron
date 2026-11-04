r"""The network figure of slide 9a: the invertible network drawn as a chain of
alternating linear and spline coupling layers, with its exact inverse.

The chain has six blocks $h_1, h_2, h_3, h_4, \dots, h_{N-1}, h_N$, linear and coupling
in turn from a linear block, as the experiment's network starts with a linear layer.
Double-headed arrows join consecutive blocks, the forward map $H$ runs above the chain
and the inverse $H^{-1}$ below it, and the words at its ends name the two spaces.
Under the first linear block and under the first coupling block an inset pair shows
one sheet before and after that layer: a regular grid on a square carrying
trajectories of the Bautin normal form, drawn by `coupling_warp.draw_grid_sheet` in
the neutral grays of `WarpStyle`, with a frame in the layer's color around the
after-panel. The start triangles turn the same way in both panels, which shows that
neither layer flips the plane. `schematic_linear` builds the linear layer of the inset
as a real `BiLipschitzLinear` with hand-set leaves. `StackLabels` holds every string
of the figure and `StackStyle` its drawing constants, with the geometry in inches;
`plot_inn_stack` draws the figure on an overlay axes whose data units are inches.

The design and its decisions are in `docs/changes/2026-11-04-inn-schematics.md`.
"""

import copy
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Float
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrow, FancyArrowPatch, FancyBboxPatch, Rectangle

from ..model.invertible import BiLipschitzLinear, CouplingFlow, MonotonicRQSpline
from .coupling_warp import draw_grid_sheet, schematic_sheet, WarpStyle


_NUM_BLOCKS = 6
"""The blocks of the chain: four from the start, two at the end."""


def schematic_linear(
    angle_raw: float = 0.3, stretch_raw: float = 1.0, max_lipschitz: float = 3.0
) -> BiLipschitzLinear:
    r"""The linear layer of the figure: a rotation followed by a bounded stretch along
    one axis and a matching squeeze along the other.

    The layer is the identity `BiLipschitzLinear` of dimension 2 without bias, with
    `raw_U[1, 0]` set to `angle_raw` and `raw_s` to `(stretch_raw, -stretch_raw)`. So
    `U` is the rotation by $-2 \arctan$ `angle_raw`, `V` is the identity, and the
    singular values are the images of $\pm$ `stretch_raw` in $(1/L, L)$. Their
    product is 1 for the defaults, and $\det W > 0$ for every value.

    **Arguments:**

    - `angle_raw`: The raw generator of the rotation `U`.
    - `stretch_raw`: The raw of the first singular value; the second is its negative.
    - `max_lipschitz`: $L > 1$, the bound of the singular values.

    **Returns:**

    A `BiLipschitzLinear` of dimension 2 without bias.
    """
    identity = BiLipschitzLinear(
        2, max_lipschitz, use_bias=False, key=jax.random.PRNGKey(0)
    )
    raw_U = jnp.zeros((2, 2)).at[1, 0].set(angle_raw)
    raw_s = jnp.array([stretch_raw, -stretch_raw])
    return eqx.tree_at(lambda m: (m.raw_U, m.raw_s), identity, (raw_U, raw_s))


@dataclass(frozen=True)
class StackLabels:
    r"""The strings of the network figure, as raw mathtext; `copy.replace` changes
    one.

    **Attributes:**

    - `blocks`: The labels of the six blocks, in chain order.
    - `ellipsis`: The text between the fourth and fifth block.
    - `forward`: The label of the forward arrow, above the chain.
    - `inverse`: The label of the inverse arrow, below the chain.
    - `input`: The word left of the chain, for the space the forward map starts in.
    - `output`: The word right of the chain, for the space the forward map ends in.
    - `linear`: The type name under the inset of the linear layer.
    - `coupling`: The type name under the inset of the coupling layer.

    **Raises:**

    `ValueError` if `blocks` does not have six entries.
    """

    blocks: tuple[str, ...] = (
        r"$h_1$",
        r"$h_2$",
        r"$h_3$",
        r"$h_4$",
        r"$h_{N-1}$",
        r"$h_N$",
    )
    ellipsis: str = r"$\cdots$"
    forward: str = r"$H$"
    inverse: str = r"$H^{-1}$"
    input: str = "oscillator state"
    output: str = "normal-form coordinates"
    linear: str = "linear"
    coupling: str = "spline coupling"

    def __post_init__(self) -> None:
        """Check the number of block labels."""
        if len(self.blocks) != _NUM_BLOCKS:
            raise ValueError(
                f"blocks must have {_NUM_BLOCKS} entries, got {len(self.blocks)}."
            )


@dataclass(frozen=True)
class StackStyle:
    """The drawing constants of the network figure; `copy.replace` changes one.

    **Attributes:**

    - `labels`: The strings of the figure.
    - `linear_color`: The color of the linear blocks, their type name and the frame
        of the linear after-panel; Okabe–Ito vermilion.
    - `coupling_color`: The color of the coupling blocks, their type name and the
        frame of the coupling after-panel; Okabe–Ito bluish green.
    - `block_text_color`: The color of the block labels, drawn on the block fill.
    - `arrow_color`: The color of every arrow and of the ellipsis.
    - `block_size`: The width and height of each block, in inches.
    - `block_gap`: The gap between consecutive blocks, in inches.
    - `ellipsis_gap`: The gap between the fourth and fifth block, which holds the
        ellipsis, in inches.
    - `arrow_offset`: The distance between the chain and each long arrow, and between
        the inverse's label and the insets, in inches.
    - `label_pad`: The distance between an arrow, block or inset and the text next to
        it, in inches.
    - `inset_size`: The edge of each square inset panel, in inches.
    - `inset_gap`: The gap between the two panels of an inset pair, which holds a
        small arrow, in inches.
    - `inset_separation`: The least gap between the two inset pairs, in inches; the
        pairs move apart from under their blocks when they would come closer.
    - `inset_cells`: The number of grid cells along each axis of the inset sheet.
    - `inset_margin`: The fraction of an inset's box added on each side.
    - `inset_line_width`: The width of the grid lines of the insets, in points.
    - `inset_curve_width`: The width of the trajectories of the insets, in points.
    - `inset_halo_width`: The width of the white stroke under each trajectory of the
        insets, in points.
    - `inset_marker_size`: The size of the start triangles of the insets, in points.
    - `frame_width`: The line width of the frame around each after-panel, in points.
    - `arrow_width`: The line width of the arrows, in points.
    - `arrow_head`: The size of the arrow heads, in points.
    - `block_label_size`: The font size of the block labels and the ellipsis, in
        points.
    - `arrow_label_size`: The font size of the labels of the long arrows, in points.
    - `endpoint_label_size`: The font size of the words at the chain's ends, in
        points.
    - `inset_label_size`: The font size of the type names, in points.
    - `figsize`: The size of the figure, in inches: the width of a 13.33 × 7.5 inch
        slide less its margins, with the chain in the upper part and the inset pairs
        below it.

    **Raises:**

    `ValueError` naming the field if `inset_margin` is negative, a size, gap, width,
    offset or `figsize` entry is not positive, `inset_cells` is less than 1, or
    `ellipsis_gap` is less than `block_gap`.
    """

    labels: StackLabels = StackLabels()
    linear_color: str = "#D55E00"
    coupling_color: str = "#009E73"
    block_text_color: str = "white"
    arrow_color: str = "0.2"
    block_size: tuple[float, float] = (0.9, 1.6)
    block_gap: float = 0.35
    ellipsis_gap: float = 0.7
    arrow_offset: float = 0.22
    label_pad: float = 0.08
    inset_size: float = 1.15
    inset_gap: float = 0.3
    inset_separation: float = 0.3
    inset_cells: int = 10
    inset_margin: float = 0.12
    inset_line_width: float = 0.6
    inset_curve_width: float = 1.1
    inset_halo_width: float = 2.2
    inset_marker_size: float = 4.5
    frame_width: float = 2.0
    arrow_width: float = 1.2
    arrow_head: float = 12.0
    block_label_size: float = 15.0
    arrow_label_size: float = 16.0
    endpoint_label_size: float = 12.0
    inset_label_size: float = 12.0
    figsize: tuple[float, float] = (12.0, 5.5)

    def __post_init__(self) -> None:
        """Check every validated field's range."""
        if self.inset_margin < 0:
            raise ValueError(
                f"inset_margin must not be negative, got {self.inset_margin}."
            )
        for name in (
            "block_size",
            "block_gap",
            "ellipsis_gap",
            "arrow_offset",
            "label_pad",
            "inset_size",
            "inset_gap",
            "inset_separation",
            "inset_line_width",
            "inset_curve_width",
            "inset_halo_width",
            "inset_marker_size",
            "frame_width",
            "arrow_width",
            "arrow_head",
            "block_label_size",
            "arrow_label_size",
            "endpoint_label_size",
            "inset_label_size",
            "figsize",
        ):
            value = getattr(self, name)
            if min(value if isinstance(value, tuple) else (value,)) <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")
        if self.inset_cells < 1:
            raise ValueError(f"inset_cells must be at least 1, got {self.inset_cells}.")
        if self.ellipsis_gap < self.block_gap:
            raise ValueError(
                f"ellipsis_gap must be at least block_gap ({self.block_gap}), got "
                f"{self.ellipsis_gap}."
            )


def _text_height(size: float) -> float:
    """The height in inches given to one line of text of font size `size` points."""
    return 1.2 * size / 72


def _inset_style(style: StackStyle) -> WarpStyle:
    """The `WarpStyle` of the inset panels: the default grays, every line of constant
    $x_1$ drawn, and the inset's margin and line sizes."""
    default = WarpStyle()
    return copy.replace(
        default,
        margin=style.inset_margin,
        grid_width=style.inset_line_width,
        fixed_every=1,
        sheet=copy.replace(
            default.sheet,
            curve_width=style.inset_curve_width,
            halo_width=style.inset_halo_width,
            marker_size=style.inset_marker_size,
        ),
    )


def plot_inn_stack(
    linear: BiLipschitzLinear,
    coupling: CouplingFlow[MonotonicRQSpline],
    trajectories: Float[np.ndarray, "n samples 2"],
    style: StackStyle = StackStyle(),
) -> Figure:
    r"""The network figure: the chain of six blocks with the forward and inverse
    arrows, the words at its ends, and an inset pair under the first linear and the
    first coupling block.

    The figure is drawn on an overlay axes spanning the figure whose data units are
    inches, and the four inset panels are placed by rect from the same inch geometry.
    The content is centered vertically. The row of the two words and the chain is
    centered horizontally, with the words measured as drawn. Each inset pair is
    centered under its block and moved outward when the two pairs would come closer
    than `style.inset_separation`. Both pairs draw one `schematic_sheet` of
    `style.inset_cells` cells on the spline's range, carrying `trajectories`, before and
    after their layer, with `draw_grid_sheet` in the grays of `WarpStyle`. A rectangle
    of `style.frame_width` points in the block's color frames the after-panel of each
    pair, on the edges of its axes; the before-panels have no frame.

    **Arguments:**

    - `linear`: The linear layer of the left inset pair, of dimension 2.
    - `coupling`: The spline coupling layer of the right inset pair, of dimension 2;
        the sheet is drawn on its spline's range.
    - `trajectories`: The trajectories on the sheet, such as `schematic_trajectories`
        of the spline's range.
    - `style`: The drawing constants.

    **Returns:**

    The figure, at `style.figsize`, with the overlay axes first and the four inset
    axes after it.
    """
    labels = style.labels
    width, height = style.figsize
    block_width, block_height = style.block_size
    forward_height = _text_height(style.arrow_label_size)
    inset_label_height = _text_height(style.inset_label_size)

    # Vertical layout, top down, as offsets from the top of the content.
    forward_y = forward_height + style.label_pad
    chain_top = forward_y + style.arrow_offset
    chain_bottom = chain_top + block_height
    inverse_y = chain_bottom + style.arrow_offset
    inset_top = inverse_y + style.label_pad + forward_height + style.arrow_offset
    inset_bottom = inset_top + style.inset_size
    content = inset_bottom + style.label_pad + inset_label_height
    top = height - 0.5 * (height - content)

    def y(offset: float) -> float:
        """The overlay ordinate of an offset from the top of the content."""
        return top - offset

    fig = plt.figure(figsize=style.figsize)
    overlay = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    overlay.set_xlim(0, width)
    overlay.set_ylim(0, height)
    overlay.axis("off")
    pad = style.label_pad
    chain_y = y(0.5 * (chain_top + chain_bottom))
    endpoints = [
        overlay.text(
            0.0,
            chain_y,
            text,
            ha=ha,
            va="center",
            fontsize=style.endpoint_label_size,
        )
        for text, ha in [(labels.input, "right"), (labels.output, "left")]
    ]
    input_width, output_width = (
        text.get_window_extent().width / fig.dpi for text in endpoints
    )

    # Horizontal layout: block left edges, the ellipsis gap after the fourth block,
    # with the row of the two words and the chain centered.
    gaps = [style.block_gap] * (_NUM_BLOCKS - 1)
    gaps[3] = style.ellipsis_gap
    lefts = np.cumsum([0.0, *(block_width + gap for gap in gaps)])
    row_width = input_width + 4 * pad + lefts[-1] + block_width + output_width
    lefts = lefts + 0.5 * (width - row_width) + input_width + 2 * pad
    rights = lefts + block_width
    centers = lefts + 0.5 * block_width
    endpoints[0].set_x(lefts[0] - 2 * pad)
    endpoints[1].set_x(rights[-1] + 2 * pad)

    colors = [style.linear_color, style.coupling_color] * (_NUM_BLOCKS // 2)
    for left, center, color, label in zip(
        lefts, centers, colors, labels.blocks, strict=True
    ):
        overlay.add_patch(
            FancyBboxPatch(
                (left, y(chain_bottom)),
                block_width,
                block_height,
                boxstyle="round,pad=0,rounding_size=0.08",
                facecolor=color,
                edgecolor="none",
            )
        )
        overlay.text(
            center,
            chain_y,
            label,
            ha="center",
            va="center",
            color=style.block_text_color,
            fontsize=style.block_label_size,
        )

    def arrow(start: float, end: float, at: float, arrowstyle: str) -> None:
        """Add an arrow of `arrowstyle` along the overlay row `at`."""
        overlay.add_patch(
            FancyArrowPatch(
                (start, at),
                (end, at),
                arrowstyle=arrowstyle,
                mutation_scale=style.arrow_head,
                linewidth=style.arrow_width,
                color=style.arrow_color,
                shrinkA=0,
                shrinkB=0,
            )
        )

    for right, next_left in zip(rights[:-1], lefts[1:], strict=True):
        arrow(right + pad, next_left - pad, chain_y, "<->")
    overlay.text(
        0.5 * (rights[3] + lefts[4]),
        chain_y,
        labels.ellipsis,
        ha="center",
        va="center",
        color=style.arrow_color,
        fontsize=style.block_label_size,
        bbox=dict(facecolor="white", edgecolor="none", pad=2.0),
        zorder=3,
    )

    middle = 0.5 * (lefts[0] + rights[-1])
    arrow(lefts[0], rights[-1], y(forward_y), "->")
    overlay.text(
        middle,
        y(forward_y) + pad,
        labels.forward,
        ha="center",
        va="bottom",
        fontsize=style.arrow_label_size,
    )
    arrow(rights[-1], lefts[0], y(inverse_y), "->")
    overlay.text(
        middle,
        y(inverse_y) - pad,
        labels.inverse,
        ha="center",
        va="top",
        fontsize=style.arrow_label_size,
    )
    # The inset pairs, centered under blocks 0 and 1 and moved apart if too close.
    pair_width = 2 * style.inset_size + style.inset_gap
    shortfall = style.inset_separation - (centers[1] - centers[0] - pair_width)
    shift = 0.5 * max(0.0, shortfall)
    pair_lefts = [
        centers[0] - 0.5 * pair_width - shift,
        centers[1] - 0.5 * pair_width + shift,
    ]

    def inset(left: float) -> Axes:
        """Add a square inset axes with its left edge at `left` and its top at
        `inset_top`."""
        return fig.add_axes(
            (
                left / width,
                y(inset_bottom) / height,
                style.inset_size / width,
                style.inset_size / height,
            )
        )

    box = coupling.template.xy_range
    inset_y = y(0.5 * (inset_top + inset_bottom))
    sheet = schematic_sheet(box, trajectories, cells=style.inset_cells)
    warp_style = _inset_style(style)
    for pair_left, layer, color in [
        (pair_lefts[0], linear, style.linear_color),
        (pair_lefts[1], coupling, style.coupling_color),
    ]:
        draw_grid_sheet(inset(pair_left), sheet, box, warp_style)
        after = inset(pair_left + style.inset_size + style.inset_gap)
        draw_grid_sheet(
            after, sheet.map(eqx.filter_jit(layer.__call__)), box, warp_style
        )
        after.add_patch(
            Rectangle(
                (0.0, 0.0),
                1.0,
                1.0,
                transform=after.transAxes,
                fill=False,
                edgecolor=color,
                linewidth=style.frame_width,
                clip_on=False,
                zorder=6,
            )
        )

    head = 0.5 * style.arrow_head / 72
    for pair_left, name, color in [
        (pair_lefts[0], labels.linear, style.linear_color),
        (pair_lefts[1], labels.coupling, style.coupling_color),
    ]:
        overlay.add_patch(
            FancyArrow(
                pair_left + style.inset_size + pad,
                inset_y,
                style.inset_gap - 2 * pad,
                0.0,
                width=style.arrow_width / 72,
                head_width=head,
                head_length=head,
                length_includes_head=True,
                color=style.arrow_color,
            )
        )
        overlay.text(
            pair_left + 0.5 * pair_width,
            y(inset_bottom) - pad,
            name,
            ha="center",
            va="top",
            color=color,
            fontsize=style.inset_label_size,
        )
    return fig
