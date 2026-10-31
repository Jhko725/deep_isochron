"""The project's matplotlib style, and the drawing helpers the figures share: a curve
colored by a value, a panel's box and format, and a colorbar with chosen ticks."""

from collections.abc import Sequence

import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from jaxtyping import Float
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Colormap, Normalize
from matplotlib.ticker import MaxNLocator, NullLocator


def configure_rcparams():
    """Set the project's default plot style: a grid, open spines, the glasbey
    color cycle, and 600 dpi figures."""
    plt.rc("axes", grid=True, prop_cycle=cycler(color=cc.cm.glasbey_light.colors))
    plt.rc("axes.spines", top=False, right=False)
    plt.rc("grid", color="lightgray", linestyle="--")
    plt.rc("font", size=12)
    plt.rc("savefig", dpi=600)


def colored_curve(
    ax: Axes,
    points: Float[np.ndarray, "n 2"],
    values: Float[np.ndarray, " n"],
    cmap: Colormap,
    norm: Normalize,
    linewidth: float = 1.0,
    linestyle: str = "solid",
) -> LineCollection:
    """Draw `points` as a curve on `ax`, each segment colored by the mean of `values`
    at its ends.

    **Returns:**

    The collection, for a colorbar.
    """
    segments = list(np.stack([points[:-1], points[1:]], axis=1))
    collection = LineCollection(
        segments, cmap=cmap, norm=norm, linewidths=linewidth, linestyles=linestyle
    )
    collection.set_array(0.5 * (values[:-1] + values[1:]))
    ax.add_collection(collection)
    return collection


def set_limits(ax: Axes, limits: Sequence[float]) -> None:
    """Set the box of `ax` to `limits` and format it as a map panel: equal aspect, at
    most seven ticks per axis at round steps, and no grid.

    **Arguments:**

    - `ax`: The axes.
    - `limits`: The box `(x_lo, y_lo, x_hi, y_hi)`, such as the result of
        `sheet_limits`.

    **Raises:**

    `ValueError` if `limits` does not have four entries.
    """
    if len(limits) != 4:
        raise ValueError(
            f"limits must be (x_lo, y_lo, x_hi, y_hi), got {len(limits)} entries."
        )
    x_lo, y_lo, x_hi, y_hi = (float(v) for v in limits)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_aspect("equal")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(MaxNLocator(nbins=7, steps=[1, 2, 5]))
    ax.grid(False)


def add_colorbar(
    fig: plt.Figure,
    axes: Sequence[Axes],
    mappable: ScalarMappable,
    *,
    label: str,
    ticks: Sequence[float],
    tick_labels: Sequence[str],
    shrink: float = 0.85,
) -> None:
    """Add one colorbar of `mappable` at the right of `axes`, with the given ticks and
    no minor ticks.

    **Arguments:**

    - `fig`: The figure that holds `axes`.
    - `axes`: The panels that share the colorbar.
    - `mappable`: The artist whose colormap and norm the colorbar shows, such as the
        fill that `draw_sheet` returns.
    - `label`: The colorbar's label.
    - `ticks`: The tick positions, in the units of the mappable's norm.
    - `tick_labels`: The text of each tick.
    - `shrink`: The colorbar's length as a fraction of the panels' height.
    """
    bar = fig.colorbar(
        mappable, ax=list(axes), location="right", shrink=shrink, label=label
    )
    bar.set_ticks(list(ticks), labels=list(tick_labels))
    bar.ax.yaxis.set_minor_locator(NullLocator())
