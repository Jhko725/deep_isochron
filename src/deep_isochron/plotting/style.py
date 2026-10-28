"""The project's matplotlib style, and the drawing helper the figures share."""

import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from jaxtyping import Float
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import Colormap, Normalize


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
