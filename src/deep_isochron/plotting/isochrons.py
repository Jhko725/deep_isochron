"""A phase-plane plot of the isochrons that AUTO-07p traced (roadmap O7)."""

import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from ..analysis.ode import IsochronFoliation


def plot_isochrons(
    foliation: IsochronFoliation, n_cycle_points: int = 512
) -> plt.Figure:
    """The isochrons of `foliation` in the phase plane, each colored by its phase on a
    cyclic colormap, with the limit cycle in black, the base points marked and a
    colorbar for the phase.

    **Arguments:**

    - `foliation`: The traced isochrons.
    - `n_cycle_points`: The number of points of the drawn cycle.

    **Returns:**

    The figure.
    """
    cmap = cc.cm.CET_C6  # a cyclic map: phases -π and π share a color
    norm = Normalize(-np.pi, np.pi)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    for iso in foliation.isochrons:
        ax.plot(*iso.points.T, color=cmap(norm(iso.phase)), linewidth=0.8)
    _, cycle = foliation.cycle.sample(n_cycle_points)
    cycle = np.asarray(cycle)
    ax.plot(*np.vstack([cycle, cycle[:1]]).T, color="black", linewidth=1.5)
    if foliation.isochrons:
        bases = np.array([iso.base_point for iso in foliation.isochrons])
        ax.scatter(*bases.T, s=8, color="black", zorder=3)
    fig.colorbar(plt.cm.ScalarMappable(norm, cmap), ax=ax, label="phase $\\theta$")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_aspect("equal")
    return fig
