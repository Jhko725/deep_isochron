"""A phase-plane plot of the trajectories of a dataset, and `cycle_curve`, the closed
polyline of a limit cycle that the phase-plane figures draw."""

import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Float

from ..analysis.ode import LimitCycle
from ..data import TimeSeriesDataSource


def cycle_curve(cycle: LimitCycle, n_points: int = 512) -> Float[np.ndarray, "n 2"]:
    """Sample `cycle` at `n_points` evenly spaced phases and close the curve.

    **Arguments:**

    - `cycle`: The limit cycle.
    - `n_points`: The number of samples before closing.

    **Returns:**

    The sampled cycle of `n_points` points, closed by repeating the first, so the
    array has `n_points + 1` rows.
    """
    _, points = cycle.sample(n_points)
    points = np.asarray(points)
    return np.vstack([points, points[:1]])


def plot_source_trajectories(
    source: TimeSeriesDataSource, traj_downsample: int = 1
) -> plt.Figure:
    """The planar trajectories of `source`, every `traj_downsample`-th one, with their
    initial conditions marked."""
    ys = source.ys[::traj_downsample]

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    for y_i in ys:
        ax.plot(*y_i.T, alpha=0.9, linewidth=1.0)
    ax.scatter(*ys[:, 0].T, s=3, color="black", alpha=0.9)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_aspect("equal")
    return fig
