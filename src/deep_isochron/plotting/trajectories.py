"""A phase-plane plot of the trajectories of a dataset."""

import matplotlib.pyplot as plt

from ..data import TimeSeriesDataSource


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
