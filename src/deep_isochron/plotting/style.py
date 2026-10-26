"""The project's matplotlib style."""

import colorcet as cc
import matplotlib.pyplot as plt
from cycler import cycler


def configure_rcparams():
    """Set the project's default plot style: a grid, open spines, the glasbey
    color cycle, and 600 dpi figures."""
    plt.rc("axes", grid=True, prop_cycle=cycler(color=cc.cm.glasbey_light.colors))
    plt.rc("axes.spines", top=False, right=False)
    plt.rc("grid", color="lightgray", linestyle="--")
    plt.rc("font", size=12)
    plt.rc("savefig", dpi=600)
