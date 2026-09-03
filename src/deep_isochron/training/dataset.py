from dataclasses import replace
from typing import Self

import equinox as eqx
import numpy as np
from jaxtyping import Float


class TimeSeriesDataset(eqx.Module):
    """
    Class representing the collection of trajectories from a dynamical system.

    All data manipulation is done with numpy instead of jax.numpy to not overload the
    GPU RAM.
    """

    t: Float[np.ndarray, "samples time"]
    u: Float[np.ndarray, "samples time dim"]
    metadata: dict | None = None

    def __init__(
        self,
        t: Float[np.ndarray, "#samples time"],
        u: Float[np.ndarray, "samples time dim"],
        metadata: dict | None = None,
    ):
        t = np.atleast_2d(np.asarray(t))
        self.t = np.tile(t, (u.shape[0], 1)) if t.shape[0] == 1 else t
        self.u = np.asarray(u)
        self.metadata = metadata

    def __check_init__(self):
        if self.t.shape != self.u.shape[:2]:
            raise ValueError("t and u do not have compatible shapes!")

    @property
    def dt(self) -> float:
        # Assumes that all trajectories are equispaced with the same time increment
        return self.t[0, 1] - self.t[0, 0]

    @property
    def trajectory_length(self) -> int:
        return self.t.shape[1]

    def __len__(self) -> int:
        return self.t.shape[0]

    def __getitem__(self, idx):
        return self.t[idx], self.u[idx]

    def downsample(self, downsample_factor: int) -> Self:
        return replace(
            self, t=self.t[:, ::downsample_factor], u=self.u[:, ::downsample_factor]
        )

    def split_along_time(self, split_index: int) -> tuple[Self, Self]:
        t1, t2 = self.t[:, :split_index], self.t[:, split_index:]
        u1, u2 = self.u[:, :split_index], self.u[:, split_index:]
        return replace(self, t=t1, u=u1), replace(self, t=t2, u=u2)

    def add_noise(self, noise_std_relative: float, *, seed: int = 0) -> Self:
        rng = np.random.default_rng(seed)
        std = np.std(self.u, axis=(0, 1))
        noise = rng.normal(size=self.u.shape) * std * noise_std_relative
        return replace(self, u=self.u + noise)


# TODO: implement save/load logic. Use pandas or np.loadz
