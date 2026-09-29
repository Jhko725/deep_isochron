from typing import Self

import numpy as np
from jaxtyping import Float


# TODO: implement save/load from npz files
class TimeSeriesDataSource:
    ts: Float[np.ndarray, " time"]
    ys: Float[np.ndarray, "N time dim"]
    window_size: int

    def __init__(self, ts, ys, window_size: int | None = None):
        self.ts = np.asarray(ts)
        self.ys = np.asarray(ys)

        if window_size is None:
            self.window_size = self.trajectory_length
        else:
            if window_size > self.trajectory_length:
                raise ValueError(
                    "window_size cannot be larger than the trajectory length"
                )
            self.window_size = window_size

    @property
    def dim(self) -> int:
        return self.ys.shape[-1]

    @property
    def num_trajectories(self) -> int:
        return len(self.ys)

    @property
    def trajectory_length(self) -> int:
        return len(self.ts)

    @property
    def windows_per_trajectory(self) -> int:
        return self.trajectory_length - self.window_size + 1

    def __len__(self) -> int:
        return self.num_trajectories * self.windows_per_trajectory

    def __getitem__(
        self, idx: int
    ) -> tuple[Float[np.ndarray, " window_size"], Float[np.ndarray, "window_size dim"]]:
        idx_traj, idx_window = divmod(idx, self.windows_per_trajectory)
        sel = slice(idx_window, idx_window + self.window_size)
        return self.ts[sel], self.ys[idx_traj, sel]

    def split(self, idx: int) -> tuple[Self, Self]:
        ds = [
            TimeSeriesDataSource(t, y, window_size=self.window_size)
            for t, y in zip(np.split(self.ts, [idx]), np.split(self.ys, [idx], axis=1))
        ]
        return tuple(ds)
