"""Trajectory datasets: an ``xarray.Dataset`` on disk and in memory; training windows.

On disk a dataset is **one netCDF4 file** (HDF5 underneath, written through
``h5netcdf``): the arrays and their metadata cannot drift apart because they are one
file. Layout::

    dims      trajectory, time, dim
    coords    time (float, the sample instants)
    data      ys  (trajectory, time, dim)
    attrs     DatasetMetadata, flattened (nested fields JSON-encoded)

``TimeSeriesDataSource`` wraps such a dataset and serves fixed-length **windows** —
``(ts[k:k+W], ys[i, k:k+W])`` — through the random-access protocol grain expects
(``__len__``/``__getitem__``). Splits are by time (``split_time``) or by trajectory
(``split_trajectories``); both return new sources over views of the same arrays. The
weighted-window and mixture samplers are in ``data/sampling.py``.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from jaxtyping import Float


_JSON_FIELDS = ("system_params", "ic_sampler_params", "extra")
ENGINE = "h5netcdf"


@dataclass(frozen=True)
class DatasetMetadata:
    """Everything needed to regenerate the dataset, plus provenance.

    Flat scalars become netCDF attributes as they are; the dict-valued fields are
    JSON-encoded into string attributes (netCDF attributes are flat).
    """

    system: str
    system_params: dict[str, Any]
    ic_sampler: str
    ic_sampler_params: dict[str, Any]
    seed: int
    n_trajectories: int
    n_time: int
    dim: int
    t0: float
    t1: float
    solver: str
    rtol: float
    atol: float
    max_steps: int
    strategy: str = ""
    dtype: str = "float64"
    config_hash: str = ""
    created: str = ""
    git_sha: str = ""
    git_dirty: int = 0
    package_version: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_attrs(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        for k in _JSON_FIELDS:
            d[k] = json.dumps(d[k], sort_keys=True, default=str)
        return d

    @classmethod
    def from_attrs(cls, attrs: dict[str, Any]) -> "DatasetMetadata":
        names = {f.name for f in dataclasses.fields(cls)}
        d = {k: v for k, v in attrs.items() if k in names}
        for k in _JSON_FIELDS:
            if isinstance(d.get(k), str):
                d[k] = json.loads(d[k])
        # netCDF round-trips Python ints/floats as numpy scalars
        for k, v in list(d.items()):
            if isinstance(v, np.generic):
                d[k] = v.item()
        return cls(**d)


class TimeSeriesDataSource:
    """Windows over a set of trajectories sampled on a common time grid.

    **Arguments:**

    - ``ts``: ``(time,)`` sample instants.
    - ``ys``: ``(trajectory, time, dim)`` states.
    - ``window_size``: length of the windows served by ``__getitem__`` (default: whole
      trajectories).
    - ``metadata``: ``DatasetMetadata`` (optional in memory, written on ``save``).
    """

    def __init__(
        self,
        ts,
        ys,
        window_size: int | None = None,
        metadata: DatasetMetadata | None = None,
    ):
        ts, ys = np.asarray(ts), np.asarray(ys)
        if ts.ndim != 1 or ys.ndim != 3 or ys.shape[1] != ts.shape[0]:
            raise ValueError(
                "expected ts (time,) and ys (trajectory, time, dim) with matching "
                f"time; got {ts.shape} and {ys.shape}."
            )
        self._ds = xr.Dataset(
            {"ys": (("trajectory", "time", "dim"), ys)},
            coords={"time": ts},
            attrs=metadata.to_attrs() if metadata is not None else {},
        )
        self._metadata = metadata
        self.window_size = self._check_window(window_size)

    @classmethod
    def from_xarray(
        cls, ds: xr.Dataset, window_size: int | None = None
    ) -> "TimeSeriesDataSource":
        meta = DatasetMetadata.from_attrs(ds.attrs) if "system" in ds.attrs else None
        return cls(ds["time"].values, ds["ys"].values, window_size, meta)

    # ----------------------------------------------------------------- views -----
    @property
    def dataset(self) -> xr.Dataset:
        return self._ds

    @property
    def metadata(self) -> DatasetMetadata | None:
        return self._metadata

    @property
    def ts(self) -> Float[np.ndarray, " time"]:
        return self._ds["time"].values

    @property
    def ys(self) -> Float[np.ndarray, "trajectory time dim"]:
        return self._ds["ys"].values

    @property
    def u0(self) -> Float[np.ndarray, "trajectory dim"]:
        return self.ys[:, 0]

    @property
    def dim(self) -> int:
        return self.ys.shape[-1]

    @property
    def num_trajectories(self) -> int:
        return self.ys.shape[0]

    @property
    def trajectory_length(self) -> int:
        return self.ts.shape[0]

    # --------------------------------------------------------------- windows -----
    def _check_window(self, window_size: int | None) -> int:
        if window_size is None:
            return self.trajectory_length
        if not 1 <= window_size <= self.trajectory_length:
            raise ValueError(
                f"window_size must be in [1, {self.trajectory_length}], got "
                f"{window_size}."
            )
        return int(window_size)

    @property
    def windows_per_trajectory(self) -> int:
        return self.trajectory_length - self.window_size + 1

    def __len__(self) -> int:
        return self.num_trajectories * self.windows_per_trajectory

    def window_index(self, idx: int) -> tuple[int, int]:
        """``(trajectory, start)`` of window ``idx``; windows are enumerated trajectory-
        major."""
        return divmod(int(idx), self.windows_per_trajectory)

    def __getitem__(
        self, index: int
    ) -> tuple[Float[np.ndarray, " window"], Float[np.ndarray, "window dim"]]:
        i, k = self.window_index(index)
        sel = slice(k, k + self.window_size)
        return self.ts[sel], self.ys[i, sel]

    def window_start_times(self) -> Float[np.ndarray, " windows"]:
        """Start time of every window, in ``__getitem__`` order (for sampling
        weights)."""
        starts = self.ts[: self.windows_per_trajectory]
        return np.tile(starts, self.num_trajectories)

    # ---------------------------------------------------------------- splits -----
    def _with(self, ts, ys) -> "TimeSeriesDataSource":
        return type(self)(ts, ys, metadata=self._metadata)

    def split_time(
        self, idx: int
    ) -> tuple["TimeSeriesDataSource", "TimeSeriesDataSource"]:
        """``(before, after)``: trajectories cut at time index ``idx``. Both halves keep
        this window size when it fits, else whole trajectories."""
        parts = []
        for t, y in zip(np.split(self.ts, [idx]), np.split(self.ys, [idx], axis=1)):
            src = self._with(t, y)
            src.window_size = src._check_window(
                self.window_size if self.window_size <= len(t) else None
            )
            parts.append(src)
        return parts[0], parts[1]

    def split_trajectories(
        self, frac: float, seed: int = 0
    ) -> tuple["TimeSeriesDataSource", "TimeSeriesDataSource"]:
        """``(train, held_out)``: a random ``frac`` of the trajectories held out."""
        if not 0 < frac < 1:
            raise ValueError("frac must be in (0, 1).")
        n = self.num_trajectories
        perm = np.random.default_rng(seed).permutation(n)
        n_out = max(1, int(round(frac * n)))
        out, keep = np.sort(perm[:n_out]), np.sort(perm[n_out:])
        a, b = self._with(self.ts, self.ys[keep]), self._with(self.ts, self.ys[out])
        a.window_size = b.window_size = self.window_size
        return a, b

    # ------------------------------------------------------------------ disk -----
    def save(self, path: str | Path) -> Path:
        """Write the single netCDF4/HDF5 file (arrays + metadata)."""
        if self._metadata is None:
            raise ValueError("save() requires metadata; pass DatasetMetadata.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._ds.to_netcdf(path, engine=ENGINE)
        return path

    @classmethod
    def load(
        cls, path: str | Path, window_size: int | None = None, *, dtype=None
    ) -> "TimeSeriesDataSource":
        """Read a file written by ``save``. ``dtype`` (e.g. ``np.float64``) makes a
        mismatch loud instead of silently mixing precisions with the model."""
        with xr.open_dataset(path, engine=ENGINE) as ds:
            ds = ds.load()
        if dtype is not None and ds["ys"].dtype != np.dtype(dtype):
            raise TypeError(
                f"{path} holds {ds['ys'].dtype} trajectories, expected "
                f"{np.dtype(dtype)}."
            )
        return cls.from_xarray(ds, window_size)

    def __repr__(self) -> str:
        return (
            f"TimeSeriesDataSource(n={self.num_trajectories}, "
            f"T={self.trajectory_length}, dim={self.dim}, window={self.window_size}, "
            f"system={self._metadata.system if self._metadata else None})"
        )
