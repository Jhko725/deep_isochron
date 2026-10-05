"""Trajectory datasets: whole trajectories on a common time grid; one netCDF4 file each.

``TimeSeriesDataSource`` is a frozen dataclass holding ``ts (time,)``, ``ys (trajectory,
time, dim)`` and a ``DatasetMetadata``. It is a grain random-access source over *whole
trajectories* — ``len`` is the number of trajectories and ``source[i]`` is
``{"t": ts, "u": ys[i]}`` — and nothing more: windowing, weighting and mixing are grain
transforms in ``data/windows/`` (the swirl-dynamics pattern), so the source has no
window bookkeeping. Splits (``split_time``, ``split_trajectories``) are ``copy.replace``
with sliced arrays; ``dataset`` builds the ``xarray.Dataset`` view on demand.

On disk a dataset is **one netCDF4 file** (HDF5 underneath, written through
``h5netcdf``), so arrays and metadata cannot drift apart::

    dims      trajectory, time, dim
    coords    time (the sample instants)
    data      ys  (trajectory, time, dim)
    attrs     one JSON-string attribute per metadata group (netCDF attributes are flat):
              system, sampling, grid, solve, provenance, extra; plus config_hash

``DatasetMetadata`` is grouped by what the fields describe — ``system`` (which ODE, its
constrained parameters), ``sampling`` (initial conditions), ``grid`` (the time axis),
``solve`` (``SolverConfig`` + integration), ``provenance`` (when, which code, dtype) —
and ``config_hash`` is a property over the first four groups: everything that
determines the arrays and nothing that merely records when they were made.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from jaxtyping import Float


ENGINE = "h5netcdf"


# ------------------------------------------------------------------ metadata -------
@dataclass(frozen=True)
class SystemSpec:
    name: str
    params: dict[str, Any]


@dataclass(frozen=True)
class SamplingSpec:
    ic_sampler: str
    params: dict[str, Any]
    seed: int
    n_trajectories: int


@dataclass(frozen=True)
class GridSpec:
    t0: float
    t1: float
    n: int


@dataclass(frozen=True)
class SolveSpec:
    solver: str
    rtol: float
    atol: float
    max_steps: int
    integration: str = ""
    """Integration method name for normal forms; empty for other systems."""


@dataclass(frozen=True)
class Provenance:
    created: str = ""
    git_sha: str = ""
    git_dirty: bool = False
    package_version: str = ""
    dtype: str = "float64"


_GROUPS = ("system", "sampling", "grid", "solve", "provenance")
_HASHED = ("system", "sampling", "grid", "solve")


@dataclass(frozen=True)
class DatasetMetadata:
    """Everything needed to regenerate a dataset, grouped, plus provenance."""

    system: SystemSpec
    sampling: SamplingSpec
    grid: GridSpec
    solve: SolveSpec
    provenance: Provenance = field(default_factory=Provenance)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def config_hash(self) -> str:
        """First 8 hex digits of SHA-256 over ``system``, ``sampling``, ``grid`` and
        ``solve`` — the generation-defining groups; regenerating an unchanged config
        gives the same hash, provenance does not enter."""
        blob = json.dumps(
            {g: dataclasses.asdict(getattr(self, g)) for g in _HASHED},
            sort_keys=True,
            default=str,
        ).encode()
        return hashlib.sha256(blob).hexdigest()[:8]

    def to_attrs(self) -> dict[str, str]:
        attrs = {
            g: json.dumps(
                dataclasses.asdict(getattr(self, g)), sort_keys=True, default=str
            )
            for g in _GROUPS
        }
        attrs["extra"] = json.dumps(self.extra, sort_keys=True, default=str)
        attrs["config_hash"] = self.config_hash  # informational; recomputed on read
        return attrs

    @classmethod
    def from_attrs(cls, attrs: dict[str, Any]) -> DatasetMetadata:
        load = lambda g: json.loads(attrs[g])  # noqa: E731
        return cls(
            system=SystemSpec(**load("system")),
            sampling=SamplingSpec(**load("sampling")),
            grid=GridSpec(**load("grid")),
            solve=SolveSpec(**load("solve")),
            provenance=Provenance(**load("provenance")),
            extra=json.loads(attrs.get("extra", "{}")),
        )

    @staticmethod
    def is_present(attrs: dict[str, Any]) -> bool:
        return all(g in attrs for g in _GROUPS)


# ------------------------------------------------------------------- source --------
@dataclass(frozen=True)
class TimeSeriesDataSource:
    """Whole trajectories on a common time grid; a grain random-access source.

    **Fields:** ``ts (time,)``, ``ys (trajectory, time, dim)``, optional ``metadata``.
    Elements are ``{"t": ts, "u": ys[i]}``; windowing is done by the transforms in
    ``data/windows/``.
    """

    ts: Float[np.ndarray, " time"]
    ys: Float[np.ndarray, "trajectory time dim"]
    metadata: DatasetMetadata | None = None

    def __post_init__(self):
        ts, ys = np.asarray(self.ts), np.asarray(self.ys)
        if ts.ndim != 1 or ys.ndim != 3 or ys.shape[1] != ts.shape[0]:
            raise ValueError(
                "expected ts (time,) and ys (trajectory, time, dim) with matching "
                f"time; got {ts.shape} and {ys.shape}."
            )
        object.__setattr__(self, "ts", ts)
        object.__setattr__(self, "ys", ys)

    # grain RandomAccessDataSource protocol
    def __len__(self) -> int:
        return self.ys.shape[0]

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        return {"t": self.ts, "u": self.ys[index]}

    # ----------------------------------------------------------------- views -----
    @property
    def dim(self) -> int:
        return self.ys.shape[-1]

    @property
    def num_trajectories(self) -> int:
        return self.ys.shape[0]

    @property
    def trajectory_length(self) -> int:
        return self.ts.shape[0]

    @property
    def u0(self) -> Float[np.ndarray, "trajectory dim"]:
        return self.ys[:, 0]

    @property
    def dataset(self) -> xr.Dataset:
        """The ``xarray.Dataset`` view (what ``save`` writes)."""
        return xr.Dataset(
            {"ys": (("trajectory", "time", "dim"), self.ys)},
            coords={"time": self.ts},
            attrs=self.metadata.to_attrs() if self.metadata is not None else {},
        )

    @classmethod
    def from_xarray(cls, ds: xr.Dataset) -> TimeSeriesDataSource:
        meta = (
            DatasetMetadata.from_attrs(ds.attrs)
            if DatasetMetadata.is_present(ds.attrs)
            else None
        )
        return cls(ds["time"].values, ds["ys"].values, meta)

    # ---------------------------------------------------------------- splits -----
    def split_time(self, idx: int) -> tuple[TimeSeriesDataSource, TimeSeriesDataSource]:
        """``(before, after)``: every trajectory cut at time index ``idx``."""
        if not 0 < idx < self.trajectory_length:
            raise ValueError(
                f"idx must be in (0, {self.trajectory_length}), got {idx}."
            )
        return (
            copy.replace(self, ts=self.ts[:idx], ys=self.ys[:, :idx]),
            copy.replace(self, ts=self.ts[idx:], ys=self.ys[:, idx:]),
        )

    def split_trajectories(
        self, frac: float, seed: int = 0
    ) -> tuple[TimeSeriesDataSource, TimeSeriesDataSource]:
        """``(train, held_out)``: a random ``frac`` of the trajectories held out."""
        if not 0 < frac < 1:
            raise ValueError("frac must be in (0, 1).")
        n = self.num_trajectories
        perm = np.random.default_rng(seed).permutation(n)
        n_out = max(1, int(round(frac * n)))
        out, keep = np.sort(perm[:n_out]), np.sort(perm[n_out:])
        return copy.replace(self, ys=self.ys[keep]), copy.replace(self, ys=self.ys[out])

    # ------------------------------------------------------------------ disk -----
    def save(self, path: str | Path) -> Path:
        """Write the single netCDF4/HDF5 file (arrays + metadata)."""
        if self.metadata is None:
            raise ValueError("save() requires metadata; pass DatasetMetadata.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.dataset.to_netcdf(path, engine=ENGINE)
        return path

    @classmethod
    def load(cls, path: str | Path, *, dtype=None) -> TimeSeriesDataSource:
        """Read a file written by ``save``. ``dtype`` (e.g. ``np.float64``) makes a
        mismatch loud instead of silently mixing precisions with the model."""
        with xr.open_dataset(path, engine=ENGINE) as ds:
            ds = ds.load()
        if dtype is not None and ds["ys"].dtype != np.dtype(dtype):
            raise TypeError(
                f"{path} holds {ds['ys'].dtype} trajectories, expected "
                f"{np.dtype(dtype)}."
            )
        return cls.from_xarray(ds)

    def __repr__(self) -> str:
        system = self.metadata.system.name if self.metadata else None
        return (
            f"TimeSeriesDataSource(n={self.num_trajectories}, "
            f"T={self.trajectory_length}, dim={self.dim}, system={system})"
        )
