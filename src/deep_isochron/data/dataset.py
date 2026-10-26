"""Trajectory datasets: whole trajectories on a common time grid, one netCDF4 file each.

``TimeSeriesDataSource`` is a frozen dataclass holding ``ts``, ``ys`` and a
``DatasetMetadata``. It is a grain random-access source over whole trajectories and
nothing more: ``len`` is the number of trajectories, ``source[i]`` is
``{"t": ts, "u": ys[i]}``, and windowing, weighting and mixing are grain transforms in
``data/windows/``, so the source has no window bookkeeping. The splits are
``copy.replace`` with sliced arrays, and ``dataset`` builds the ``xarray.Dataset`` view
on demand.

On disk a dataset is one netCDF4 file (HDF5 underneath, written through ``h5netcdf``),
so arrays and metadata cannot drift apart: dimensions ``trajectory, time, dim``, the
coordinate ``time``, the data ``ys``, and one JSON-string attribute per metadata group,
since netCDF attributes are flat. ``DatasetMetadata`` is grouped by what the fields
describe: ``system``, ``sampling``, ``grid``, ``solve`` and ``provenance``. Its
``config_hash`` covers the first four groups, everything that determines the arrays
and nothing that records when they were made. ADR-0008 has the format decisions.
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
"""The xarray engine; netCDF4 through HDF5."""


# ------------------------------------------------------------------ metadata -------
@dataclass(frozen=True)
class SystemSpec:
    """Which ODE the trajectories come from.

    **Attributes:**

    - `name`: The system's class name.
    - `params`: Its constrained parameters, from ``AbstractODE.params()``.
    """

    name: str
    params: dict[str, Any]


@dataclass(frozen=True)
class SamplingSpec:
    """How the initial conditions were drawn.

    **Attributes:**

    - `ic_sampler`: The sampler's class name.
    - `params`: Its configuration, from ``AbstractICSampler.params()``.
    - `seed`: The seed of the draw.
    - `n_trajectories`: How many were drawn.
    - `outside_basin`: What `generate` did with initial conditions outside the
        system's basin, ``"error"`` or ``"resample"``. It defines the generation, so
        it is hashed.
    """

    ic_sampler: str
    params: dict[str, Any]
    seed: int
    n_trajectories: int
    outside_basin: str = "error"


@dataclass(frozen=True)
class GridSpec:
    """The time axis: `n` samples from `t0` to `t1`."""

    t0: float
    t1: float
    n: int


@dataclass(frozen=True)
class SolveSpec:
    """The solver settings, from ``SolverConfig.params()``, plus `integration`, the
    integration method's name for a normal form and empty for other systems."""

    solver: str
    rtol: float
    atol: float
    max_steps: int
    integration: str = ""


@dataclass(frozen=True)
class Provenance:
    """When and by which code the arrays were made; not hashed.

    **Attributes:**

    - `created`: The ISO timestamp.
    - `git_sha`, `git_dirty`: The repository state.
    - `package_version`: The installed package version.
    - `dtype`: The arrays' dtype.
    - `rejected_ics`: How many initial conditions outside the basin were drawn and
        replaced under ``outside_basin="resample"``; an outcome, so not hashed.
    """

    created: str = ""
    git_sha: str = ""
    git_dirty: bool = False
    package_version: str = ""
    dtype: str = "float64"
    rejected_ics: int = 0


_GROUPS = ("system", "sampling", "grid", "solve", "provenance")  # one attribute each
_HASHED = ("system", "sampling", "grid", "solve")  # what config_hash covers


@dataclass(frozen=True)
class DatasetMetadata:
    """Everything needed to regenerate a dataset, grouped, plus provenance.

    **Attributes:**

    - `system`, `sampling`, `grid`, `solve`: The generation-defining groups.
    - `provenance`: When and by which code the arrays were made.
    - `extra`: Free JSON-able metadata, such as the resolved generation config.
    """

    system: SystemSpec
    sampling: SamplingSpec
    grid: GridSpec
    solve: SolveSpec
    provenance: Provenance = field(default_factory=Provenance)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def config_hash(self) -> str:
        """The first 8 hex digits of the SHA-256 over the four generation-defining
        groups: regenerating an unchanged config gives the same hash, and provenance
        does not enter."""
        blob = json.dumps(
            {g: dataclasses.asdict(getattr(self, g)) for g in _HASHED},
            sort_keys=True,
            default=str,
        ).encode()
        return hashlib.sha256(blob).hexdigest()[:8]

    def to_attrs(self) -> dict[str, str]:
        """The netCDF attributes: one JSON string per group, ``extra``, and the hash,
        which is informational and recomputed on read."""
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
        """The metadata read back from the attributes `to_attrs` wrote."""
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
        """Whether `attrs` carries every metadata group."""
        return all(g in attrs for g in _GROUPS)


# ------------------------------------------------------------------- source --------
@dataclass(frozen=True)
class TimeSeriesDataSource:
    """Whole trajectories on a common time grid, as a grain random-access source
    whose elements are ``{"t": ts, "u": ys[i]}``.

    **Attributes:**

    - `ts`: The sample times, shape ``(time,)``.
    - `ys`: The trajectories, shape ``(trajectory, time, dim)``.
    - `metadata`: The `DatasetMetadata`, or ``None`` for a source built in memory.

    **Raises:**

    `ValueError` at construction when the shapes do not match."""

    ts: Float[np.ndarray, " time"]
    ys: Float[np.ndarray, "trajectory time dim"]
    metadata: DatasetMetadata | None = None

    def __post_init__(self):
        """Coerce the arrays and validate the shapes."""
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
        """The number of trajectories."""
        return self.ys.shape[0]

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        """Trajectory `index` as ``{"t": ts, "u": ys[index]}``."""
        return {"t": self.ts, "u": self.ys[index]}

    # ----------------------------------------------------------------- views -----
    @property
    def dim(self) -> int:
        """The state dimension."""
        return self.ys.shape[-1]

    @property
    def num_trajectories(self) -> int:
        """The number of trajectories."""
        return self.ys.shape[0]

    @property
    def trajectory_length(self) -> int:
        """The number of samples per trajectory."""
        return self.ts.shape[0]

    @property
    def u0(self) -> Float[np.ndarray, "trajectory dim"]:
        """The initial conditions, shape ``(trajectory, dim)``."""
        return self.ys[:, 0]

    @property
    def dataset(self) -> xr.Dataset:
        """The ``xarray.Dataset`` view, which is what `save` writes."""
        return xr.Dataset(
            {"ys": (("trajectory", "time", "dim"), self.ys)},
            coords={"time": self.ts},
            attrs=self.metadata.to_attrs() if self.metadata is not None else {},
        )

    @classmethod
    def from_xarray(cls, ds: xr.Dataset) -> TimeSeriesDataSource:
        """The source read from an ``xarray.Dataset`` in the layout `dataset` makes;
        the metadata is ``None`` when the attributes do not carry it."""
        meta = (
            DatasetMetadata.from_attrs(ds.attrs)
            if DatasetMetadata.is_present(ds.attrs)
            else None
        )
        return cls(ds["time"].values, ds["ys"].values, meta)

    # ---------------------------------------------------------------- splits -----
    def split_time(self, idx: int) -> tuple[TimeSeriesDataSource, TimeSeriesDataSource]:
        """``(before, after)``: every trajectory cut at time index `idx`, which must
        lie strictly inside the time axis.

        **Raises:**

        `ValueError` otherwise."""
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
        """``(train, held_out)``: a random fraction `frac` of the trajectories held
        out, at least one, drawn with `seed`.

        **Raises:**

        `ValueError` for `frac` outside ``(0, 1)``."""
        if not 0 < frac < 1:
            raise ValueError("frac must be in (0, 1).")
        n = self.num_trajectories
        perm = np.random.default_rng(seed).permutation(n)
        n_out = max(1, int(round(frac * n)))
        out, keep = np.sort(perm[:n_out]), np.sort(perm[n_out:])
        return copy.replace(self, ys=self.ys[keep]), copy.replace(self, ys=self.ys[out])

    # ------------------------------------------------------------------ disk -----
    def save(self, path: str | Path) -> Path:
        """Write the netCDF4 file, arrays and metadata together, creating the parent
        directory; the path is returned.

        **Raises:**

        `ValueError` without metadata."""
        if self.metadata is None:
            raise ValueError("save() requires metadata; pass DatasetMetadata.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.dataset.to_netcdf(path, engine=ENGINE)
        return path

    @classmethod
    def load(cls, path: str | Path, *, dtype=None) -> TimeSeriesDataSource:
        """Read a file written by `save`.

        **Arguments:**

        - `path`: The file.
        - `dtype`: When given, the dtype the trajectories must have, so that a
            mismatch is loud instead of silently mixing precisions with the model.

        **Raises:**

        `TypeError` on a dtype mismatch."""
        with xr.open_dataset(path, engine=ENGINE) as ds:
            ds = ds.load()
        if dtype is not None and ds["ys"].dtype != np.dtype(dtype):
            raise TypeError(
                f"{path} holds {ds['ys'].dtype} trajectories, expected "
                f"{np.dtype(dtype)}."
            )
        return cls.from_xarray(ds)

    def __repr__(self) -> str:
        """The counts and the system name."""
        system = self.metadata.system.name if self.metadata else None
        return (
            f"TimeSeriesDataSource(n={self.num_trajectories}, "
            f"T={self.trajectory_length}, dim={self.dim}, system={system})"
        )
