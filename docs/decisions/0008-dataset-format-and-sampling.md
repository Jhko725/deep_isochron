# ADR-0008 — Datasets are one netCDF4 file via xarray; transient oversampling by weighted windows (and `mix`, for comparison)

**Status**: accepted (2026-10-02)

## Context

Trajectory datasets are generated, not collected: a dataset is fully determined by
(system + parameters, initial-condition sampler + seed, time grid, solver + tolerances,
integration strategy, code version). Two requirements: the arrays and the metadata that
describes them must not be able to go out of sync, and the training loader must oversample
the transient part of each trajectory (where the amplitude dynamics are visible) relative to
the near-periodic tail.

## Decision 1 — one netCDF4 file, written and read through xarray

`TimeSeriesDataSource` wraps an `xarray.Dataset` with dims `(trajectory, time, dim)`, the
time grid as the `time` coordinate, and a `DatasetMetadata` flattened into the attributes
(dict-valued fields JSON-encoded, since netCDF attributes are flat). `save` writes one
`.nc` file with the `h5netcdf` engine; `load` reads it and can insist on a dtype. The file
is named `<name>-<config_hash>.nc`, the hash being the first 8 hex digits of SHA-256 over
the generation-defining fields (provenance such as timestamps and git state excluded, so
regenerating an unchanged config overwrites the same file).

Why netCDF4 and not the alternatives:

| | single file | structured metadata | verdict |
|---|---|---|---|
| netCDF4 via xarray | yes (HDF5 underneath; readable by `h5py`) | attrs | **chosen** — named dims, `time` coordinate, lazy loading and `.isel` for free |
| HDF5 via `h5py` | yes | attrs | same file format with hand-written save/load; nothing gained |
| Zarr | no (directory store; single file only via `ZipStore`) | attrs | chunked/cloud use case, not ours |
| `.npz` + `.json` | two files | — | the roadmap's earlier plan; fails the one-file requirement |
| safetensors / Parquet | yes | string header / tabular | wrong shape for `(N, T, d)` + nested metadata |

(Formats: xarray I/O guide — netCDF4 is HDF5-based, Zarr is a directory store; h5py —
attributes are "the official way to store metadata in HDF5", kept under ~64 KiB.)

**DVC** was considered and *not* adopted: it versions large or irreproducible data by
committing hash metafiles and keeping bytes in a cache/remote. Our datasets are cheap and
deterministic to regenerate; `(git_sha, config_hash, seed)` in the file's own metadata is the
version. If data ever becomes expensive, shared, or too big for git, `dvc add data/` is a
retrofit that changes nothing in this format.

## Decision 2 — weighted window sampling, with the two-loader `mix` kept alongside

`weighted_windows(source, weights, seed)` draws window indices i.i.d. with probability
proportional to a per-window weight (`transient_weights`: `1 + boost·exp(−(t_start −
t₀)/τ)`), implemented as `grain.MapDataset.range(n).repeat().seed(s).random_map(...)` with
an inverse-CDF lookup. One dataset, one loader, a smooth policy described by two numbers.
`mixed_split(source, split_idx, weights, seed)` is the previous design — cut in time,
shuffle each half, `grain.MapDataset.mix` — kept so the two can be compared on the same
data. `mix` is a deterministic interleave by weights whose length is that of the shortest
input and which needs its inputs shuffled beforehand (grain 0.2.18 docstring); hence the
`.shuffle().repeat()` on each half.

- *Rejected as the only option*: `mix`. The transient/attractor boundary is an index cut,
  and the policy is baked into dataset objects rather than being a function.
- *Rejected for now*: loss-side weighting (curriculum) — changes loss semantics and spends
  compute on down-weighted samples; stratified K-way `mix` — the K=2 case is `mixed_split`.
- Survey: grain has `mix`, `WindowShuffle*`, `InterleaveIterDataset` and no weighted
  sampler; tf.data `sample_from_datasets` and HF `interleave_datasets(probabilities)` are
  `mix`-shaped; torch `WeightedRandomSampler` is `weighted_windows`-shaped.

## Consequences

- `xarray` and `h5netcdf` become runtime dependencies.
- `TimeSeriesDataSource.split(idx)` is now `split_time(idx)`; `split_trajectories(frac,
  seed)` added for held-out validation.
- A per-trajectory settling time (`t_settle`, from the normal form's `amplitude` or a
  numerical distance to the cycle) is a natural extra variable for the file and a better
  basis for weights than wall-clock start time; parked until the analysis module exists.
