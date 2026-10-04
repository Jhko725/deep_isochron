---
type: decision
id: ADR-0008
status: accepted; amended
updated: 2026-10-03
verified_by: joon
---

# ADR-0008 — Datasets are one netCDF4 file via xarray; transient oversampling by weighted windows (and `mix`, for comparison)

**Status**: accepted (2026-10-02); amended 2026-10-03 after review — metadata grouped; the source holds whole trajectories and windowing is a grain transform (Decision 2 rewritten).

## Context

Trajectory datasets are generated, not collected: a dataset is fully determined by
(system + parameters, initial-condition sampler + seed, time grid, solver + tolerances,
integration strategy, code version). Two requirements: the arrays and the metadata that
describes them must not be able to go out of sync, and the training loader must oversample
the transient part of each trajectory (where the amplitude dynamics are visible) relative to
the near-periodic tail.

## Decision 1 — one netCDF4 file, written and read through xarray

`TimeSeriesDataSource` is a frozen dataclass (`ts`, `ys`, `metadata`) whose `dataset`
property is the `xarray.Dataset` view with dims `(trajectory, time, dim)` and the time grid
as the `time` coordinate. `DatasetMetadata` is **grouped** — `system` (name, constrained
`params()`), `sampling` (IC sampler, its params, seed, `n_trajectories`), `grid` (`t0`,
`t1`, `n`), `solve` (`SolverConfig.params()` + integration name), `provenance` (created,
git SHA/dirty, package version, dtype), `extra` — and each group is written as one
JSON-string attribute, since netCDF attributes are flat (netCDF4 *groups* were rejected as
too heavy for six small dicts, and xarray reads them only with `group=`). `config_hash` is
a property over the first four groups. `save` writes one
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

## Decision 2 — the source holds whole trajectories; windowing is a grain transform

The source is a random-access dataset of *whole trajectories* (`source[i] = {"t": ts,
"u": ys[i]}`), and windows are cut by `grain.transforms.RandomMap`s applied after
`.shuffle().repeat()` — the swirl-dynamics pattern (`ArrayDataSource` + `RandomSection`
in the DySLIM code). grain seeds the per-element generator by the global index, so a
repeated trajectory gets a fresh window every epoch (verified). This removes all window
bookkeeping from the source (window size, enumeration, start times), which had made the
first version of `TimeSeriesDataSource` carry sampling policy in its state.

Two transforms implement transient oversampling, kept side by side for comparison:

- `WeightedWindow(length, weight)` draws the start index with probability ∝
  `weight(t_start)` by inverse-CDF lookup; `transient_weight(boost, tau)` is the smooth
  default `1 + boost·exp(−(t − t₀)/τ)`. In distribution this equals the earlier
  index-level weighted sampler (the weight depends on start time only), with trajectories
  visited uniformly.
- `mixed_windows(source, length, split_idx, weights, seed)` is the two-loader design:
  `RandomWindow` restricted to starts before `split_idx` and `RandomWindow` restricted to
  starts at/after it, interleaved by `grain.MapDataset.mix`. `mix` is a deterministic
  interleave by weights whose length is that of the shortest input and which needs its
  inputs shuffled beforehand (grain 0.2.18 docstring); hence `.shuffle().repeat()` on each.

- *Rejected*: window-level source + functions emitting `MapDataset`s (the first version).
  Not idiomatic grain; the sampling policy leaked into the source.
- *Rejected*: a custom `IndexSampler`. It would have to know window geometry, which is
  exactly what was removed from the source; the transform is the cleaner home.
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
  seed)` added for held-out validation; both are `copy.replace` on the frozen dataclass.
- Batches are dicts `{"t": (B, L), "u": (B, L, dim)}`; `ConjugacyTrajectoryLoss` reads
  them by key.
- A per-trajectory settling time (`t_settle`, from the normal form's `amplitude` or a
  numerical distance to the cycle) is a natural extra variable for the file and a better
  basis for weights than wall-clock start time; parked until the analysis module exists.
