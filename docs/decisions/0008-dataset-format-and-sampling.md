---
type: decision
id: ADR-0008
status: accepted; amended
updated: 2026-10-05
verified_by: joon (through 2026-10-03); Decision 3 pending review
---

# ADR-0008 — Datasets are one netCDF4 file via xarray; transient oversampling by weighted windows (and `mix`, for comparison)

**Status**: accepted (2026-10-02); amended 2026-10-03 after review — metadata grouped; the source holds whole trajectories and windowing is a grain transform (Decision 2 rewritten); amended 2026-10-05 — training runs read from a *batched* window source (Decision 3), the per-element transforms remaining the reference semantics.

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

## Decision 3 — training runs read batches from a `WindowBatchSource` (2026-10-05, roadmap C9)

Decision 2's pipeline costs grain one Python `__getitem__` round per *element* — index
mapping, shuffle, a per-element `np.random.Generator`, the RandomMap call, then
`batch`'s stack — ≈ 0.1–0.17 ms per window, i.e. 50–90 ms per batch of 512. Once the
training step itself came down to 33 ms on a V100 (ADR-0010) the fetch was the bound:
`to_device` ran at the fetch rate, 93 ms per step (change document 2026-10-04, round 2).

Training runs therefore read from **`WindowBatchSource(source, length, batch, seed=,
epochs=)`**, a `RandomAccessDataSource` whose element `i` is the `i`-th *batch* of the run,
built by one vectorized gather `ys[traj[:, None], start[:, None] + arange(L)]` (≈ 1 ms per
batch of 512, measured). It is still a *source* — the data are batches of windows, and
`__getitem__(i)` is a pure function of `(seed, i)` — not a transform that injects data:

- **An epoch is every window once.** `num_windows = N · (hi − lo)` over the valid starts;
  `batches_per_epoch = num_windows // batch` (the remainder is dropped — different windows
  each epoch since the permutation differs, and uniform batch shapes for `jit`);
  `len = epochs · batches_per_epoch`. Batch `i` is the `j`-th slice of a fresh permutation
  of epoch `e = i // batches_per_epoch` (`rng([seed, e])`). This is the original
  "enumerate all windows" semantics rather than Decision 2's "one window per trajectory
  visit", and the loader is **finite**: the data define the run (`Trainer.train(num_steps=
  None)`), `window_batches(..., num_steps=)` sizes it in steps, and resuming at step `s` is
  the slice `[s:]`.
- **Weighted and mixed starts are draws with replacement** (a trajectory uniformly, the
  start by inverse-CDF on `weight` or a Bernoulli between the two ranges of
  `mixed_window_batches`), `num_windows` of them per epoch, so `len` means the same amount
  of data in every mode. The marginals equal `WeightedWindow`'s and `mixed_windows`'s
  (tests compare them); `grain.MapDataset.mix` of two pipelines is no longer needed.
- Measured on the V100 (2026-10-05): fetch 88.5 → 0.7 ms per batch of 512; `to_device`
  32.6 ms per step against 31.0 with device-resident batches — the loader is no longer
  measurable next to the step.
- `windows`/`mixed_windows` (Decision 2) stay as the per-element reference and for
  `validation_windows`, which keeps the per-element `_FixedWindows` source (an occasional
  pass; the same gather with a stride would speed it up if it ever matters).

- *Rejected*: a `RandomMap` on a `range` source that holds the arrays and gathers (first
  proposal). Same speed, but a transform that *injects* data blurs the transform/source
  distinction grain is built on, and it cannot see the epoch (it gets a per-element rng,
  not the global index), so epochs become i.i.d. draws.
- *Rejected*: a custom `MapDataset.batch` node that reaches past its parent into the
  arrays — only works with one specific parent; the blur moves one level.
- *Rejected*: a nominal huge `len` with `repeat()`. grain's `RepeatMapDataset` hands the
  source `i % len`, so the source cannot see the epoch and every epoch would replay the same
  batches; the epoch count belongs in the source, where it also makes `len` concrete.

## Consequences

- `xarray` and `h5netcdf` become runtime dependencies.
- `TimeSeriesDataSource.split(idx)` is now `split_time(idx)`; `split_trajectories(frac,
  seed)` added for held-out validation; both are `copy.replace` on the frozen dataclass.
- Batches are dicts `{"t": (B, L), "u": (B, L, dim)}` from either path;
  `ConjugacyTrajectoryLoss` reads them by key.
- Training loaders are finite (Decision 3); `Trainer.train`'s `num_steps` is optional and
  an upper bound.
- A per-trajectory settling time (`t_settle`, from the normal form's `amplitude` or a
  numerical distance to the cycle) is a natural extra variable for the file and a better
  basis for weights than wall-clock start time; parked until the analysis module exists.
