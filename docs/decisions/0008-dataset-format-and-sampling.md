---
type: decision
id: ADR-0008
status: accepted; amended
updated: 2026-10-27
verified_by: joon (through 2026-10-03); Decision 3 pending review
---

# ADR-0008 — Datasets are one netCDF4 file via xarray; transient oversampling by weighted windows (and `mix`, for comparison)

## Context

### Motivation

- Generated datasets needed a save format. `.npz` and `.npy` cannot store metadata
  effectively, so a format that keeps the arrays and their description together was
  needed (Joon, review of 2026-10-27).
- The first version's flat metadata read as unorganized; the review asked for groups
  such as one for the `diffeqsolve` configuration
  ([`2026-10-02-data-generation.md`](../changes/2026-10-02-data-generation.md):492).
- The first version's window-level `TimeSeriesDataSource` carried sampling policy in its
  state (window size, enumeration, start times), and `data/sampling.py` emitted a
  different `MapDataset` per policy, which is not grain's idiom; the DySLIM code cuts
  windows by a transform instead (`2026-10-02-data-generation.md`:494).
- The training loader must oversample the transient part of each trajectory, where the
  amplitude dynamics are visible, relative to the near-periodic tail.
- Decision 2's per-element pipeline costs ≈ 0.1–0.17 ms per window, that is, 50–90 ms
  per batch of 512. Once the training step came down to 33 ms on a V100
  ([ADR-0010](0010-rotations-by-cayley-transform.md)) the fetch was the bound:
  `to_device` ran at the fetch rate, 93 ms per step (change document 2026-10-04, round
  2).

### Forces

Trajectory datasets are generated, not collected: a dataset is fully determined by
(system + parameters, initial-condition sampler + seed, time grid, solver + tolerances,
integration strategy, code version). The arrays and the metadata that describes them
must not be able to go out of sync. The datasets are cheap and deterministic to
regenerate.

## Decision

### 1. One netCDF4 file, written and read through xarray

`TimeSeriesDataSource` is a frozen dataclass (`ts`, `ys`, `metadata`) whose `dataset`
property is the `xarray.Dataset` view with dims `(trajectory, time, dim)` and the time
grid as the `time` coordinate. `DatasetMetadata` is **grouped**: `system` (name,
constrained `params()`), `sampling` (IC sampler, its params, seed, `n_trajectories`),
`grid` (`t0`, `t1`, `n`), `solve` (`SolverConfig.params()` + integration name),
`provenance` (created, git SHA/dirty, package version, dtype), `extra`. Each group is
written as one JSON-string attribute, since netCDF attributes are flat; netCDF4 *groups*
were rejected as too heavy for six small dicts, and xarray reads them only with
`group=`. `config_hash` is a property over the first four groups. `save` writes one
`.nc` file with the `h5netcdf` engine; `load` reads it and can insist on a dtype. The
file is named `<name>-<config_hash>.nc`, the hash being the first 8 hex digits of
SHA-256 over the generation-defining fields. Provenance such as timestamps and git state
is excluded from the hash, so regenerating an unchanged config overwrites the same file.

Why netCDF4 and not the alternatives:

| | single file | structured metadata | verdict |
|---|---|---|---|
| netCDF4 via xarray | yes (HDF5 underneath; readable by `h5py`) | attrs | **chosen**: named dims, `time` coordinate, lazy loading and `.isel` for free |
| HDF5 via `h5py` | yes | attrs | same file format with hand-written save/load; nothing gained |
| Zarr | no (directory store; single file only via `ZipStore`) | attrs | chunked/cloud use case, not ours |
| `.npz` + `.json` | two files | — | the roadmap's earlier plan; fails the one-file requirement |
| safetensors / Parquet | yes | string header / tabular | wrong shape for `(N, T, d)` + nested metadata |

(Formats: the xarray I/O guide says that netCDF4 is HDF5-based and Zarr a directory
store; the h5py documentation says that attributes are "the official way to store
metadata in HDF5", kept under ~64 KiB.)

### 2. The source holds whole trajectories; windowing is a grain transform

The source is a random-access dataset of *whole trajectories*
(`source[i] = {"t": ts, "u": ys[i]}`), and windows are cut by
`grain.transforms.RandomMap`s applied after `.shuffle().repeat()`, the swirl-dynamics
pattern (`ArrayDataSource` + `RandomSection` in the DySLIM code). grain seeds the
per-element generator by the global index, so a repeated trajectory gets a fresh window
every epoch (verified). The transform removes all window bookkeeping from the source
(window size, enumeration, start times).

Two transforms implement transient oversampling, kept side by side for comparison:

- `WeightedWindow(length, weight)` draws the start index with probability ∝
  `weight(t_start)` by inverse-CDF lookup; `transient_weight(boost, tau)` is the smooth
  default `1 + boost·exp(−(t − t₀)/τ)`. In distribution this equals the earlier
  index-level weighted sampler (the weight depends on start time only), with
  trajectories visited uniformly.
- `mixed_windows(source, length, split_idx, weights, seed)` is the two-loader design:
  `RandomWindow` restricted to starts before `split_idx` and `RandomWindow` restricted
  to starts at/after it, interleaved by `grain.MapDataset.mix`. `mix` is a deterministic
  interleave by weights whose length is that of the shortest input and which needs its
  inputs shuffled beforehand (grain 0.2.18 docstring); hence `.shuffle().repeat()` on
  each.

This per-element pipeline is the reference semantics: validation and the tests use it,
and training runs use Decision 3.

- *Rejected*: a custom `IndexSampler`. It would have to know window geometry, which is
  exactly what was removed from the source; the transform is the cleaner home.
- *Rejected as the only option*: `mix`. The transient/attractor boundary is an index
  cut, and the policy is baked into dataset objects rather than being a function.
- *Rejected for now*: loss-side weighting (curriculum), which changes loss semantics and
  spends compute on down-weighted samples; stratified K-way `mix`, whose K=2 case is
  `mixed_windows` (the first version's `mixed_split`).
- Survey: grain has `mix`, `WindowShuffle*`, `InterleaveIterDataset` and no weighted
  sampler; tf.data `sample_from_datasets` and HF `interleave_datasets(probabilities)`
  are `mix`-shaped; torch `WeightedRandomSampler` is `WeightedWindow`-shaped (the first
  version's `weighted_windows`).

### 3. Training runs read batches from a `WindowBatchSource` (roadmap C9)

**In one paragraph: there are two ways to get batched windows out of a
`TimeSeriesDataSource`, and both are kept on purpose.**

- The *per-element* way (Decision
2; [`elementwise.py`](../../src/deep_isochron/data/windows/elementwise.py): `windows()`,
`mixed_windows()`, `validation_windows()`) is grain's idiom: one trajectory per element,
a `RandomMap` cuts one window, `.batch(B)` stacks. It is short, easy to see to be
correct, and the reference that the tests compare against; it is also slow, ≈ 0.1–0.2 ms
of grain overhead per window, 50–90 ms per batch of 512.

- The *batched* way (this
decision; [`batched.py`](../../src/deep_isochron/data/windows/batched.py):
`window_batches()`, `mixed_window_batches()`) serves whole batches from a
`WindowBatchSource` by one vectorized gather, ≈ 1 ms per batch, with the same start
distributions and a concrete epoch structure.

**Training runs use the batched way;
validation and tests use the per-element way.**
[`common.py`](../../src/deep_isochron/data/windows/common.py) holds what both share.

Decision 2's pipeline costs grain one Python `__getitem__` round per *element* (index
mapping, shuffle, a per-element `np.random.Generator`, the RandomMap call, then
`batch`'s stack). Training runs therefore read from
**`WindowBatchSource(source, length, batch, seed=, epochs=)`**, a
`RandomAccessDataSource` whose element `i` is the `i`-th *batch* of the run, built by
one vectorized gather `ys[traj[:, None], start[:, None] + arange(L)]` (≈ 1 ms per batch
of 512, measured). It is still a *source*, since the data are batches of windows and
`__getitem__(i)` is a pure function of `(seed, i)`, not a transform that injects data:

- **An epoch is every window once.** `num_windows = N · (hi − lo)` over the valid
  starts; `batches_per_epoch = num_windows // batch`, the remainder dropped, which costs
  different windows each epoch since the permutation differs and gives uniform batch
  shapes for `jit`; `len = epochs · batches_per_epoch`. Batch `i` is the `j`-th slice of
  a fresh permutation of epoch `e = i // batches_per_epoch` (`rng([seed, e])`). This is
  the original "enumerate all windows" semantics rather than Decision 2's "one window
  per trajectory visit", and the loader is **finite**: the data define the run
  (`Trainer.train(num_steps= None)`), `window_batches(..., num_steps=)` sizes it in
  steps, and resuming at step `s` is the slice `[s:]`.
- **Weighted and mixed starts are draws with replacement** (a trajectory uniformly, the
  start by inverse-CDF on `weight` or a Bernoulli between the two ranges of
  `mixed_window_batches`), `num_windows` of them per epoch, so `len` means the same
  amount of data in every mode. The marginals equal `WeightedWindow`'s and
  `mixed_windows`'s (tests compare them); `grain.MapDataset.mix` of two pipelines is no
  longer needed.
- Measured on the V100 (2026-10-05): fetch 88.5 → 0.7 ms per batch of 512; `to_device`
  32.6 ms per step against 31.0 with device-resident batches, so the loader is no longer
  measurable next to the step.
- `windows`/`mixed_windows` (Decision 2) stay as the per-element reference and for
  `validation_windows`, which keeps the per-element `_FixedWindows` source (an
  occasional pass; the same gather with a stride would speed it up if it ever matters).
- `num_steps` is a constructor argument of `WindowBatchSource`, mutually exclusive with
  `epochs` (`epochs = ceil(num_steps / batches_per_epoch)`).
- The package `data/windows/` (`common`, `elementwise`, `batched`) has a file per way,
  so the file structure says which code serves which; `single_threaded`, `to_device` and
  `resolve_device` live in [`device.py`](../../src/deep_isochron/data/device.py), since
  they apply to any pipeline.

- *Rejected*: a `RandomMap` on a `range` source that holds the arrays and gathers (first
  proposal). Same speed, but a transform that *injects* data blurs the transform/source
  distinction grain is built on, and it cannot see the epoch (it gets a per-element rng,
  not the global index), so epochs become i.i.d. draws.
- *Rejected*: a custom `MapDataset.batch` node that reaches past its parent into the
  arrays. It only works with one specific parent; the blur moves one level.
- *Rejected*: a nominal huge `len` with `repeat()`. grain's `RepeatMapDataset` hands the
  source `i % len`, so the source cannot see the epoch and every epoch would replay the
  same batches; the epoch count belongs in the source, where it also makes `len`
  concrete.

## Consequences

- `xarray` and `h5netcdf` become runtime dependencies.
- `TimeSeriesDataSource.split(idx)` is now `split_time(idx)`;
  `split_trajectories(frac, seed)` added for held-out validation; both are
  `copy.replace` on the frozen dataclass.
- Batches are dicts `{"t": (B, L), "u": (B, L, dim)}` from either path;
  `ConjugacyTrajectoryLoss` reads them by key.
- Training loaders are finite (Decision 3); `Trainer.train`'s `num_steps` is optional
  and an upper bound.
- A per-trajectory settling time (`t_settle`, from the normal form's `amplitude` or a
  numerical distance to the cycle) is a natural extra variable for the file and a better
  basis for weights than wall-clock start time; parked until the analysis module exists.

## Alternatives rejected

- **DVC** was considered and *not* adopted. It versions large or irreproducible data by
  committing hash metafiles and keeping bytes in a cache or remote. Our datasets are
  cheap and deterministic to regenerate; `(git_sha, config_hash, seed)` in the file's
  own metadata is the version. If data ever becomes expensive, shared, or too big for
  git, `dvc add data/` is a retrofit that changes nothing in this format.
- The other formats, in the table of Decision 1.
- *Superseded 2026-10-03*: a window-level source + functions emitting `MapDataset`s (the
  first version). Not idiomatic grain; the sampling policy leaked into the source.
- *Superseded 2026-10-05*: training runs reading through Decision 2's per-element
  pipeline. Replaced because its 50–90 ms per batch bounded a 33 ms step; it stays as
  the reference.
- *Superseded 2026-10-05*: the single module `data/windows.py`, with the device helpers
  in it. Replaced by the package and `data/device.py` so that the file structure names
  the two ways.

## History

- 2026-10-02 (`data-generation`): accepted.
- 2026-10-03 (`data-generation`, review): metadata grouped; the source holds whole
  trajectories and windowing is a grain transform (Decision 2 rewritten).
- 2026-10-05 (`trainer`, roadmap C9): training runs read from a batched
  `WindowBatchSource` (Decision 3), the per-element transforms remaining the reference
  semantics; `data/windows.py` split into the package `data/windows/` and
  `data/device.py`; `num_steps` a constructor argument of `WindowBatchSource` (review
  round on C9).
