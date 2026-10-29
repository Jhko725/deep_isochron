# `scripts/` — the command-line entry points

All run from the repository root in the dev environment (`uv sync`). The three Hydra
entry points take `key=value` overrides and `--config-name`; the configs are in
`configs/`, and the same work from a notebook is
[`docs/guides/experiments.md`](../docs/guides/experiments.md).

| Script | What it does | Typical call |
|---|---|---|
| `generate_data.py` | Integrates trajectories from a `configs/data/*.yaml` generation config and writes one netCDF4 file, `<out_dir>/<name>-<config_hash>.nc` (ADR-0008). The hash is over the generation-defining fields, so an unchanged config overwrites the same file. **Training never generates**: run this first. | `uv run python scripts/generate_data.py --config-name bautin` <br> `uv run python scripts/generate_data.py --config-name fhn n_trajectories=512 seed=3` |
| `train.py` | One training run from `configs/train.yaml` (ADR-0011): builds the loader, model, loss, schedule and optimizer, trains, logs (`PrintLogger`; wandb with `wandb=online\|offline`), checkpoints into the Hydra run directory `runs/<date>/<time>` (resolved `config.yaml`, `metadata.json`, `checkpoints/`), and the final step is always saved. `resume=<run_dir>` restores that run's latest checkpoint and continues the same data stream. `-m` sweeps (`runs/multirun/...`, one wandb group). | `uv run python scripts/train.py` <br> `uv run python scripts/train.py model=autoencoder loss=autoencoder schedule=yawata` <br> `uv run python scripts/train.py num_steps=20000 windows.batch=4096 wandb=online` <br> `uv run python scripts/train.py resume=runs/2026-10-05/12-00-00` <br> `uv run python scripts/train.py -m seed=0,1,2` |
| `compute_isochrons.py` | Traces the isochrons that a `configs/isochrons/*.yaml` config describes by AUTO-07p and writes one netCDF file, `<out_dir>/<name>-<hash>.nc` (ADR-0017); the config takes its system from a data config. An existing file is reported and left alone; failed phases are saved, printed, and give exit status 1. Needs a built AUTO tree (`docs/guides/auto07p.md`). | `uv run python scripts/compute_isochrons.py --config-name isochrons/fhn` <br> `uv run python scripts/compute_isochrons.py --config-name isochrons/fhn -m hydra/launcher=slurm hydra.launcher.partition=<name>` |
| `install_auto.sh` | Builds AUTO-07p's Fortran tree at the commit `uv.lock` pins, for `compute_isochrons.py`; `just install-auto` runs it, and [`docs/guides/auto07p.md`](../docs/guides/auto07p.md) is the guide. `--commit` prints the pinned commit. | `uv run just install-auto` |

## `diagnostics/` — instruments for a run

| Script | What it does | Typical call |
|---|---|---|
| `bench_dataloader.py` | Times the data pipeline against the training step on the device you name (`--device`, else `resolve_device`): fetch rows, the dispatch floor, dispatch vs total for several loaders, `--hlo-stats` (loop/launch/conditional census of the compiled step), `--profile DIR`. `--config <overrides>` benchmarks a training config's own pipeline, model and loss. How to read the table: `docs/design/training-step-performance.md` §4–5. | `uv run python scripts/diagnostics/bench_dataloader.py --device 0 --hlo-stats --no-mp` <br> `uv run python scripts/diagnostics/bench_dataloader.py --config data=fhn windows.batch=4096` |
| `find_nan.py` | Hunts the first non-finite gradient of a run from its last (or a given) checkpoint: rebuilds the same data stream from that step, prints the batch's latent extent (`latent_extent`) every 25 steps, and at the first non-finite gradient reports the loss terms, the offending parameter leaves, the windows with non-finite predictions and the first bad window's start, its image under the bijection and its latent trajectory; `--debug-nans` then re-runs that batch so JAX names the primitive. Writes nothing. | `uv run python scripts/diagnostics/find_nan.py runs/winfree-1` <br> `uv run python scripts/diagnostics/find_nan.py runs/winfree-1 --step 2500 --max-steps 300 --debug-nans` |

## `checks/` — the deterministic checks

`prek` runs these on every commit and CI runs them on every push (`prek.toml`; `just
check` runs them all). What each one catches, and why it exists, is in
[`docs/coding-standards.md`](../docs/coding-standards.md) § Deterministic checks and
[`docs/document-conventions.md`](../docs/document-conventions.md): `check_spelling.py`
(the `-ise` families), `check_md_math.py` (Markdown that renders in both previews; the frontmatter),
`check_references.py` (every citation and relative link resolves) and
`check_docstrings.py` (docstring headers, private functions documented, modules under
600 lines). Each takes paths and exits 1 when it flags anything.
