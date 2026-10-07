# `scripts/` — the command-line entry points

All run from the repository root with the dev environment (`uv sync --group dev`). The
Hydra scripts take `key=value` overrides and `--config-name`; see `configs/`.

| Script | What it does | Typical call |
|---|---|---|
| `generate_data.py` | Integrates trajectories from a `configs/data/*.yaml` generation config and writes one netCDF4 file, `<out_dir>/<name>-<config_hash>.nc` (ADR-0008). The hash is over the generation-defining fields, so an unchanged config overwrites the same file. **Training never generates**: run this first. | `uv run python scripts/generate_data.py --config-name bautin` <br> `uv run python scripts/generate_data.py --config-name fhn n_trajectories=512 seed=3` |
| `train.py` | One training run from `configs/train.yaml` (ADR-0011): builds the loader, model, loss, schedule and optimizer, trains, logs (`PrintLogger`; wandb with `wandb=online\|offline`), checkpoints into the Hydra run directory `runs/<date>/<time>` (resolved `config.yaml`, `metadata.json`, `checkpoints/`), and the final step is always saved. `resume=<run_dir>` restores that run's latest checkpoint and continues the same data stream. `-m` sweeps (`runs/multirun/...`, one wandb group). | `uv run python scripts/train.py` <br> `uv run python scripts/train.py model=autoencoder loss=autoencoder schedule=yawata` <br> `uv run python scripts/train.py num_steps=20000 windows.batch=4096 wandb=online` <br> `uv run python scripts/train.py resume=runs/2026-10-05/12-00-00` <br> `uv run python scripts/train.py -m seed=0,1,2` |
| `bench_dataloader.py` | Times the data pipeline against the training step on the device you name (`--device`, else `resolve_device`): fetch rows, the dispatch floor, dispatch vs total for several loaders, `--hlo-stats` (loop/launch/conditional census of the compiled step), `--profile DIR`. `--config <overrides>` benchmarks a training config's own pipeline, model and loss. How to read the table: `docs/design/training-step-performance.md` §4–5. | `uv run python scripts/bench_dataloader.py --device 0 --hlo-stats --no-mp` <br> `uv run python scripts/bench_dataloader.py --config data=fhn windows.batch=4096` |
| `check_md_math.py` | Emulates GitHub's Markdown-before-MathJax rendering over `docs/**/*.md` and reports constructs that break there (`CLAUDE.md` rules). Must report 0 problems before a Markdown commit. | `uv run python scripts/check_md_math.py` |

Runs from Python (notebooks, analysis). `setup` builds what the script builds and owns
the run directory; the loop is yours or the default:

```python
from deep_isochron.experiment import compose, load_model, load_or_generate_source, load_run, setup, train

cfg = compose("data=fhn", "data.n_trajectories=500", "model=conjugacy", "num_steps=2500")
load_or_generate_source(cfg.data)            # notebooks may generate; setup/train never do
exp = setup(cfg, "runs/notebook/try-1")      # refuses a directory that already holds a run
state = exp.trainer.train(                   # your loop, your logger …
    exp.state, exp.loaders.train,
    logger=DelayedLogger(WandbLogger(wandb.init(entity="jhko725", project="isochron"), every=25)),
    checkpointer=exp.run.checkpointer(save_every=200),   # … but the run's checkpointer
    evaluate=exp.evaluator, eval_every=200,
)
state = exp.train()                          # or the default loop (Print + wandb per config)
state = train(cfg, "runs/notebook/try-2")    # = setup(...).train()

cfg, state = load_run("runs/notebook/try-1")             # any run directory, latest step
model = load_model("runs/2026-10-05/12-00-00", step=5000)
cfg, state = load_run("results/bare-ckpt", template=my_state)  # a model built without a config

cfg_more = compose(..., "num_steps=5000")
state = train(cfg_more, "runs/notebook/try-1", resume=True)   # continue in place: explicit; the
                                                              # config change is recorded
```
