"""Hunt the first non-finite gradient of a run, from its last (or a given) checkpoint.

    uv run python scripts/find_nan.py runs/notebook/winfree-1
    uv run python scripts/find_nan.py runs/winfree-1 --step 2500 --max-steps 300
    uv run python scripts/find_nan.py runs/notebook/winfree-1 --debug-nans

Reopens the run (``load_run``), rebuilds the same data stream from that step
(``Loaders.from_config(..., start_step=step)``, so the batches are the ones the run
saw), and steps the trainer by hand. Every 25 steps it prints the batch's latent extent
(``training.evaluation.latent_extent``: where the bijection sends the batch); after
every step it checks the gradients leaf by leaf, and at the first non-finite one reports
the step, the loss terms, the batch's extent, the offending parameter leaves and the
batch's non-finite predictions, then stops. ``--debug-nans`` re-runs that batch under
``jax_debug_nans`` so JAX names the primitive. Nothing is written to the run directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from deep_isochron.data import TimeSeriesDataSource
from deep_isochron.experiment import (
    build_source,
    build_trainer,
    configure_jax,
    load_run,
    Loaders,
)
from deep_isochron.training.evaluation import latent_extent


def nonfinite_leaves(tree) -> list[str]:
    out = []
    for path, leaf in jax.tree_util.tree_leaves_with_path(tree):
        if isinstance(leaf, jax.Array) and not bool(jnp.all(jnp.isfinite(leaf))):
            out.append(jax.tree_util.keystr(path))
    return out


def describe_extent(model, batch) -> str:
    u = jnp.asarray(batch["u"])
    ext = latent_extent(model, u.reshape(-1, u.shape[-1]))
    parts = [f"{k}={float(v):.4g}" for k, v in ext.items() if k != "n"]
    return "  ".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--step", type=int, default=None, help="checkpoint step (latest)")
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--debug-nans", action="store_true")
    args = ap.parse_args()

    cfg, state = load_run(args.run_dir, args.step)
    assert cfg is not None
    device = configure_jax(cfg)
    source: TimeSeriesDataSource = build_source(cfg.data)
    trainer = build_trainer(cfg)
    start = int(state.step)
    loaders = Loaders.from_config(cfg, source, device, start_step=start)
    print(
        f"run {args.run_dir}: checkpoint step {start}; stepping up to {args.max_steps}"
    )

    grad_fn = eqx.filter_jit(
        lambda model, batch, weights: eqx.filter_value_and_grad(
            trainer.loss, has_aux=True
        )(model, batch, weights)
    )
    for i, batch in enumerate(loaders.train):
        if i >= args.max_steps:
            print("no non-finite gradient within --max-steps")
            return
        step = int(state.step)
        weights = trainer.schedule.weights(state.schedule_state, state.step)
        model = eqx.nn.inference_mode(state.model, False)
        (loss, terms), grads = grad_fn(model, batch, weights)
        bad = nonfinite_leaves(grads)
        if i % 25 == 0 or bad:
            print(
                f"step {step}: loss {float(loss):.4g}  {describe_extent(model, batch)}"
            )
        if bad or not bool(jnp.isfinite(loss)):
            print(f"\nFIRST NON-FINITE at step {step}")
            print("  loss terms:", {k: float(v) for k, v in terms.items()})
            print("  gradient leaves non-finite:", bad or "(none; loss itself)")
            x = jnp.asarray(batch["u"])
            x_pred, y_pred = eqx.filter_vmap(model)(jnp.asarray(batch["t"]), x[:, 0])
            bad_windows = np.flatnonzero(
                ~np.asarray(jnp.all(jnp.isfinite(x_pred), axis=(1, 2)))
            )
            print(
                f"  windows with non-finite predictions: {len(bad_windows)} "
                f"(indices {bad_windows[:10]})"
            )
            if len(bad_windows):
                j = int(bad_windows[0])
                print("  first bad window start x0 =", np.asarray(x[j, 0]))
                print("  its image H(x0) =", np.asarray(model.bijection(x[j, 0])))
                print("  its latent trajectory (first 3):", np.asarray(y_pred[j, :3]))
            if args.debug_nans:
                print("\nre-running the batch with jax_debug_nans ...")
                jax.config.update("jax_debug_nans", True)
                eqx.filter_value_and_grad(trainer.loss, has_aux=True)(
                    model, batch, weights
                )
            return
        state = trainer.train_step(state, batch)[0]
    print("loader exhausted without a non-finite gradient")


if __name__ == "__main__":
    main()
