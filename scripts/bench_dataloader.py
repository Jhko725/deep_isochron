"""Benchmark the data pipeline against the training step (roadmap C7).

Answers three questions on the machine it runs on:

1. How long does one batch take to *fetch* from the window pipeline (host-side slicing)?
2. How long is one *training step*, and does the fetch hide behind it with
   ``to_device`` (grain's two-stage prefetch, one CPU thread)?
3. Does ``mp_prefetch`` (worker processes) pickle our ``TimeSeriesDataSource`` and, if
   so, does it shorten the step further?

Usage (from the repository root, with the dev environment)::

    uv run python scripts/bench_dataloader.py                       # Bautin, conjugacy
    uv run python scripts/bench_dataloader.py --model ae --batch 2048
    uv run python scripts/bench_dataloader.py --data data/fhn-<hash>.nc --workers 4

Prints one row per configuration: ms per batch fetched, ms per training step, and the
ratio. Read it as: if ``step (to_device)`` ≈ ``step (device batches)`` (the lower bound,
batches already resident), the thread keeps up and ``mp_prefetch`` is unnecessary.
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from collections.abc import Iterable
from pathlib import Path

import grain
import jax
import jax.numpy as jnp
import optax
from absl import flags
from deep_isochron.data import (
    generate,
    OnCycleGaussian,
    TimeSeriesDataSource,
    to_device,
    windows,
)
from deep_isochron.model import (
    ConjugateLatentDynamics,
    PhaseAmplitudeAutoencoder,
    PhaseAmplitudeLatentDynamics,
)
from deep_isochron.model.invertible import (
    BiLipschitzLinear,
    CouplingFlow,
    MonotonicRQSpline,
    SequentialINN,
)
from deep_isochron.systems import BautinNormalForm
from deep_isochron.training import (
    ConjugacyTrajectoryLoss,
    PhaseAutoencoderLoss,
    Trainer,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data", type=Path, help=".nc dataset; default: generate Bautin")
    p.add_argument("--model", choices=["conjugacy", "ae"], default="conjugacy")
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--length", type=int, default=50, help="window length")
    p.add_argument("--blocks", type=int, default=8, help="INN coupling blocks")
    p.add_argument("--steps", type=int, default=50, help="timed steps per config")
    p.add_argument("--workers", type=int, default=4, help="mp_prefetch workers")
    p.add_argument("--no-mp", action="store_true", help="skip the mp_prefetch test")
    return p.parse_args()


def make_source(args: argparse.Namespace) -> TimeSeriesDataSource:
    if args.data is not None:
        return TimeSeriesDataSource.load(args.data)
    nf = BautinNormalForm(1.0, 0.5, 2.0, 1.0)
    period = float(nf.period())
    return generate(
        nf,
        OnCycleGaussian.from_normal_form(nf, 1000, 0.5),
        jnp.arange(0.0, 3 * period, period / 40),
        512,
        seed=0,
    )


def make_model_and_loss(args: argparse.Namespace, dim: int):
    key = jax.random.key(0)
    if args.model == "ae":
        model = PhaseAmplitudeAutoencoder(
            dim, PhaseAmplitudeLatentDynamics(1.0, -0.5), key=key
        )
        return model, PhaseAutoencoderLoss()
    layers = []
    for i, k in enumerate(jax.random.split(key, args.blocks)):
        kl, kc = jax.random.split(k)
        layers.append(BiLipschitzLinear(dim=dim, max_lipschitz=2.0, key=kl))
        layers.append(
            CouplingFlow(
                dim=dim,
                bijection=MonotonicRQSpline(num_bins=9, xy_range=(-4.0, 4.0)),
                mlp_width=32,
                mlp_depth=2,
                flip=(i % 2 == 0),
                key=kc,
            )
        )
    model = ConjugateLatentDynamics(
        BautinNormalForm(1.0, 0.5, 2.0, 1.0), SequentialINN(layers)
    )
    return model, ConjugacyTrajectoryLoss()


def time_fetch(loader: Iterable, n: int) -> float:
    it = iter(loader)
    next(it)
    t0 = time.perf_counter()
    for _ in range(n):
        next(it)
    return 1e3 * (time.perf_counter() - t0) / n


def time_steps(trainer: Trainer, model, loader: Iterable, n: int) -> float:
    it = iter(loader)
    state = trainer.init(model, key=jax.random.key(1))
    state, metrics = trainer.train_step(state, next(it))  # compile
    jax.block_until_ready(metrics["loss"])
    t0 = time.perf_counter()
    for _ in range(n):
        state, metrics = trainer.train_step(state, next(it))
    jax.block_until_ready(metrics["loss"])
    return 1e3 * (time.perf_counter() - t0) / n


def main() -> None:
    args = parse_args()
    flags.FLAGS(sys.argv[:1])  # grain's multiprocessing reads absl flags
    jax.config.update("jax_enable_x64", True)
    source = make_source(args)
    model, loss = make_model_and_loss(args, source.ys.shape[-1])
    trainer = Trainer(optax.adam(1e-3), loss)
    device = jax.devices()[0]
    print(
        f"device: {device.platform}:{device.id}; {len(source)} trajectories × "
        f"{source.trajectory_length} steps; batch {args.batch} × window {args.length}; "
        f"model {args.model}"
    )

    def pipeline() -> grain.MapDataset:
        return windows(source, args.length, seed=0).batch(
            args.batch, drop_remainder=True
        )

    rows: list[tuple[str, float | None, float | None]] = []

    # 1. fetch alone (host slicing + batching, no transfer)
    fetch = time_fetch(pipeline(), args.steps)
    rows.append(("fetch only (host)", fetch, None))

    # 2a. lower bound: batches already on the device
    resident = [
        jax.device_put(b, device)
        for b in (lambda it: [next(it) for _ in range(8)])(iter(pipeline()))
    ]

    def cycle(items: list) -> Iterable:
        while True:
            yield from items

    rows.append(
        (
            "step, device-resident batches",
            None,
            time_steps(trainer, model, cycle(resident), args.steps),
        )
    )
    # 2b. option A: NumPy batches, jit transfers them synchronously in the loop
    rows.append(
        (
            "step, NumPy batches (option A)",
            None,
            time_steps(trainer, model, pipeline(), args.steps),
        )
    )
    # 2c. option C: to_device (thread prefetch + device buffer)
    rows.append(
        (
            "step, to_device (option C)",
            None,
            time_steps(trainer, model, to_device(pipeline(), device), args.steps),
        )
    )

    # 3. mp_prefetch: pickling and speed
    if not args.no_mp:
        try:
            pickle.loads(pickle.dumps(source))
            mp = (
                pipeline()
                .to_iter_dataset()
                .mp_prefetch(grain.MultiprocessingOptions(num_workers=args.workers))
            )
            rows.append(
                (
                    f"fetch only, mp_prefetch({args.workers})",
                    time_fetch(mp, args.steps),
                    None,
                )
            )
            mp = (
                pipeline()
                .to_iter_dataset()
                .mp_prefetch(grain.MultiprocessingOptions(num_workers=args.workers))
            )
            rows.append(
                (
                    f"step, mp_prefetch({args.workers}) + to_device",
                    None,
                    time_steps(trainer, model, to_device(mp, device), args.steps),
                )
            )
        except Exception as e:  # noqa: BLE001 — report, do not crash the benchmark
            rows.append(
                (f"mp_prefetch FAILED: {type(e).__name__}: {str(e)[:80]}", None, None)
            )

    print(f"\n{'configuration':<48}{'ms/batch':>10}{'ms/step':>10}")
    for name, f, s in rows:
        fs = f"{f:10.2f}" if f is not None else f"{'':>10}"
        ss = f"{s:10.2f}" if s is not None else f"{'':>10}"
        print(f"{name:<48}{fs}{ss}")
    print(
        "\nRead: if 'to_device' ≈ 'device-resident', the prefetch thread keeps up and "
        "mp_prefetch is unnecessary;\nif 'to_device' ≈ 'fetch only' the fetch is the "
        "bottleneck and worker processes are the remedy."
    )


if __name__ == "__main__":
    main()
