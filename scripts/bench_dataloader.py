"""Benchmark the data pipeline against the training step (roadmap C7).

Answers, on the machine it runs on and the device the *user* names:

1. how long one batch takes to *fetch* from the window pipeline (host-side slicing),
   with and without ``mp_prefetch`` worker processes;
2. how long one *training step* takes, split into **host dispatch** (the Python loop
   returning, JAX's asynchronous dispatch) and **total** (device finished), for batches
   that are (a) already on the device, (b) NumPy — transferred by ``jit`` in the loop,
   grain's "option A" — and (c) delivered by ``data.to_device`` (grain's option C);
3. optionally a profiler trace of one configuration (``--profile DIR``; open it in
   TensorBoard or Perfetto) when the numbers above do not explain themselves.

Configurations are run **interleaved and repeated** (``--repeats``), after a throwaway
warm-up, and the table reports the **min** and **median** over repeats — a single
back-to-back pass is not a measurement on a shared machine.

Usage (from the repository root, with the dev environment; ``--device`` is the index
into ``jax.devices()`` and is required — the script never picks a GPU on its own)::

    uv run python scripts/bench_dataloader.py --device 0
    uv run python scripts/bench_dataloader.py --device 0 --model ae --batch 2048
    uv run python scripts/bench_dataloader.py --device 0 --data data/fhn-<hash>.nc
    uv run python scripts/bench_dataloader.py --device 0 --profile /tmp/trace --no-mp

Reading the table: ``dispatch ≈ total`` means the host is the bottleneck (Python
dispatch of the step, GIL contention with loader threads); ``dispatch ≪ total`` means
the device is. If ``to_device`` ≈ ``device-resident``, the prefetch thread keeps up and
``mp_prefetch`` is unnecessary; if ``to_device`` ≈ ``fetch only``, the fetch is the
bottleneck and worker processes are the remedy.
"""

from __future__ import annotations

import os


if __name__ == "__mp_main__":  # a grain worker re-importing this script (spawn start)
    os.environ.setdefault("JAX_PLATFORMS", "cpu")  # workers must never touch the GPU

import argparse  # noqa: E402
import pickle  # noqa: E402
import statistics  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections.abc import Callable, Iterable, Iterator  # noqa: E402
from pathlib import Path  # noqa: E402

import equinox as eqx  # noqa: E402
import grain  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402
from absl import flags  # noqa: E402
from deep_isochron.data import (  # noqa: E402
    generate,
    OnCycleGaussian,
    TimeSeriesDataSource,
    to_device,
    windows,
)
from deep_isochron.model import (  # noqa: E402
    ConjugateLatentDynamics,
    PhaseAmplitudeAutoencoder,
    PhaseAmplitudeLatentDynamics,
)
from deep_isochron.model.invertible import (  # noqa: E402
    BiLipschitzLinear,
    CouplingFlow,
    MonotonicRQSpline,
    SequentialINN,
)
from deep_isochron.systems import BautinNormalForm  # noqa: E402
from deep_isochron.training import (  # noqa: E402
    ConjugacyTrajectoryLoss,
    PhaseAutoencoderLoss,
    Trainer,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--device",
        type=int,
        required=True,
        help="index into jax.devices(); the script never chooses a GPU by itself",
    )
    p.add_argument("--data", type=Path, help=".nc dataset; default: generate Bautin")
    p.add_argument("--model", choices=["conjugacy", "ae"], default="conjugacy")
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--length", type=int, default=50, help="window length")
    p.add_argument("--blocks", type=int, default=8, help="INN coupling blocks")
    p.add_argument("--steps", type=int, default=30, help="timed steps per measurement")
    p.add_argument("--repeats", type=int, default=3, help="interleaved repeats")
    p.add_argument("--workers", type=int, default=4, help="mp_prefetch workers")
    p.add_argument("--no-mp", action="store_true", help="skip the mp_prefetch rows")
    p.add_argument(
        "--profile",
        type=Path,
        help="write a jax.profiler trace of the to_device configuration here",
    )
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


# ------------------------------------------------------------------- measurements --
def time_fetch(loader: Iterable, n: int) -> float:
    it = iter(loader)
    next(it)
    t0 = time.perf_counter()
    for _ in range(n):
        next(it)
    return 1e3 * (time.perf_counter() - t0) / n


def time_steps(
    trainer: Trainer, state, loader: Iterable, n: int
) -> tuple[float, float]:
    """``(dispatch, total)`` ms/step: host loop time without waiting, and with the final
    ``block_until_ready`` (device done). Same state throughout; the warm-up step is not
    timed."""
    it = iter(loader)
    state, metrics = trainer.train_step(state, next(it))
    jax.block_until_ready(metrics["loss"])
    t0 = time.perf_counter()
    for _ in range(n):
        state, metrics = trainer.train_step(state, next(it))
    t1 = time.perf_counter()
    jax.block_until_ready(metrics["loss"])
    t2 = time.perf_counter()
    return 1e3 * (t1 - t0) / n, 1e3 * (t2 - t0) / n


def cycle(items: list) -> Iterator:
    while True:
        yield from items


def main() -> None:
    args = parse_args()
    flags.FLAGS(sys.argv[:1])  # grain's multiprocessing reads absl flags
    jax.config.update("jax_enable_x64", True)
    device = jax.devices()[args.device]
    jax.config.update("jax_default_device", device)  # model and state live there too
    source = make_source(args)
    model, loss = make_model_and_loss(args, source.ys.shape[-1])
    trainer = Trainer(optax.adam(1e-3), loss)
    state = trainer.init(model, key=jax.random.key(1))
    print(
        f"device: {device.platform}:{device.id} ({device.device_kind}); "
        f"{len(source)} trajectories × {source.trajectory_length} steps; batch "
        f"{args.batch} × window {args.length}; model {args.model}; "
        f"{args.steps} steps × {args.repeats} repeats, interleaved"
    )

    def pipeline() -> grain.MapDataset:
        return windows(source, args.length, seed=0).batch(
            args.batch, drop_remainder=True
        )

    def mp_pipeline() -> grain.IterDataset:
        return (
            pipeline()
            .to_iter_dataset()
            .mp_prefetch(grain.MultiprocessingOptions(num_workers=args.workers))
        )

    # one fetch of a few batches, three ways to hold them
    first = [next(it) for it in [iter(pipeline())] for _ in range(8)]
    resident = [jax.device_put(b, device) for b in first]  # committed to `device`
    uncommitted = [{k: jnp.asarray(v) for k, v in b.items()} for b in first]

    fetch_rows: list[tuple[str, Callable[[], Iterable]]] = [
        ("fetch only (host)", pipeline),
    ]
    step_rows: list[tuple[str, Callable[[], Iterable]]] = [
        ("step, device-resident (committed) batches", lambda: cycle(resident)),
        ("step, device-resident (uncommitted) batches", lambda: cycle(uncommitted)),
        ("step, NumPy batches (option A)", pipeline),
        ("step, to_device (option C)", lambda: to_device(pipeline(), device)),
    ]
    mp_ok = not args.no_mp
    if mp_ok:
        try:
            pickle.loads(pickle.dumps(source))
            next(iter(mp_pipeline()))
            fetch_rows.append((f"fetch only, mp_prefetch({args.workers})", mp_pipeline))
            step_rows.append(
                (
                    f"step, mp_prefetch({args.workers}) + to_device",
                    lambda: to_device(mp_pipeline(), device),
                )
            )
        except Exception as e:  # noqa: BLE001 — report, do not crash the benchmark
            print(f"mp_prefetch unavailable: {type(e).__name__}: {str(e)[:120]}")
            mp_ok = False

    # dispatch floor: a trivial jitted function over the same (state, batch) pytrees —
    # what the host pays per call just to hand the arguments to JAX
    @eqx.filter_jit
    def trivial(s, b):
        return s.step + 1, jnp.sum(b["u"])

    def time_floor(loader: Iterable, n: int) -> float:
        it = iter(loader)
        jax.block_until_ready(trivial(state, next(it)))
        t0 = time.perf_counter()
        for _ in range(n):
            out = trivial(state, next(it))
        jax.block_until_ready(out)
        return 1e3 * (time.perf_counter() - t0) / n

    # warm-up (compile, allocator, lazy CUDA state; several steps, not one), then
    # interleaved repeats
    time_steps(trainer, state, cycle(resident), 5)
    time_floor(cycle(resident), 5)
    floor: dict[str, list[float]] = {"committed": [], "uncommitted": [], "numpy": []}
    fetch: dict[str, list[float]] = {name: [] for name, _ in fetch_rows}
    dispatch: dict[str, list[float]] = {name: [] for name, _ in step_rows}
    total: dict[str, list[float]] = {name: [] for name, _ in step_rows}
    for _ in range(args.repeats):
        floor["committed"].append(time_floor(cycle(resident), args.steps))
        floor["uncommitted"].append(time_floor(cycle(uncommitted), args.steps))
        floor["numpy"].append(time_floor(cycle(first), args.steps))
        for name, make in fetch_rows:
            fetch[name].append(time_fetch(make(), args.steps))
        for name, make in step_rows:
            d, t = time_steps(trainer, state, make(), args.steps)
            dispatch[name].append(d)
            total[name].append(t)

    def cell(values: list[float]) -> str:
        return f"{min(values):8.1f} /{statistics.median(values):7.1f}"

    print(f"\n{'ms per batch or step: min / median':<48}{'fetch':>18}")
    for name, _ in fetch_rows:
        print(f"{name:<48}{cell(fetch[name]):>18}")
    print(f"\n{'dispatch floor (trivial jit on state + batch)':<48}{'ms/call':>18}")
    for name, values in floor.items():
        print(f"{'  ' + name + ' batch':<48}{cell(values):>18}")
    print(f"\n{'':<48}{'dispatch':>18}{'total':>18}")
    for name, _ in step_rows:
        print(f"{name:<48}{cell(dispatch[name]):>18}{cell(total[name]):>18}")
    print(
        "\nRead: dispatch ≈ total → the host is the bottleneck: Python dispatch / GIL "
        "if the floor is of the order of the step, otherwise kernel launches or a sync "
        "inside the executable (launch-bound dispatch is flat in --batch; device-bound "
        "scales with it); dispatch ≪ total → the device is.\n"
        "to_device ≈ device-resident → the prefetch thread keeps up and mp_prefetch is "
        "unnecessary; to_device ≈ fetch only → the fetch is the bottleneck."
    )

    if args.profile is not None:
        with jax.profiler.trace(str(args.profile)):
            time_steps(trainer, state, to_device(pipeline(), device), args.steps)
        print(
            f"\nprofiler trace of the to_device configuration written to {args.profile}"
        )


if __name__ == "__main__":
    main()
