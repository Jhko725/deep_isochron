"""Benchmark the data pipeline against the training step (roadmap C7).

Answers, on the machine it runs on and the device the *user* names:

1. how long one batch takes to *fetch* from the window pipeline (host-side slicing),
   with and without ``mp_prefetch`` worker processes;
2. how long one *training step* takes, split into **host dispatch** (the Python loop
   returning, JAX's asynchronous dispatch) and **total** (device finished), for batches
   that are (a) already on the device, (b) NumPy — transferred by ``jit`` in the loop,
   grain's "option A" — and (c) delivered by ``data.to_device`` (grain's option C);
3. optionally (``--hlo-stats``) the shape of the compiled step: instruction count, the
   ``while`` loops with their trip counts, and an estimate of kernel launches per step —
   a step whose host time is flat in ``--batch`` is launch-bound, and this says by how
   many launches;
4. optionally a profiler trace of one configuration (``--profile DIR``; open it in
   TensorBoard or Perfetto) when the numbers above do not explain themselves.

Configurations are run **interleaved and repeated** (``--repeats``), after a throwaway
warm-up, and the table reports the **min** and **median** over repeats — a single
back-to-back pass is not a measurement on a shared machine. The header prints the CPU
allowance of the process (affinity, cgroup quota, Slurm variables): every loader thread
or worker process competes with the host thread that launches the step's kernels, so a
one-core allocation serializes them all.

Usage (from the repository root, with the dev environment; ``--device`` is the index
into ``jax.devices()`` and is required — the script never picks a GPU on its own)::

    uv run python scripts/bench_dataloader.py --device 0
    uv run python scripts/bench_dataloader.py --device 0 --model ae --batch 2048
    uv run python scripts/bench_dataloader.py --device 0 --data data/fhn-<hash>.nc
    uv run python scripts/bench_dataloader.py --device 0 --hlo-stats --no-mp
    uv run python scripts/bench_dataloader.py --device 0 --profile /tmp/trace --no-mp

Reading the table: ``dispatch ≈ total`` means the host is the bottleneck — Python
dispatch and GIL contention if the dispatch floor is of the order of the step, otherwise
the kernel launches (or a synchronization) inside the executable; ``dispatch ≪ total``
means the device is. If ``to_device`` ≈ ``device-resident``, the prefetch thread keeps
up and ``mp_prefetch`` is unnecessary; if ``to_device`` ≈ ``fetch only``, the fetch is
the bottleneck and worker processes are the remedy — provided the process has the cores
for them. The ``grain default readers`` row iterates the ``MapDataset`` bare (16 reader
threads, 500-batch buffer) and is there to show what that costs next to the step.
"""

from __future__ import annotations

import os


if __name__ == "__mp_main__":  # a grain worker re-importing this script (spawn start)
    os.environ.setdefault("JAX_PLATFORMS", "cpu")  # workers must never touch the GPU

import argparse  # noqa: E402
import pickle  # noqa: E402
import re  # noqa: E402
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
    single_threaded,
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
        "--hlo-stats",
        action="store_true",
        help="print the compiled step's while loops and a kernel-launch estimate",
    )
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


def device_put_each(loader: Iterable, device: jax.Device) -> Iterator:
    """grain's option A literally: a synchronous ``jax.device_put`` per batch."""
    for batch in loader:
        yield jax.device_put(batch, device)


def cpu_allowance() -> str:
    """What the process may actually run on: CPU affinity, cgroup quota, Slurm."""
    parts = [f"affinity {len(os.sched_getaffinity(0))} of {os.cpu_count()} CPUs"]
    for path in ("/sys/fs/cgroup/cpu.max", "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"):
        try:
            parts.append(f"{path.rsplit('/', 1)[1]} {Path(path).read_text().strip()}")
            break
        except OSError:
            continue
    slurm = {k: v for k, v in os.environ.items() if k.startswith("SLURM_CPUS")}
    if slurm:
        parts.append(" ".join(f"{k}={v}" for k, v in sorted(slurm.items())))
    return "; ".join(parts)


# ------------------------------------------------------------- compiled-step shape --
_HLO_HEADER = re.compile(r"^(?:ENTRY\s+)?%(\S+)\s*\(")
_HLO_INSTR = re.compile(r"^\s*(?:ROOT\s+)?%(\S+)\s*=\s*.+?\s([a-z\-]+)\(")
_HLO_NO_LAUNCH = {"parameter", "constant", "get-tuple-element", "tuple", "bitcast"}


def hlo_launch_estimate(
    text: str,
) -> tuple[int, list[tuple[str, int | None, int]], int, int]:
    """``(instructions, loops, launches, conditionals)`` from optimized HLO text: the
    module's instruction count, one ``(name, trip_count, launches_per_iteration)`` per
    ``while`` (trip count from XLA's ``known_trip_count`` annotation when present), an
    *estimate* of kernel launches per call — every instruction other than a parameter,
    constant, tuple or bitcast counts one, a loop body counts ``trip_count`` times (once
    when unknown), ``call``s are inlined, ``conditional`` branches are summed — and the
    number of ``conditional`` executions per call (loop multiplicity included): on the
    GPU backend each one needs its predicate on the host, i.e. a device-to-host copy the
    launching thread waits for."""
    comps: dict[str, list[str]] = {}
    entry: str | None = None
    current: list[str] | None = None
    for line in text.splitlines():
        m = _HLO_HEADER.match(line)
        if m:
            current = comps.setdefault(m.group(1), [])
            if line.startswith("ENTRY"):
                entry = m.group(1)
        elif current is not None and _HLO_INSTR.match(line):
            current.append(line)
    if entry is None:
        raise ValueError("no ENTRY computation in the HLO text")
    loops: list[tuple[str, int | None, int]] = []
    conditionals = [0]

    def ref(line: str, key: str) -> str | None:
        m = re.search(rf"{key}=%(\S+?)[,\s)]", line + " ")
        return m.group(1) if m else None

    def cost(name: str, mult: int = 1) -> int:
        n = 0
        for line in comps.get(name, []):
            m = _HLO_INSTR.match(line)
            assert m is not None
            op = m.group(2)
            if op in _HLO_NO_LAUNCH:
                continue
            if op == "call":
                target = ref(line, "to_apply")
                n += cost(target, mult) if target else 1
            elif op == "while":
                body, cond = ref(line, "body"), ref(line, "condition")
                t = re.search(r'known_trip_count\\?":\{\\?"n\\?":\\?"(\d+)', line)
                trip = int(t.group(1)) if t else None
                inner = mult * (trip or 1)
                per = cost(body, inner) if body else 0
                per += cost(cond, inner) if cond else 0
                loops.append((m.group(1), trip, per))
                n += per * (trip or 1)
            elif op == "conditional":
                conditionals[0] += mult
                branches = re.findall(r"(?:true|false)_computation=%(\S+?)[,\s)]", line)
                branches += re.findall(r"branch_computations=\{([^}]*)\}", line)
                n += 1 + sum(cost(b.strip().lstrip("%"), mult) for b in branches)
            else:
                n += 1
        return n

    launches = cost(entry)
    return sum(len(v) for v in comps.values()), loops, launches, conditionals[0]


def report_hlo_stats(trainer: Trainer, state, batch, device: jax.Device) -> None:
    text = trainer.train_step.lower(state, batch).compile().compiled.as_text()
    instructions, loops, launches, conditionals = hlo_launch_estimate(text)
    print(
        f"\ncompiled step: {instructions} HLO instructions, {len(loops)} while loops, "
        f"≈{launches} kernel launches per step (estimate; a launch costs the host of "
        f"the order of 5–10 µs), {conditionals} conditional executions per step (on "
        f"the GPU backend each waits for its predicate from the device)"
    )
    for name, trip, per in sorted(loops, key=lambda x: -(x[1] or 1) * x[2])[:12]:
        trip_s = "?" if trip is None else str(trip)
        print(f"  {name:<32} trip count {trip_s:>5} × {per:>5} launches/iteration")

    # the suspect: jax.scipy.linalg.expm (BiLipschitzLinear's rotations) is a 16-step
    # scan of lax.cond; time one 2×2 expm against the closed-form rotation
    a = jax.device_put(jnp.array([[0.0, -0.3], [0.3, 0.0]]), device)
    expm = jax.jit(jax.scipy.linalg.expm)

    def rotation(m):
        c, s = jnp.cos(m[1, 0]), jnp.sin(m[1, 0])
        return jnp.array([[c, -s], [s, c]])

    rot = jax.jit(rotation)

    def ms(fn, n=100) -> tuple[float, float]:
        fn(a).block_until_ready()
        t0 = time.perf_counter()
        for _ in range(n):
            out = fn(a)
        t1 = time.perf_counter()
        out.block_until_ready()
        return 1e3 * (t1 - t0) / n, 1e3 * (time.perf_counter() - t0) / n

    for name, fn in [("expm (2×2)", expm), ("closed-form rotation", rot)]:
        d, t = ms(fn)
        print(f"  {name:<32} dispatch {d:7.3f} ms   total {t:7.3f} ms per call")


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
        f"{args.steps} steps × {args.repeats} repeats, interleaved\n"
        f"host: {cpu_allowance()}; jax {jax.__version__}"
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
    first = [next(it) for it in [iter(single_threaded(pipeline()))] for _ in range(8)]
    resident = [jax.device_put(b, device) for b in first]  # committed to `device`
    uncommitted = [{k: jnp.asarray(v) for k, v in b.items()} for b in first]
    cached = grain.MapDataset.source(first).repeat()  # a pipeline with no fetch work

    fetch_rows: list[tuple[str, Callable[[], Iterable]]] = [
        ("fetch only (host, 1 reader thread)", lambda: single_threaded(pipeline())),
    ]
    step_rows: list[tuple[str, Callable[[], Iterable]]] = [
        ("step, device-resident (committed) batches", lambda: cycle(resident)),
        ("step, device-resident (uncommitted) batches", lambda: cycle(uncommitted)),
        (
            "step, NumPy batches, 1 reader (option A)",
            lambda: single_threaded(pipeline()),
        ),
        (
            "step, NumPy + device_put in loop (option A)",
            lambda: device_put_each(single_threaded(pipeline()), device),
        ),
        ("step, NumPy batches, grain default readers", pipeline),
        ("step, to_device (option C)", lambda: to_device(pipeline(), device)),
        ("step, to_device over cached batches", lambda: to_device(cached, device)),
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
    if args.hlo_stats:
        report_hlo_stats(trainer, state, resident[0], device)
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
        "unnecessary; to_device ≈ fetch only → the fetch is the bottleneck.\n"
        "to_device ≫ to_device over cached batches → the fetch thread's CPU work is "
        "what slows the step (too few cores for the loader threads), not the transfer."
    )

    if args.profile is not None:
        with jax.profiler.trace(str(args.profile)):
            time_steps(trainer, state, to_device(pipeline(), device), args.steps)
        print(
            f"\nprofiler trace of the to_device configuration written to {args.profile}"
        )


if __name__ == "__main__":
    main()
