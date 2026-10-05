---
type: design
status: tentative
updated: 2026-10-05
verified_by: joon (2026-10-05; §6.3 C9 update pending)
sources: [JAX docs (async dispatch, data placement, profiling), XLA GPU runtime source, grain source and tutorial, CPython docs, NVIDIA CUDA Graphs blog, Serino et al. 2025]
---

# Where a JAX training step spends its time — and where ours did

This document records what the Phase C benchmarking (`scripts/bench_dataloader.py`, change
document `2026-10-04-trainer.md` round 2, ADR-0009 §1, ADR-0010) taught us about the
machinery under a JAX training loop, written for a reader who has not worked with JAX's
asynchronous dispatch, XLA's GPU runtime, or Python's threading model. Each section states
the mechanism, cites where it is documented, and then says what it meant for `deep_isochron`.
Statements are marked **[cited]** when the source says it in so many words and **[deduced]**
when we inferred it from a measurement or from reading source code.

Sections 1–3 are the background; §4 defines what the benchmark script measures; §5 is the
taxonomy of bottlenecks those measurements tell apart; §6 the case history; §7 the rules we
adopted.

## 1. Asynchronous dispatch: the host runs ahead of the device

When Python calls a JAX operation — a jitted function above all — JAX does not wait for the
result. It *enqueues* the work on the device and returns a `jax.Array` that is a future for
the value; the Python program continues, and only an operation that needs the actual numbers
(printing, `np.asarray`, `float()`, `.block_until_ready()`) blocks until the device has
produced them **[cited: JAX, *Asynchronous dispatch*]**. The documentation's own framing is
that this "allows Python code to 'run ahead' of an accelerator device, keeping Python code
out of the critical path." The same page warns that timing a JAX call without blocking
measures only the *dispatch* — the host-side cost of enqueueing — not the computation.

Three consequences shape everything below.

- **A training loop has two clocks.** The host clock is how long the Python loop takes to
  come back from `train_step(state, batch)` and go round again; the device clock is how long
  the device takes to finish the step. Throughput is set by the slower of the two. The
  benchmark reports both — *dispatch* (host loop time without waiting) and *total* (with a
  final `block_until_ready`) — because their relation is the first diagnostic (§5).
- **Reading a metric is a synchronization.** `float(loss)` makes the host wait for the step
  that produced it, and with it for everything queued before. This is why the loop never
  reads device values and the loggers do it (`to_floats`), and why the training cookbook's
  one-step-delayed logging (`DelayedLogger`) exists: read the *previous* step's numbers after
  enqueueing the current one, so the wait overlaps device work (ADR-0009 §2).
- **"Asynchronous" is at the level of the stream, not of the launching thread.** What the
  returning call has done is hand every kernel of the compiled program to the device's work
  queue (its *stream*). That hand-over is itself host work, done by the calling thread, and
  for a program of many small kernels it is not free (§5.2) **[deduced from the XLA runtime
  source and the measurements of §6; the JAX documentation describes the effect from the
  user's side only]**.

## 2. Where an array lives: committed and uncommitted

Every `jax.Array` is on some device. It is either *committed* there or not
**[cited: JAX, *Data placement*]**:

- `jax.device_put(x, device)` (or with a sharding) **commits** the array: "the array is
  committed … and it stays put. Computations follow their data." Combining arrays committed
  to *different* devices in one operation is an error — "Commitment is a promise JAX
  enforces."
- An array created without naming a device — `jnp.asarray(np_array)`, the output of a
  computation on uncommitted inputs — is placed "somewhere sensible" (the default device) and
  left **uncommitted**: "if it's used in a computation together with data that *is*
  committed somewhere, JAX moves it there implicitly." A computation whose inputs are all
  uncommitted runs on the default device; `jax.config.update("jax_default_device", d)` and
  the `jax.default_device` context manager set which one **[cited: JAX API reference,
  `jax.default_device`]**.

What it meant here. `data.to_device(dataset, device)` delivers *committed* batches (grain's
`device_put` stage), while `TrainerState` is created uncommitted on the default device. The
benchmark carried a committed/uncommitted pair of rows because our first reading of a bad
V100 table was that handing committed inputs to a jitted function next to an uncommitted
state cost host time. It does not: the dispatch floors are 0.7 ms for both and the steps are
identical (33.4 ms each on the V100, 2026-10-05) **[measured]**. The pair stays in the
script as a guard. Two practical rules survive: the device is chosen once, by the caller
(`to_device(dataset, device)` takes it explicitly), and the model and state are created
under `jax_default_device` set to the same card so nothing is moved implicitly.

*Which default, then?* `jax.devices()[0]` is JAX's own default device — the first device of
the default backend, "generally `'gpu'` or `'tpu'` if available, otherwise `'cpu'`"
**[cited: JAX, `jax.devices`]** — and it is the right choice exactly when the visible devices
are the right ones: a CPU machine, tests, or a job whose scheduler set
`CUDA_VISIBLE_DEVICES` (Slurm's `--gpus`/`--gres` do), so that the one visible card is the
assigned one. It is the wrong choice on a shared node where several cards are visible and
none assigned. `data.resolve_device(index=None)` encodes this: an explicit index wins;
otherwise JAX's default when one device is visible, the backend is CPU, or a
`*_VISIBLE_DEVICES` variable is set; otherwise a refusal listing the devices. The benchmark
script and the notebook go through it.

## 3. Threads, processes and the GIL

CPython's **global interpreter lock** lets only one thread execute Python bytecode at a time;
extension code may release it "when doing computationally intensive tasks", and it is always
released around I/O **[cited: Python glossary, *global interpreter lock*]**. A thread that
holds the GIL keeps it for the interpreter's *switch interval*, 5 ms by default, after which
a waiting thread may take over — and "which thread becomes scheduled at the end of the
interval is the operating system's decision" **[cited: `sys.setswitchinterval`]**. So Python
threads doing Python work do not run in parallel however many cores there are, and a thread
that needs the GIL back waits behind whichever runnable threads the OS happens to favor.

grain's data pipeline is Python. Its reader threads slice windows in Python; its
`ReadOptions` default to **16 threads** and a **500-element** prefetch buffer, and its own
docstring says: "If the data are already loaded in memory, we recommend setting this to 0 to
avoid Python GIL contention by multiple threads" **[cited: grain `options.py`]**. Iterating a
`MapDataset` directly, or `to_iter_dataset()` without options, uses those defaults.

Processes do not share a GIL. grain's `mp_prefetch` runs the pipeline in worker processes,
started with Python's `spawn` method: "The parent process starts a fresh Python interpreter
process", which re-imports the main module, hence the `if __name__ == "__main__"` guard
**[cited: Python, *multiprocessing*, the spawn start method and *Safe importing of main
module*]**. Two things follow for a JAX program. Anything that happens at import time
happens in every worker — our constraint primitives used to initialize a CUDA backend on
import, so each worker claimed GPU memory and died of `CUDA_ERROR_OUT_OF_MEMORY` (change
document, *Bugs fixed*; fixed by computing the shift lazily, ADR-0005 amended, and by the
script setting `JAX_PLATFORMS=cpu` when it runs as `__mp_main__`). And the batches still
have to cross back into the main process, where a Python thread receives them — so
`mp_prefetch` removes the slicing from the main process but not the GIL traffic of the
hand-over.

## 4. The instrument: what `scripts/bench_dataloader.py` measures

The script times one `Trainer.train_step` under controlled conditions and prints a table.
Every number is milliseconds per batch or per step, reported as **min / median** over
`--repeats` interleaved repetitions of `--steps` steps each, after a multi-step warm-up
(compilation, allocator, lazy CUDA state). Interleaving matters on a shared machine: a
configuration measured in one block can be flattered or punished by whatever else the node
was doing in that minute; the min is the best guess at the undisturbed cost, the median at
the typical one.

**The header** names the device (`--device`, an index into `jax.devices()`; the script never
chooses a GPU), the dataset and batch shape, and the process's CPU allowance (affinity,
cgroup quota, `SLURM_CPUS_*`), because §3's thread effects must be told apart from plain
core starvation.

**Two timings per step row.** For a loop of *n* steps over a loader,

- **dispatch** is the wall time for the Python loop to come back from the *n*-th
  `train_step` call, divided by *n* — the host clock of §1, including the loader's `next()`;
- **total** is the same, plus a final `block_until_ready` on the last step's loss — the time
  until the device has actually finished everything, divided by *n*.

With *n* = 30 the difference between the two is the device's backlog at the end of the loop
spread over the steps: dispatch ≪ total means the host ran far ahead (device-bound);
dispatch ≈ total means it could not (host-bound, launch-bound, or synchronized — §5).

**The dispatch floor.** A deliberately trivial jitted function over the *same* `(state,
batch)` pytrees — it adds one to the step counter and sums the batch — timed like a step.
Its cost is everything the host pays per call *before any kernel of the real step exists*:
`eqx.filter_jit`'s partitioning of the state's leaves, hashing the static parts, pushing a
few hundred array arguments through the dispatch path. It is reported for committed,
uncommitted and NumPy batches, so a difference between those three is visible here in
isolation (§2). If the floor is of the order of the step, the step's host time is Python
overhead and nothing inside the executable can fix it (§5.1); if it is small, as it has
always been here (0.7–1.3 ms), the host time is being spent *inside* the executable.

**Fetch rows** time the loader alone — `next()` on the window pipeline, no step: *fetch
only (host, 1 reader thread)* for `single_threaded(windows(...).batch(B))`, and *fetch only,
`mp_prefetch(k)`* with worker processes. They bound what any prefetching can achieve
(§5.5).

**Step rows**, each the same `train_step` over a different loader:

| Row | Loader | What it isolates |
|---|---|---|
| device-resident (committed) | a cycle of 8 batches `jax.device_put` to the device | the step itself, no fetch, no transfer — the reference |
| device-resident (uncommitted) | the same batches as `jnp.asarray` on the default device | placement effects (§2) against the row above |
| NumPy batches, 1 reader (option A) | `single_threaded(pipeline)`; `jit` transfers each batch synchronously | fetch + implicit transfer in the loop |
| NumPy + `device_put` in loop (option A) | the same with an explicit `jax.device_put` per batch | grain's "option A" literally, against the implicit path |
| NumPy batches, grain default readers | the `MapDataset` iterated bare: 16 reader threads, 500-batch buffer | what grain's defaults cost the launching thread (§3) |
| `to_device` (option C) | `data.to_device(pipeline, device)` | the production path: one prefetch thread, transfer, device buffer |
| `to_device` over cached batches | `to_device` over a dataset of 8 pre-fetched batches | the same threads and transfers with **no slicing work** — the control for the row above |
| `mp_prefetch(k)` + `to_device` | slicing in `k` worker processes, then `to_device` | whether moving the fetch out of the process pays |

Reading them pairwise is the method: *to_device* against *cached* separates the fetch's
CPU/GIL cost from the thread mechanics; *to_device* against *fetch only* says whether the
loader is the bound; *committed* against *uncommitted* tests placement; *1 reader* against
*default readers* tests the GIL.

**`--hlo-stats`** lowers and compiles the step for the device, then parses the optimized HLO
text: number of instructions; every `while` loop with its `known_trip_count` and launches
per iteration; the number of `conditional` executions per step (loop multiplicity included);
and an *estimate* of kernel launches per step (every instruction other than a parameter,
constant, tuple or bitcast counts one; loop bodies count once per trip; `call`s are inlined).
It also times a lone 2×2 `expm` against the closed-form rotation on the device — the
micro-benchmark that settled §6.2. The estimate is a census of the program, not a
measurement of the device, and the same program compiles differently for CPU and GPU, so it
is run on the target device.

**`--profile DIR`** wraps the `to_device` configuration in `jax.profiler.trace` for a
timeline in TensorBoard's profiler **[cited: JAX, *Profiling computation*]**. One caveat
worth knowing: XLA's GPU backend switches off command buffers while profiling unless told
otherwise (`--xla_enable_command_buffers_during_profiling`), so a traced run can differ from
the run it is meant to explain.

## 5. Five ways a step can be bound, and how to tell them apart

| Regime | Signature in the table | Where the time goes | Remedy |
|---|---|---|---|
| Python-bound (§5.1) | dispatch ≈ total; **floor ≈ step** | the loop's own Python per call | fewer leaves, bigger batches, less filtering |
| Launch-bound (§5.2) | dispatch ≈ total; floor small; step **flat in batch size**; launch count × ~5–20 µs ≈ step | the calling thread enqueueing thousands of small kernels | remove or unroll loops, closed forms, command buffers, larger batches |
| Synchronized (§5.3) | dispatch ≈ total; floor small; conditionals in the census; not explained by launch count | host waits for a predicate from the device at every `conditional` / unknown-trip `while` | remove `lax.cond` / dynamic `while` from the hot path |
| Device-bound (§5.4) | **dispatch ≪ total**; total scales with batch | the accelerator | the ordinary ones; this is the regime to be in |
| Fetch-bound (§5.5) | `to_device` ≈ *fetch only* while *cached* ≈ resident | the loader's Python, one batch at a time | faster sampler (vectorize), then worker processes |

Each regime leaves a distinct signature because the rows were chosen to make it so. *Floor*
below is the dispatch floor of §4.

### 5.1 Host-bound by Python

The loop's own Python — partitioning the pytree, hashing static arguments, the loader's
`next()` — takes longer than the device needs. Signature: dispatch ≈ total **and** floor of
the order of the step. Remedy: less Python per step (bigger batches, fewer leaves,
`eqx.filter_jit` over a plain `jax.jit` only where the filtering is needed). We never saw
this regime: the floor was 0.7–0.9 ms against steps of 30–90 ms.

### 5.2 Launch-bound

The compiled program consists of many small kernels, and the calling thread's cost of
enqueueing them exceeds the device's cost of running them. NVIDIA puts the per-launch
overhead at the microsecond scale and notes that "Modern GPUs are so fast that, in many cases
of interest, the time taken by each GPU operation … is now measured in microseconds. However,
there are overheads associated with the submission of each operation to the GPU – also at the
microsecond scale – which are now becoming significant" **[cited: NVIDIA, *CUDA Graphs*
blog; it measures 9.6 µs per short kernel including overheads]**. Signature: dispatch ≈
total, floor small, and the step **nearly flat in batch size** — a kernel over 2 048 points
launches as fast as one over 512. The `--hlo-stats` census (§4) counts the launches.

Where do thousands of launches come from in a model with a dozen layers? From **loops**. A
`lax.fori_loop`/`lax.scan` becomes an XLA `while`; on the GPU backend the runtime executes
its body once per iteration, re-launching the body's kernels each time **[cited: XLA
`while_thunk.cc` — when the trip count is known it runs `for (i < trip_count)` over the body
thunks]**. A `vmap` does not change this: it widens each kernel, it does not remove the
iterations. So a 48-iteration Newton solve with 4 kernels per iteration is ≈ 200 launches
per call, however many points it is vectorized over.

Remedies, in order of preference: remove the loop (a closed form, ADR-0010); fewer
iterations; `unroll=k` on the loop (k iterations per launch set); XLA command buffers (CUDA
graphs — "a mechanism to launch multiple GPU operations through a single CPU operation"
**[cited: NVIDIA, *CUDA Graphs*]** — `--xla_gpu_enable_command_buffer=+WHILE`, flag names
present in jaxlib 0.11; untested here); and simply larger batches, which are nearly free in
this regime.

### 5.3 Synchronization inside the executable

Some XLA operations cannot be enqueued blind: the runtime needs a value from the device to
decide what to enqueue next. On the GPU backend a **`conditional`** (what `lax.cond` and
`lax.switch` become) copies its predicate from device to host and *blocks the host until the
copy is done* before launching the chosen branch **[cited: XLA `conditional_thunk.cc`:
`stream.Memcpy(…, branch_index_address, …)` then `stream.BlockHostUntilDone()`]**. A `while`
whose trip count XLA could not determine does the same once per iteration **[cited: XLA
`while_thunk.cc`, the unknown-trip-count branch]**; a `fori_loop` with static bounds carries
a `known_trip_count` annotation and does not. Each such wait is a full round trip to the
device and back — tens of microseconds on a PCIe card — and, worse than the time, it
forbids the host from running ahead: the host must sit at every conditional until the device
reaches it, so dispatch ≈ total even when the device is the real limit.

Signature: dispatch ≈ total, floor small, step not flat in batch size *and* not explained by
the launch count. `--hlo-stats` reports the number of `conditional` executions per step
(loop multiplicity included).

Not every `cond` in user code ends up here — on CPU, and for small branches, XLA may fold a
conditional into a `select`, and the same program compiled for CPU showed far fewer loops
and conditionals than for GPU **[measured: 7 loops / 40 conditionals on CPU against 114 /
1 696 on the V100 for the same step before ADR-0010]**. Only the compiled program for the
target backend tells; that is why the census is run on the device in question.

### 5.4 Device-bound

The device's own work per step exceeds the host's. Signature: dispatch ≪ total, with the
total scaling with batch size. This is the regime a training loop *wants* to be in — the
accelerator is the bottleneck and everything else is hidden behind it. The remedies are the
ordinary ones (mixed precision, smaller model, fewer solver steps), none of which this
project has needed yet.

### 5.5 And the data: fetch-bound

Orthogonal to the four above: the loader cannot produce batches as fast as the step consumes
them. grain's training tutorial names the plain pattern — `jax.device_put` in the loop,
"option A" — as one where "the host blocks while the device receives data", and recommends
`grain.experimental.device_put` (option C: a CPU prefetch thread, the transfer, a
device-side buffer; "ThreadPrefetch -> map(jax.device_put) -> ThreadPrefetch") "for real
training" **[cited: grain, *JAX training tutorial*]**. Prefetching overlaps the fetch with the
step; it does **not** make the fetch faster, so the loop runs at max(step, fetch).
Signature: `to_device` ≈ *fetch only* while the same batches served from a cache (the
"to_device over cached batches" row) run at the device-resident speed.

## 6. Case history: three bottlenecks, found in the wrong order

The V100 numbers are from Joon's runs of `scripts/bench_dataloader.py --device 0` on
2026-10-04/05 (batch 512 × window 50, conjugacy model, 8 blocks); the CPU numbers from the
dev container.

1. **Reader threads (§3), found on CPU first.** The training loader was `windows(...).batch(B)`
   iterated bare — grain's 16 reader threads. The step's host time went from ≈ 15 ms with
   device-resident batches to ≈ 90 ms, dispatch ≈ total: the launching thread waited for the
   GIL behind sixteen slicing threads. One reader thread (`data.single_threaded`) under
   `to_device` brought it back to ≈ 22 ms **[measured, CPU]**. On the 24-core V100 host the
   same contrast is 361–400 ms against 93 ms — four times slower with cores to spare, which
   is what rules out CPU time-sharing and leaves the GIL **[deduced]**.

2. **`expm`'s conditionals (§5.3), found with the loop census.** With the data already on
   the device the step still took 88 ms with dispatch ≈ total and a 0.8 ms floor. The census
   gave 114 `while` loops and ≈ 1 700 `conditional` executions per step; 112 loops were
   "16 × 10", and 16 is `jax.scipy.linalg.expm`'s `max_squarings` — its squaring step is a
   `lax.scan` of 16 `lax.cond`s **[cited: JAX source, `jax/_src/scipy/linalg.py`,
   `_squaring`]**. `BiLipschitzLinear` built its two rotations with `expm(raw − rawᵀ)`: two
   per layer × 8 layers × (`__call__` and `inverse` both read `params`) × (primal and the
   Fréchet-derivative `expm` of the custom JVP) ≈ 112. A lone 2×2 `expm` measured **0.65–0.84
   ms per call** on the V100 against **0.05–0.07 ms** for the closed-form rotation — sixteen
   round trips of ≈ 40 µs **[measured]**; ≈ 1 700 of them per step is the 80 ms. The Cayley
   transform `(I − A)(I + A)⁻¹` replaced `expm` (ADR-0010): 0 conditionals, 2 loops, step
   **88 → 33 ms** **[measured]**.

3. **The fetch (§5.5), exposed by 2.** With the step at 33 ms, `to_device` sits at 93 ms —
   the same as *fetch only* (89 ms): one thread slicing 512 windows in grain's per-element
   pipeline takes 0.17 ms per window (`shuffle` → `random_map`, which constructs an
   `np.random.Generator` per element → `batch`'s `np.stack`), and the prefetch thread can
   hide the step behind the fetch but not the fetch behind the step. The same batches from a
   cache run at 33 ms. Fixed by C9: `WindowBatchSource`, a source whose element is a whole
   batch built by one fancy-indexed gather — 0.6 ms against 49 ms per batch on the dev CPU
   **[measured]**, with an epoch defined as every window once (ADR-0008 Decision 3). On the
   V100: fetch 88.5 → 0.7 ms, `to_device` 32.6 ms against 31.0 device-resident
   **[measured]** — the loader is out of the picture; the step (§5.2) is what remains.

Two hypotheses were raised and refuted on the way, and are kept as negative results: that
*committed* batches dispatch slower than uncommitted ones (§2; floors and steps identical),
and that the host was *CPU-starved* by loader threads (§3; 24 cores, same 4× slowdown).
The remaining step is launch-bound (§5.2) on the root solve's two loops (64 × 4 and 48 × 4
launches, ≈ 17 µs per launch) and nearly flat in batch size.

## 7. Rules adopted

- The loop never reads device values; loggers do (`to_floats`), one step late
  (`DelayedLogger`). ADR-0009 §1–2.
- The training loader is `to_device(window_batches(...), device)` — a batch-level source,
  so grain does no per-element work; a grain dataset is never iterated bare in the
  training process, and the `Evaluator` reads through
  `single_threaded`. No Python thread but `to_device`'s prefetch thread runs beside the
  launching thread; `mp_prefetch` is not in the default pipeline. ADR-0009 §1.
- The device is chosen by the caller, once; nothing auto-selects a GPU.
- No `lax.cond` or unknown-trip-count `while` in the model's hot path; loops with static
  bounds only, and as few as the mathematics needs. `tests/test_linear.py` pins the absence
  of conditionals in the linear layer; the census (`--hlo-stats`) is the check for anything
  new. ADR-0010.
- Importing the package must not initialize a JAX backend (worker processes re-import it);
  `tests/test_data.py` pins it. ADR-0005 amended.
- Before believing a table: interleaved repeats, min and median, dispatch and total, the
  floor, and the loop census — `scripts/bench_dataloader.py` is the instrument.

## References

- JAX, *Asynchronous dispatch* — https://docs.jax.dev/en/latest/async_dispatch.html
- JAX, *Data placement* (committed / uncommitted) — https://docs.jax.dev/en/latest/201/placement.html
- JAX, `jax.default_device` — https://docs.jax.dev/en/latest/_autosummary/jax.default_device.html
- JAX, *Profiling computation* — https://docs.jax.dev/en/latest/profiling.html
- JAX, *The training cookbook*, the training loop (one-step-delayed logging) — https://docs.jax.dev/en/latest/the-training-cookbook.html#the-training-loop
- JAX source, `jax/_src/scipy/linalg.py` (`expm`, `_squaring`) — https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/linalg.py
- XLA GPU runtime, `conditional_thunk.cc` — https://github.com/openxla/xla/blob/main/xla/backends/gpu/runtime/conditional_thunk.cc
- XLA GPU runtime, `while_thunk.cc` — https://github.com/openxla/xla/blob/main/xla/backends/gpu/runtime/while_thunk.cc
- grain, *JAX training tutorial*, moving batches to the accelerator — https://google-grain.readthedocs.io/en/latest/tutorials/jax_training_tutorial.html
- grain source, `grain/_src/python/options.py` (`ReadOptions`) — https://github.com/google/grain/blob/main/grain/_src/python/options.py
- Python, glossary: *global interpreter lock* — https://docs.python.org/3/glossary.html#term-global-interpreter-lock
- Python, `sys.setswitchinterval` — https://docs.python.org/3/library/sys.html#sys.setswitchinterval
- Python, *multiprocessing* — start methods and *Safe importing of main module* — https://docs.python.org/3/library/multiprocessing.html
- NVIDIA, *Getting Started with CUDA Graphs* — https://developer.nvidia.com/blog/cuda-graphs/
- D. A. Serino et al., *Fast-slow neural networks for learning singularly perturbed dynamical systems*, J. Comput. Phys. 537, 114090 (2025) — the bi-Lipschitz linear layer.
