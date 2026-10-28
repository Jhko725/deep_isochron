---
type: decision
id: ADR-0009
status: accepted; amended
updated: 2026-10-29
verified_by: joon (review round 1, 2026-10-04); round 2 pending
---

# ADR-0009 — Trainer: injected logging and checkpointing, losses as weighted terms, schedules in the state

## Context

### Motivation

The first trainer owned everything. The code before the decision
(`git show fef302e:src/deep_isochron/training/trainer.py`) shows each failure:

- the Orbax checkpoint directory named from `datetime.now()` (`:114`);
- `wandb.init` inside the trainer (`:145`), so the loop could not run without wandb;
- the "best" checkpoint selected by `BestN` on the logged *training* loss (`:150-151`),
  saving only the model's weights, so a run could not resume (`:183`, `:198`);
- a one-step-delayed logging scheme interleaved with the loop body and duplicated in a
  final flush (`:183` and `:198`);
- the two losses each carried their own schedule logic (`PhaseAutoencoderLoss` switched
  its weights per batch), so a loss was not a pure function and its state was not
  checkpointed.

### Forces

Phase C had to make the loop testable without wandb or a filesystem, make runs
resumable, evaluate on held-out data, and let the trainer, not the loss, own curricula.
JAX dispatches asynchronously, and the training loop is host-sensitive: any Python
thread that holds the GIL next to the launching thread slows every step.

## Decision

### 1. The step is pure; the loop injects its side effects

`Trainer(optimizer, loss, schedule)` defines a step; `TrainerState` (model, optimizer
state, step, key, schedule state; optimizer and trainable filter static) is everything
that changes; `train_step(state, batch) -> (state, metrics)` is a jitted pure function.
`train(model_or_state, loader, *, num_steps, key, logger, checkpointer, evaluate, eval_every)`
is the loop, with `Logger` and `Checkpointer` **abstract base classes** (`NullLogger`,
`NullCheckpointer` for tests; `PrintLogger`, `ListLogger`, `WandbLogger` wrapping a run
the caller created; `OrbaxCheckpointer` taking a directory the caller chose) and an
`evaluate(model) -> {name: float}` callable. The loop never reads a device value: the
logger decides when to synchronize. The base classes carry the lifetime logic. `close()`
is a no-op unless overridden, and both are context managers (`with log, ckpt:`; exit
calls `close`), so they close, and the `DelayedLogger` flushes, even when a step raises;
`close()` remains for callers managing lifetimes themselves.

The loop does **not** move batches to the device. grain's JAX training tutorial calls a
`jax.device_put` in the loop its option A, where "the host blocks while the device
receives data", and recommends its option C, `grain.experimental.device_put` (a CPU
prefetch thread, the transfer, and a device-side buffer) "for real training".
`data.to_device(dataset, device)` wraps that, with the device given by the caller. There
is no auto-selection on a shared cluster; `data.resolve_device()` is the one sanctioned
default, JAX's own `jax.devices()[0]` when the scheduler scoped the process to a card
and a refusal otherwise. The trainer accepts whatever the loader yields; NumPy batches
still work, since `jit` transfers them synchronously, which is fine for tests.

**Loaders must not start grain's default reader pool** (measured on CPU in
[`bench_dataloader.py`](../../scripts/diagnostics/bench_dataloader.py); the mechanics of
asynchronous dispatch, the GIL and the benchmark's rows are explained in
[`training-step-performance.md`](../design/training-step-performance.md)). The training
loop is host-sensitive: its cost per step is the Python dispatch of one jitted call, ≈15
ms here. Iterating a `MapDataset` directly, or `to_iter_dataset()` with default
`ReadOptions`, starts 16 reader threads and a 500-element prefetch buffer; those threads
contend with the loop for the GIL and the dispatch alone grew to ≈90 ms, with the device
idle (`dispatch == total`). With one reader thread (`data.single_threaded`,
`ReadOptions(num_threads=1, prefetch_buffer_size=1)`) under `to_device`'s two
single-thread prefetch stages the step returned to ≈22 ms. `to_device` and the
`Evaluator` read grain datasets this way; a training loader is
`to_device(windows(...).batch(B), device)`, never the `MapDataset` itself. Deduced from
the measurement and grain's `ReadOptions` defaults, not from a documented grain
recommendation.

**GPU outcome (V100, 2026-10-05; C7 closed).** `scripts/diagnostics/bench_dataloader.py --device 0`,
batch 512 × window 50, conjugacy model, 30 steps × 3 interleaved repeats: the dispatch
floor is 0.8 ms for committed, uncommitted and NumPy batches alike, so handing the
pytrees to JAX costs nothing and the earlier "committed inputs are slow" pattern was the
old script's artifact; the state is **not** committed at `Trainer.init`. Device-resident
batches step in 92 ms, `to_device` in 94 ms: the loader-side prefetch keeps up and is
the training-run path. NumPy batches iterated from the bare `MapDataset` take 488 ms,
which is grain's 16 reader threads next to the launch thread, the round-2 finding again
(reproduced 5× on CPU), and `mp_prefetch(4) + to_device` 464 ms although the fetch alone
drops 27 → 7 ms; at batch 2048 even the single-thread `to_device` stopped hiding the
fetch (356 ms vs 122 resident + 104 fetch). The host has 24 cores, so this is not
time-sharing: it is the GIL. A Python thread slicing windows holds the interpreter in 5
ms slices, and the launching thread, which takes the GIL back several times per step,
waits behind every runnable Python thread each time. So no Python thread but
`to_device`'s one prefetch thread runs in the training process, and `mp_prefetch` is not
in the default pipeline. The step itself is host-bound (dispatch ≈ total, ≈ 80 ms fixed
+ 10 ms per 512 points), a model/XLA matter, not the loader's (roadmap C8).

- *Rejected*: committing the `TrainerState` to the device at `init`. The floors show no
  committed/uncommitted difference to fix.
- *Rejected*: `mp_prefetch` in the default pipeline. Measured 5× slower end to end on
  the V100 even though the fetch itself is faster; its main-process reader thread
  contends for the GIL like any other.

- *Rejected*: the trainer constructing the wandb run and the run directory. Both belong
  to the experiment script (Hydra owns directories, Phase D), and tests must not need
  them.

### 2. One-step-delayed logging, as a logger adapter

JAX dispatches asynchronously; reading the current step's metrics (`float()`) blocks the
host before the next step is enqueued, and with a lazily evaluated grain pipeline the
host then also spends the batch-slicing time (~15 ms measured) with the device idle. The
training cookbook's remedy is to log the *previous* step's metrics after enqueueing the
current one (`RecordWriter`). We keep it, as `DelayedLogger(inner)`: one pending record,
forwarded on the next call, flushed on `close()`. The loop body stays free of it.

- *Rejected*: dropping the delay for `log_every`. On CPU the two are indistinguishable
  (measured), but the mechanism matters on an accelerator whenever the loader does not
  prefetch in the background; the adapter costs nothing and composes with `every`.
- *Noted*: the real fix for the data-fetch cost is prefetching in the pipeline
  (`mp_prefetch`, `ThreadPrefetchIterDataset`); the trainer accepts device batches as
  they come.

### 3. Losses are weighted sums of named terms; weights are an argument

`AbstractLoss.terms(model, batch) -> (terms, aux)`: `terms` holds **exactly** the
weighted scalars, keyed by `weight_names` (enforced), `aux` the unweighted diagnostics
(`final` error, learned `omega`/`kappa`);
`__call__(model, batch, weights=None) -> (Σ wᵢ·termᵢ, terms | aux)`. The building blocks
(`trajectory_mse`, `final_mse`, `step_weighted_consistency`, `alpha_schedule`,
`batch_center_of_mass`) are plain functions. The weights reach the loss from the trainer
every step, so they are logged (`w/<name>`) and scheduled outside the loss.

`final` is a diagnostic, not a loss term: the whole window should fit, but the error at
the window's end exposes a frequency drift that the mean hides.

- `default_weights` is an `init=True` static field (an instance may override it; equinox
  warns about float leaves in `init=False` fields), a deliberate exception to
  [ADR-0004](0004-abstractvar-fields.md)'s `init=False` rule, documented on the class.

### 4. Schedules live in the trainer state

`AbstractLossSchedule[S]` with `init() -> S`, `weights(state: S, step)`,
`update(state: S, step, terms) -> S`, the state type `S` a type parameter (`None` for
the stateless schedules, a boolean scalar for the switch), all inside the jitted step,
so that the state is in `TrainerState` and survives a checkpoint. The schedules are
`Constant`; `StepSchedule(fn)` for curricula (any traceable `step -> weights`, optax
schedules per weight); and `ThresholdSwitch(before, after, thresholds)`, Yawata et al.'s
once-and-for-all switch (Sec. IV.A), which with the exact chart reaches a circular phase
error of 0.017 rad where the per-batch approximation reached 0.08.

### 5. Evaluation on held-out data, through the model contract

`Evaluator(val_batches, reference)` works on any `AbstractPhaseAmplitudeModel`:
`val/mse` (the checkpoint-selection metric, Joon's decision) and `val/final_mse`; with
`t_split=…`, `val/mse_early` / `val/mse_late`, the same error over windows starting
before / at-or-after `t_split`, because the overall mean is dominated by near-cycle
windows and selects for the cycle (`checkpoint.metric` can point at `val/mse_early`;
design document [`validation-split.md`](../design/validation-split.md)); learned
`period` and `kappa` for both model kinds; the normal form's parameters and, on a grid
over the validation data's bounding box (Joon's decision), the bijection's round-trip
error and Jacobian singular values (per layer for a `SequentialINN`) for the conjugacy
model; and against a reference normal form the phase's circular standard deviation (both
orientations) and the amplitude's `|corr|`, the comparisons that the design document's
§5.4 says are meaningful (phase up to a constant, amplitude up to scale). The validation
data is a **finite, deterministic, re-iterable** dataset,
`validation_windows(source, length, stride)`: every window at fixed starts of every
held-out trajectory, no shuffle, no repeat, iterated to exhaustion on each call
(levanter's eval-loop shape). So every evaluation sees the same windows and `val/mse` is
comparable across steps. The metrics are accumulated as sufficient statistics over
batches (sums of `exp(iΔ)` and second moments), so nothing is held in memory and the
grid's bounding box comes from a first pass.

`Trainer.train(stop_on_nonfinite=True)` raises `FloatingPointError` at the first
evaluation with a non-finite metric, before that state reaches the checkpointer, so that
the last checkpoint on disk is finite, which is what
[`find_nan.py`](../../scripts/diagnostics/find_nan.py) needs; an empty validation bucket's `nan` (§5
split) is exempt.

`Evaluator` also reports the **latent extent** over every validation sample:
`latent/amp_min|amp_max` (the model's amplitude), for conjugacy models
`latent/r_min|r_max` (the latent radius `|H(x)|`) and `latent/nonfinite_image` (fraction
of samples whose image `H(x)` is not finite), and `latent/nonfinite` (fraction of
non-finite amplitudes; with a finite image this means that the chart overflowed). These
are the detector for a learned map that sends data toward the latent origin, where the
normal form's closed forms overflow, or far outside the cycle, where `Ψ` saturates and
the chart's inverse is ill-conditioned, which is the mechanism behind the `nan` runs
against Winfree's $a = 0$ member ([ADR-0013](0013-normal-form-basins.md) §5).

- *Rejected*: streaming `n` batches from the infinite shuffled training-style loader.
  Different windows each time, so `BestN` would select on sampling noise.

**Why the circular standard deviation for the phase.** The learned phase is determined
up to a constant (design document §5.4) and the difference lives on the circle, so an
MSE of angle differences needs the offset removed and breaks at `±π`. `sqrt(−2 ln R̄)`,
`R̄` the mean resultant length (Mardia & Jupp, *Directional Statistics*, 1999; SciPy's
`circstd`), is invariant to both and equals the ordinary standard deviation for small
spread, the RMS phase error in radians. For FitzHugh–Nagumo only prediction error and
period are available until Phase E.

### 6. Whole-state checkpoints; the caller owns the directory

`OrbaxCheckpointer(directory, save_every, metric, custom_metadata)` saves the array
leaves of the whole `TrainerState` with `save_async` under `FixedIntervalPolicy` and
keeps `AnyPreservationPolicy([LatestN(1), BestN(metric)])`; `restore(template, step)`
rebuilds the state via a `ShapeDtypeStruct` tree (with sharding) and `eqx.combine`;
`train` accepts a restored state to resume. Orbax's `BestN(reverse=False)` keeps the
*largest* metric, so minimization passes `reverse=True`.

## Consequences

- `scripts/training/` deleted; Phase D rebuilds the entry point on this contract.
- `tests/test_training.py` runs the whole loop with `NullLogger`/`ListLogger` and a
  temporary Orbax directory; the slow B12 test is a `Trainer` run.
- `ConjugacyTrajectoryLoss` weights `data` and `latent` and reports `final`; its data
  term is the squared Euclidean norm per point (formerly the per-coordinate mean, a
  factor `dim`).
- Training runs put the transfer in the loader (`to_device`); `validation_windows` is
  the validation source.

## Alternatives rejected

- *Superseded 2026-10-04*: saving the model only (the previous behavior): not resumable,
  and the schedule state would be lost.
- *Superseded 2026-10-04*: schedule state inside the loss module (the B12 per-batch
  switch). A loss is then not a pure function of `(model, batch)` and its state is not
  checkpointed.
- *Superseded 2026-10-04* (review round 1): terms and diagnostics in one dictionary with
  `weight_names` selecting a subset. Implicit; a reader could not tell a weighted term
  from a logged number.
- *Superseded 2026-10-04* (review round 1): a list of validation batches collected once.
  Correct, but it holds the set in memory and bypasses the pipeline; kept only as
  `collect_batches` for convenience.
- *Superseded 2026-10-04* (review round 2): `Logger`/`Checkpointer` as `Protocol`s with
  a shared `_Closing` mixin for the context-manager methods (round 1's shape). Every
  concrete class had to inherit the mixin anyway, so the protocol only restated what the
  mixin provided and the mixin was duplicated in two modules; a base class with the
  methods baked in is the same contract with one definition.

## History

- 2026-10-04 (`trainer`): accepted; decisions taken with Joon on 2026-10-04.
- 2026-10-04 (`trainer`, review round 1): loader-side transfer, finite validation
  dataset, context managers, terms/aux split, typed schedule state.
- 2026-10-04/05 (`trainer`, review round 2): `Logger`/`Checkpointer` as base classes;
  loaders must not start reader pools; the GPU outcome (C7 closed); `resolve_device`.
- 2026-10-06 (`experiment-config`): `Evaluator(t_split=…)` reports `val/mse_early` /
  `val/mse_late`.
- 2026-10-14 (`normal-forms-basin`): `Trainer.train(stop_on_nonfinite=True)`;
  `Evaluator` reports the latent extent.
- 2026-10-29 (`docs-layout`): links follow the scripts into `scripts/diagnostics/`.
