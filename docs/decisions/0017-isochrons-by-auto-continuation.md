---
type: decision
id: ADR-0017
status: accepted (review pending)
updated: 2026-10-30
verified_by: pending (Joon; the decisions were taken in discussion 2026-10-27, plan `claude/isochron-auto-plan-2026-10-22`)
---

# ADR-0017 — Isochrons by AUTO-07p continuation of a rotating multi-segment boundary value problem

## Context

### Motivation

- The learned conjugacy carries the normal form's isochrons back to the observed system,
  and nothing in the package computed the observed system's own isochrons to check them
  against. Roadmap O7 asks for them on demand, as a result that the comparison with the
  learned isochrons loads instead of recomputing (plan decision H7).
- Andrus Giraldo had reproduced the FitzHugh–Nagumo isochrons of Langfield et al.
  (2014) in AUTO-07p, in a script and a hand-written Fortran file for that one system
  ([`src/Isochron2PBVP`](https://github.com/Jhko725/deep_isochron/tree/3af99f2/src/Isochron2PBVP)
  at commit `3af99f2`; the directory is the coworker's working copy, outside the
  project's checks). One phase took 140–290 s on two cores,
  and the 100 phases ran overnight in sequence.
- His bundle integrated $+\tau A^{\mathsf T}w$, which is not the adjoint equation. On
  the Hopf and Bautin forms it walks the multiplier from 1.2 to $-8.7$ without a branch
  point, and on FitzHugh–Nagumo its end direction is $58°$ from the stable direction
  ([`docs/design/isochrons.md`](../design/isochrons.md) §3.1).
- His phase origin was the crossing of the cycle with the line $y = 0.5$, a condition
  written for FitzHugh–Nagumo. The package's cycles carry their own phase origin,
  `LimitCycle.x0s[0]`, and their own section through it
  ([ADR-0014](0014-limit-cycle-by-shooting.md)).
- AUTO-07p is a Fortran program with a Python layer that is not on PyPI, and it compiles
  each equation file with `gfortran` when it runs. Neither the package nor its CI had
  either.

### Forces

The isochrons are an oracle for learned ones, so they have to be accurate to well below
the error of a trained model, and their provenance has to be recorded with them. The
method has to work for every planar system the package declares, from the one SymPy
definition of [ADR-0016](0016-symbolic-systems.md), with no Fortran written per system.
A foliation of 100 phases is hours of computation, so it is saved once and loaded many
times, and the phases are independent. AUTO writes its files into the working
directory, which belongs to the whole process. The package runs on Joon's Linux cluster
and on WSL2, without `sudo` on the cluster.

## Decision

### 1. The method: the rotating multi-segment set-up of Hannam et al. (2022)

One generic Fortran file,
[`isochron_multiseg.f90`](../../src/deep_isochron/analysis/ode/auto/fortran/isochron_multiseg.f90),
states the boundary value problem in four nested sizes, selected by `NDIM`, and
[`stages.py`](../../src/deep_isochron/analysis/ode/auto/stages.py) runs it in four
stages ([design document](../design/isochrons.md) §3), each a function of a `Runner`
and the previous stage's `AutoSolution`
([`runner.py`](../../src/deep_isochron/analysis/ode/auto/runner.py)), which
[`foliation.py`](../../src/deep_isochron/analysis/ode/auto/foliation.py) strings
together:

1. **Two segments** (`NDIM = 4`): O1's cycle, cut at half its period.
2. **The adjoint bundle** (`NDIM = 8`): $\dot w = -\tau A^{\mathsf T}w$ on both
   segments, continued in the multiplier $\lambda$ from 1.2 to the branch point at
   $\lambda = 1$, then switched onto the bundle normalized to $\vert w\vert = 1$ at the
   junction. The rotation $Jw$ is the stable direction there.
3. **The rotation** of the junction: the length of segment 2 is continued to
   $\tau_2 = (1 - \theta/2\pi)T$ for every requested phase $\theta$ (since 2026-10-30
   $\tau_2 = (1 - \Theta)T$ for a phase $\Theta$ in turns, ADR-0020), which moves the
   base point and its bundle together around the cycle.
4. **The isochron** (`NDIM = 10`, per phase): a segment of `n_periods` periods that ends
   at the base point plus $\eta Jw$, continued in $\eta$ in both directions; its start
   traces the isochron of the base point.

Each isochron has its own base point and its own stable direction from the bundle
(decision D5), rather than Langfield's single base point with segments of
$(k + 1 - \vartheta)T$. The isochron segment has a fixed number of periods (decision
D4); Langfield's staged extension, which appends a period when a curve needs to reach
further, is not built. The method is planar: the end direction $Jw$ is a planar
construction.

### 2. The bundle integrates the adjoint equation

$\dot w = -\tau A^{\mathsf T}w$ (decision D1). Its solutions keep $w\cdot F$ constant
along the cycle, and the one with multiplier 1 is parallel to the gradient of the
asymptotic phase. AUTO's bundle stays, rather than O2's Floquet vectors passed in as
data (decision 11): it is computed on AUTO's own mesh, and it rotates with the base
point in stage 3.

### 3. Phase 0 is the cycle's own origin, anchored on its section

Segment 1 starts on O1's section through `x0s[0]`:
$(u_1(0) - p)\cdot n = 0$ is the boundary condition `FB(6)` (decision 3), with $p$ and
$n$ in parameter slots. A caller moves phase 0 with `reanchor(cycle, point)` before
tracing. Phases are radians in $(-\pi, \pi]$, increasing with time from that origin
(decision 1; since 2026-10-30 fractions of the period in $[0, 1)$, ADR-0020, and the
netCDF `phase` variable carries `units = "turns"`, which `load` requires).

### 4. AUTO is a locked dependency, its Fortran tree built by a script

AUTO's Python layer is the dependency group `auto` of `pyproject.toml`, from git at a
commit that `uv.lock` pins, and the `dev` group includes it (decision 4).
[`scripts/install_auto.sh`](../../scripts/install_auto.sh) builds the Fortran tree at
the same commit with a serial `make`; `just setup` and `just install-auto` call it. The
tree is found through `DEEP_ISOCHRON_AUTO_DIR`, then AUTO's own `AUTO_DIR`, then
`~/.local/opt/auto-07p`, and never through an argument: it is a property of the machine,
not of a computation (R2). The machine needs `gfortran` whatever installs AUTO, because
AUTO compiles each equation file at run time (R1).

### 5. The phases run in a pool of spawned processes

Stages 1–3 run once in the calling process; stage 4 runs for each phase in a
`ProcessPoolExecutor` started with `spawn`, one temporary working directory per worker
(decisions 17 and G2). Processes, because AUTO writes
into the working directory of its process; `spawn`, because JAX runs threads and a
forked copy of a threaded process can deadlock. The pool has
`os.process_cpu_count()` workers unless `IsochronSettings.max_workers` says otherwise
(H3).

Every AUTO run sets `OMP_NUM_THREADS=1`, since AUTO's OpenMP
build otherwise starts one thread per core, which on a 24-core node kept stages 1–3 of
one phase busy for 8.5 minutes. Each worker also runs JAX on the CPU, since a worker
that started a GPU backend preallocated most of the GPU's memory.

A phase that fails comes back as an `IsochronFailure` with AUTO's last 50 lines
of output and its run constants; the other phases go on (G4).

### 6. A foliation is one netCDF file, loaded against its cycle

`IsochronFoliation.save` writes one netCDF file through `h5netcdf`: the isochrons
concatenated along one dimension and delimited by offsets, the cycle's mesh, period
and section, and the system, settings, failures and provenance (the git state, the
AUTO commit, the hash of the Fortran sources, the compiler and its flags) as JSON
attributes (decision 16). `save` refuses an existing file.
`IsochronFoliation.load(path, cycle)` takes the cycle the foliation was traced for,
because the file stores the cycle as data and not as a `LimitCycle` with its system and
solver, and it refuses a cycle of another system, other parameters, another period or
another phase origin, listing every mismatch (L1).

## Consequences

- A planar system declared in SymPy gets its isochrons with no Fortran of its own.
  The pipeline's bundle branches at multiplier 1 on FitzHugh–Nagumo, Hopf, Bautin and
  Winfree, and is parallel to O2's Floquet vectors and to the closed-form phase
  gradient to $10^{-11}$.
- The FitzHugh–Nagumo isochrons reproduce Andrus's files 26, 51 and 100 within
  $2\times 10^{-5}$ in both directions. The Bautin and Winfree isochrons carry their
  closed-form phase to $4\times 10^{-10}$.
- A FitzHugh–Nagumo foliation with the default settings costs 47 s for stages 1–3
  and about 290 s of stage 4 per phase on one core. On the two-core session, four
  phases took 1208 s with one worker and 700 s with two, a speed-up of 1.8 on stage 4.
  By that rate the 100 phases of the coworker's overnight run would take about 20
  minutes on a 24-core node **[deduced]**, provided every AUTO run is single-threaded
  (§5 and the History line of 2026-10-28).
- An isochron that needs to reach further than its `n_periods` allow stops at
  `"max_steps"` and says so (D7); the staged extension is the remedy if it is ever
  needed.
- Systems with more than two states are out of reach: their isochrons are surfaces,
  and their stable directions a subspace.
- Every machine that traces isochrons needs `gfortran`, `make` and `git`, and a built
  AUTO tree; the `auto` tests skip without one, and CI's `auto` job builds it from
  cache.
- Bumping AUTO means changing the pin in `pyproject.toml`, relocking, and rebuilding
  the tree; the `auto` job gates the change.

## Alternatives rejected

- **Langfield's single base point** (Langfield et al. 2014, §II.B): one stable
  direction at phase 0, and segments of $(k + 1 - \vartheta)T$ for the other phases.
  It needs the stable direction at one point only, but every phase other than 0 is then
  traced through a longer segment, so the phases are treated unequally; with the
  corrected bundle, every base point has an accurate direction of its own (D5).
- **The coworker's $+A^{\mathsf T}$ bundle.** A sign slip, not a method: no branch point
  on the normal forms (D1).
- **O2's Floquet vectors as the stable direction**, passed to AUTO as data. They are
  computed on O1's shooting mesh, not on AUTO's, and would not rotate with the base
  point; deferred until O4's $Z(\theta)$ exists (decision 11).
- **Phase 0 on a fixed line** such as $y = 0.5$, as the coworker had. It is particular
  to one system, and it is ill-conditioned where the line meets the cycle at a small
  angle; the section through `x0s[0]` is transversal by construction (decision 3).
- **AUTO from conda-forge or pixi.** pixi installs native dependencies as well as
  Python ones, but the machine still needs `gfortran` for AUTO's run-time compiles, and
  one native dependency does not justify a second environment manager (R1).
- **An f2py extension for the generated Fortran**, which would also have tested
  `rhs.inc` from Python. The meson build could not find the included files; a
  `gfortran` driver tests the same thing (decision 6).
- **Threads, or `fork`.** Threads share one working directory; `fork` copies JAX's
  threads into the child (G2).
- **A cache of stages 1–3 on disk.** They take about a minute, against minutes per
  phase in stage 4 (decision 10).
- **Interpolating the stage-3 solution for the stage-4 start**, by Lagrange evaluation
  within AUTO's mesh intervals or by the coworker's SciPy splines. Stage 4 starts on
  AUTO's own grid, with no interpolation and no SciPy (D2).
- **A point-to-point tolerance for the regression oracle**, the plan's $2\times 10^{-3}$.
  It measured the spacing of the points, not the distance between the curves; the
  oracle compares each file's points with the other curve's polyline in both directions,
  at $2\times 10^{-5}$ (D3).
- **Langfield's staged extension of an isochron** beyond one segment. Stage 4 keeps the
  coworker's single segment of `n_periods` periods; the extension is deferred until an
  isochron needs it (D4).
- **A mesh independent of the number of periods.** `ntst` defaults to `60 * n_periods`,
  so a longer segment keeps its resolution per period (D6); the sweep of 2026-10-27
  found 30 per period as good at half the time, and the default is Joon's to settle.

## History

- 2026-10-28 (`auto-isochrons`): accepted.
- 2026-10-28 (`auto-isochrons`): every AUTO run, in the calling process as in the
  workers, sets `OMP_NUM_THREADS=1`; the workers run JAX on the CPU.
- 2026-10-28 (`auto-isochrons`, review of Steps C–E): `isochron.py` split into
  `results`, `runner`, `stages` and `foliation`; every stage returns an
  `AutoSolution`, and only the branch switch of stage 2 restarts from AUTO's own
  point.
- 2026-10-29 (`docs-layout`): D2, D3, D4 and D6 moved here from the design document's
  retired *Decisions* section.
- 2026-10-29 (`isochron2pbvp`): the coworker's directory restored from his last commit
  `d3edbfe` as `isochron2pbvp/`, a project of its own in this repository; the links stay
  pinned at the version compared.
- 2026-10-30 (`phase-turns`, ADR-0020): phases in turns; the saved `phase` carries `units = "turns"`
  and `load` refuses a file without it; the shipped hashes repinned (`fhn`
  `d9ded7af → 11305875`, `fhn-easy` `b688cbfe → fb823a3f`).
- 2026-10-30 (`auto-docs`): AUTO's `EPSL`, `EPSU`, `EPSS` are `IsochronSettings`
  fields written into the constants file (defaults the coworker's), so they enter the
  hash: `fhn` `11305875 → 1d0ffe43`, `fhn-easy` `fb823a3f → defa5f18`.
