---
type: guide
status: current
updated: 2026-10-30
verified_by: "pending: the tests deriving the slot table and the per-stage constants from the code land later on auto-docs"
sources:
  - "src/deep_isochron/analysis/ode/auto/codegen.py"
  - "src/deep_isochron/analysis/ode/auto/runner.py"
  - "src/deep_isochron/analysis/ode/auto/stages.py"
  - "src/deep_isochron/analysis/ode/auto/fortran/isochron_multiseg.f90"
  - "isochron2pbvp/fncMultiSeg/ (Andrus Giraldo's reproduction: fncMultiSegIso.f90, c.fncMultiSeg, mainIsochronMultiSegFNC.py)"
  - "AUTO-07p manual, chapter 'Description of AUTO-Constants' (doc/auto.tex of the installed tree)"
---

# Reading the pipeline's AUTO runs

This guide is for someone who knows AUTO-07p and wants to see exactly what the isochron
pipeline hands it: which files it generates, which `PAR` slot holds what, which
constants every run starts from and what each stage changes. The method itself, and why
the stages are what they are, is [`design/isochrons.md`](../design/isochrons.md) §3;
installing AUTO is [`auto07p.md`](auto07p.md), and computing a foliation from a config
is [`isochrons.md`](isochrons.md). The code is
`src/deep_isochron/analysis/ode/auto/`: `codegen.py` writes the files, `runner.py` runs
AUTO, `stages.py` holds the four stages.

Andrus Giraldo's reproduction of Hannam, Krauskopf and Osinga's set-up, in
`isochron2pbvp/fncMultiSeg/`, is the pipeline's ancestor, and the tables below give his
numbering beside ours so that the two can be read against each other.

## The generated files

`write_problem(ode, workdir)` lays four files into a work directory, named after the
equation name (`isochron` by default):

| File | What it is | Where it comes from |
|---|---|---|
| `isochron.f90` | the equation file: `FUNC`, `BCND`, `PVLS` and `STPNT` for the multi-segment boundary value problem in its four sizes `NDIM` 4, 8, 10 | `fortran/isochron_multiseg.f90`, shipped as package data, copied verbatim |
| `rhs.inc` | the body of `SUBROUTINE RHS`: the system's `F(i)` and exact Jacobian `A(i,j)`, printed by SymPy | `ode.symbolic()` |
| `indices.inc` | one `INTEGER, PARAMETER :: IP_<NAME> = <slot>` per pipeline slot | `internal_indices(n_params)` |
| `c.isochron` | the base constants file that every run starts from | `constants_file` |

`scripts/diagnostics/dump_auto_problem.py isochrons/fhn <dir>` writes the four files
of a config into a directory and prints its slot table, without running AUTO.

The equation file is the same for every system; the two includes carry everything
system-specific, so a reader who wants to see what AUTO integrates reads `rhs.inc`, and
one who wants to know which slot a `PAR(k)` in the Fortran refers to reads
`indices.inc`. `AutoProblem.fortran_sha`, the hash of the three Fortran files together,
is saved with every foliation as provenance.

## The parameter slots

The system's parameters occupy `PAR(1..n)` in the order of its symbolic definition
(`parameter_map`), and the pipeline's own slots form one block starting at
$b = \max(n, 10)$ (`internal_indices`; the reasoning is
[ADR-0016](../decisions/0016-symbolic-systems.md) §5). For every system of up to
ten parameters the block therefore starts at `PAR(11)`, the slot AUTO's `IPS = 2` stage
reserves for the period, and the slots between the last parameter and `PAR(11)` are
unused: FitzHugh–Nagumo has four parameters, so `PAR(5..10)` are empty. A system of more
than ten parameters would push the block up, and `internal_indices` raises instead.

For FitzHugh–Nagumo (`a`, `b`, `c`, `z`):

| Slot | `Idx` attribute, Fortran constant | Meaning | Set by | Andrus's slot |
|---|---|---|---|---|
| 1–4 | — | the system's parameters `a`, `b`, `c`, `z` | `auto_pars` from the instance | 1–4 |
| 5–10 | — | unused | — | 5–10 |
| 11 | `period`, `IP_PERIOD` | the period $T$ | stage 1 from the cycle; free in every stage | 11 |
| 12 | `seg1`, `IP_SEG1` | the integration time of segment 1, `U(1:2)` | stage 1, $T/2$; free in every stage | 12 |
| 13 | `seg2`, `IP_SEG2` | the integration time of segment 2, `U(3:4)` | stage 1, $T/2$; stage 3 continues it to $(1 - \Theta)T$ | 13 |
| 14 | `copies`, `IP_COPIES` | the integration time of the isochron segment `U(9:10)`, `n_periods` periods | stage 4's start | 14 |
| 15 | `mult`, `IP_MULT` | the multiplier of the adjoint bundle between the segments | stage 2, from 1.2 down to 1 | 34 |
| 16 | `norm`, `IP_NORM` | $\vert w(0)\vert^2$ of the bundle at the junction | stage 2, 0 to 1 | 35 |
| 17 | `eta`, `IP_ETA` | the distance $\eta$ along the stable direction at the end of the isochron segment | stage 4's continuation parameter | 36 |
| 18–19 | `section_point`, `IP_SECTION_POINT_X/Y` | the point of the cycle's section through phase 0 | stage 1 from the cycle | — (his anchor is $y = $ const in `FB(6)`) |
| 20–21 | `section_normal`, `IP_SECTION_NORMAL_X/Y` | the normal of that section | stage 1 from the cycle | — |
| 22–23 | `base`, `IP_BASE_X/Y` | the base point `U(3:4)` at $t = 0$, printed by `PVLS` | AUTO, at every stage-4 point | 72–73 |
| 24–25 | `point`, `IP_POINT_X/Y` | the traced isochron point `U(9:10)` at $t = 0$, printed by `PVLS` | AUTO, at every stage-4 point | 74–75 |

`NPAR` is 36, AUTO's default (`MIN_NPAR`), which covers the block; Andrus's file has
`NPAR = 80` for his slots up to 75 and `PAR(80)`, the arclength `GETP('STA')`, which the
pipeline does not print. Slots 22–25 are output: `PVLS` writes them at every step and
stage 4 reads the traced isochron from the branch's `PAR(24)`, `PAR(25)` columns. Every
other slot is input, set on the start solution's `PAR` dict and handed to `run` as the
`PAR` keyword.

## The base constants file

`constants_file` writes `c.isochron` for a planar system as

```text
IPS  =   1, IRS =   0, ILP =   1
NDIM = 2
NTST = 300, NCOL=   5, IAD =   3, ISP =   2, ISW = 1, IPLT= 0, NBC= 0, NINT= 0
NMX  = 1000, NPR= 100, MXBF=   0, IID =   2, ITMX= 10, ITNW= 5, NWTN= 3, JAC= 0
EPSL = 1e-10, EPSU = 1e-10, EPSS = 1e-9
DS   = 1.0, DSMIN= 1e-13, DSMAX= 1e-1, IADS=   1
NPAR = 36, THL =  {11: 0.0, 12: 0.0, 13: 0.0}, THU =  {}
```

Apart from `NDIM` and `NPAR` it is Andrus's `c.fncMultiSeg` line for line: the
collocation mesh (`NTST`, `NCOL`, `IAD`), the Newton and convergence tolerances
(`EPSL`, `EPSU`, `EPSS`, `ITMX`, `ITNW`, `NWTN`), the step-size bounds (`DSMIN`,
`DSMAX`, `IADS`) and the zero weights `THL` on the three time slots, which keep the
period and the segment times out of the continuation's arclength. The three
tolerances are the one part of the file a config sets: `IsochronSettings.epsl`,
`epsu` and `epss` (`settings.epss=1e-12` on the command line) are written into the
file through `auto_session(ode, tolerances=...)`, so they hold on every run of every
stage and enter the foliation's hash. The values `IPS`, `NDIM`, `NBC`, `NINT`, `ICP`,
`DS`, `NMX` and `NPR` in the file are placeholders that every run overrides.

## How a run is made

Every stage calls `Runner.__call__(start, **constants)`, which calls AUTO's
`run(start, e="isochron", c="isochron", **constants)`. AUTO's `run` applies the
constants file first and every keyword on top, and nothing survives from the run
before, so a stage states what differs from the file and must restate it on every run.
`bvp(ndim, nbc)` is the part all runs share, `IPS = 4` (a boundary value problem),
`NDIM`, `NBC` and `NINT = 0`; the rest is the stage's own. A start that is an
`AutoSolution` contributes its `PAR` dict as the `PAR` keyword, so the slot values
travel with the solution from stage to stage, and `Runner.constants` keeps the
keywords of the last run for a failure report.

`NTST` is the one constant with two sources. Stages 1–3 take the file's 300 intervals;
stage 4 passes `NTST = IsochronSettings.mesh_intervals`, which is `ntst` when set and
`60 * n_periods` otherwise, 300 for the default five periods. The two agree at the
defaults and differ as soon as `n_periods` or `ntst` changes.

## What each stage sets

The rows are the keywords each run passes on top of the file, so the tolerances and
the mesh of stages 1–3 are the file's. `period`, `seg1` and so on stand for the slots of
the table above.

| Stage | Run | `NDIM`, `NBC` | `ICP` | Other constants | Stops at |
|---|---|---|---|---|---|
| 1 `periodic_orbit` | store the cycle | 4, 6 | `seg2, seg1, period` | `NMX = 1` | the start, interpolated onto the mesh and not corrected |
| 2 `adjoint_bundle` | the trivial bundle in `mult` | 8, 11 | `mult, norm, seg1, period` | `DS = -0.1`, `DSMAX = 0.1`, `NMX = 100`, `SP = [BP1]` | the first branch point, checked to be at `mult` $= 1$ |
| 2 | the switch onto the bundle | 8, 11 | the same | `ISW = -1`, `UZSTOP = {norm: 1.0}`, `NMX = 200`; restarted from AUTO's own `BP` point | `norm` $= 1$ |
| 3 `rotate_phase` | `seg2` down and up, one run each | 8, 11 | `seg2, mult, seg1, period` | `DS = ∓0.01`, `DSMAX = 0.05`, `NMX = 2000`, `NPR = 0`, `SP = []`, `UZR = {seg2: [(1 − Θ)T, …]}`, `RL0 = 0`, `RL1 = 1.01T` | the ends of $[0, 1.01T]$; one `UZ` point per requested phase |
| 4 `isochron` | correct the start at $\eta = 0$ | 10, 13 | `eta, mult, seg1, period, base_x, base_y, point_x, point_y` | `NTST = mesh_intervals`, `NMX = 1` | the corrected start |
| 4 | the two halves in $\eta$ | 10, 13 | the same | `NTST = mesh_intervals`, `DS = ∓ds` (`1e-5`), `DSMAX = 10`, `NMX = max_steps` (`8000`), `NPR = 200`, `UZSTOP = {eta: ±eta_stop, point_x: box, point_y: box}`, `SP = [LP0, BP0]` | $\vert\eta\vert =$ `eta_stop`, the traced point on the box, or `max_steps` |

Stage 3 accepts a `UZ` point whose `seg2` is within `1e-8 * period` of its target, so a
base point sits up to $10^{-8}$ turns off its recorded phase; the phase whose target is
the bundle's own `seg2` ($\Theta = \tfrac12$) takes the bundle itself, since AUTO
reports a user point only where its function changes sign between steps. Stage 4's
`ICP` lists eight parameters for a problem with four free ones: AUTO prints every
parameter in `ICP`, and the four extra slots are how the base point and the traced
point reach the branch's columns (AUTO manual, § ICP, parameter overspecification).

## The constants, in brief

What the pipeline's constants mean, with the section of the AUTO-07p manual's chapter
"Description of AUTO-Constants" that defines each; the manual is `doc/auto.tex` (or
`auto.pdf`) in the installed tree.

| Constant | Meaning here | Manual |
|---|---|---|
| `IPS` | the problem type: 4, a boundary value problem with `BCND` and `ICND` | § IPS |
| `NDIM` | the number of state components: 4, 8 or 10 by stage | § NDIM |
| `NBC` | the number of boundary conditions in `BCND`: 6, 11 or 13 | § NBC |
| `NINT` | integral constraints: none | § NINT |
| `NPAR` | the number of `PAR` slots: 36 | § NPAR |
| `JAC` | 0: AUTO differences the Jacobian; `rhs.inc` prints the exact `A(i,j)` for the adjoint equation, not for AUTO | § JAC |
| `NTST`, `NCOL` | the collocation mesh: intervals and collocation points per interval | § NTST, § NCOL |
| `IAD` | mesh adaptation every 3 steps | § IAD |
| `EPSL`, `EPSU` | Newton's relative convergence criterion on the parameters and on the solution, $10^{-10}$ by default (`IsochronSettings.epsl`, `epsu`) | § EPSL, § EPSU |
| `EPSS` | the relative arclength criterion for locating special points such as user points, $10^{-9}$ by default (`IsochronSettings.epss`) | § EPSS |
| `ITMX`, `NWTN`, `ITNW` | the Newton iterations allowed to locate a special point; the full-Newton steps before the Jacobian is frozen; the Newton–Chord iterations allowed to a correction before the step is halved | § ITMX, § NWTN, § ITNW |
| `DS` | the first continuation step, with its sign the direction | § DS |
| `DSMIN`, `DSMAX`, `IADS` | the step-size bounds and the adaptation period | § DSMIN, § DSMAX, § IADS |
| `THL`, `THU` | the parameter and state weights in the pseudo-arclength: the three time slots weigh 0 | § THL, § THU |
| `NMX` | the bound on continuation steps of one run | § NMX |
| `RL0`, `RL1` | the bounds on the first `ICP` parameter, used by stage 3 on `seg2` | § RL0, § RL1 |
| `ICP` | the free parameters, with the overspecified ones printed | § ICP, § Parameter overspecification |
| `ISW` | 1, an ordinary branch; −1, branch switching at a `BP` | § ISW |
| `ISP` | 2, every special point detected, as in the file; `SP` narrows it per run | § ISP |
| `SP` | which special points are detected and how many stop the run: `BP1` stops at the first branch point, `LP0` and `BP0` turn detection off | § SP |
| `NPR` | print a solution every `NPR` steps; 0 only at special points | § NPR |
| `UZR`, `UZSTOP` | user points: a parameter value AUTO locates and labels `UZ`, and the variant that ends the run there | § UZR, § UZSTOP |
| `IPLT`, `IID`, `MXBF`, `ILP`, `IRS` | plotting column, diagnostic verbosity, no automatic branch following, fold detection as in the file, no restart label (the start is passed as an array) | § IPLT, § IID, § MXBF, § ILP, § IRS |

## Inspecting a run

- AUTO's output is captured and logged at `DEBUG` on the logger
  `deep_isochron.analysis.ode.auto.runner`; `logging.getLogger("deep_isochron.analysis.ode.auto").setLevel(logging.DEBUG)`
  before a session shows every run as AUTO prints it.
- `Runner.tail()` is the last `LOG_TAIL` (50) lines, and `Runner.constants` the
  keywords of the last run. A phase that fails in the foliation comes back as an
  `IsochronFailure` carrying both, so a failed run can be read without rerunning it.
- `Runner.output` is every line AUTO printed in the session.
- A session's directory, `deep_isochron_auto_*` under the system's temporary
  directory, is deleted when its block returns and kept when the block raises, with
  the generated files and AUTO's `fort.*`, `b.*`, `s.*` and `d.*` files; an
  `IsochronFailure.workdir` names it, and the foliation's log line and
  `compute_isochrons.py` print it. `auto_session(ode, keep=True)` keeps the
  directory of a block that succeeds, for running AUTO on the files by hand.
