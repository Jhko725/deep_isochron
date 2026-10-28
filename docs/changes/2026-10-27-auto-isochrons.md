---
type: change
status: Steps A–E reviewed 2026-10-28; the review's changes in progress
updated: 2026-10-28
branch: auto-isochrons
---

# `auto-isochrons` — isochrons of a planar system from AUTO-07p (roadmap O7, step 2)

## Summary

This branch builds the AUTO-07p isochron pipeline of roadmap O7. The plan is the Project
document "Isochrons from AUTO-07p" (`claude/isochron-auto-plan-2026-10-22`, revised
2026-10-27), with decisions 1–17 on the plan and D1–D9 on the design document, all
settled with Joon on 2026-10-27.

The first commit is the design document [`design/isochrons.md`](../design/isochrons.md),
which derives the method and breaks down the published version (Langfield et al. 2014),
the coworker's version and ours. Writing it found that the coworker's bundle stage
integrates $\dot w = +A^{\mathsf T}w$ instead of the adjoint equation
$\dot w = -A^{\mathsf T}w$. The slip is invisible on FitzHugh–Nagumo, whose Jacobian is
similar to its transpose, and fatal on the normal forms, where the stage finds no
branch point.

The second commit is Step A of the plan: the corrected bundle (D1), the generator's two
compile bugs, and tests that compile the generated Fortran with gfortran, replacing the
f2py test that had never run. The third commit is Step B: AUTO as a dependency, with
its Python layer locked by commit in `uv.lock` and its Fortran tree built at the same
commit by `scripts/install_auto.sh`. Joon reviewed Steps A and B on 2026-10-28 and
asked three questions in the review table, which became decisions R1–R3. The commit
after the review, Step B′, applies them: `just` as the task runner, the AUTO tree located
only through the environment, and the setup documented for a cluster. It also corrects
a false claim in `docs/auto.md`, adds the README, and acknowledges Andrus Giraldo, whose
reproduction the pipeline is built from. Step C starts with `reanchor`, the O1
function that moves a cycle's phase origin (plan decision 2). The pipeline follows:
the section anchor, the four stages and `isochron_foliation`, the results with netCDF
persistence and a plot (Step D), and tests against the coworker's stored isochrons.
With the quick settings, two FitzHugh–Nagumo isochrons lie within $7\times 10^{-6}$ of
the coworker's curves, traced in 72 s.

Step E adds the remaining oracles of the design document's §8 and
[ADR-0017](../decisions/0017-isochrons-by-auto-continuation.md). Stages 1–3 agree with
O1's flow and with O2's Floquet vectors on all four planar systems; the coworker's Hopf
route reaches O1's cycle; the Bautin and Winfree isochrons carry their closed-form
phase; and FitzHugh–Nagumo's traced points, flowed five periods, end at their base
points. Writing them found two defects of the pipeline (*Bugs fixed*). Ruff's `W605`
now rejects the backslash in a docstring that is not raw, which had slipped through
three times.

After Joon's cluster runs and a design round on 2026-10-28, the branch adds a Hydra
script that computes and saves a foliation from a config, `scripts/compute_isochrons.py`
with `configs/isochrons/`, runnable as a SLURM job through Hydra's submitit launcher,
and `Isochron.resample` for evenly spaced points.

## Files

- `docs/design/isochrons.md` (**new**): the design document, §1–11, with D1–D9 decided.
- `src/deep_isochron/analysis/ode/auto/fortran/isochron_multiseg.f90`: the bundle
  integrates $-\tau A^{\mathsf T}w$; the header and the `BCND` comment say so.
- `src/deep_isochron/analysis/ode/auto/codegen.py`: `indices_block` declares one slot
  per line; `_wrap` cuts only at blanks and raises if it cannot; the docstrings of the
  module, `Idx.mult` and `LINE_WIDTH` follow.
- `tests/test_auto_codegen.py`: `_statements` joins continuations with a blank; new
  tests for the wrap, `indices.inc`, the adjoint sign, the compile of the generic file,
  and the gfortran driver, which replaces the f2py test.
- `.github/workflows/ci.yml`: the `slow` job installs gfortran; a new `auto` job builds
  AUTO, cached under the pinned commit, and runs `pytest -m auto`.
- `pyproject.toml` / `uv.lock`: the dependency group `auto` (`auto-07p` from git at
  `44cc1c4`), included by `dev`; the pytest marker `auto`.
- `scripts/install_auto.sh` (**new**): builds the Fortran tree at the commit in
  `uv.lock`; `--commit` prints the commit.
- `src/deep_isochron/analysis/ode/auto/_env.py` (**new**): `auto_dir`, `pinned_commit`,
  `load_auto` returning an `AutoHandle`; `AutoNotInstalled`.
- `src/deep_isochron/analysis/ode/auto/__init__.py`: the docstring describes `_env`.
- `tests/test_auto_env.py` (**new**): see *Tests*.
- `docs/auto.md` (**new**): installing AUTO, the variables, the pin, the tests,
  troubleshooting.
- `docs/architecture.md`: `_env.py` in the module map.
- `justfile` (**new**): the recipes `setup`, `install-auto`, `check`, `test`,
  `test-slow`, `test-auto`, `smoke` (R3).
- `pyproject.toml` / `uv.lock`, Step B′: `rust-just` and `prek` in the `dev` group.
- `prek.toml`: excludes `src/Isochron2PBVP/` and `tests/golden/`, so that no hook
  rewrites the coworker's files or golden data; Step D drops the first.
- `.github/workflows/ci.yml`, Step B′: every job runs `just` recipes.
- `src/deep_isochron/analysis/ode/auto/_env.py`, Step B′: `auto_dir` and `load_auto`
  take no path (R2).
- `tests/test_auto_env.py`, Step B′: the resolution tests set the variables.
- `docs/auto.md`, Step B′: `just setup`; the `uv sync` correction; the platforms; "On a
  cluster"; the network note.
- `README.md`: what the package does, `just setup`, the documentation, the
  acknowledgment.
- `docs/design/isochrons.md`, Step B′: the acknowledgment.
- `CLAUDE.md`, Step B′: the recipes in "Repository facts"; `prek` and its exclusions.
- `src/deep_isochron/analysis/ode/limit_cycle.py`, Step C: `reanchor`; the shooting
  guards of `find_limit_cycle` moved into `_check_shooting`, which both use.
- `src/deep_isochron/analysis/ode/__init__.py`: exports `reanchor`; the docstring.
- `tests/test_analysis_ode.py`: three `reanchor` tests and the fixture `hopf_cycle`.
- `docs/architecture.md`, Step C: `reanchor` in the module map.
- `src/deep_isochron/analysis/ode/auto/fortran/isochron_multiseg.f90`, Step C: `FB(6)`
  anchors segment 1 on the section through phase 0, $(u - p)\cdot n = 0$ (decision 3);
  `STPNT` is empty and `PVLS` no longer stores an anchor or AUTO's stability flag.
- `src/deep_isochron/analysis/ode/auto/codegen.py`, Step C: `Idx` trades `anchor` and
  `stability` for `section_point` and `section_normal`.
- `tests/test_auto_codegen.py`, Step C: the slot layout with the section slots.
- `src/deep_isochron/analysis/ode/auto/isochron.py` (**new**): `IsochronSettings`,
  `Isochron`, `IsochronFailure`, `IsochronFoliation` (with `dataset`, `save`, `load`),
  the four stages and `isochron_foliation`.
- `src/deep_isochron/plotting/isochrons.py` (**new**): `plot_isochrons`; the plotting
  package exports it.
- `tests/test_isochron.py` (**new**): see *Tests*.
- `tests/helpers.py`: `TOL["isochron_curve"]`, `TOL["branch_point"]`, and the named
  settings `ISOCHRON_SETTINGS` (`quick`, `full`).
- `tests/golden/fncMultiSeg_Iso_{100,26,51}.csv` (**new**): three of the coworker's
  isochrons, byte for byte (decision G8).
- `justfile`, `.github/workflows/ci.yml`, Step D: `test-auto` takes `quick` or `full`;
  CI runs `full`, since it runs only on pull requests and on `master`.
- `docs/architecture.md`, Step D: `isochron.py` and `plot_isochrons` in the module map.
- `src/Isochron2PBVP/` (**deleted**), Step D: the coworker's scripts, Fortran and 100
  CSVs. The full oracle reproduces his isochrons, so the pipeline replaces them; the
  directory stays readable at commit `3af99f2`, and *Design* quotes the parts the
  pipeline is built from.
- `pyproject.toml`, `prek.toml`, `scripts/check_docstring_headers.py`, Step D: the
  exclusions of `src/Isochron2PBVP` removed (ty, ruff, `per-file-ignores`, codespell,
  the hooks, the docstring-header check).
- `CLAUDE.md`, `docs/coding-standards.md`, Step D: the exclusion lines name only
  `tests/golden/`.
- `docs/decisions/0016-symbolic-systems.md`, Step D: the links to the coworker's files
  point at commit `3af99f2`; a History line.
- `docs/design/isochrons.md`, Step D: the coworker's files linked at commit `3af99f2`.
- `docs/roadmap.md`, Step D: the O7 row; the Done ledger records Steps C and D.
- `src/deep_isochron/analysis/ode/__init__.py`, Step C: exports the pipeline.
- `CLAUDE.md`: `-n 2` in the cloud sessions; the AUTO setup line.
- `docs/references.md`: entries for Hannam et al. 2022, Krauskopf & Rieß 2008 and
  Osinga & Moehlis 2010.
- `docs/index.md`: the rows of the design document and `auto.md`.
- `docs/roadmap.md`: the O7 row points to the design document and names the gfortran
  tests; the Done ledger records Steps A and B.
- `docs/changes/2026-10-27-auto-isochrons.md` (**new**): this document.
- `docs/decisions/0017-isochrons-by-auto-continuation.md` (**new**), Step E: ADR-0017.
- `docs/index.md`, Step E: the ADR-0017 row replaces the reservation.
- `src/deep_isochron/analysis/ode/auto/isochron.py`, Step E: `_stop_reason` decides a
  box stop by the box edge; the docstrings of stage 1 say that AUTO stores its start
  uncorrected.
- `docs/design/isochrons.md`, Step E: stage 1 does not correct the orbit; the oracle of
  O1's cycle is the orbit after stage 2.
- `tests/test_isochron.py`, Step E: the oracles of §8 (see *Tests*).
- `tests/helpers.py`, Step E: the tolerances `auto_orbit`, `bundle_direction`,
  `bundle_scale`, `isochron_phase`, `isochron_forward`, and the settings
  `ISOCHRON_SETTINGS["reach"]`.
- `pyproject.toml`, `docs/coding-standards.md`, Step E: ruff's `W605` selected, and its
  row in the table of checks.
- `docs/roadmap.md`, Step E: the O7 row; the Done ledger records Step E.
- `src/deep_isochron/analysis/ode/auto/isochron.py`, after Joon's cluster runs: every
  AUTO run on one OpenMP thread (`_one_thread` in `_Runner`); phase π taken from the
  bundle in stage 3; pool workers with JAX on the CPU.
- `tests/test_isochron.py`, `tests/helpers.py`: the stage test includes phase π; the
  named settings run one worker.
- `src/deep_isochron/analysis/ode/auto/isochron.py`, `docs/design/isochrons.md`,
  `tests/helpers.py`: `box_scale` defaults to 2.5 (Joon, 2026-10-28).
- `src/deep_isochron/experiment/isochrons.py` (**new**): `isochron_phases`,
  `isochron_settings`, `isochron_cycle`, `isochron_hash`, `isochron_file`,
  `compute_isochrons`, `load_isochrons`; exported from `experiment`.
- `scripts/compute_isochrons.py` (**new**): the Hydra entry point; `scripts/README.md`
  lists it.
- `configs/isochrons/fhn.yaml`, `configs/isochrons/fhn-easy.yaml` (**new**): the
  isochron configs, each composing its data config.
- `configs/hydra/launcher/slurm.yaml` (**new**): the SLURM launcher's settings.
- `pyproject.toml`, `uv.lock`: the dependency group `slurm`
  (`hydra-submitit-launcher`), included by `dev`.
- `src/deep_isochron/analysis/ode/auto/isochron.py`: `Isochron.resample`; `wrap_phases`
  public (it was `_check_phases`), exported from `analysis.ode`; INFO lines for stages
  1–3 and for every finished phase.
- `tests/test_experiment_isochrons.py` (**new**), `tests/test_isochron.py`: see
  *Tests*.
- `docs/auto.md`: "Computing and saving a foliation", the SLURM launcher.
- `docs/decisions/0011-experiment-layer.md`: Decision 3 covers the isochron configs; a
  History line.
- `docs/architecture.md`: `experiment/isochrons.py`, the script and the configs.
- `CLAUDE.md`: a test outcome is stated only after the test has run.
- `src/deep_isochron/analysis/ode/auto/isochron.py` (**deleted**), review round 2:
  split into `results.py`, `runner.py`, `stages.py` and `foliation.py` (**new**, all
  four); `auto/__init__.py` and `analysis/ode/__init__.py` re-export from them.
- `src/deep_isochron/utils/numerics.py`: `arclength(points)`, which the stage, the
  resample tests and the golden comparison had each written.
- `tests/test_auto_stages.py` (**new**): the §8 oracles of the stages, out of
  `tests/test_isochron.py`, which keeps the results, `resample`, persistence, the plot
  and the foliation against the coworker's curves; `tests/conftest.py` holds the one
  `auto` fixture; `tests/helpers.py` the planar systems and `anchored_cycle`.
- `docs/design/isochrons.md`: §6 with a subheading per stage, stage 2 written out.
- `docs/architecture.md`, `docs/decisions/0017-isochrons-by-auto-continuation.md`:
  the four modules; the ADR links the coworker's scripts at `3af99f2`.
- `src/deep_isochron/analysis/ode/limit_cycle.py`, `poincare.py`, review round 2:
  `t_settle`, the period estimate and `PoincareMap.t_max` reach the jitted stages as
  arrays; `nearest_cycle_time` on `optimistix.Newton`; `reanchor` without the polish.
- `src/deep_isochron/systems/normal_forms/integration.py`: the `r_squared` field as a
  module-level function with the normal form in `args`; the closed form jitted.
- `tests/helpers.py` (`CompileCounter`), `tests/test_analysis_ode.py`,
  `tests/test_normal_forms.py`: the guard tests that a second call compiles nothing;
  `docs/coding-standards.md` lists them.
- `docs/design/analysis.md`: `nearest_cycle_time` and `reanchor` in §2.2; the array
  rule in §2.4.
- `README.md`: the AUTO directory variables; the two citations in full.

## Design

- **The design document precedes the pipeline.** The numerical-method skill puts the
  derivation before the code, and the bundle defect shows why: the plan's oracle "bundle
  against O2" would have failed against the coworker's equation, with no document to say
  which side was wrong.
- **The document describes three versions of the method side by side** (§4–6). A
  document about our version alone was rejected, because the review needs to see what
  each of our replacements replaces.
- **The bundle integrates the adjoint equation** (D1; ADR-0017 to come). The variational
  equation, whose branch point sits at the stable multiplier ($1.7\times 10^{-12}$ for
  FitzHugh–Nagumo), and O2's Floquet vectors passed in as parameters were the
  alternatives (design document §3).
- **The generated Fortran is tested by compiling it with gfortran**, the compiler AUTO
  uses (plan decision 6). The f2py route needed `meson` and `ninja` in the dev group, and
  meson builds in a subdirectory where `INCLUDE 'rhs.inc'` is not found. The driver test
  checks `rhs.inc` numerically; the compile test checks the generic file with its
  includes, and needs no AUTO because `-c` does not link.
- **`_wrap` cuts only at blanks** (D8). SymPy's printer puts blanks only between tokens,
  so no token is split. Cutting mid-token with no blank on either side of the `&` is also
  legal Fortran; it was rejected because it keeps the generated file harder to read and
  rests on a finer point of the standard.
- **AUTO's Python layer is a locked git dependency, loaded only by `_env.load_auto`**
  (plan decision 4; ADR-0017 to come). Installing it into the environment from the tree
  was rejected because `uv sync` removes what the lock does not list; putting the tree's
  `python/` on `sys.path` was rejected because it leaves the version unrecorded. The
  commit lives once, in `pyproject.toml`; the install script and CI's cache key read it
  from `uv.lock`.
- **`load_auto` sets `AUTO_DIR` before importing.** AUTO's Python layer reads the
  variable at import time and fails with `FileNotFoundError` without it.
- **The bundle's branch-point test moves to Step C.** The plan put it in Step B's smoke
  tests, but it needs the two-segment start that stage 1 builds; written now, it would
  be a second, test-only construction of that start.
- **`just` runs the project's commands, and CI calls the same recipes** (R3). Make was
  the first proposal, since AUTO's build already needs it; just was chosen for recipes
  that read as shell scripts, `just --list` as the help, and arguments that read like a
  command's. It installs from PyPI as `rust-just`, so `uv sync` brings it. Poe the Poet,
  which keeps tasks in `pyproject.toml`, was the runner-up; Task, invoke, nox, hatch,
  pixi and mise each bring more than a command list needs.
- **The install script stays** (R1): AUTO's own procedure is `configure && make`, and
  AUTO compiles equation files at run time, so the machine needs gfortran whichever
  tool installs AUTO. pixi with conda-forge is the established route for many native
  dependencies and is revisited if a second one arrives.
- **The AUTO tree is chosen by the environment only** (R2). A path argument to
  `load_auto` and a directory field on `IsochronSettings` were rejected: the tree is a
  property of the machine, a setting would be saved with every foliation, and two ways
  to choose it could disagree with the install script.
- **`prek` is a dev dependency.** `just setup` installs its git hook, which then calls
  the copy in the environment rather than a global one.
- **The cluster** (G1, H1–H4): `just setup` names `apt-get`, `module load gcc` and conda
  when gfortran is missing; `docs/auto.md` says to load the compiler in every job script,
  since AUTO compiles at run time, and that nothing needs the network at run time.
- **`reanchor` finds the nearest orbit point by Newton in time, then polishes on the
  flow-normal section there** (plan decision 2). The first version polished on the
  section through `point` itself, with the normal of the nearest of 1024 samples; on
  Hopf's unit circle that missed the nearest point by $2.8\times 10^{-4}$ for a point at
  radius 0.6, because the sampled normal is off by up to half a sample spacing. Newton
  on $g(t) = (\gamma(t) - p)\cdot f(\gamma(t))$ converges in a few steps, and the
  section is then exactly the one `find_limit_cycle` would build through that point.
  The guards of `find_limit_cycle` became `_check_shooting` rather than a copy.
- **`load` takes the cycle and checks it against the file** (Step D). The file stores
  the cycle as data (mesh, period, section, system name and parameters), not as a
  `LimitCycle`, which would need the system object and its solver. Rebuilding the system
  from its name through a registry was rejected as machinery nothing else needs; the
  caller rebuilds the cycle with `find_limit_cycle` and `reanchor` at the stored origin,
  and `load` raises when the system, parameters, period or origin disagree.
- **CI runs the full `auto` suite.** Decision G3 split a quick tier for every push from
  the full suite for pull requests; CI runs only on pull requests and on `master`, so
  its `auto` job runs `just test-auto full`, and `just test-auto` is the quick tier for
  a session.
- **The remaining decisions are design-document decisions for later steps:** D2 (the
  stage-4 start on AUTO's grid), D3 (the oracle's tolerance), D4 (one segment per
  isochron), D5 (the rotating base point, for ADR-0017), D6 (`ntst` scaling with
  `n_periods`) and D7 (the stop reason of each half).

- **The coworker's code, quoted before its deletion (Step D).** `src/Isochron2PBVP` is
  Andrus Giraldo's work, and the pipeline keeps its structure, so the parts it is built
  from are quoted here as they stood at commit `3af99f2`. His run sequence,
  `fncMultiSeg/mainIsochronMultiSegFNC.py`, lines 39–42, 45 and 54:

  ```text
  rPre[2] = run(rPre[1]('UZ1'), IPS= 4, ISW= 1, NDIM=4, NBC=6, NINT=0, ICP =  [13,12,11], PAR={12:rPre[1]('UZ1')['PAR(11)']}, UZSTOP={13:1.0})
  rPre[3] = run(rPre[2]('UZ1'), ICP =  [71,12,11], UZSTOP={71:0.5})
  rPre[4] = run(rPre[3]('UZ1'), NDIM=8, NBC=11, ICP =  [34,35,12,11], SP=['BP2'], PAR={34:1.2,35:0}, DS=-0.1,DSMAX=1e-01)
  rPre[5] = run(rPre[4]('BP1'), ISW=-1, UZSTOP={35:1.0},SP=['LP0','BP0'])
  rPre[6]  = run(rPre[5]('UZ1'), ICP =  [13,34,12,11], SP=[], DS=-0.01, DSMAX=0.05, NMX=2000,NPR=0,UZR={13:rPre[5]('UZ1')['PAR(11)']*phaseVec},RL0=0, RL1=rPre[5]('UZ1')['PAR(11)']*1.01)
  rIso[3]   = run(rIso[1], DS=1e-05,DSMAX=10,NMX=8000,NPR=200,UZSTOP={36:[1e-06,-1e-6],74:[10,-10],74:[10,-10]},SP=['LP0','BP0'])
  ```

  The pipeline's stages 1–4 are these runs: the orbit split into two segments (runs 2–3,
  with the section anchor of decision 3 in place of his $u_2 = 0.5$), the adjoint bundle
  and its branch point (4–5), the rotation to each phase (6–7), and the two halves of
  each isochron (`rIso[3]`, `rIso[4]`). The right-hand side and boundary conditions of
  `fncMultiSeg/fncMultiSegIso.f90`, lines 49–53 and 82–91:

  ```fortran
  F(5:6)  =  PAR(12)*(MATMUL(TRANSPOSE(A),U(5:6)))
  F(7:8)  =  PAR(13)*(MATMUL(TRANSPOSE(B),U(7:8)))
  F(9:10)  = PAR(14)*F(9:10)
  FB(1:2)  = U0(1:2)-U1(3:4)
  FB(3:4)  = U1(1:2)-U0(3:4)
  FB(5)    = PAR(11)-PAR(12)-PAR(13)
  FB(6)    = U0(2)-PAR(71)
  FB(7:8)  = U1(7:8)- U0(5:6)
  FB(9:10) = U1(5:6)- PAR(34)*U0(7:8)
  FB(11)   = DOT_PRODUCT(U0(7:8),U0(7:8)) - PAR(35)
  FB(12:13)= U1(9:10)- (U0(3:4) + PAR(36)*(/ -U0(8), U0(7) /))
  ```

  `isochron_multiseg.f90` keeps these conditions but two: the bundle is
  $-\tau A^{\mathsf T}w$ (D1, *Bugs fixed*), and `FB(6)` anchors segment 1 on the
  section through phase 0.

- **`box_scale` defaults to 2.5, not 5 (Joon, 2026-10-28).** Five reproduced the
  coworker's $\vert x\vert = 10$ on FitzHugh–Nagumo; 2.5 shortens the outward halves,
  which carry little of the structure, while the inward halves toward the unstable
  fixed point are unchanged. The slow two-way oracle failed with the new default: ours
  covered 1243 of the 7124 points of the coworker's file 100, below the half the test
  requires, though every one of our points lay on his curve. The oracle therefore runs
  with his box, `box_scale = 5` (`ISOCHRON_SETTINGS["full"]`), and passes (*Tests*).

- **The isochron script (decisions of 2026-10-28, Q1–Q19).** The configs live in
  `experiment/`, since library code never reads configs (ADR-0011): `compute_isochrons`
  and `load_isochrons` are the counterparts of `generate_dataset` and `build_source`.
  Each config composes its data config through Hydra's defaults
  (`/data@data: fhn`), so the reference isochrons are of the trained system; a config
  restating the system was rejected, since the two could drift apart. The file is
  `data/isochrons/<data name>-<hash>.nc`, the hash over the system, the cycle block,
  the wrapped phases and the settings other than `max_workers`; Hydra's dated run
  directory was rejected because a config could not find it again. An existing file
  is reported and left alone, so a resubmitted job costs nothing; an `overwrite` flag
  was rejected as a second way to delete a file. Failed phases are saved with the
  others and give exit status 1. The anchor is required, since without it phase 0 is
  wherever settling ended. `compute_isochrons` returns the foliation rather than the
  path, so that the script can report failures without reading the file back. The
  `fhn-easy` anchor, $(-0.0977, 0.5)$, is the left crossing of its cycle with
  $y = 0.5$, the rule that gives the coworker's anchor on `fhn` (to $2\times 10^{-7}$).
  The script runs JAX on the CPU and logs one line per finished phase.
- **SLURM through Hydra's submitit launcher**, in a dependency group `slurm` that `dev`
  includes, with `configs/hydra/launcher/slurm.yaml`; the partition is given on the
  command line. Plain `sbatch` was the alternative; Joon may return to it. The launcher
  submits multiruns only, so even one job takes `-m`.
- **`Isochron.resample(spacing)`** returns points at an exact arclength spacing on a
  grid anchored at the base point, interpolated linearly in arclength; $\eta$ is
  interpolated in $\log\vert\eta\vert$ on each half. Linear interpolation errs by
  under $10^{-5}$ on the coworker's curves, estimated from their curvature and steps;
  a spline would add a dependency for less than the oracle's tolerance. A fixed number
  of points per half, which would stack into one array, waits for the metric that needs
  it. The script stores the traced points; resampling is the reader's choice.

- **The pipeline in four modules (review round 2, Q1–Q8 of 2026-10-28).** `isochron.py`
  held four responsibilities in 850 lines, and `isochron_foliation` called a
  housekeeping function rather than the stage. Now `results.py` holds the settings and
  the result types, `runner.py` an AUTO session (`auto_session(ode)` yields a `Runner`:
  AUTO loaded, the problem written into a temporary working directory, one thread,
  output captured) and `AutoSolution`, `stages.py` the four numerical stages, and
  `foliation.py` the driver, so that the chain `isochron_foliation → trace_isochron →
  isochron` is visible. Two files, numerics and housekeeping, were rejected: the
  housekeeping half would still be 400 lines of three things. The stages are public
  within the subpackage, since tests and the design document name them, and
  `analysis.ode` re-exports only the result types, `isochron_foliation` and
  `wrap_phases`. Every stage returns an `AutoSolution` (mesh, components, parameters
  by slot, label, step), the one shape at every seam and across the pool; stage 2's
  branch switch is the one run that restarts from AUTO's own point, since `ISW = -1`
  needs the null vector stored with it, which an array start loses. The stages are
  functions of the runner, not methods of a pipeline class, so the data flow between
  them is on the page. `results.py` stays separate from the driver (300 lines against
  150; a notebook imports from `analysis.ode`, not from these modules).

- **The recompile fixes (Q9).** Three sources, each measured with `jax_log_compiles`:
  `find_limit_cycle` compiled three programs for every new `t_settle` (the float into
  `_settle`, `t_max = 10 t_settle` as a static field of `PoincareMap`, the period
  estimate into `_polish`), one for every new parameter set; the `r_squared`
  integration compiled `diffeqsolve` on every call, its field being a closure built
  per call; the closed form compiled its two `fori_loop`s on every call, run op by op.
  The fixes follow diffrax's own conventions: values are arrays (`PoincareMap.t_max`
  is a converted array leaf, so a new bound is not a new program), a vector field is a
  module-level function with its parameters in `args`, and the closed form is jitted
  as `diffeqsolve` is. A guard test per driver (`CompileCounter` in `tests/helpers.py`)
  asserts zero compilations on a second call with new values; both fail on the code
  before the fix. Chex's `assert_max_traces` does the same and was not added, one
  assertion not being worth a dependency.
- **`reanchor` on Optimistix, without the polish (Q10).** `nearest_cycle_time` ran a
  hand-rolled Newton scan; it is now `optimistix.root_find(Newton)` on the
  stationarity condition, differentiating through the flow, with its first step in
  closed form at offset zero: diffrax's derivative of a flow in its end time is
  identically zero for a solve of zero length (measured: `[0, 0]` at $\delta = 0$,
  $f(\gamma)$ at $10^{-3}$), so a solve started there would not move. `reanchor` no
  longer re-solves the shooting problem, since the orbit is converged: it flows the
  new origin for the mesh and evaluates the shooting residual once there.

## Bugs fixed

- **`docs/auto.md` said that a plain `uv sync` uninstalls AUTO's Python layer.** It does
  not: uv syncs the `dev` group by default, and `dev` includes `auto`
  (`uv sync --dry-run` reports no change). The claim had been measured in a scratch
  project whose `auto` group was not in `dev`.

- **The bundle integrated $+A^{\mathsf T}w$.** Symptom: on `HopfNormalForm(1.3, 1.5, 0.4)`
  and `BautinNormalForm(1.0, 0.3, 1.5, 0.5)`, the coworker's runs 4–5 walk from
  $\lambda = 1.2$ to $-8.7$ without a branch point; on FitzHugh–Nagumo the end direction
  is $58°$ from the stable direction. With the sign fixed, the branch point is at
  $\lambda = 0.99999999998$ and $w$ equals the closed-form $\nabla\Theta$ direction to
  round-off.
- **`indices.inc` did not compile.** Symptom: gfortran's "Line truncated" on a
  declaration of about 160 characters, past the 132 of free form.
- **Bautin's `rhs.inc` did not compile.** Symptom: "Expected a right parenthesis",
  because `_wrap` cut `PAR(1)` into `P &` and `& AR(1)`, two tokens. The test helper
  `_statements` joined continuations without a blank and hid it; Hopf's file happened to
  be cut between tokens.

- **A half that stopped at `eta_stop` was reported as a box stop.** Symptom: Bautin's
  inner half at phase 0.5 ended at $\eta = -9.9999999532\times 10^{-7}$, a point that
  AUTO located to its own tolerance, and `_stop_reason` compared $\vert\eta\vert$ with
  `eta_stop` to a relative $10^{-9}$, so the half was labeled `"box"` at radius 0.96.
  The box is now decided by the box: a user point is a box stop when the traced point
  lies on an edge, which AUTO reaches to round-off.
- **Stage 1 does not correct the orbit, although its docstring and the design document
  said so.** Symptom: on Bautin's unit circle the stored orbit of stage 1 is off by
  $3.4\times 10^{-6}$ in radius, oscillating within each mesh interval: the sag of
  chords between 601 points. AUTO stores the first point of a run with `NMX = 1` as the
  start interpolated onto its mesh, before Newton. Nothing downstream was wrong, since
  stage 2's first step corrects the orbit (to $7\times 10^{-12}$); the docstrings and
  the design document now say what happens, and the oracle against O1 reads the orbit
  after stage 2. The set-up run of stage 4 behaves the same way, as the coworker's did.

- **Phase π always failed** ("no user point at this phase in stage 3", Joon's notebook,
  `linspace(-π, π, 1, endpoint=False)`). Its target $\tau_2 = T/2$ is where stage 1
  leaves segment 2, and AUTO reports a user point only where its function changes sign
  between two steps, never at the start. Stage 3 now takes the bundle itself for it.
- **Stages 1–3 ran AUTO on every core.** Symptom: 24 `isochron.exe` threads on a
  24-core node for about 8.5 minutes, one thread once stage 4 began. AUTO compiles with
  `-fopenmp` and only the pool workers set `OMP_NUM_THREADS=1`; `_Runner` now sets it
  for every run.
- **Five phases ran a GPU out of memory five times.** Each pool worker unpickled the
  cycle's arrays and integrated on the default backend, the GPU, each preallocating
  most of its memory. `_init_worker` puts the workers' JAX on the CPU. Not tested here,
  which has no GPU.

## Tests

Only `tests/test_auto_codegen.py` covers the modules this step touches. The full tier
has not been run for this commit: a quick-tier run at `-n 4` on the session's two cores
and 7 GB lost two workers to memory and was stopped. `uv run pytest
tests/test_auto_codegen.py` gives 35 passed, 8 of them `slow` and skipped without
gfortran (`apt-get update && apt-get install -y gfortran` installs it in
these sessions). With the generator of `master`, 10 of the 35 fail: both
compile-dependent tests for Bautin and Winfree, the compile test for all four systems,
the wrap and `indices.inc` tests, and the parse test for Hopf and Bautin once
`_statements` joins with a blank.

- `test_generic_file_compiles_with_the_generated_includes` (slow): `gfortran -c` on the
  generic file with each planar system's includes.
- `test_compiled_rhs_agrees_with_jax` (slow): the driver's `F` and `A` against `rhs` and
  `jacfwd(rhs)` at four points, to `TOL["identity"]`.
- `test_long_statements_are_cut_between_tokens`, `test_indices_block_has_one_declaration_per_line`,
  `test_bundle_integrates_the_adjoint_equation`, and the strengthened
  `test_rhs_block_lines_fit_the_width`.

The design document's numbers were measured on 2026-10-27 with AUTO-07p built at commit
`44cc1c4` (serial `make`; `make -j4` fails on `support.mod`) and GNU Fortran 13.3.0, by
running the coworker's script and variants of it in a scratch directory:

- the coworker's runs 0–5, the base point $(-0.19007, -0.12319)$ and the bundle
  $w = (-0.58385, -0.81186)$, compared with $DF/\vert DF\vert$ and with $v_s$ from the
  variational transition matrix integrated by `Dopri8` at `rtol = 1e-11`;
- the same runs with the sign negated: $w = (-0.37971, 0.92511)$, $Jw = -v_s$ to eight
  digits;
- the bundle stage on generated problems for FitzHugh–Nagumo, Hopf and Bautin, with both
  signs (the table under *Bugs fixed*);
- isochrons 1, 26 and 51 rerun, timed and compared with the stored files by the distance
  from points to the other curve's polyline ($3.6\times 10^{-6}$); isochron 26 with the
  negated bundle ($8.1\times 10^{-6}$ from the stored file; 272 s) and with the start on
  AUTO's grid ($3.3\times 10^{-6}$ from the spline start);
- the stored points of files 26, 51 and 100 integrated forward four periods.

Step B adds `tests/test_auto_env.py`, 4 passed with a built tree:

- `test_missing_tree_raises_and_names_the_install_script` and
  `test_explicit_path_then_variables_then_default`: how `auto_dir` resolves the tree,
  with fake trees; no AUTO needed.
- `test_the_python_layer_is_pinned_by_commit`: the installed `auto-07p` has a
  40-character commit.
- `test_fitzhugh_nagumo_problem_finds_its_hopf_point` (`auto`): the generated problem,
  built and run by AUTO, stops at the Hopf point; the oracle is the $c$ where the
  Jacobian's trace vanishes at the equilibrium, computed from the cubic, to a relative
  $10^{-6}$.

`scripts/install_auto.sh` built the tree in 37 s, and a second run reported it already
built.

Step B′ ran `uv run just check` (all hooks on every file, then `ty`: clean) and
`uv run just test dev tests/test_auto_env.py` (4 passed). The first `just check` run
showed that the end-of-file and whitespace hooks rewrite the coworker's files; those
edits were reverted, and `prek.toml` now excludes them and the golden data.

Step C's `reanchor` adds three tests to `tests/test_analysis_ode.py`, 3 passed at the
`dev` profile: the origin moves to the nearest cycle point for points drawn at radius
0.6–1.6 on Hopf's cycle, with every invariant of `find_limit_cycle`; reanchoring at
``x0s[0]`` returns the same mesh; a point of the wrong shape raises. On
FitzHugh–Nagumo, `reanchor` at the coworker's anchor $(-1.72170711, 0.5)$ returns that
point as ``x0s[0]`` and the period to $10^{-10}$, in 15 s for the first call (compiling)
and 0.1 s after.

**The pipeline (Steps C and D).** `tests/test_isochron.py` holds 18 tests. Without AUTO
(12, about 25 s): the settings' ranges and defaults; the phases wrapped and checked;
a hand-built Hopf foliation round-tripping through netCDF, `load` refusing a cycle of
other parameters or another phase origin, and the plot drawing every isochron. With
AUTO (marker `auto`): the bundle's branch point at multiplier 1 and its normalization
for FitzHugh–Nagumo, Hopf, Bautin and Winfree (10–22 s each), and FitzHugh–Nagumo's
isochrons at the phases of the coworker's files 100 and 26, with the quick settings,
within $2\times 10^{-5}$ of the stored curves (about 90 s; measured 1.7e-6 and 6.7e-6).
The full five-period suite against files 100, 26 and 51 (`auto` and `slow`) takes
11 minutes on two cores.

Three things went wrong on the way and are fixed. Every AUTO run passes `IPS`, `NDIM`,
`NBC` and `NINT`, because each run reloads the constants file; the first version relied
on constants carried over from the previous run, and stage 2 failed at its first step
(`MX`). The first two-way comparison failed with a distance of 0.82 because the
pipeline's box, relative to the cycle, ends file 100's outward half at
$x = -9.18$ where the coworker's ends at $x = -10$; the comparison of the stored curve
against ours is now restricted to the arclength ours covers. And decision G3's
test-sized settings were wrong twice, as the next paragraph records.

**The cost of an isochron is the length of its segment, not `eta_stop`.** A first
"test-sized" `eta_stop = 1e-4` was meant to stop each half sooner; a second guess,
`1e-9`, was meant to stop the inward half near the cycle. Neither works: with five
periods, both halves of isochron 26 ran 4800–6300 continuation steps while
$\vert\eta\vert$ stayed between $10^{-14}$ and $6\times 10^{-11}$. The step count
follows the length of the traced curve, which grows with `n_periods`. The quick settings
are therefore `n_periods = 1`, `box_scale = 2` and `max_steps = 1500`
(`ISOCHRON_SETTINGS["quick"]` in `tests/helpers.py`): two FitzHugh–Nagumo phases in
72 s end to end, stages 1–3 included.

**Step D baseline.** After the deletion of `src/Isochron2PBVP`: `uv run just check`
clean; `uv run pytest -n 2 -m "not slow" tests/test_isochron.py
tests/test_auto_codegen.py tests/test_auto_env.py --hypothesis-profile=dev` gives 48
passed in 160 s with AUTO installed. The slow two-way oracle passed before the deletion
(682 s). The full tier has not been run on this branch since Step B (plan decision 8).

**Step E: the oracles of the design document's §8**, in `tests/test_isochron.py`, with
the tolerances named in `tests/helpers.py`:

| Test | What it pins | Measured |
|---|---|---|
| `test_stages_one_to_three_agree_with_o1_and_o2` (four systems) | the branch point and the normalization at 1; the period against O1's; the orbit after stage 2 against O1's flow; $Jw$ along O2's Floquet vector at phase 0 and at three rotated base points; $w\cdot F$ constant; each rotated base point at O1's mesh point of its phase; on the normal forms, $w$ along the closed-form $\nabla\Theta$ and the closed-form phase of each base point | orbit $3\times 10^{-9}$ (FitzHugh–Nagumo), $6\times 10^{-11}$ (Hopf); the sines $10^{-11}$; $w\cdot F$ spread $7\times 10^{-10}$ relative; base points $3\times 10^{-10}$ |
| `test_hopf_route_reaches_o1s_cycle` | the coworker's runs 0–1, equilibrium to Hopf point to the cycle at $c = 3$, against O1's period and flow (decision 12) | period $9\times 10^{-11}$ relative, orbit $2.5\times 10^{-9}$ |
| `test_normal_form_isochrons_carry_the_closed_form_phase` (Bautin, Winfree) | the closed-form phase at every traced point, with phase 0 at $(1, 0)$, signed by the sense of rotation (decision 14); halves reaching inside $r = 0.5$ and outside $r = 1.5$ | $4\times 10^{-10}$; radii 0.038–2.0 (Bautin), 0.36–2.08 (Winfree) |
| `test_fitzhugh_nagumo_isochrons_lie_on_the_coworkers` | now also each traced point flowed five periods by O1's solver, against its base point | $4\times 10^{-6}$ at one point, $10^{-9}$ for most |
| `test_a_user_point_on_the_box_is_a_box_stop` (no AUTO) | the stop reason of a half by the box edge, with the $\eta$ that was misread | — |

Winfree rotates clockwise, so its closed-form phase is the negative of the time phase;
the tests compare against the requested phase times the sign of $\omega$. With one
period, Winfree's halves stay within 0.002 of the cycle, since its multiplier is about
$4\times 10^{-4}$; the closed-form test therefore runs with two periods
(`ISOCHRON_SETTINGS["reach"]`).

**Step E baseline.** `uv run pytest -q -n 2 tests/test_isochron.py -m "not slow"`:
21 passed in 504 s. The new tests make the quick `auto` tier slow on two cores: the
stage test of FitzHugh–Nagumo takes 228 s, the FitzHugh–Nagumo foliation 164 s, the
normal-form foliations 107 s and 118 s, because each foliation's own pool of two
processes shares the two cores with the second pytest worker. Run alone, stages 1–3 of
FitzHugh–Nagumo take 82 s and the quick foliation 70 s.

**After the cluster fixes**: `uv run pytest -q -n 2 tests/test_isochron.py
tests/test_auto_codegen.py tests/test_auto_env.py -m "not slow"
--hypothesis-profile=dev`: 52 passed in 351 s, against 504 s for the isochron file
alone before; phase π passes. One worker per foliation removed the oversubscription,
but the tier is still 5.8 minutes, above the 3.5–4 estimated: the foliation tests now
trace their phases one after the other, and JAX's own threads in each pytest worker
still share the two cores.

**Stage-4 settings against cost and reach** (Joon's question 9, 2026-10-28): the
isochron of the coworker's file 26 on FitzHugh–Nagumo with `box_scale = 2.5`, one
worker, run alone on the two-core session; the time includes about 50 s of cycle and
stages 1–3. The fixed point is at $(0.9066, -0.2582)$, and the coworker's curve comes
within $1.6\times 10^{-4}$ of it.

| `n_periods` | `ntst` (per period) | Time | Points | Closest to the fixed point | Farthest from the coworker's curve |
|---|---|---|---|---|---|
| 5 | 300 (60) | 491 s | 7005 | $1.6\times 10^{-4}$ | $8.6\times 10^{-6}$ |
| 5 | 200 (40) | 331 s | 7005 | $1.6\times 10^{-4}$ | $8.6\times 10^{-6}$ |
| 5 | 150 (30) | 264 s | 7010 | $1.6\times 10^{-4}$ | $8.0\times 10^{-6}$ |
| 4 | 240 (60) | 294 s | 4950 | $6.5\times 10^{-4}$ | $8.8\times 10^{-6}$ |
| 4 | 160 (40) | 221 s | 4948 | $6.5\times 10^{-4}$ | $8.8\times 10^{-6}$ |
| 3 | 180 (60) | 198 s | 3708 | $2.7\times 10^{-3}$ | $8.8\times 10^{-6}$ |

The number of periods sets how close the inward half reaches: each period fewer
leaves it about four times farther from the fixed point. The mesh per period does not
change the curve, its reach or its agreement with the coworker's down to 30 intervals,
and the time falls with it: five periods at 30 intervals per period take 264 s against
491 s at 60. The first row ran while other work used the session for a few minutes,
so its time may be high. Whether to lower the default `ntst` is Joon's decision.

**The isochron script and `resample`**: `tests/test_experiment_isochrons.py` composes
both configs against their data configs, checks the phases from `count` or `values`,
that the hash follows the system, anchor, phases and settings and not `max_workers`,
that `load_isochrons` refuses a missing file and `compute_isochrons` an existing one,
and (marker `auto`) that a one-phase foliation computed from a config loads back
through it. `tests/test_isochron.py` checks `resample` on a circular arc (exact
spacing, the base point kept, within the traced steps' sag of the circle, $\eta$ signed
and monotone) and on a line (exact). The eight tests pass in 132 s at `-n 2`, the
`auto` one included.

**The slow oracle with the coworker's box** (`ISOCHRON_SETTINGS["full"]`,
`box_scale = 5`): passes, in 1026 s alone on the session.

**Per-file timing at the `ci` profile** (decision 8; Joon asked for it on
2026-10-28): `pytest -n 2 <file> -m "not slow and not auto" --hypothesis-profile=ci`,
as CI's `checks` job runs the tier, one file at a time and alone on the two-core, 7 GB
session:

| File | Time | Result |
|---|---|---|
| `test_bijections.py` | 1450 s | 271 passed |
| `test_splines.py` | 1190 s | 20 passed |
| `test_analysis_ode.py` | 1095 s | 42 passed, 1 failed (worker killed) |
| `test_normal_forms.py` | 1057 s | 26 passed, 2 failed (workers killed) |
| `test_experiment.py` | 188 s | 23 passed |
| `test_analytic.py` | 129 s | 8 passed, 1 xfailed |
| `test_training.py` | 57 s | 21 passed |
| `test_baseline.py` | 35 s | 9 passed |
| `test_constraints.py` | 33 s | 54 passed |
| `test_systems.py` | 26 s | 8 passed |
| `test_analysis.py` | 24 s | 12 passed |
| `test_data.py` | 21 s | 14 passed |
| `test_isochron.py` | 20 s | 15 passed |
| `test_linear.py` | 20 s | 13 passed |
| `test_models.py`, `test_symbolic.py` | 16 s each | 4 and 36 passed |
| `test_registry.py`, `test_windows.py` | 12 s, 11 s | 5 and 15 passed |
| `test_auto_codegen.py`, `test_auto_env.py`, `test_experiment_isochrons.py` | 8 s, 7 s, 7 s | 27, 3 and 5 passed |

The tier takes about 90 minutes at `-n 2`; four files hold 4800 s of it. The three
failures are workers that the kernel killed for memory: `dmesg` records two "Memory
cgroup out of memory" kills of a pytest worker at 3.8 GB and 3.1 GB resident, so two
workers at the `ci` profile exceed the session's 7 GB. The failed tests are
`test_limit_cycle_on_random_hopf_forms`, `test_cartesian_integration_is_finite_at_the_origin`
and `test_integrations_agree`. Run alone at the `ci` profile:

- `test_integrations_agree` passes, in 714 s at a 4.2 GB peak: alone it fits, beside a
  second worker it does not.
- `test_cartesian_integration_is_finite_at_the_origin` passes, in 10 s at 0.6 GB: its
  worker was killed for what earlier tests had left in it.
- `test_limit_cycle_on_random_hopf_forms` aborts (SIGABRT, exit 134) after 673 s at a
  4.0 GB peak, inside XLA's `backend_compile_and_load` under `find_limit_cycle`
  (`limit_cycle.py:364`). It aborts in Hypothesis's generation phase as well as while
  shrinking, so whether any example fails is not known.

**Every call of `find_limit_cycle` with a new `t_settle` or new parameters compiles
again.** Counted with `jax_log_compiles`: the first call compiles 45 programs, a repeat
0, a new `t_settle` 3 (`settle`, the return map's `filter_custom_jvp`, `polish`), and new
parameters with the same `t_settle` 1 (`polish`). Python floats such as `t_settle` and
the return-map period reach `eqx.filter_jit` as static arguments, so each value is a new
program. Over the 300 examples of the `ci` profile the compiled programs accumulate,
which fits the memory growth and the abort, and likely much of the time of the four
slowest files **[deduced]**. Passing them as arrays should remove it; that is O1's code
and decision 8's question, left for Joon.

**Reassessed 2026-10-28 (Fable 5.1), measured [code]:**

- The abort is the kernel's mapping limit, not memory. With pytest's fd capture off
  (`-s`) LLVM's message survives: "allocateMappedMemory failed with error: Cannot
  allocate memory … LLVM ERROR: Unable to allocate section memory!". XLA's CPU backend
  mmaps each compiled program's sections and keeps them while the program is cached;
  `vm.max_map_count` is 65530, and a `find_limit_cycle` call with a new `t_settle` adds
  about 470 mappings, so the limit falls near example 140, about 600 s in: the abort came
  at 673 s. A larger runner does not help; a count, not a size.
- The session's shell cgroup limit is 5.84 GiB (`memory.limit_in_bytes`), not 7 GB;
  the two killed workers, 3.8 and 3.1 GB, exceeded it together.
- Per call after warm-up: `find_limit_cycle` with a new `t_settle` 3 compiles, 22 MB,
  about 4 s; with the same `t_settle` 1 compile (`polish`, since `period_hat` always
  differs), 11 MB, 2 s. `t_max = 10 t_settle` is a static field of `PoincareMap`, the
  third program.
- Two more sources, recompiling on every call even for the same instance: the
  `r_squared` integration's `rhs_s`, a closure built inside `__call__`
  (`integration.py:117`), so diffrax's `filter_jit` sees a new function each time; and
  the closed form's `_radius_from_isostable`, whose two `fori_loop`s run op-by-op and
  compile their closure each call. `test_integrations_agree` makes about 5 compiles per
  example, 12 MB; 300 examples give its 4.2 GB.
- `test_splines.py` and `test_bijections.py` have a different profile: about 2500
  tiny op-by-op compiles per file at the `dev` profile (`jit(add)`, `jit(multiply)`,
  `jit(searchsorted)`), one per new shape the strategies draw, and heavy per-example
  work; no program-level recompile. Their cost is eager execution over shape-varying
  draws, not static floats.
- Fixes, in order: `t_settle`, `period_hat` and `t_max` as arrays; `rhs_s` as a method
  (`rhs_r_squared`, as `rhs_polar`); `filter_jit` on the closed form or on
  `radius_from_isostable`; a test that runs each driver twice with new values and
  asserts zero compilations the second time (`jax_log_compiles` plus a logging handler),
  recorded in `docs/coding-standards.md`; for the spline and bijection files, jitted
  checkers and strategies drawing shapes from a few sizes, or a lower `max_examples` on
  the heaviest tests. JAX's persistent cache would not lift the mapping limit.

**After the split**: `pytest -n 2 tests/test_isochron.py tests/test_auto_stages.py
tests/test_experiment_isochrons.py tests/test_auto_env.py -m "auto and not slow"`:
10 passed in 303 s; the AUTO-free tests of the same files 15 passed in 19 s. Stages 2
and 3 now restart from arrays rather than AUTO's objects, and every oracle holds to the
same tolerances.

**After the recompile fixes**: `find_limit_cycle` with a new `t_settle` and new
parameters compiles 0 programs (3 and 1 before); each of the four integrations with
new parameters compiles 0 (1, 0, 1, 2 before). The reanchor and phase tests pass (13
in 383 s at `-n 2`). The two files that lost workers at the `ci` profile, rerun at
`-n 2`:

| File | Before | After |
|---|---|---|
| `test_analysis_ode.py` | 1095 s, 42 passed, 1 worker killed | 274 s, 44 passed, run alone |
| `test_normal_forms.py` | 1057 s, 26 passed, 2 workers killed | 2577 s, 28 passed, 1 worker killed (3.0 GB) |

`test_analysis_ode.py` is four times faster and whole; `test_limit_cycle_on_random_hopf_forms`,
which aborted at 673 s, takes 21 s. `test_normal_forms.py` is not fixed, and its
earlier figure was short because the two killed tests, `test_integrations_agree`
among them (714 s alone), had stopped early; its first ten minutes also ran beside
the reanchor tests. Its remaining source is `WinfreeNormalForm`, whose `a` and `w`
are Python floats (`eqx.field(static=True)`, by ADR-0013 §5), so every Winfree draw
of the strategies is a new program for every jitted function it reaches; Hopf and
Bautin keep theirs as array leaves and compile nothing on a new draw (measured over
twelve draws of `all_normal_forms`: Winfree 1 `diffeqsolve` program each, the others
0). A test-side cost beside it: `test_hopf_floquet_multiplier_is_the_polar_monodromy`
calls `diffeqsolve` with a lambda built per example, 15 programs per 10 examples.

**Timing of a full foliation** (FitzHugh–Nagumo, the default settings, phases of the
coworker's files 100, 76, 51 and 26, run alone on the two-core session, before the one-thread fix, which on two cores
changes little):

| | Stages 1–3 | Four phases, total | Stage 4 per phase |
|---|---|---|---|
| one worker | 47 s | 1208 s | 290 s |
| two workers | — | 700 s | — |

No phase failed; the curves have 4749–7058 points, and every half stopped at
`eta_stop` or at the box. Two workers cut stage 4 from 1161 s to 653 s, a speed-up of
1.8.

## Open issues

- **`WinfreeNormalForm`'s parameters are static floats** (ADR-0013 §5), the last
  per-value recompile in the tree; it keeps `test_normal_forms.py` at 43 minutes and
  3 GB per worker at the `ci` profile. Joon's decision: array leaves with the
  $a = 0$ limit selected by `jnp.where`, as Bautin's closed forms handle their poles
  (recommended); or bounded draws in the Winfree strategy; or the nightly tier.

- **`find_limit_cycle` recompiles for every new `t_settle` and parameter set** (*Tests*,
  per-file timing): the likely cause of the `ci` tier's memory kills and of much of its
  90 minutes. The fix, floats passed to the jitted stages as arrays, is O1's and
  decision 8's.

- **The canceled `checks` job** is fixed on its own branch before this one is reviewed
  (plan decision 8).
- **`AbstractNormalForm.floquet_multiplier` is above 1 for a clockwise cycle.** It is
  $e^{\kappa T}$ with the signed period, so Winfree's attracting cycle reports 2576
  instead of $3.9\times 10^{-4}$; `test_normal_form_closed_forms_are_consistent` and
  `test_hopf_floquet_multiplier_is_the_polar_monodromy` (which integrates to the signed
  period) pin the same formula. Found by Step E; nothing in O7 uses it. The fix is
  $e^{\kappa\vert T\vert}$ and the two tests, on the normal-forms code's own branch.
- **The quick `auto` tier takes about 8 minutes on two cores** (*Tests*). Options: run
  the stage test on FitzHugh–Nagumo and Bautin only, give the foliation tests
  `max_workers=1` so that pytest's workers do not oversubscribe the cores, or mark the
  normal-form foliations `slow`.

## Review notes

| Change | Thoughts | Modifications |
|---|---|---|
| `docs/design/isochrons.md` — the design document: three versions of the method, AUTO background, the stable direction and the $+A^{\mathsf T}$ defect, accuracy, oracles, D1–D5 | Reads well. Very good. | None |
| `docs/references.md` — Hannam et al. 2022, Krauskopf & Rieß 2008, Osinga & Moehlis 2010 | Confirmed. | None |
| `docs/index.md` — the design document's row | Simple change. | None |
| `docs/roadmap.md` — O7 row points to the design document | Confirmed. | None |
| `docs/changes/2026-10-27-auto-isochrons.md` — this change document | Read through. Looks good. | None |
| `src/deep_isochron/analysis/ode/auto/fortran/isochron_multiseg.f90` — the bundle integrates the adjoint equation $-\tau A^{\mathsf T}w$ (D1) | Looks good. | None |
| `src/deep_isochron/analysis/ode/auto/codegen.py` — `indices_block` one slot per line; `_wrap` cuts only at blanks (D8) | Confirmed. | None |
| `tests/test_auto_codegen.py` — gfortran driver and compile tests in place of f2py; tests for the wrap, `indices.inc` and the adjoint sign  | Skimmed through. Will scrutinize later if needed. | None |
| `.github/workflows/ci.yml` — the `slow` job installs gfortran | Confirmed. | None |
| `docs/roadmap.md` — Done ledger row for Step A; the O7 row names the gfortran tests | Confirmed. | None |
| `pyproject.toml`, `uv.lock` — the `auto` dependency group (AUTO's Python layer from git at `44cc1c4`), the `auto` marker | Confirmed. | None |
| `scripts/install_auto.sh` — builds AUTO's Fortran tree at the locked commit | Had a look. For uv projects that have additional non-uv dependencies (compilers for c, c++, fortran, etc, interop with other languages, etc.), is using shell scripts the established way? | To be discussed. |
| `src/deep_isochron/analysis/ode/auto/_env.py` — `load_auto`, `auto_dir`, `pinned_commit`, `AutoNotInstalled` | Do we bake the auto installation dir in the module or expose it so that the user may provide a preferred path, with the current choice as the default? | To be discussed. |
| `src/deep_isochron/analysis/ode/auto/__init__.py` — docstring | Looks good. | None |
| `tests/test_auto_env.py` — tree resolution; the FitzHugh–Nagumo Hopf point through AUTO | Confirmed. | None |
| `.github/workflows/ci.yml` — the `auto` job | Confirmed. | None |
| `docs/auto.md` — installing AUTO | Looks good for now. Can we even ease the entire repo setup process even easier with say, a makefile? | To be discussed with Claude. |
| `docs/architecture.md` — `_env.py` in the module map | Simple change. | None |
| `docs/index.md` — the `auto.md` row | Confirmed. | None |
| `CLAUDE.md` — `-n 2` in the cloud sessions; the AUTO setup line | Confirmed. | None |
| `docs/roadmap.md` — Done ledger row for Step B | Confirmed. | None |
| `justfile` — the recipes `setup`, `install-auto`, `check`, `test`, `test-slow`, `test-auto`, `smoke` (R3) | `setup` is additional steps around `install-auto`. Remind me the rationale behind keeping both. | None |
| `pyproject.toml`, `uv.lock` — `rust-just` and `prek` in the `dev` group | Confirmed. | None |
| `prek.toml` — excludes `src/Isochron2PBVP/` and `tests/golden/` | Confirmed. | None |
| `.github/workflows/ci.yml` — every job runs `just` recipes | Confirmed. | None |
| `src/deep_isochron/analysis/ode/auto/_env.py` — the tree only from the environment (R2) | Looks mostly good, but `_env`'s auto installation behavior depends on the environment variables `DEEP_ISOCHRON_AUTO_DIR` and `AUTO_DIR`, but these are not named in the top-level package `README.md`. Instead, they are mentioned in passing in `docs/auto.md`. Shouldn't these be more explicitly named in the top-level readme along with the installation instructions? | To be discussed. |
| `tests/test_auto_env.py` — resolution tests through the variables | Like the tests but conflicted on its organization. Does the current grouping make more sense, or splitting into installation-related and auto functionality-related test files make more sense? | To be discussed. |
| `docs/auto.md` — `just setup`, the `uv sync` correction, platforms, "On a cluster" | Good, but couple thoughts. `auto.md` is not immediately relatable, especially to people not familiar with the continuation software. For example installing_auto07p might be clearer. (auto07p_installation.md? etc?) Also, need to decide on the scope of the document. Script instructions were being gathered in `/scripts/README.md`. Move to there? Create a hyperlink from there to here? De we need a `docs/guides/` directory? | To be discussed. |
| `README.md` — the package, setup, documentation, acknowledgment | The Hannam paper in the acknowledgement is missing proper citation (Its probably J. Hannam, B. Krauskopf, and H. Osinga. Isochron folications and flobal bifurcations: a case study. Transactions of Mathematics and Its Applications, 6, 2 (2022).) Also `index.md` is helpful for humans too, but its opening paragraphs make it clear that it is primarily intended for Claude. Either move the Claude instructions in `index.md` (Start from here and read what's relevant for each task) to CLAUDE.md and/or add a section for scripts - what exist, and how to use the main ones; for details link to `scripts/README.md`. | Docs organization principle to be discussed with Claude. |
| `docs/design/isochrons.md` — acknowledgment of Andrus Giraldo | Confirmed. | None |
| `CLAUDE.md` — the recipes; `prek` and its exclusions | Are there potential improvements / trimming do be done on the current `CLAUDE.md`? | To be discussed. |
| `docs/roadmap.md` — Done ledger row for Step B′; the O7 row names the consumer of the saved foliation | Looks good. | None |
| `src/deep_isochron/analysis/ode/limit_cycle.py` — `reanchor`; `_check_shooting` shared with `find_limit_cycle` | `nearest_cycle_time` rolls its own newton iterations. Why not optimistix? Also, `reanchor` runs Newton steps, followed by `_polish` which is the multiple shooting for limit cycle detection. Since we started with a converged limit cycle, and is looking to just tare the phase thought the process would be simpler than this. Do we have docs for `reanchor`? `analysis.md` is getting quite long and difficult to find relevant info in one pass. Maybe split into smaller documents per analysis? | To be discussed. |
| `src/deep_isochron/analysis/ode/__init__.py` — exports `reanchor` | Simple change. | None |
| `tests/test_analysis_ode.py` — the `reanchor` tests | Confirmed. | None |
| `docs/architecture.md` — `reanchor` in the module map | Confirmed. | None |
| `docs/roadmap.md` — Done ledger row for `reanchor` | Confirmed. | None |
| `src/deep_isochron/analysis/ode/auto/fortran/isochron_multiseg.f90` — phase 0 anchored on the section (decision 3) | Confirmed. | None |
| `src/deep_isochron/analysis/ode/auto/codegen.py` — section slots replace the anchor and stability slots | Confirmed. | None  |
| `tests/test_auto_codegen.py` — the slot layout | Confirmed. | None |
| `src/deep_isochron/analysis/ode/auto/isochron.py` — the pipeline (work in progress, untested) | Reviewing the file after all changes. Skipped.  | None |
| `src/deep_isochron/analysis/ode/__init__.py` — exports the pipeline | Simple change. | None |
| `src/deep_isochron/analysis/ode/auto/isochron.py`, Step D — `IsochronFoliation.dataset`, `save`, `load`; the stages finished | The code works well. However the file contains a lot of functions and is hard to grasp at once. Wondering if this could be refactered to enhance readability and make it easier to understand overall. Namings could be improved too. Expect `_isochron` to be called by `isochron_foliation`, but its actually `_trace_phase` that is called. Would be nice if the actual numerics and the housekeeping around them could be decoupled more somehow in the code. | Need discussions and grilling sessions with Claude. |
| `src/deep_isochron/plotting/isochrons.py` — `plot_isochrons` | Not that happy with the function. The plots in `prototype_analysis.ipynb` are more to my liking. Also the how the functions are used in the notebook may contain hints as to which methods are needed on various parts of the repo for better ergonomics. | To be discussed. |
| `src/deep_isochron/plotting/__init__.py` — exports `plot_isochrons` | Simple change. | None |
| `tests/test_isochron.py` — settings, phases, persistence, plot; bundle at multiplier 1; FitzHugh–Nagumo against the coworker | Do we need testing for plotting functions? Does some of the utility function `_arclength_from_base` have use elsewhere as well (for resampling or as plot helpers). If so, move to `utils`. | To be discussed. |
| `tests/helpers.py` — `TOL["isochron_curve"]`, `TOL["branch_point"]`, `ISOCHRON_SETTINGS` | Okay for now. May revisit later to reassess if these tolerance values are reasonable. | None |
| `tests/golden/fncMultiSeg_Iso_{100,26,51}.csv` — three golden isochrons | Confirmed. | None. |
| `justfile`, `.github/workflows/ci.yml` — `test-auto` takes `quick` or `full` | Confirmed. | None |
| `docs/architecture.md` — the pipeline and the plot in the module map | Confirmed. | None |
| `src/Isochron2PBVP/` — deleted; the coworker's run sequence and Fortran quoted in *Design* | Confirmed. | None |
| `pyproject.toml`, `prek.toml`, `scripts/check_docstring_headers.py` — the exclusions of `src/Isochron2PBVP` removed | Confirmed. | None |
| `CLAUDE.md`, `docs/coding-standards.md` — the exclusion lines name only `tests/golden/` | Makes sense. How much overlap are there between `CLAUDE.md` and other docs? Is this needed duplication? | To be discussed. |
| `docs/decisions/0016-symbolic-systems.md` — links at commit `3af99f2`; History line | Like the link. | None |
| `docs/design/isochrons.md` — the coworker's files linked at commit `3af99f2` | Looks good. | None |
| `docs/roadmap.md` — the O7 row; Done ledger row for Steps C–D | Confirmed. | None |
| `docs/decisions/0017-isochrons-by-auto-continuation.md` — ADR-0017: the method, the adjoint bundle, the section anchor, AUTO as a dependency, the pool, the netCDF file | Looks good, but the same link to commit containing Andrus' original scripts so that readers can directly inspect Andrus' original script if they want. | To be discussed. |
| `docs/index.md` — the ADR-0017 row | Confirmed. | None |
| `src/deep_isochron/analysis/ode/auto/isochron.py` — a box stop decided by the box edge; stage 1's docstrings | Private functions here starting with `_` don't have documentation compliant with our rules. This is a key analysis so its worth providing docs for private functions as well. | To be discussed. |
| `docs/design/isochrons.md` — stage 1 stores the orbit uncorrected; the O1 oracle after stage 2 | Want the 6. Our algorithm section to explicitly contain subheadings for stage 1-4, so that each stage of the algorithm really pops out and is legible to the reader. Seems that stage 2 is not even bolded currently. | To be discussed. |
| `tests/test_isochron.py` — the §8 oracles: stages 1–3 vs O1 and O2, the Hopf route, the closed-form phase, forward integration, the stop reason | Confirmed. | None |
| `tests/helpers.py` — five tolerances and `ISOCHRON_SETTINGS["reach"]` | Confirmed. | None |
| `pyproject.toml`, `docs/coding-standards.md` — ruff `W605` and its row | Confirmed. | None  |
| `docs/roadmap.md` — O7 step 2 done; Done ledger row for Step E | Confirmed. | None |
| `docs/decisions/0016-symbolic-systems.md` — the History line dated 2026-10-28 | Confirmed. | None |
| `src/deep_isochron/analysis/ode/auto/isochron.py` — one OpenMP thread per AUTO run; phase π; workers' JAX on the CPU | Confirmed. | None |
| `tests/test_isochron.py`, `tests/helpers.py` — phase π in the stage test; one worker in the named settings | Confirmed. | None |
| `docs/decisions/0017-isochrons-by-auto-continuation.md` — the timing; one thread per AUTO run, workers on the CPU | Sometimes, docs have a tendency to group unrelated ideas into a conjoined paragraph. As long as a paragraph conveys its one theme clearly, it is okay for it to be short for documentation writing. | Linebreaks and reordered text for better readability. |
| `isochron.py`, `docs/design/isochrons.md`, `tests/helpers.py` — `box_scale` defaults to 2.5 | Confirmed. | None |
| `tests/helpers.py` — `ISOCHRON_SETTINGS["full"]` keeps the coworker's box | Confirmed. | None |
| `src/deep_isochron/experiment/isochrons.py`, `experiment/__init__.py` — the isochron configs: phases, settings, cycle, hash, file, compute, load | Looking at functions like `isochron_phases` or `isochron_settings` makes me feel like dataclasses configs may be useful (since these would then be methods of a dataclass). Worth it? Can dataclass configs be used with yaml configs? | To be discussed. |
| `scripts/compute_isochrons.py`, `scripts/README.md` — the Hydra entry point | Script looks good. For the README, now that there are quite a bit of scripts, would it be better to ditch the table and switch to one script docs per subheading? Also, currently there are two types of scripts - the config related ones that run experiments, and the housekeeping ones for benchmarking or deterministic checks. These are different in nature so makes sense to sepaeate them. | Need discussion and grilling to determine better organization strategy for scripts. |
| `configs/isochrons/fhn.yaml`, `configs/isochrons/fhn-easy.yaml` — the isochron configs | Confirmed. Is phase_field's t_end an appropriate place for this? | To be discussed. |
| `configs/hydra/launcher/slurm.yaml`, `pyproject.toml`, `uv.lock` — the SLURM launcher and the `slurm` group | Confirmed. | None |
| `src/deep_isochron/analysis/ode/auto/isochron.py`, `analysis/ode/__init__.py` — `Isochron.resample`; `wrap_phases` public; INFO per phase | Confirmed. | None |
| `tests/test_experiment_isochrons.py`, `tests/test_isochron.py` — the configs and the script; `resample` | Confirmed. | None |
| `docs/auto.md`, `docs/decisions/0011-experiment-layer.md`, `docs/architecture.md` — the script, the launcher, the module map | Confirmed. | None |
| `CLAUDE.md` — a test outcome only after the test has run | Confirmed. | None |
| `src/deep_isochron/analysis/ode/auto/{results,runner,stages,foliation}.py` — the pipeline in four modules; `AutoSolution` at every seam; `auto_session`; `trace_isochron` | Much easier to read, that allowed me to find the following - `wrap_phases` is horrible code. Pretty sure everything can be done in vectoried numpy: `values=np.atleast_1d(np.asarray(phases, dtype=float))`. Cehck for ValueError on `np.any(bad)`; `bad=np.isfinite(values)`. `wrap_phase` works on arrays, and checking for duplicates is easily done bu `np.unique`. Never use python loops/list comps/etc to perform array operations elementwise in python loop. It's probably expressible in numpy/JAX logic. Also we are going to use x64 anyways, so why not set as the global JAX config at package import (instead of contextmanagers)? | Deferred to Claude. |
| `src/deep_isochron/analysis/ode/auto/__init__.py`, `analysis/ode/__init__.py` — re-exports | Confirmed. | None |
| `src/deep_isochron/utils/numerics.py` — `arclength` | Minor, but `jnp.cumulative_sum(..., include_initial=True)` should remove the need for a concatenation. | Deferred to Claude. |
| `tests/test_auto_stages.py`, `tests/test_isochron.py`, `tests/conftest.py`, `tests/helpers.py` — the tests split as the modules; one `auto` fixture; `PLANAR_SYSTEMS`, `anchored_cycle` | Confirmed. | None |
| `docs/design/isochrons.md` — §6 with a subheading per stage; stage 2 written out | Looks good. | None |
| `docs/architecture.md`, `docs/decisions/0017-isochrons-by-auto-continuation.md` — the four modules; the link to the coworker's scripts; History | Looks good. | None |
| `src/deep_isochron/analysis/ode/limit_cycle.py`, `poincare.py` — times as arrays into the jitted stages; `nearest_cycle_time` on Optimistix; `reanchor` without the polish | Does it make more sense for `reanchor` to be a method of `LimitCycle`? Also, not sure why we still retain `newton_step` when using Optimistix's Newton solver. `optx.Newton` has a `.step` method; furthermore, isn't initializing with a manual Newton step same as just running the solver one more time? | To be discussed. |
| `src/deep_isochron/systems/normal_forms/integration.py` — the `r_squared` field through `args`; the closed form jitted | Makes sense. | None |
| `tests/helpers.py`, `tests/test_analysis_ode.py`, `tests/test_normal_forms.py`, `docs/coding-standards.md` — the compile guards | Looks good, but take a look at `equinox.debug`. Anything useful there we are reinventing? | To be discussed. |
| `docs/design/analysis.md` — `nearest_cycle_time`, `reanchor`, the array rule | Looks good. | None |
| `README.md` — `DEEP_ISOCHRON_AUTO_DIR` and `AUTO_DIR` named; Langfield et al. and Hannam et al. cited in full | Confirmed. | None |
