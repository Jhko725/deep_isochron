---
type: change
status: in progress; Steps A and B done, the design document out for review
updated: 2026-10-27
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
commit by `scripts/install_auto.sh`. The pipeline itself (Steps C–E) follows.

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
- `CLAUDE.md`: `-n 2` in the cloud sessions; the AUTO setup line.
- `docs/references.md`: entries for Hannam et al. 2022, Krauskopf & Rieß 2008 and
  Osinga & Moehlis 2010.
- `docs/index.md`: the rows of the design document and `auto.md`.
- `docs/roadmap.md`: the O7 row points to the design document and names the gfortran
  tests; the Done ledger records Steps A and B.
- `docs/changes/2026-10-27-auto-isochrons.md` (**new**): this document.

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
- **The remaining decisions are design-document decisions for later steps:** D2 (the
  stage-4 start on AUTO's grid), D3 (the oracle's tolerance), D4 (one segment per
  isochron), D5 (the rotating base point, for ADR-0017), D6 (`ntst` scaling with
  `n_periods`) and D7 (the stop reason of each half).

## Bugs fixed

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

## Open issues

- **The canceled `checks` job** is fixed on its own branch before this one is reviewed
  (plan decision 8).
- **Steps B–E** of the plan: the AUTO dependency, `reanchor` and the pipeline, the
  results and persistence, the oracles and ADR-0017.

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
