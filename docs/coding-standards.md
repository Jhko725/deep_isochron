---
type: reference
status: current
updated: 2026-10-29
---

# Coding standards

This document says where this repository enforces the conventions for code, tests and
docstrings, and which checks stand in for prose. The conventions themselves, with the
reasoning behind them, are in the `code-conventions` skill, and the pervasive ones are in
the global `CLAUDE.md`; the conventions for the documents under `docs/` are in
[`document-conventions.md`](document-conventions.md). Read this document when a task
touches tiers, tolerances, contracts or checks.

## Tiers and commands

The commands are the recipes of the `justfile` (`uv run just --list`), and CI runs the
same recipes. `just test [profile] [paths]` runs pytest at `-n auto`, the machine's core
count; the Hypothesis profiles are `dev` (10 examples), `default` (50) and `ci` (300,
derandomized).

| Purpose | Recipe |
|---|---|
| Fast, while editing | `just test dev <test files of the touched modules>` (`slow` tests excluded; `dev` is the default profile) |
| Full, once at the end | `just test default`, then `just test ci` when the branch is about to merge |
| Slow tier | `just test-slow [profile]`: FitzHugh–Nagumo's limit cycle, the autoencoder training test |
| AUTO | `just test-auto quick` or `full`: the tests marked `auto`, which skip without a built tree |
| Every hook, then types | `just check`: the `prek` hooks on every file, then `ty check src`, clean, with no `# ty: ignore` |

Read the baseline before editing rather than running the full suite to discover what is
already red: the branch change document's *Tests* and *Open issues*, the pinned
`xfail`s, and the `UNTESTED` entries in `tests/registry.py`.

## Named values

- Tolerances are the entries of `TOL` in `tests/helpers.py`, one per *kind of claim*
  rather than per test (`jit_eager`, `identity`, `closed_form`, `join`, `jacobian`,
  `bspline_inverse`, `flow`, `floquet`, the `cycle_*` family), each carrying its reason.
  A number that *is* a law stays inline with a comment saying so: the asymptotic-slope
  bound in `test_sinh_asymptotically_linear`, finite-difference steps, a strategy's
  magnitude range.
- Solver configurations are the entries of `SOLVERS` in `tests/helpers.py` (`tight`,
  `tight_stiff`, `tight_long`, `stiff`, `data`), with one line of reason each. `Tsit5` is
  the non-stiff default, on diffrax's own recommendation.
- Exceptions to a law belong in `tests/registry.py` with a stated reason, never in a
  silent tolerance bump. `IDENTITY_TOL` was deleted for that reason: since every vector
  bijection is the identity at init to round-off, an entry in it would have meant "this
  class is not", which is what the universal law forbids.

## Contracts this repository fixes

| Contract | Where | ADR |
|---|---|---|
| One unconstrained `raw` leaf; `constrain(raw)` the single conversion site | `model/invertible/base.py` | 0001 |
| `constrain(0)` is the identity, so a fresh layer is the identity map | | 0002 |
| `smoothness` declared on every bijection; containers take the minimum | | 0003 |
| `eqx.AbstractVar` as `eqx.field(static=True, init=False)`, with `default=` when the class fixes the value and assigned in `__init__` when derived — never a bare attribute, `ClassVar` or property, which type checkers reject as overrides | | 0004 |
| Constraint primitives are plain Python objects created *inside* `constrain`, never stored as fields; nothing JAX runs at construction or import | `model/invertible/constraints.py` | 0005 |
| Templates are instances with `raw=None`, hashable, with zero trainable size | | 0001 |
| Rotations by Cayley transform, never `jax.scipy.linalg.expm` | `model/invertible/linear.py` | 0010 |
| Every concrete `AbstractBijection` registered in `tests/registry.py`, or in `UNTESTED` with a reason; `tests/test_registry.py` enforces it | `tests/registry.py` | — |
| A system is declared once in SymPy; the JAX `rhs` and the AUTO Fortran are derived from the same expressions | `systems/`, `analysis/ode/auto/` | 0016 |

`periodic_schur` raises `NotImplementedError` above two dimensions on purpose, naming
`slycot.math.mb03wd` as the route if a three-dimensional system appears.

## Deterministic checks

The configuration lives in `pyproject.toml`, in `[tool.codespell]` and in the `banned-api`
table with `TID251` in ruff's `select`, with a comment there on each non-obvious setting.
This section says what each check is for; the file has the current values.

| Rule | Check | Runs in |
|---|---|---|
| American English, irregulars and typos | `codespell` 2.4.3 | `prek`, CI |
| American English, the productive families | `scripts/checks/check_spelling.py` | `prek`, CI |
| `copy.replace` (this project's floor is Python 3.13); Cayley rotations | ruff `TID251` banned-api | `prek`, CI |
| the documents: both Markdown previews, every citation and relative link resolves | `scripts/checks/check_md_math.py`, `scripts/checks/check_references.py`; [`document-conventions.md`](document-conventions.md) | `prek`, CI |
| a docstring on every public module, class, method and function | ruff `D100`–`D103` (notebooks exempt) | `prek`, CI |
| `**Arguments:**`, `**Returns:**`, `**Raises:**`, `**Attributes:**` each on its own line with a blank line after; a docstring on every `_`-prefixed function and class under `src/` (dunders exempt), since ruff's `D103` skips them; no module under `src/` over 600 lines | `scripts/checks/check_docstrings.py` | `prek`, CI |
| no invalid escape such as `\Phi` in a docstring that is not raw (Python only warns) | ruff `W605` | `prek`, CI |
| a driver called again with new values compiles nothing: times and parameters reach `filter_jit` as arrays, a vector field is a module-level function or a bound method, loops run under a jit | `tests/helpers.py` `CompileCounter` in `test_find_limit_cycle_compiles_nothing_for_new_values` and `test_integrations_compile_nothing_for_new_parameters` | `prek`, CI (the tests) |
| no `# ty: ignore` under `src/` | `! grep -rn 'ty: ignore' src` (ty itself honors the comments) | CI |

The one deliberate exemption is in `scripts/diagnostics/bench_dataloader.py`, which times
`jax.scipy.linalg.expm` against the closed-form rotation. That timing is the measurement
ADR-0010 rests on, so that one line carries `# noqa: TID251` with the reason. The ban is
for code that *uses* `expm`, not for the benchmark that justified banning it.

**Two spelling checks, because one tool cannot do both jobs.** `codespell` is a
535-entry British-to-American word list: the right shape for irregulars (*behaviour*, <!-- codespell:ignore -->
*centre*, *artefact*, *modelled*) and for typos, and it is what caught `centerd`, <!-- codespell:ignore -->
`implemention`, `documention` and `constract`. It is the wrong shape for this <!-- codespell:ignore -->
repository's vocabulary, because `-ise` is productive morphology and no list ships
`discretise`, `parametrise`, `orthogonalise`, `diagonalise`, `linearise` or <!-- codespell:ignore -->
`denormalise` [code: measured against codespell 2.4 and `breame`, 2026-10-25; both miss <!-- codespell:ignore -->
all six]. `scripts/checks/check_spelling.py` inverts the problem. It applies the rule
(`-ise`/`-isation`/`-isable`/`-yse` take a `z`) and keeps the exception list, which is
the part that *is* closed: the sixty-odd words where `-ise` is also correct in American
English, from `advertise` to `otherwise`. `-isation` and `-isable` have no exceptions at
all, and `-yses` is left out of the rule on purpose, since it is also the plural of
`-ysis` (*the analyses*, which is correct).

**`docs/changes/` is skipped on purpose.** A landed change document is a record of what
was said, and its *Review notes* table holds Joon's words verbatim, typos included
(*annotationns*, *spefics*, *useing*). `CLAUDE.md` says Claude never edits those columns, <!-- codespell:ignore: quoted typos -->
and several lines quote a spelling *as* the thing that was fixed (the `centerd` typo, <!-- codespell:ignore: quoted typos -->
`normalise_phase_plane` → `normalize_phase_plane`), so correcting them would destroy the
history they exist to carry. The current documents are in scope: `design/`,
`decisions/`, `roadmap.md`, `architecture.md` and `index.md`.

A quotation keeps its source's spelling even when that spelling is British. Where one
appears, the line carries `<!-- codespell:ignore: cited quotation -->`; `design/floquet.md`
quotes diffrax's phrase for forward-mode differentiation and is the current instance.
Both checks honor that marker, and `spelling:ignore` as well.

The `ignore-words-list` in `pyproject.toml` covers the false positives this domain
produces: the LaTeX contour-integral command, the numerator and denominator locals in
`rational_quadratic.py`, two journal abbreviations, an author's surname, and a Fortran
intent keyword.

## Docstrings

A docstring is read in an editor hover or at the top of a file, so it is short, and it
tells the reader what the code is for and how it is shaped, not why it was chosen. The
reasoning lives in the ADRs and the design documents, and a docstring cites them
(ADR-0018). The prose rules are the `clear-prose` skill's; this section gives the shape.

**The module docstring is the map.** Its first sentence says what the module provides.
A paragraph then gives the design at the level a reader needs to navigate the file: the
public objects, how they relate, and the invariants that span several functions (a
direction rule, a jit rule, "planar only"). It names each route or variant in a phrase
and gives none of them its accuracy, arguments or failure modes. The last line names the
ADR or design document section that holds the reasoning and the citations. A module
docstring is at most about 25 lines; this is a standard, not a check, so a module that
needs 30 lines gets them, with the reason in its review row.

**A function docstring is the contract.** It opens with one sentence saying what the
function computes or does, in the active voice with the function as the actor, and then
carries everything the caller needs for that call, in the equinox form: `**Arguments:**`,
one bullet per argument; `**Returns:**`; `**Raises:**` when the function raises on its
own. Each header stands on its own line, followed by a blank line and then the bullets
or the paragraph, as diffrax writes them (`scripts/checks/check_docstrings.py`); a
one-argument function whose opening sentence names its argument needs no
`**Arguments:**` list. The precision or validity limits of that function, and the
citation for a formula it implements (author, year and locator), belong here, not in
the module docstring.

**A class docstring** opens the same way and documents the fields in an
`**Attributes:**` list, as diffrax does; a field carries no attribute docstring of its
own, so that each attribute is described once.

The test for a sentence: if deleting it from the module docstring would leave a caller
of one function uninformed, it belongs in that function's docstring; if deleting it from
a function docstring would leave a reader of the module unable to choose between
functions, it belongs in the module docstring. A sentence that passes neither test is
deleted. Material cut from a docstring that no document holds is moved rather than
dropped: a finding to the design document, with its provenance label; a decision to the
ADR, as a dated amendment.

A test's docstring is one sentence: the property it pins down, and the oracle or fixture
it is checked against. Math in a docstring is written as `$…$`, the same form as the
documents; nothing renders docstrings, and the form reads in a hover.

Every module, class, function and method has a docstring: the public ones by ruff
`D100`–`D103`, the `_`-prefixed ones under `src/` by `scripts/checks/check_docstrings.py`,
since ruff skips them and the review of the isochron pipeline asked for the private
stages to be documented like the public ones. The same script bounds a module under
`src/` at 600 lines: the one that reached 850 was "hard to grasp at once", and the
signal to split comes before that.

## Deliberate exceptions

- `tests/golden/` holds reference data byte for byte; `prek.toml` excludes it from every
  hook.
- `uv.lock` is not regenerated in a cloud session: the container's `uv` rewrites the lock
  format and produces thousands of noise lines for a handful of real version changes
  (`CLAUDE.md` carries the rule).
