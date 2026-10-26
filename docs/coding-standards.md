---
type: reference
status: current
updated: 2026-10-26
---

# Coding standards

Where this repository enforces the conventions, and which checks stand in for prose. The
conventions themselves, with the reasoning behind them, are in the `code-conventions`
skill; the pervasive ones are in the global `CLAUDE.md`. This document is the place to
look when a task touches tiers, tolerances, contracts or checks.

## Tiers and commands

| Purpose | Command |
|---|---|
| Fast, while editing | `uv run pytest <files touched> -m "not slow" --hypothesis-profile=dev` |
| Full, once at the end | `uv run pytest -n 4` (default profile, 50 examples) |
| Derandomized | `uv run pytest -n 4 --hypothesis-profile=ci` (300 examples) |
| Slow tier | add `-m slow`: FitzHugh–Nagumo's limit cycle, the autoencoder training test |
| Types | `uv run ty check src/deep_isochron/model/invertible`, clean, with no `# ty: ignore` |
| Markdown | `uv run python scripts/check_md_math.py`, 0 problems |
| Citations | `uv run python scripts/check_references.py`, every key resolves |
| Docstrings | `uv run ruff check`, with `D100`–`D103` selected for the modules the prose pass has reached |

Read the baseline before editing — the branch change document's *Tests* and *Open
issues*, the pinned `xfail`s, the `UNTESTED` entries in `tests/registry.py` — rather than
running the full suite to discover what is already red.

## Named values

- **Tolerances**: `TOL` in `tests/helpers.py`, one entry per *kind of claim* rather than
  per test — `jit_eager`, `identity`, `closed_form`, `join`, `jacobian`,
  `bspline_inverse`, `flow`, `floquet`, the `cycle_*` family — each carrying its reason.
  A number that *is* a law stays inline with a comment saying so: the asymptotic-slope
  bound in `test_sinh_asymptotically_linear`, finite-difference steps, a strategy's
  magnitude range.
- **Solver configurations**: `SOLVERS` in `tests/helpers.py` — `tight`, `tight_stiff`,
  `tight_long`, `stiff`, `data` — one line of reason each. `Tsit5` is the non-stiff
  default, on diffrax's own recommendation.
- **Exceptions to a law** belong in `tests/registry.py` with a stated reason, never in a
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

## Markdown that renders in both previews

GitHub runs Markdown before MathJax; VS Code's preview parses `$…$` first. The rules
`scripts/check_md_math.py` enforces, and the reason each exists:

- No line-length limit (ruff formats Python only), but an inline formula `$…$` never
  crosses a line.
- Display math only as a standalone block: `$$` alone on a line, the formula, `$$` alone
  on a line. Never `$$…$$` inside a paragraph.
- Inside inline math, no backslash followed by ASCII punctuation: `\,` `\;` `\{` `\|`
  `\\` become `\thinspace` `\medspace` `\lbrace` `\Vert`, and matrices move into blocks.
- No `*` inside inline math — use `\ast`, since `x^*` loses its star and italicizes the
  rest of the paragraph.
- Spaces around `<` and `>`.
- No `_` immediately after `}` `)` `|`: write `\mathbf e_r`, `C_{\rm loc}^{k}`,
  `\vert_{E}`, because such an `_` opens Markdown emphasis.

Every span has also been checked against KaTeX as VS Code configures it, so the two
previews agree.

## Checks that replace prose

The configuration lives in `pyproject.toml` (`[tool.codespell]`, and the `banned-api` table
with `TID251` in ruff's `select`), with a comment there on each non-obvious setting.
This section says what each check is for; read the file for the current values.

| Rule | Check | Runs in |
|---|---|---|
| American English, irregulars and typos | `codespell` 2.4.3 | `prek`, CI |
| American English, the productive families | `scripts/check_spelling.py` | `prek`, CI |
| `copy.replace` (this project's floor is Python 3.13); Cayley rotations | ruff `TID251` banned-api | `prek`, CI |
| both Markdown previews | `scripts/check_md_math.py` | `prek`, CI |
| every citation resolves to `docs/references.md` | `scripts/check_references.py` | `prek`, CI |
| a docstring on every public module, class, method and function | ruff `D100`–`D103`, `per-file-ignores` naming the modules not yet reached | `prek`, CI |
| no `# ty: ignore` under `src/` | `! grep -rn 'ty: ignore' src` (ty itself honors the comments) | CI |

One deliberate exemption: `scripts/bench_dataloader.py` times `jax.scipy.linalg.expm` against
the closed-form rotation, which is the measurement ADR-0010 rests on, so that one line
carries `# noqa: TID251` with the reason. The ban is for code that *uses* `expm`, not for
the benchmark that justified banning it.

**Two spelling checks, because one tool cannot do both jobs.** `codespell` is a
535-entry British-to-American word list: the right shape for irregulars (*behaviour*, <!-- codespell:ignore -->
*centre*, *artefact*, *modelled*) and for typos — it is what caught `centerd`, <!-- codespell:ignore -->
`implemention`, `documention` and `constract`. It is the wrong shape for this <!-- codespell:ignore -->
repository's vocabulary, because `-ise` is productive morphology and no list ships
`discretise`, `parametrise`, `orthogonalise`, `diagonalise`, `linearise` or <!-- codespell:ignore -->
`denormalise` [code: measured against codespell 2.4 and `breame`, 2026-10-25; both miss <!-- codespell:ignore -->
all six]. `scripts/check_spelling.py` inverts the problem: it applies the rule
(`-ise`/`-isation`/`-isable`/`-yse` take a `z`) and keeps the exception list, which is
the part that *is* closed — the sixty-odd words where `-ise` is also correct in American
English, from `advertise` to `otherwise`. `-isation` and `-isable` have no exceptions at
all; `-yses` is left out of the rule on purpose, since it is also the plural of `-ysis`
(*the analyses*, which is correct).

**`docs/changes/` is skipped on purpose.** A landed change document is a record of what
was said, and its *Review notes* table holds Joon's words verbatim — typos included
(*annotationns*, *spefics*, *useing*). `CLAUDE.md` says Claude never edits those columns, <!-- codespell:ignore: quoted typos -->
and several lines quote a spelling *as* the thing that was fixed (the `centerd` typo, <!-- codespell:ignore: quoted typos -->
`normalise_phase_plane` → `normalize_phase_plane`), so correcting them would destroy the
history they exist to carry. The current documents — `design/`, `decisions/`,
`roadmap.md`, `architecture.md`, `index.md` — are in scope.

A quotation keeps its source's spelling even when that spelling is British. Where one
appears, the line carries `<!-- codespell:ignore: cited quotation -->`; `design/analysis.md`
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
(ADR-0018). The prose rules are the `clear-prose` skill's; what follows is the shape.

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
or the paragraph, as diffrax writes them; a one-argument function whose opening
sentence names its argument needs no `**Arguments:**` list. The precision or validity limits of that function, and the citation for a formula
it implements (author, year and locator), belong here, not in the module docstring.

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

Every public module, class, function and method has a docstring (ruff `D100`–`D103`); a
private helper gets one when its name does not say what it does. The rule is switched
on module group by module group as the prose pass of 2026-10-26 reaches them, through
`per-file-ignores` in `pyproject.toml`; the list there is the current frontier.

## References

One bibliography, `docs/references.md`, and one in-text form. The key is `Author Year`
for one author (`Lust 2001`), `Author & Author Year` for two (`Wilson & Moehlis 2016`),
`Author et al. Year` for three or more (`Dieci et al. 1997`), with a letter suffix on a
collision (`Langfield et al. 2025a`). <!-- references:ignore: the example --> A locator follows the key after a comma, in one of
the forms `Eq. 7`, `Eqs. 9–14`, `§3`, `Prop. 5`, `Theorem 2`, `Table 5`, `Fig. 9`,
`pp. 210–214`, `Ch. 8`, `App. D`, `Remark 4`; several citations are separated by
semicolons inside one pair of parentheses. Every key resolves to an entry in
`docs/references.md`, which `scripts/check_references.py` enforces; the entry carries the
full reference, a DOI or arXiv identifier, and the locators the repository uses. Design
documents and ADRs carry no reference list of their own; their frontmatter `sources:`
lists keys.

## Deliberate exceptions

- `uv.lock` is not regenerated here: the container's `uv` rewrites the lock format and
  produces thousands of noise lines for a handful of real version changes. Joon runs
  `uv lock` after pulling.
- `prototype.ipynb` and `prototype_analysis.ipynb` are Joon's own records. Claude updates
  them to follow an API change and does not restructure them.
- `src/Isochron2PBVP` is a coworker's reproduction, read as reference rather than edited;
  its Fortran is excluded from the spelling check.
