---
type: change
status: reviewed 2026-10-29; the review's change in
updated: 2026-10-29
---

# `docs-layout` — guides, document conventions, one design document per analysis, the scripts in three kinds

## Summary

The review of `auto-isochrons` asked five questions about the documentation rather than
the code: `auto.md` was not relatable by name and mixed installation with usage;
`scripts/README.md` held a notebook guide and listed only one of four checks; the
scripts were of two kinds in one directory; `index.md` opened by addressing Claude; and
`CLAUDE.md` restated what other files carried. The grilling round of 2026-10-29 (Q1–Q8,
plus a conventions document) settled the layout, and this branch applies it. Nothing in
the package's behavior changes; the three check scripts gain two rules and one path.

Reversible: every move is a `git mv`, and the two new rules are two functions in one
script.

## Files

- `docs/guides/auto07p.md` — `docs/auto.md` renamed; the foliation section moved out.
- `docs/guides/isochrons.md` — new: computing a foliation, reading it back, SLURM.
- `docs/guides/experiments.md` — new: the notebook block of `scripts/README.md`, as a
  guide to `experiment` from Python.
- `docs/document-conventions.md` — new: what each document kind is for; frontmatter and
  status (from `index.md`); change documents (from `CLAUDE.md`); decision records,
  references, the Markdown rules (from `coding-standards.md`); design documents and
  provenance labels; paragraphs; the document checks.
- `docs/coding-standards.md` — the three moved sections removed; the tiers table names
  the `just` recipes instead of raw pytest commands, which had gone stale; the checks
  table carries the two new rules; the docstring rule covers private functions.
- `CLAUDE.md` — 87 lines to 50: the session protocol and the facts the code cannot carry;
  everything else a pointer.
- `docs/index.md` — opens for a human reader; a Guides table; four design rows replace
  the `analysis.md` row; `../CLAUDE.md` row dropped.
- `docs/design/cycle-estimation.md`, `limit-cycle.md`, `floquet.md`, `phase-field.md` —
  `analysis.md` split per item, each renumbered from §1 (`Cycle` is §1 of
  `cycle-estimation.md`; E1 is its §2; O3 and O4 are §1 and §2 of `phase-field.md`).
- `docs/design/analysis.md` — deleted.
- `docs/architecture.md` — the scripts block as a tree; `guides/` and `design/` named in
  the opening; `_env.py` and `nearest_phase` point at the new files.
- `docs/roadmap.md` — row P9; the O3 row and the method paragraph point at
  `phase-field.md`; Done ledger row.
- `docs/decisions/0012-analysis-layout.md` — Decision 4 and Consequences name one
  document per item; History.
- `docs/decisions/0009`, `0010`, `0011`, `0014`, `0015`, `0018`, `0019` — links follow
  the moved files; one History line each.
- `docs/references.md`, `docs/design/training-step-performance.md` — paths.
- `scripts/README.md` — the three directories, one table each; the notebook block gone.
- `scripts/checks/check_docstrings.py` — `check_docstring_headers.py` renamed, plus a
  docstring required on every `_`-prefixed function and class under `src/` (dunders
  exempt) and a 600-line bound on a module under `src/`.
- `scripts/checks/check_md_math.py` — moved, plus the frontmatter check (round 1): the
  block parses, carries `type`, `status`, `updated`, and writes `sources` in block form.
- the seventeen documents with a `sources:` line — the flow lists rewritten as block
  lists of quoted items (round 1).
- `scripts/checks/check_spelling.py`, `check_references.py` — moved;
  `check_references.py` resolves the root two levels up and reads `README.md` and
  `CLAUDE.md` by default.
- `scripts/diagnostics/bench_dataloader.py`, `find_nan.py` — moved.
- `prek.toml` — the four hook entries follow; `check-docstring-headers` is
  `check-docstrings`.
- `pyproject.toml`, `justfile`, `scripts/install_auto.sh`, `README.md` — paths in
  comments and prose.
- `src/deep_isochron/analysis/{data/cycle,ode/limit_cycle,ode/floquet,ode/phase,ode/poincare,ode/shooting}.py`,
  `ode/auto/_env.py`, `training/trainer.py` — docstrings point at the new documents and
  paths.
- `src/deep_isochron/systems/base.py`, `fitzhugh_nagumo.py`, `hodgekin_huxley.py` — the
  three private functions the new rule flagged get docstrings.
- `.gitignore`, `.DS_Store`, `src/.DS_Store` — the two tracked files removed; the
  pattern ignored.

## Design

- **Three kinds of document, by what the reader wants.** `design/` is why the
  mathematics is what it is, `decisions/` why the structure is, `guides/` how to do
  something. `auto.md` carried an installation and a usage, so it is two guides, each
  named for what the reader wants to do. Alternative rejected: one `auto.md` with both
  halves, because the installation is read once per machine and the usage once per
  experiment.
- **One design document per analysis**, as `isochrons.md` already was. Each is
  renumbered from §1, so the section numbers cited elsewhere changed, and every citation
  in code, ADRs, the roadmap's current plan and `references.md` was followed; change
  documents keep the old numbers as history. Alternative rejected: keeping the original
  numbering inside each split, which would have left documents opening at §2.
- **A conventions document for documents**, so that what `coding-standards.md` says
  about code it does not also say about Markdown, and `CLAUDE.md` can point rather than
  restate. `coding-standards.md` keeps docstrings, which ruff and the script enforce.
- **`CLAUDE.md` holds the protocol and the facts no file can carry**: the cloud session's
  core count and `apt-get` line, float64, the jaxtyping hook, the `uv.lock` and
  notebook rules. Recipes, profiles, hooks and document rules are pointers now; a
  document that restates configuration is a cache that goes stale (the tiers table had).
- **Two new checks rather than two rules in prose**, from the retro of the
  `auto-isochrons` review: private functions documented (asked for in the isochron
  pipeline; ruff's `D103` skips `_` names by design), a module under 600 lines (the
  pipeline reached 853 before review called it hard to grasp; the largest module now is
  543). Each was planted with a violation and watched fire before the commit.
- **`scripts/` by who calls it**: the entry points and `install_auto.sh` at the top,
  where a user and CI look; `diagnostics/` for instruments a person runs on a run;
  `checks/` for what `prek` runs. Alternative rejected: a flat directory with a README
  partitioned by subheading, since `ls scripts/` is what a newcomer sees first.
- **Frontmatter `sources` as a block list of quoted items** (round 1). A YAML flow list
  splits at every comma, so `ADR-0012 (analysis layout, amendment)` rendered as two
  sources on GitHub. Quoting every item is uniform and needs no judgment about which
  items contain a comma, colon or quote; the check in `check_md_math.py` refuses the
  flow form so the rule holds without a reviewer.
- No ADR: the layout is reversible and not surprising; this section and
  `document-conventions.md` § What goes where record it.

## Bugs fixed

- `docs/coding-standards.md` § Tiers listed `uv run pytest -n 4` commands that the
  `justfile` had replaced; the table now names the recipes.
- `docs/coding-standards.md` § Docstrings said the `D` rules were switched on "module
  group by module group" through `per-file-ignores`; only notebooks are exempt now.
- `README.md` was outside `check_references.py`'s scope, which is how its Hannam
  acknowledgment went uncited until review.

## Tests

No test changes. The checks that cover this branch, each run on the final tree:

- `uv run python scripts/checks/check_docstrings.py`: 0 problems after the three
  docstrings were added; a planted `_x` without a docstring and a planted 601-line
  module each flagged (then removed).
- `uv run python scripts/checks/check_references.py`: all 88 cited keys resolve, no
  broken relative links; a planted `(Nobody 1999)` in `README.md` flagged (then
  removed).
- `uv run python scripts/checks/check_md_math.py`: 53 files, 0 problems; a planted flow
  `sources` list, a missing `updated` and a block with an unquoted quote each flagged
  (then removed).
- `uv run just check`: every hook passed, `ty` clean (after ten docstring lines that the
  longer paths pushed past 88 characters were reflowed); after round 1: every hook
  passed, `ty` clean.
- `uv run just test dev tests/test_analysis.py tests/test_analysis_ode.py tests/test_symbolic.py`
  (the modules whose docstrings changed): 92 passed in 300 s on two cores.

## Open issues

- `docs/changes/2026-10-28-phase-field.md` (review pending) cites `analysis.md §4–5`;
  change documents are history and keep the old names.
- The `configs/phase_field/` group for O3's `t_end`, deferred from `auto-isochrons`.
- The uncited entries that `check_references.py` lists are informational and predate
  this branch.

## Review notes

| Change | Thoughts | Modifications |
|---|---|---|
| `docs/guides/auto07p.md`, `docs/guides/isochrons.md` — `auto.md` renamed and split into installation and usage | Good, but the sources frontmatter contains characters , and - that mess up how the the frontmatter is rendered. | TDeferred to Claude. |
| `docs/guides/experiments.md` — the notebook guide to `experiment`, out of `scripts/README.md` | Looks good. | None |
| `docs/document-conventions.md` — what each document kind is for, and the forms, gathered from four files | Looks good and agree on the convention. | None |
| `docs/coding-standards.md` — document sections moved out; tiers as recipes; the new check rows | Looks good. | None |
| `CLAUDE.md` — protocol and facts only, 50 lines | Looks good. | None |
| `docs/index.md` — for a human reader; Guides table; four design rows | Looks good. | None |
| `docs/design/cycle-estimation.md`, `limit-cycle.md`, `floquet.md`, `phase-field.md` — `analysis.md` split per item, renumbered | Looks good. Would it be worth setting down a more concrete format for design docs? (This would then be added to the documents-convention file as well.) | To be discussed. |
| `docs/architecture.md` — the scripts tree; the directories named | Looks good. | None |
| `docs/roadmap.md` — P9; the O3 row; the ledger |  Looks good. | None |
| `docs/decisions/0012-analysis-layout.md` and the six ADRs with moved links — History lines | Looks good. | None |
| `scripts/README.md` — three tables for three directories | Confirmed. | None |
| `scripts/checks/check_docstrings.py` — renamed; private docstrings; the 600-line bound | Looks good. | None |
| `scripts/checks/check_references.py` — root two levels up; `README.md` and `CLAUDE.md` in scope | Confirmed. | None |
| `scripts/checks/`, `scripts/diagnostics/` — the moves; `prek.toml` | Confirmed. | None |
| `src/deep_isochron/systems/{base,fitzhugh_nagumo,hodgekin_huxley}.py` — the three docstrings the rule flagged | Confirmed. | None |
| the docstrings of `analysis/` and `training/trainer.py` — paths and section numbers | Confirmed. | None |
| `.gitignore`, `.DS_Store` — untracked and ignored | Confirmed. | None |
| `scripts/checks/check_md_math.py`, `docs/document-conventions.md` § Frontmatter, and the seventeen `sources:` lists — round 1: `sources` as a block list of quoted items, checked | | |
