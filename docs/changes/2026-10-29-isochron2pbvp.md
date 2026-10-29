---
type: change
status: implemented 2026-10-29; review pending
updated: 2026-10-29
---

<<<<<<< HEAD
# `isochron2pbvp` — the coworker's directory restored as his working copy
=======
# `isochron2pbvp` — the coworker's AUTO-07p reproduction as a project of its own
>>>>>>> 2e58f90ba3b74b1a88a3a38084c476e9e6b1b522

## Summary

Step D of `auto-isochrons` deleted `src/Isochron2PBVP`, Andrus Giraldo's AUTO-07p
reproduction, once the pipeline reproduced it (`1749851`), leaving links pinned at
<<<<<<< HEAD
`3af99f2`. Andrus is to keep working on it in this repository, so the directory comes
back exactly as his last commit `d3edbfe` left it, which is byte for byte the version
at `3af99f2` (`git diff --stat d3edbfe 3af99f2 -- src/Isochron2PBVP` is empty), minus
the three `.DS_Store` files, which `.gitignore` now refuses. The checks exclude it again
as they did before Step D.

Reversible: one `git rm -r` and the five exclusion lines.

## Files

- `src/Isochron2PBVP/` — 110 files from `d3edbfe`: `isoFunctions.py`, `plotIsochron.m`
  and `.asv`, `fncMultiSeg/` (`c.fncMultiSeg`, `fncMultiSegIso.f90`,
  `mainIsochronMultiSegFNC.py`, `dataIso/` with the 100 isochron CSVs), `deprecated/`.
- `prek.toml` — the exclude regex covers it again.
- `pyproject.toml` — `[tool.ty.src] exclude`, `[tool.ruff] exclude`, the `codespell`
  skip.
- `scripts/checks/check_docstrings.py` — `EXCLUDED`.
- `scripts/checks/check_spelling.py` — `SKIP_PARTS`.
- `docs/design/isochrons.md`, `docs/decisions/0016`, `0017` — "removed from the tree"
  becomes "his working copy; the link pins the version compared"; History lines.
- `docs/coding-standards.md` — the exception listed under *Deliberate exceptions*.
- `docs/architecture.md` — the directory in the map.
=======
`3af99f2`. Andrus is to keep iterating on it in this repository, in a monorepo shape:
one clone, one history, two projects side by side. The directory comes back as his last
commit `d3edbfe` left it, which is byte for byte the version at `3af99f2`
(`git diff --stat d3edbfe 3af99f2 -- src/Isochron2PBVP` is empty), minus the three
`.DS_Store` files that `.gitignore` now refuses, at the repository's root as
`isochron2pbvp/`, with a README in his name. The package's checks stop at its edge.

Reversible: one `git rm -r`, the README, and four exclusion lines.

## Files

- `isochron2pbvp/` — 110 files from `d3edbfe`: `isoFunctions.py`, `plotIsochron.m` and
  `.asv`, `fncMultiSeg/` (`c.fncMultiSeg`, `fncMultiSegIso.f90`,
  `mainIsochronMultiSegFNC.py`, `dataIso/` with the 100 isochron CSVs), `deprecated/`.
- `isochron2pbvp/README.md` (**new**) — what the directory is, its layout, how the
  driver runs under AUTO's Python, and which of its files the package is checked against.
- `prek.toml` — the exclude regex covers it.
- `pyproject.toml` — `[tool.ruff] exclude` and the `codespell` skip. No `ty` exclusion:
  `ty check src` never sees the root directory.
- `scripts/checks/check_spelling.py` — `SKIP_PARTS`, since it walks the whole tree.
- `src/deep_isochron/analysis/ode/auto/stages.py` — the module docstring names the
  project.
- `docs/design/isochrons.md`, `docs/decisions/0016`, `0017` — "removed from the tree"
  becomes "his project, read at commit `3af99f2`"; History lines.
- `docs/coding-standards.md` — the exception under *Deliberate exceptions*.
- `docs/document-conventions.md` — one paragraph under *What goes where*: the
  conventions stop at the directory's edge.
- `docs/architecture.md` — the directory as the second project in the map.
- `CLAUDE.md` — Claude does not edit it unless asked.
>>>>>>> 2e58f90ba3b74b1a88a3a38084c476e9e6b1b522
- `docs/roadmap.md` — Done ledger row.

## Design

<<<<<<< HEAD
- **Restored at its original path**, `src/Isochron2PBVP`, rather than under a new
  directory, so that Andrus's own checkout, scripts and paths keep working, and the
  pinned links resolve to the same path at the old commit. Alternative rejected: a
  separate repository or a `coworker/` directory, which would move his files out from
  under his scripts.
- **Excluded from every check** (`prek`, ruff, `ty`, `codespell`, the two check
  scripts), as before Step D: the directory is his working copy, not this project's
  code, and the project's conventions do not apply to it. `check_references.py` walks
  `src/deep_isochron` only and needs no line; `check_md_math.py` walks `docs/`.
- **The `.DS_Store` files are left out.** They are Finder metadata, `.gitignore` refuses
  them since `docs-layout`, and nothing reads them.
- **The documents keep the pinned links.** `design/isochrons.md` compares the pipeline
  with the version at `3af99f2`; Andrus's later commits may change the directory, so the
  link stays on the commit and the prose says the directory is a working copy.
=======
- **A directory beside the package, not inside it** (Q1 of 2026-10-29). `src/` holds the
  package alone, so `hatchling`, `ty` and `check_docstrings.py` never see his files by
  construction; only the tools that walk the whole tree (`prek`'s formatters, ruff,
  `codespell`, `check_spelling.py`) need a line. Alternative rejected: the original
  path `src/Isochron2PBVP`, which needed six exclusions and sat where a wheel build
  looks; also a separate repository or submodule, which would cost a second clone for
  a comparison that reads both sides.
- **One README and nothing else** (Q2): no `pyproject`, tests or CI job, since the `dev`
  environment already carries AUTO's Python layer and his scripts need nothing else.
  Alternative rejected: a workspace member, which would make his scripts a package
  nobody imports.
- **The links stay pinned at `3af99f2`** (Q3): `design/isochrons.md` reads a particular
  version run by run, and his later commits may change the directory, so the prose
  names the live path and the link the version compared.
- **The `.DS_Store` files are left out.** Finder metadata, refused by `.gitignore` since
  `docs-layout`, read by nothing.
>>>>>>> 2e58f90ba3b74b1a88a3a38084c476e9e6b1b522

## Bugs fixed

None.

## Tests

<<<<<<< HEAD
- `uv run just check`: every hook passed, `ty` clean, with the directory present and
  excluded.
=======
- `uv run just check`: every hook passed, `ty` clean, with the directory present.
>>>>>>> 2e58f90ba3b74b1a88a3a38084c476e9e6b1b522
- No test touches the directory; the three golden CSVs the oracle reads live in
  `tests/golden/`.

## Open issues

<<<<<<< HEAD
- Whether Andrus's work should also be excluded from `hatchling`'s wheel: `src/` has
  two top-level directories, and the build picks `deep_isochron`; a `uv build` run
  would confirm it, and nothing ships a wheel today.
=======
None.
>>>>>>> 2e58f90ba3b74b1a88a3a38084c476e9e6b1b522

## Review notes

| Change | Thoughts | Modifications |
|---|---|---|
<<<<<<< HEAD
| `src/Isochron2PBVP/` — 110 files from `d3edbfe`, no `.DS_Store` | | |
| `prek.toml`, `pyproject.toml`, `scripts/checks/check_docstrings.py`, `check_spelling.py` — the exclusions | | |
| `docs/design/isochrons.md`, `docs/decisions/0016`, `0017`, `docs/coding-standards.md`, `docs/architecture.md`, `docs/roadmap.md` — the directory is a working copy again | | |
=======
| `isochron2pbvp/` — 110 files from `d3edbfe`, no `.DS_Store`; the README | | |
| `prek.toml`, `pyproject.toml`, `scripts/checks/check_spelling.py` — the four exclusions | | |
| `stages.py`, `docs/design/isochrons.md`, `docs/decisions/0016`, `0017`, `docs/coding-standards.md`, `docs/document-conventions.md`, `docs/architecture.md`, `CLAUDE.md`, `docs/roadmap.md` — the project named | | |
>>>>>>> 2e58f90ba3b74b1a88a3a38084c476e9e6b1b522
