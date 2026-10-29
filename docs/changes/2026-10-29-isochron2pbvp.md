---
type: change
status: implemented 2026-10-29; review pending
updated: 2026-10-29
---

# `isochron2pbvp` — the coworker's directory restored as his working copy

## Summary

Step D of `auto-isochrons` deleted `src/Isochron2PBVP`, Andrus Giraldo's AUTO-07p
reproduction, once the pipeline reproduced it (`1749851`), leaving links pinned at
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
- `docs/roadmap.md` — Done ledger row.

## Design

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

## Bugs fixed

None.

## Tests

- `uv run just check`: every hook passed, `ty` clean, with the directory present and
  excluded.
- No test touches the directory; the three golden CSVs the oracle reads live in
  `tests/golden/`.

## Open issues

- Whether Andrus's work should also be excluded from `hatchling`'s wheel: `src/` has
  two top-level directories, and the build picks `deep_isochron`; a `uv build` run
  would confirm it, and nothing ships a wheel today.

## Review notes

| Change | Thoughts | Modifications |
|---|---|---|
| `src/Isochron2PBVP/` — 110 files from `d3edbfe`, no `.DS_Store` | | |
| `prek.toml`, `pyproject.toml`, `scripts/checks/check_docstrings.py`, `check_spelling.py` — the exclusions | | |
| `docs/design/isochrons.md`, `docs/decisions/0016`, `0017`, `docs/coding-standards.md`, `docs/architecture.md`, `docs/roadmap.md` — the directory is a working copy again | | |
