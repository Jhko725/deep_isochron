# Working conventions for Claude in this repository

Joon's personal rules are in his global `CLAUDE.md`; the conventions they rest on are in
the `code-conventions` skill. This file carries the session protocol and the facts Claude
cannot infer from the code. Where each convention is enforced here is
[`docs/coding-standards.md`](docs/coding-standards.md) for code and
[`docs/document-conventions.md`](docs/document-conventions.md) for documents; both name
the checks that stand in for prose.

## Start of a session

1. Read `docs/index.md`, then `docs/architecture.md`, then the current phase of
   `docs/roadmap.md`.
2. Find the branch's change document `docs/changes/<YYYY-MM-DD>-<branch>.md`. Its
   **Review notes** table is the current state of the review: rows with *Thoughts* or
   *Modifications* filled in are instructions for this session, and those two columns
   are never edited except when asked.
3. Read the ADRs in `docs/decisions/` that cover the area being touched.

## During and at the end of a session

- The change document follows `docs/document-conventions.md` § Change documents and is
  updated in the same commit as the code; a test's outcome is stated only after that
  test has run on the code the document describes.
- `docs/roadmap.md` is the single current plan: finished items are appended to its
  **Done** ledger, never inserted or deleted, in the same commit as the work.
  `git log -p docs/roadmap.md` is the project diary.
- `docs/architecture.md` follows every module move, rename, addition or removal;
  `docs/index.md` lists every document under `docs/` with its status.
- `just check` is clean, and the test files of the touched modules pass at the `dev`
  profile; the full tier runs once at the end (`docs/coding-standards.md` § Tiers).

## Repository facts

- Python 3.13; `uv sync` (the `dev` group is uv's default). Every command is a recipe in
  the `justfile`, and CI runs the same recipes. `just test` runs at `-n auto`: 2 cores in
  the cloud sessions, which have 7 GB, and 4 in CI.
- AUTO-07p (`docs/guides/auto07p.md`): in a cloud session, `apt-get update && apt-get
  install -y gfortran make`, then `uv run just install-auto` (about 40 s). Tests marked
  `auto` skip without a tree; the gfortran tests of `tests/test_auto_codegen.py` are
  `slow` and skip without gfortran.
- float64 throughout: library drivers raise when `jax_enable_x64` is off; notebooks and
  scripts enable it at the top.
- The jaxtyping/beartype import hook runs in the tests, so shape annotations are checked
  at runtime: the first occurrence of an axis name is a plain name (`" K"`, not
  `" K+1"`), `{self.attr}` expressions are allowed, and `typing.Self` is written as a
  string forward reference.
- `uv.lock` is not regenerated here; Joon runs `uv lock` after pulling. The notebooks are
  Joon's own records: Claude updates them to follow an API change and does not
  restructure them.
- `isochron2pbvp/` is Andrus Giraldo's project, kept in this repository under his
  conventions; Claude does not edit it unless Joon or Andrus asks.
