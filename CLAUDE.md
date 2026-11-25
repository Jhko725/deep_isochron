# Working conventions for Claude in this repository

Joon's working rules are in `.claude/rules/`: `working-rules.md` for every session, and
`python.md` and `notebooks.md` when a file of that kind is read. The conventions they
rest on are in the `code-conventions` skill. Where a skill and this repository's files
disagree, this repository's files take precedence: the skill serves several repositories
and can describe a format that this one has replaced. This file carries the session
protocol and the facts Claude cannot infer from the code. Where each convention is
enforced here is [`docs/coding-standards.md`](docs/coding-standards.md) for code and
[`docs/document-conventions.md`](docs/document-conventions.md) for documents; both name
the checks that stand in for prose.

## Start of a session

This reading list is for a session that plans, reviews or orchestrates. A subagent that
implements one commit from a brief reads what its brief names, and nothing else here.

1. Read `docs/index.md`, then `docs/architecture.md`, then the current phase of
   `docs/roadmap.md`.
2. Find the branch's change document `docs/changes/<YYYY-MM-DD>-<branch>.md`. Its
   *Review notes* is the current state of the review: rows with *Thoughts* or
   *Modifications* filled in are instructions for this session, and those two columns
   are never edited except when asked.
3. Read the ADRs in `docs/decisions/` that cover the area being touched, and the
   glossary before naming a quantity.

## During and at the end of a session

- The change document follows `docs/document-conventions.md` § Change documents and is
  updated in the same commit as the code: every file the commit touches gets a row in
  the open round of *Review notes*, with its *Change* cell filled, in that commit. The
  open round is the last one. A round is closed once Joon has filled a *Thoughts* cell
  of it: its rows are never edited again, and the next change opens a new round under
  ``### Round <n>: from `<commit>` ``, naming his latest review commit. Claude fills
  *Modifications* only when asked. A test's outcome is stated only after that test has
  run on the code the document describes.
- `docs/roadmap.md` is the single current plan: finished items are appended to its
  *Done* ledger, never inserted or deleted, in the same commit as the work.
  `git log -p docs/roadmap.md` is the project diary.
- `docs/architecture.md` follows every module move, rename, addition or removal;
  `docs/index.md` lists every document under `docs/` with its status.
- A decision that Claude takes without Joon is marked `[unconfirmed]` where it is
  stated; `just pending` lists every marker.
- `just check` is clean, and the test files of the touched modules pass at the `dev`
  profile; the full tier runs once at the end (`docs/coding-standards.md` § Tiers).

## Repository facts

- Python 3.13; `uv sync` (the `dev` group is uv's default). Every command is a recipe in
  the `justfile`, and CI runs the same recipes. `just test` runs at `-n auto`: 2 cores in
  the cloud sessions, which have 7 GB, and 4 in CI.
- `just check` runs `prek run --all-files`, which sees only tracked files: `git add` a
  new file before running it, or the check passes without having read the file.
- AUTO-07p (`docs/guides/auto07p.md`): in a cloud session, `apt-get update && apt-get
  install -y gfortran make`, then `uv run just install-auto` (about 40 s). Tests marked
  `auto` skip without a tree; the gfortran tests of `tests/test_auto_codegen.py` are
  `slow` and skip without gfortran.
- `tests/conftest.py` turns on JAX's persistent compilation cache (`~/.cache/jax`, or
  `JAX_COMPILATION_CACHE_DIR`) and clears the in-process caches after every test:
  `just test default` is about 15 minutes on two cores with the disk cache and over an
  hour without, and a worker that dies mid-run (`node down`, `replacing crashed worker`)
  was killed by the 7 GB memory cgroup (`dmesg | grep -i 'out of memory'`), which the
  per-test clearing prevents.
- The jaxtyping/beartype import hook runs in the tests, so shape annotations are checked
  at runtime: the first occurrence of an axis name is a plain name (`" K"`, not
  `" K+1"`), `{self.attr}` expressions are allowed, and `typing.Self` is written as a
  string forward reference.
- The session that changes a dependency runs `uv lock` and commits `uv.lock` with the
  change. It never passes `--upgrade`.
- A pushed commit is never amended or rebased, and a branch is never force-pushed: a
  change to a pushed commit is a new commit on top, because Joon may already have
  merged it. A hook in `.claude/settings.json`,
  `.claude/hooks/block_history_rewrite.py`, blocks every `git rebase` and force-push,
  and a `git commit --amend` unless it can tell that no remote branch holds `HEAD`.
- The notebooks are Joon's own records: `.claude/rules/notebooks.md` says what Claude
  may change in them.
- MP4 export of the deformation flipbook needs `ffmpeg`: a system binary on `PATH`, or
  `uv sync --extra video` for the `imageio-ffmpeg` wheel; the save call is in
  `animate_deformation`'s docstring.
- `isochron2pbvp/` is Andrus Giraldo's project, kept in this repository under his
  conventions; Claude does not edit it unless Joon or Andrus asks.
