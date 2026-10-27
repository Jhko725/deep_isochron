# Working conventions for Claude in this repository

Joon's personal rules are in his global `CLAUDE.md`; the conventions they rest on —
design, naming, typing, the JAX ecosystem — are in the `code-conventions` skill. This
file carries what is particular to this repository: how work is recorded, and the facts
Claude cannot infer from the code. [`docs/coding-standards.md`](docs/coding-standards.md)
says where each convention is enforced here, and which checks stand in for prose.

## Start of a session

1. Read `docs/index.md`, then `docs/architecture.md` (the module map and the data flow),
   then the current phase of `docs/roadmap.md`.
2. Find the branch's change document `docs/changes/<YYYY-MM-DD>-<branch>.md`. Its
   **Review notes** table is the current state of the review: rows with *Thoughts* or
   *Modifications* filled in are instructions for this session.
3. Read the ADRs in `docs/decisions/` that cover the area being touched.

## Change documents

Every branch Claude works on has a companion document
`docs/changes/<YYYY-MM-DD>-<branch>.md`, committed together with the code. It is the
reviewer's map of the change, not a changelog. Sections: **Summary**, what the branch
does and why; **Files**, every file touched, one line each, confirmable against `git diff
--stat`; **Design**, decisions not obvious from the code, each with the alternative that
was rejected, linking an ADR when the decision is structural; **Bugs fixed**, with the
symptom; **Tests**, what they pin down and how to run them; **Open issues**, with the
reason; **Review notes**.

*Review notes* is a table *Change* | *Thoughts* | *Modifications*. Claude pre-populates
one row per file touched — *Change* is the file name plus a one-line summary — and leaves
the other two columns blank. Joon fills them in while vetting; Claude reads them at the
start of the next session on that branch and never edits them except when asked.

Claude updates the change document in the same commit as any later change on the branch.
A new review row is appended after the table's last line, never inserted by matching
the text of an existing row: Joon's edits change that text, and an edit anchored on it
fails silently.

## End of a session

- The change document is updated: Files, Design, Tests, and a review row per new file.
- `docs/roadmap.md` is the single current plan. Finished items are appended to its
  **Done** ledger, never inserted and never deleted; the phase tables are kept correct
  now. It is updated in the same commit as the work that changes it, never on its own,
  and `git log -p docs/roadmap.md` is the project diary.
- `docs/architecture.md` is updated when a module moves, is renamed, added or removed.
- `docs/index.md` lists every document under `docs/`, with its status.
- `uv run python scripts/check_md_math.py` reports 0 problems when Markdown changed.
- `uv run ty check src/deep_isochron/model/invertible` stays clean.

## Repository facts

- Python 3.13; `uv sync --group dev`; `uv run pytest`, where `-n 4` (pytest-xdist) helps.
  Hypothesis profiles: `--hypothesis-profile=dev` (10 examples), default (50), `ci`
  (300, derandomized).
- The jaxtyping/beartype import hook runs in the tests, so shape annotations are checked
  at runtime: the first occurrence of an axis name is a plain name (`" K"`, not
  `" K+1"`), `{self.attr}` expressions are allowed, and `typing.Self` is written as a
  string forward reference.
- Pre-commit runs through `prek`; the hooks are listed in `prek.toml`, and what each
  check is for is in `docs/coding-standards.md`.
- Design decisions are numbered ADRs in `docs/decisions/`, and a change document's
  *Design* section links to them. The contracts those ADRs fix, together with this
  repository's tiers, tolerances and checks, are in `docs/coding-standards.md`.
- Documents under `docs/` open with a frontmatter block (`type`, `status`, `updated`,
  plus `verified_by` / `sources` / `id` where relevant), updated in the same commit as
  the document. Markdown has to render on GitHub as well as in VS Code's preview;
  `scripts/check_md_math.py` is the arbiter, and the rules it enforces are recorded in
  `docs/coding-standards.md`.
- Provenance labels in design documents: **[cited]** from a named source, **[code]**
  verified in the tree, **[deduced]** Claude's inference. Keep them.
