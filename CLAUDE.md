# Working conventions for Claude in this repository

## Change documents

Every branch that Claude works on has a companion document
`docs/changes/<YYYY-MM-DD>-<branch>.md`, committed together with the code. It is
the reviewer's map of the change, not a changelog:

- **Summary**: what the branch does and why, in a few sentences.
- **Files**: every file touched, one line each, saying what changed there. A
  reviewer should be able to confirm the list against `git diff --stat`.
- **Design**: decisions that are not obvious from the code, with the alternative
  that was rejected and why.
- **Bugs fixed**: pre-existing bugs found on the way, with the symptom.
- **Tests**: what the new tests pin down, and how to run them.
- **Open issues**: things deliberately left alone, with the reason.
- **Review notes**: a table with columns *Change* | *Thoughts* | *Modifications*.
  Claude pre-populates one row per file touched, with *Change* = the file name
  plus a one-line summary of what changed there, and leaves *Thoughts* and
  *Modifications* blank. Joon fills those in while vetting; Claude reads them
  at the start of the next session on that branch and treats them as the
  current state of the review.

Claude updates the document in the same commit as any later change on the
branch, and never edits *Review notes* except when asked to.

## Repository facts

- Python 3.13, `uv sync --group dev`; run tests with `uv run pytest`.
  Hypothesis profiles: `--hypothesis-profile=dev` (10 examples, quick),
  default (50), `ci` (300, derandomised). `-n 4` (pytest-xdist) helps.
- Tests install the jaxtyping/beartype import hook, so shape annotations are
  checked at runtime: the first occurrence of an axis name must be a plain
  name (`" K"`, not `" K+1"`), and `{self.attr}` expressions are allowed.
- Pre-commit hooks run through `prek` (ruff check --fix, ruff format,
  whitespace fixers).
- Every concrete `AbstractBijection` must be registered in `tests/registry.py`
  (or listed in `UNTESTED` with a reason); `tests/test_registry.py` enforces it.
- Implement an `eqx.AbstractVar` (`dim`, `smoothness`, `num_params`, ...) as
  `eqx.field(static=True, init=False)` — with `default=` when the value is fixed
  by the class, assigned in `__init__` when derived. Never a bare class
  attribute, `ClassVar` or property: type checkers reject those as overrides
  (ADR-0004). `ty check src/deep_isochron/model/invertible` must stay clean.
- Scalar bijections hold one unconstrained leaf `raw`; constrained parameters
  come from `constrain(raw)` (ADR-0001); `constrain(0)` is the identity
  (ADR-0002). Templates are instances with `raw=None`. Constraint primitives are
  plain Python objects created *inside* `constrain`, never stored as fields
  (ADR-0005).
- Design decisions live in `docs/decisions/` as numbered ADRs; a change
  document's *Design* section links to them. `docs/architecture.md` is the
  module map and data flow — read it first in a new session, and update it in
  the same commit as a change that moves or renames a module.
- Markdown must render on GitHub *and* in VS Code's preview. GitHub runs Markdown
  before MathJax, so: no line-length limit (ruff formats Python only) but an inline
  formula `$…$` never crosses a line; display math only as a standalone block (`$$`
  alone on a line, formula, `$$` alone on a line), never `$$…$$` inside a paragraph;
  inside inline math no backslash + ASCII punctuation (`\,` `\;` `\{` `\|` `\\` →
  `\thinspace` `\medspace` `\lbrace` `\Vert`, matrices in blocks), no `*` (use
  `\ast`), spaces around `<` `>`, and no `_` right after `}` `)` `|` (write `\mathbf e_r`,
  `C_{\rm loc}^{k}`, `\vert_{E}`), because such an `_` opens Markdown emphasis.
  `uv run python scripts/check_md_math.py` emulates GitHub and must report 0 problems;
  run it before committing a Markdown change. Every document under `docs/` starts
  with a frontmatter block (`type`, `status`, `updated`, plus
  `verified_by`/`sources`/`id` where relevant; see `docs/index.md`), updated in the
  same commit as the document.
- American English throughout (code, docstrings, documents): *center*, *normalize*,
  *optimizer*, *behavior*, *initialization*. Identifiers follow (`normalize_phase_plane`).
- `docs/roadmap.md` is the single, current plan. Finished items move to its
  *Done* ledger (not deleted), which is chronological: new rows are always
  appended at the end, never inserted; it is updated in the same commit as the
  work that changes it, never on its own; history is `git log -p docs/roadmap.md`.
