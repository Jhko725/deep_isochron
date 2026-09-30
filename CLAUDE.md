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
- **Review notes**: empty when Claude writes the document. Joon fills it in
  while vetting; Claude reads it at the start of the next session on that
  branch and treats it as the current state of the review.

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
