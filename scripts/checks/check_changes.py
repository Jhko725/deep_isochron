"""Check that a change document's *Review notes* table has a row for every *Files*
bullet, so that Joon can review every change the branch made.

`docs/document-conventions.md` § Change documents fixes the form: *Files* lists each
change as one bullet, confirmable against ``git diff --stat``, and *Review notes* is a
table with one row per *Files* bullet, appended in the commit that adds the bullet, with
the *Thoughts* and *Modifications* columns left for Joon. The review round of
`plotting` added fourteen files in three commits and no row, because the brief
enumerated the document's sections and left the table out; the check is what makes the
rule hold without a brief.

A bullet's key is the first backticked path in it (its continuation lines included).
A row covers it when a backticked token of the row's *Change* cell is that path or a
suffix of it, since a row that groups several files names the later ones by their last
components (a row for ``model/base.py`` and ``conjugacy.py`` covers both bullets).
Everything else in the two sections is free text.

The documents in `BEFORE_THE_RULE` are history and are not checked: their tables were
written before the rule and are not rewritten (`document-conventions.md` § Change
documents, last paragraph). The tuple never grows.

Usage: `uv run python scripts/checks/check_changes.py [path ...]` (default: every
Markdown file under `docs/changes`). Exits 1 when anything is flagged.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Iterator
from pathlib import Path


DEFAULT_DIR = Path("docs/changes")
FILES_HEADING = "## Files"
REVIEW_HEADING = "## Review notes"
BACKTICKED = re.compile(r"`([^`]+)`")
BEFORE_THE_RULE = (
    "2026-09-30-splines-refactor.md",
    "2026-10-01-scalar-param-refactor.md",
    "2026-10-02-data-generation.md",
    "2026-10-02-invertible-cleanup.md",
    "2026-10-04-trainer.md",
    "2026-10-05-experiment-config.md",
    "2026-10-06-analysis.md",
    "2026-10-07-run-layout.md",
    "2026-10-08-normal-forms-basin.md",
    "2026-10-15-ci-ty.md",
    "2026-10-22-symbolic-systems.md",
    "2026-10-25-conventions-docs.md",
    "2026-10-26-prose.md",
    "2026-10-27-auto-isochrons.md",
    "2026-10-28-phase-field.md",
    "2026-10-29-docs-layout.md",
    "2026-10-29-isochron2pbvp.md",
    "2026-10-29-merge-conflict-check.md",
)
"""The change documents written before the rule (2026-10-29); history, not checked."""


def section(lines: list[str], heading: str) -> tuple[int, list[str]]:
    """The lines of the section under `heading`, with the 1-based line number of its
    first line; ``(0, [])`` when the heading is absent."""
    try:
        start = next(i for i, line in enumerate(lines) if line.rstrip() == heading)
    except StopIteration:
        return 0, []
    body: list[str] = []
    for line in lines[start + 1 :]:
        if line.startswith("## "):
            break
        body.append(line)
    return start + 2, body


def bullets(body: list[str], first_line: int) -> Iterator[tuple[int, str]]:
    """The bullets of a section body, each joined with its continuation lines, with
    the line number of its first line."""
    current: list[str] = []
    start = 0
    for offset, line in enumerate(body):
        if line.startswith("- "):
            if current:
                yield start, " ".join(current)
            current = [line[2:].strip()]
            start = first_line + offset
        elif current and line.startswith("  "):
            current.append(line.strip())
        elif current:
            yield start, " ".join(current)
            current = []
    if current:
        yield start, " ".join(current)


def change_cells(body: list[str]) -> list[str]:
    """The *Change* cell of every row of the table in a *Review notes* body, the
    header and separator rows excluded."""
    rows = [line for line in body if line.startswith("|")]
    cells = []
    for row in rows[2:]:
        cells.append(row.split("|")[1])
    return cells


def problems(path: Path) -> Iterator[str]:
    """Every *Files* bullet of `path` without a *Review notes* row."""
    lines = path.read_text(encoding="utf-8").splitlines()
    files_start, files_body = section(lines, FILES_HEADING)
    _, review_body = section(lines, REVIEW_HEADING)
    if not files_body:
        return
    cells = change_cells(review_body)
    for line_number, bullet in bullets(files_body, files_start):
        match = BACKTICKED.search(bullet)
        if match is None:
            continue
        key = match.group(1)
        if not any(covers(cell, key) for cell in cells):
            yield f"{path}:{line_number}: `{key}` has no review row"


def covers(cell: str, key: str) -> bool:
    """Whether a backticked token of `cell` is `key` or a suffix of it."""
    return any(
        key == token or key.endswith("/" + token) for token in BACKTICKED.findall(cell)
    )


def main(argv: list[str]) -> int:
    """Run the check over the given paths, or `DEFAULT_DIR`; return the exit code."""
    paths = [Path(a) for a in argv] or sorted(DEFAULT_DIR.glob("*.md"))
    paths = [p for p in paths if p.name not in BEFORE_THE_RULE]
    found = [p for path in paths for p in problems(path)]
    for p in found:
        print(p)
    print(f"{len(found)} problem(s)")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
