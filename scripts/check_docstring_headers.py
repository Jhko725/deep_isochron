"""Check the form of the docstring headers `**Arguments:**`, `**Returns:**`,
`**Raises:**` and `**Attributes:**`.

`docs/coding-standards.md` § Docstrings fixes the form diffrax writes: each header
stands on its own line, followed by a blank line and then the bullets or the paragraph.
ruff's pydocstyle rules know Google and numpy sections, not these, so the check is a
regex: a header line with text after the closing stars, or without a blank line after
it, is flagged. The review of `prose` P1–P4 found the inline form written twice after
the standard said otherwise, which is what a check is for.

Usage: `uv run python scripts/check_docstring_headers.py [path ...]` (default:
`src`, `tests`, `scripts`). Exits 1 when anything is flagged.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Iterator
from pathlib import Path


HEADER = re.compile(r"^\s*\*\*(Arguments|Returns|Raises|Attributes):\*\*(.*)$")
EXCLUDED: tuple[str, ...] = ()
DEFAULT_PATHS = ("src", "tests", "scripts")


def python_files(paths: list[Path]) -> Iterator[Path]:
    """Every ``.py`` file under `paths`, minus the excluded directories."""
    for root in paths:
        files = [root] if root.is_file() else sorted(root.rglob("*.py"))
        for f in files:
            if not any(f.as_posix().startswith(e) for e in EXCLUDED):
                yield f


def problems(path: Path) -> Iterator[str]:
    """The flagged headers of `path`, one ``file:line: reason`` per header."""
    lines = path.read_text(encoding="utf-8").split("\n")
    for i, line in enumerate(lines, start=1):
        m = HEADER.match(line)
        if m is None:
            continue
        name, rest = m.groups()
        if rest.strip():
            yield f"{path}:{i}: text after `**{name}:**` on the same line"
        elif i == len(lines) or lines[i].strip():
            yield f"{path}:{i}: no blank line after `**{name}:**`"


def main(argv: list[str]) -> int:
    """Run the check over `argv` or the default paths; return the exit code."""
    paths = [Path(a) for a in argv] or [Path(p) for p in DEFAULT_PATHS]
    found = [p for f in python_files(paths) for p in problems(f)]
    for p in found:
        print(p)
    print(f"{len(found)} problem(s)")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
