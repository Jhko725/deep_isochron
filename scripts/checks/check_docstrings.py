"""Check what ruff's docstring rules leave unchecked: the form of the headers
`**Arguments:**`, `**Returns:**`, `**Raises:**` and `**Attributes:**`; a docstring on
every private function and class under ``src``; and a bound on a module's length.

`docs/coding-standards.md` § Docstrings fixes the form diffrax writes: each header
stands on its own line, followed by a blank line and then the bullets or the paragraph.
ruff's pydocstyle rules know Google and numpy sections, not these, so the check is a
regex: a header line with text after the closing stars, or without a blank line after
it, is flagged. The review of `prose` P1–P4 found the inline form written twice after
the standard said otherwise, which is what a check is for.

ruff's ``D103`` skips a name that starts with an underscore, and the review of the
isochron pipeline asked for its private stages to be documented like the public ones;
so under ``src`` every ``_``-prefixed function, method and class needs a docstring too.
Dunder methods are exempt: ``__repr__`` says what it is.

A module under ``src`` is bounded at `MAX_MODULE_LINES`. The isochron pipeline reached
853 lines in one module before review called it "hard to grasp at once" and it became
four; the bound fires before that, and a module that needs more is split by what serves
what (`code-conventions` § Code goes where it belongs).

Usage: `uv run python scripts/checks/check_docstrings.py [path ...]` (default:
`src`, `tests`, `scripts`). Exits 1 when anything is flagged.
"""

from __future__ import annotations

import ast
import re
import sys
from collections.abc import Iterator
from pathlib import Path


HEADER = re.compile(r"^\s*\*\*(Arguments|Returns|Raises|Attributes):\*\*(.*)$")
EXCLUDED: tuple[str, ...] = ("src/Isochron2PBVP",)  # the coworker's working copy
DEFAULT_PATHS = ("src", "tests", "scripts")
PRIVATE_SCOPE = "src"
MAX_MODULE_LINES = 600


def python_files(paths: list[Path]) -> Iterator[Path]:
    """Every ``.py`` file under `paths`, minus the excluded directories."""
    for root in paths:
        files = [root] if root.is_file() else sorted(root.rglob("*.py"))
        for f in files:
            if not any(f.as_posix().startswith(e) for e in EXCLUDED):
                yield f


def header_problems(path: Path, lines: list[str]) -> Iterator[str]:
    """The flagged headers of `path`, one ``file:line: reason`` per header."""
    for i, line in enumerate(lines, start=1):
        m = HEADER.match(line)
        if m is None:
            continue
        name, rest = m.groups()
        if rest.strip():
            yield f"{path}:{i}: text after `**{name}:**` on the same line"
        elif i == len(lines) or lines[i].strip():
            yield f"{path}:{i}: no blank line after `**{name}:**`"


def private_problems(path: Path, source: str) -> Iterator[str]:
    """The ``_``-prefixed functions, methods and classes of `path` without a
    docstring, dunders excepted."""
    for node in ast.walk(ast.parse(source, filename=str(path))):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            continue
        name = node.name
        if not name.startswith("_") or name.startswith("__"):
            continue
        if ast.get_docstring(node) is None:
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            yield f"{path}:{node.lineno}: private {kind} `{name}` has no docstring"


def length_problems(path: Path, lines: list[str]) -> Iterator[str]:
    """`path` when it is longer than `MAX_MODULE_LINES`."""
    count = len(lines) - (1 if lines and lines[-1] == "" else 0)
    if count > MAX_MODULE_LINES:
        yield (
            f"{path}:{count}: module has {count} lines, over {MAX_MODULE_LINES};"
            " split it by what serves what"
        )


def problems(path: Path) -> Iterator[str]:
    """Every flagged line of `path`, as ``file:line: reason``."""
    source = path.read_text(encoding="utf-8")
    lines = source.split("\n")
    yield from header_problems(path, lines)
    if path.as_posix().startswith(PRIVATE_SCOPE):
        yield from private_problems(path, source)
        yield from length_problems(path, lines)


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
