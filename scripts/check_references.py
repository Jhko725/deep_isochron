"""Check that every citation in the tree resolves to an entry in ``docs/references.md``.

A citation is an author-year key, ``Lust 2001``, ``Wilson & Moehlis 2016``,
``Langfield et al. 2014``, optionally followed by a locator after a comma:
``(Lust 2001, Eq. 10)``, ``(Wilson & Moehlis 2016, Eq. 7; Kvalheim & Revzen 2021, Prop.
5)``. The bibliography ``docs/references.md`` has one entry per key, and this script
fails on any key in the documents, docstrings or tests that is not an entry there. It
also lists the entries that nothing cites, as information rather than failure, so that
the bibliography can be pruned by hand.

Both drifts this catches happened before the bibliography existed: the same paper
cited under three forms (the full author list, the surnames joined by en dashes, and
the ``et al.`` form), and five documents each carrying their own reference list.

A line carrying ``references:ignore`` is skipped. ``docs/changes/`` is skipped as a
whole: landed change documents are records of what was said.

Usage: ``uv run python scripts/check_references.py [path ...]`` (default: ``docs``,
``src/deep_isochron``, ``tests``, ``scripts``). Exits 1 on an unresolved key.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Iterator
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
BIBLIOGRAPHY = ROOT / "docs" / "references.md"
DEFAULT_PATHS = ("docs", "src/deep_isochron", "tests", "scripts")
SKIPPED_DIRS = ("docs/changes",)
IGNORE_MARKER = "references:ignore"
TEXT_SUFFIXES = frozenset({".py", ".md"})

# A surname: a capitalized word (accented letters included), with an optional lowercase
# particle (de Boor, van der Pol). Hyphenated and apostrophized surnames are allowed.
_SURNAME = r"(?:(?:de|van|von|der|di|da) )?[A-Z][\w'’\-]+"
# A key: one surname, two joined by ``&``, or one followed by ``et al.``; then a year,
# with an optional letter suffix, and not the start of an ISO date (``2026-10-03``).
# The spaces inside a key may be line breaks, since prose is wrapped at 88 columns.
KEY = re.compile(
    rf"\b(?P<key>{_SURNAME}(?:\s+&\s+{_SURNAME}|\s+et al\.)?\s+(?:19|20)\d{{2}}[a-z]?)"
    r"\b(?!-\d)"
)
# An entry line of the bibliography: ``- **Key** — ...``.
ENTRY = re.compile(r"^- \*\*(?P<key>[^*]+)\*\* ")


def bibliography_keys(path: Path = BIBLIOGRAPHY) -> set[str]:
    """The keys defined in the bibliography, one per ``- **Key** —`` line."""
    keys: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        match = ENTRY.match(line)
        if match:
            keys.add(match.group("key").strip())
    return keys


def iter_files(paths: list[Path]) -> Iterator[Path]:
    """Every ``.py`` and ``.md`` file under `paths`, minus the excluded directories."""
    for path in paths:
        candidates = [path] if path.is_file() else sorted(path.rglob("*"))
        for file in candidates:
            if file.suffix not in TEXT_SUFFIXES or file == BIBLIOGRAPHY:
                continue
            relative = file.resolve().relative_to(ROOT).as_posix()
            if any(relative.startswith(skipped) for skipped in SKIPPED_DIRS):
                continue
            yield file


def citations(file: Path) -> Iterator[tuple[int, str]]:
    """``(line number, key)`` for every citation key in ``file``.

    The file is scanned as one string so that a key wrapped over a line break still
    matches; the key is reported with its spaces collapsed. A line carrying the ignore
    marker contributes no key.
    """
    text = file.read_text(encoding="utf-8")
    lines = text.splitlines()
    for match in KEY.finditer(text):
        number = text.count("\n", 0, match.start()) + 1
        if IGNORE_MARKER in lines[number - 1]:
            continue
        yield number, " ".join(match.group("key").split())


def main(argv: list[str] | None = None) -> int:
    """Check the citations under `argv`; return 1 if any key does not resolve."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="*", type=Path)
    arguments = parser.parse_args(argv)
    paths = arguments.paths or [ROOT / p for p in DEFAULT_PATHS]

    known = bibliography_keys()
    unresolved: list[str] = []
    cited: set[str] = set()
    for file in iter_files(paths):
        for number, key in citations(file):
            cited.add(key)
            if key not in known:
                relative = file.resolve().relative_to(ROOT).as_posix()
                unresolved.append(f"{relative}:{number}: {key}")

    for line in unresolved:
        print(line)
    uncited = sorted(known - cited)
    if uncited:
        print(f"uncited entries in {BIBLIOGRAPHY.relative_to(ROOT).as_posix()}:")
        for key in uncited:
            print(f"  {key}")
    if unresolved:
        print(f"{len(unresolved)} citation(s) not in the bibliography")
        return 1
    print(f"all {len(cited)} cited keys resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main())
