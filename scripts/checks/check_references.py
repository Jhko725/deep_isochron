# Adapted from Jhko725/UniversalForceReconstruction scripts/checks/check_references.py
# at b9e4cc3.
"""Check that every citation and every relative link in the tree resolves.

A citation resolves to an entry in ``docs/references.bib``; a relative Markdown link
resolves to a file in the repository.

A citation is an author-year key, ``Lust 2001``, ``Wilson & Moehlis 2016``,
``Langfield et al. 2014``, optionally followed by a locator after a comma:
``(Lust 2001, Eq. 10)``, ``(Wilson & Moehlis 2016, Eq. 7; Kvalheim & Revzen 2021, Prop.
5)``. The bibliography ``docs/references.bib`` has one entry per key, the key derived
from the entry's authors and year by ``scripts/docs/generate_references.py`` (or given
by the entry's ``key`` field), and this script fails on any key in the documents,
docstrings or tests that is not an entry there. It also lists the entries that nothing
cites, as information rather than failure, so that the bibliography can be pruned by
hand.

The page ``docs/references.md`` is generated from the ``.bib`` file, and this script
fails when the page differs from the generator's output, apart from its ``updated``
date; ``just references`` rewrites it. The generator is loaded by file path, and this
script writes nothing.

The check catches a paper cited under more than one form (the full author list, the
surnames joined by en dashes, or the ``et al.`` form) and a document that carries its
own reference list.

A relative link is a Markdown link ``[text](target)`` whose target is not a URL or a
bare ``#anchor``; the target, percent-decoded and without its ``#anchor``, must name an
existing file or directory relative to the linking file. Links are read from the
section tree of ``markdown.py``, each with its line, so a link quoted inside fenced or
inline code is not checked. ADR-0019 makes the first mention of an ADR, design document
or code file in a section a link, so a renamed file or a mistyped name fails here
rather than leaving a dead pointer.

Under ``docs/design/`` the provenance labels are ``[cited]``, ``[cited via X]``,
``[code]`` and ``[deduced]``; the italic forms ``*cited*``, ``*deduced*`` and
``*secondary*`` that the older documents used are refused.

A line carrying ``references:ignore`` is skipped. ``docs/changes/`` and ``docs/plans/``
are skipped as a whole: landed change documents and committed plans are records of what
was said.

Usage: ``uv run python scripts/checks/check_references.py [path ...]``, where `prek`
passes every tracked Python and Markdown file; with no path, every file in `SCOPE` is
checked. A named file outside `SCOPE`, or in a skipped directory, is not checked. The
entries that nothing cites are those that the checked files do not cite, so the list is
complete only when every file is checked, as in ``just check``. Exits 1 on an
unresolved key, a broken link, an italic provenance label, an entry that the generator
cannot list, or a page that differs from the generated one.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import markdown


ROOT = Path(__file__).resolve().parents[2]
BIBLIOGRAPHY = ROOT / "docs" / "references.md"
SOURCE = ROOT / "docs" / "references.bib"
GENERATOR = ROOT / "scripts" / "docs" / "generate_references.py"
# The files checked, relative to the root: a directory covers every file under it.
SCOPE = (
    "docs",
    "src/deep_isochron",
    "tests",
    "scripts",
    "README.md",
    "CLAUDE.md",
)
SKIPPED_DIRS = ("docs/changes", "docs/plans")
IGNORE_MARKER = "references:ignore"
LABEL_SCOPE = "docs/design"
OLD_LABEL = re.compile(r"\*(cited|deduced|secondary)\*")
TEXT_SUFFIXES = frozenset({".py", ".md"})

# A surname: a capitalized word (accented letters included), after any number of
# lowercase particles from a fixed list (de Boor, van der Pol, de la Llave). Hyphenated
# and apostrophized surnames are allowed.
_PARTICLE = r"(?:de|den|der|van|von|di|da|la|le|del|della)"
_SURNAME = rf"(?:{_PARTICLE}\s+)*[A-Z][\w'’\-]+"
# A key: one surname, two joined by ``&``, or one followed by ``et al.``; then a year
# from 1800 to 2099, with an optional letter suffix, and not the start of an ISO date
# (``2026-10-03``).
# The spaces inside a key may be line breaks, since prose is wrapped at 88 columns.
KEY = re.compile(
    rf"\b(?P<key>{_SURNAME}(?:\s+&\s+{_SURNAME}|\s+et al\.)?"
    r"\s+(?:1[89]|20)\d{2}[a-z]?)\b(?!-\d)"
)
# A target with a scheme (``https:``, ``mailto:``) or a bare anchor is not checked.
EXTERNAL = re.compile(r"^(?:[a-z][a-z0-9+.-]*:|#)")


def load_generator(path: Path = GENERATOR) -> ModuleType:
    """The module ``scripts/docs/generate_references.py``, loaded from its file, since
    ``scripts/docs`` is not a package on the import path."""
    spec = importlib.util.spec_from_file_location("generate_references", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    # A dataclass looks its module up in sys.modules while it is being defined.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def stale_page(generator: ModuleType, source: str) -> bool:
    """Whether ``docs/references.md`` differs from the page that `generator` renders
    from the BibTeX `source`, dated as the page is dated."""
    page = BIBLIOGRAPHY.read_text(encoding="utf-8") if BIBLIOGRAPHY.exists() else ""
    updated = generator.page_date(page)
    return updated is None or generator.render(source, updated) != page


def scope_files() -> Iterator[Path]:
    """Every file in `SCOPE`, the files under each directory in sorted order."""
    for entry in SCOPE:
        path = ROOT / entry
        yield from [path] if path.is_file() else sorted(path.rglob("*"))


def in_scope(file: Path) -> bool:
    """Whether `file` is a ``.py`` or ``.md`` file in `SCOPE`, outside the skipped
    directories, and not the bibliography itself."""
    if file.suffix not in TEXT_SUFFIXES or file.resolve() == BIBLIOGRAPHY:
        return False
    try:
        relative = file.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return False
    return any(
        relative == entry or relative.startswith(entry + "/") for entry in SCOPE
    ) and not any(relative.startswith(skipped + "/") for skipped in SKIPPED_DIRS)


def citations(file: Path) -> list[tuple[int, str]]:
    """``(line number, key)`` for every citation key in ``file``.

    The file is scanned as one string so that a key wrapped over a line break still
    matches; the key is reported with its spaces collapsed. A line carrying the ignore
    marker contributes no key.
    """
    text = file.read_text(encoding="utf-8")
    lines = text.splitlines()
    found = []
    for match in KEY.finditer(text):
        number = text.count("\n", 0, match.start()) + 1
        if IGNORE_MARKER in lines[number - 1]:
            continue
        found.append((number, " ".join(match.group("key").split())))
    return found


def broken_links(file: Path) -> list[tuple[int, str]]:
    """``(line number, target)`` for every relative link in the Markdown `file` whose
    target does not exist. A line carrying the ignore marker contributes nothing."""
    if file.suffix != ".md":
        return []
    text = file.read_text(encoding="utf-8")
    lines = text.splitlines()
    links = (link for part in markdown.parse(text).walk() for link in part.links)
    return [
        (link.line, link.target)
        for link in links
        if IGNORE_MARKER not in lines[link.line - 1]
        and not EXTERNAL.match(link.target)
        and not (file.parent / link.target.split("#", 1)[0]).exists()
    ]


def old_labels(file: Path) -> list[tuple[int, str]]:
    """The italic provenance labels of `file`, ``(line number, label)`` each, which
    ``docs/document-conventions.md`` § Design documents replaced with ``[cited]``,
    ``[cited via X]`` and ``[deduced]``. A line carrying the ignore marker contributes
    nothing."""
    lines = file.read_text(encoding="utf-8").splitlines()
    return [
        (number, match.group(0))
        for number, line in enumerate(lines, 1)
        if IGNORE_MARKER not in line
        for match in OLD_LABEL.finditer(line)
    ]


@dataclass(frozen=True)
class ScanResult:
    """What the scanned files hold, each flagged item as ``file:line: what``: the
    citations whose key the bibliography lacks, the broken relative links and the
    italic provenance labels under `LABEL_SCOPE`; and every cited key."""

    unresolved: list[str]
    broken: list[str]
    labels: list[str]
    cited: set[str]


def scan(files: Iterable[Path], known: set[str]) -> ScanResult:
    """Read the citations, the relative links and the provenance labels of `files`,
    where `known` holds the keys of the bibliography."""
    found = ScanResult([], [], [], set())
    for file in files:
        relative = file.resolve().relative_to(ROOT).as_posix()
        for number, key in citations(file):
            found.cited.add(key)
            if key not in known:
                found.unresolved.append(f"{relative}:{number}: {key}")
        for number, target in broken_links(file):
            found.broken.append(f"{relative}:{number}: broken link {target}")
        if relative.startswith(LABEL_SCOPE + "/"):
            for number, label in old_labels(file):
                found.labels.append(f"{relative}:{number}: italic label {label}")
    return found


def print_report(found: ScanResult, uncited: list[str], stale: bool) -> None:
    """Print each flagged line of `found`, the `uncited` keys of the bibliography, the
    counts, and whether the generated page is `stale`."""
    for line in found.unresolved + found.broken + found.labels:
        print(line)
    if uncited:
        print(f"uncited entries in {SOURCE.relative_to(ROOT).as_posix()}:")
        for key in uncited:
            print(f"  {key}")
    if found.unresolved:
        print(f"{len(found.unresolved)} citation(s) not in the bibliography")
    if found.broken:
        print(f"{len(found.broken)} broken relative link(s)")
    if found.labels:
        print(f"{len(found.labels)} italic provenance label(s) under {LABEL_SCOPE}")
    if stale:
        print(
            f"{BIBLIOGRAPHY.relative_to(ROOT).as_posix()} differs from the page "
            f"generated from {SOURCE.relative_to(ROOT).as_posix()}; "
            "run `just references`"
        )


def main(argv: list[str] | None = None) -> int:
    """Check the files named in `argv`, or every file in `SCOPE` when none is named;
    return 1 if anything is flagged."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="*", type=Path)
    arguments = parser.parse_args(argv)

    generator = load_generator()
    source = SOURCE.read_text(encoding="utf-8")
    try:
        known = {reference.key for reference in generator.references(source)}
    except ValueError as error:
        print(f"{SOURCE.relative_to(ROOT).as_posix()}: {error}")
        return 1
    stale = stale_page(generator, source)
    files = arguments.paths or scope_files()
    found = scan(filter(in_scope, files), known)
    print_report(found, sorted(known - found.cited), stale)
    if found.unresolved or found.broken or found.labels or stale:
        return 1
    print(f"all {len(found.cited)} cited keys resolve; no broken relative links")
    return 0


if __name__ == "__main__":
    sys.exit(main())
