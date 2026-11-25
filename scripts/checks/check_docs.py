# Adapted from Jhko725/UniversalForceReconstruction scripts/checks/check_docs.py at
# b9e4cc3.
"""Check the document rules of ``docs/document-conventions.md`` § Checks.

The rules, each over every Markdown file under ``docs``:

1. ``type`` and ``status`` are in the closed vocabularies, and the status is one that
   the type allows (`STATUSES`).
2. The required headings of the type are present and in order (`REQUIRED_HEADINGS`);
   other headings may stand between them.
3. ``docs/index.md`` links every document by its path relative to ``docs``.
4. The Done ledger of ``docs/roadmap.md`` is chronological: the dates that open its
   table rows never decrease.
5. Every package directory under ``src/deep_isochron`` is named in
   ``docs/architecture.md``: as a backticked path, such as ``systems/normal_forms/`` or
   ``src/deep_isochron/systems/normal_forms/``, or as a directory of a fenced module
   tree, whose path the tree's indentation gives (``normal_forms/`` drawn under
   ``systems/`` under ``deep_isochron``).
6. An ADR under ``docs/decisions`` agrees with itself, its successor and the index: a
   ``superseded`` ADR carries ``superseded_by: ADR-<NNNN>``, opens above *Context*
   with a sentence naming that ADR, and is linked under the index's *Superseded* table;
   its successor exists and carries ``supersedes`` back; an ADR in force (every status
   other than ``superseded``, ``amended`` included) carries no ``superseded_by`` and is
   linked under *In force*.
7. No ``[unconfirmed]`` marker stands in a document whose status is ``agreed``,
   ``accepted``, ``amended`` or ``merged``.
8. Each form of `REQUIRED_HEADINGS` is quoted by exactly one fenced ``markdown``
   template of ``docs/document-conventions.md``, and each template quotes exactly one
   form. A template line is a heading, then optionally two or more spaces and a
   description (``## Summary        what the branch does and why``); its heading must
   equal the form's heading at the same place.

Sections, tables, links, ATX headings and fenced templates are read from the section
tree of ``markdown.py``, so text inside fenced or inline code is never taken for
structure. The frontmatter is split off by a
regular expression and read as YAML before the body is parsed.

Rule 5's fenced trees are this repository's addition: ``docs/architecture.md`` draws its
module map as a tree in a fenced block, and UFR's rule read backticked paths only. With
no document under ``docs`` every rule passes, and rules 3 to 6 run only once their
documents exist.

The frontmatter's form (valid YAML, the required keys, quoted ``sources``) is
``check_md_math.py``'s; a block that does not parse is reported here only as the reason
the document's other rules were skipped.

Usage: ``uv run python scripts/checks/check_docs.py`` (no arguments). Exits 1 on any
problem; problems are printed one per line.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import chain, takewhile
from pathlib import Path

import markdown
import yaml


ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
CONVENTIONS = DOCS / "document-conventions.md"
PACKAGE = ROOT / "src" / "deep_isochron"
STATUSES: dict[str, tuple[str, ...]] = {
    "architecture": ("current",),
    "roadmap": ("current",),
    "glossary": ("current",),
    "findings": ("current",),
    "verification": ("current",),
    "reference": ("current",),
    "index": ("current",),
    "design": ("draft for review", "tentative", "agreed"),
    "decision": ("accepted", "amended", "superseded"),
    "change": ("planned", "in-progress", "landed", "merged"),
    "guide": ("tentative", "current"),
    "plan": ("planned", "executed"),
}
"""The closed vocabularies: each ``type`` with the statuses it allows."""
REQUIRED_HEADINGS: dict[str, tuple[str, ...]] = {
    "change": (
        "## Summary",
        "## Design",
        "## Bugs fixed",
        "## Tests",
        "## Open issues",
        "## Review notes",
    ),
    "decision": (
        "## Context",
        "### Motivation",
        "### Forces",
        "## Decision",
        "## Consequences",
        "## Alternatives rejected",
        "## History",
    ),
    "design/method": (
        "## 1. The problem",
        "## 2. What exists",
        "## 3. The method",
        "## 4. Exact and approximated",
        "## 5. Oracles",
        "## 6. Limits",
    ),
    "plan": (
        "## 1. Header",
        "## 2. Out of scope and assumptions",
        "## 3. What the implementing session does",
        "## 4. Before anything",
        "## 5. Rule text",
        "## 6. Units",
        "## 7. Baseline",
        "## 8. The documents unit",
        "## 9. Gates",
        "## 10. Decisions and where they are recorded",
    ),
}
"""The required headings of each form, in order: a ``type``, or ``design/<kind>`` for
a design document. The fenced templates of ``docs/document-conventions.md`` quote the
same headings, and rule 8 asserts that the two agree."""
SETTLED = ("agreed", "accepted", "amended", "merged")
UNCONFIRMED = "[unconfirmed]"
# Inline code, in which the marker is mentioned and not used.
CODE_SPAN = re.compile(r"`[^`\n]*`")
FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n", re.S)
TEMPLATE_GAP = re.compile(r" {2,}")
"""What separates a template line's heading from its description."""
SCHEME = re.compile(r"^[a-z][a-z0-9+.-]*:")
BACKTICKED = re.compile(r"`([^`]+)`")
TREE_ENTRY = re.compile(r"^((?:[│ ] {3})*)[├└]── (\S+)")
"""A tree line that names an entry: the indentation, then a connector and the name."""
TREE_STEP = 4
"""The columns of one level of a tree's indentation (``│   `` or four blanks)."""
TREE_DRAWING = "│├└"
ADR_ID = re.compile(r"^ADR-(\d{4})$")
ADR_FILE = re.compile(r"^(\d{4})-.+\.md$")
LEDGER_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def headings(root: markdown.Section) -> list[str]:
    """The ATX headings of the tree `root`, as written: ``## Summary``."""
    return [
        f"{section.markup} {section.title}"
        for section in root.walk()
        if section.markup.startswith("#")
    ]


def relative_links(section: markdown.Section) -> list[str]:
    """The link targets of `section` and its subsections, without their ``#anchor``;
    bare anchors and targets with a scheme are left out."""
    links = (link for part in section.walk() for link in part.links)
    targets = (link.target.split("#", 1)[0] for link in links)
    return [target for target in targets if target and not SCHEME.match(target)]


class Document:
    """One Markdown file under ``docs``: its path, frontmatter and body."""

    def __init__(self, path: Path) -> None:
        """Read `path` and split off its frontmatter block."""
        self.path = path
        self.relative = path.relative_to(DOCS).as_posix()
        text = path.read_text(encoding="utf-8")
        match = FRONTMATTER.match(text)
        self.has_frontmatter = match is not None
        self.body = text[match.end() :] if match else text
        self.offset = text[: match.end()].count("\n") if match else 0
        self.meta: dict = {}
        self.parses = True
        if match:
            try:
                loaded = yaml.safe_load(match.group(1))
            except yaml.YAMLError:
                loaded = None
            self.parses = isinstance(loaded, dict)
            self.meta = loaded if isinstance(loaded, dict) else {}

    @property
    def type(self) -> str:
        """The frontmatter ``type``, or ``""``."""
        return str(self.meta.get("type", ""))

    @property
    def status(self) -> str:
        """The frontmatter ``status``, or ``""``."""
        return str(self.meta.get("status", ""))

    def headings(self) -> list[str]:
        """The ATX headings of the body outside code, as written."""
        return headings(markdown.parse(self.body))


def documents() -> list[Document]:
    """Every Markdown file under ``docs``, sorted by path."""
    return [Document(p) for p in sorted(DOCS.rglob("*.md"))] if DOCS.exists() else []


def vocabulary_problems(docs: list[Document]) -> Iterator[str]:
    """Rule 1: a missing or unparsable frontmatter, or a type or status outside the
    closed vocabularies."""
    for doc in docs:
        if not doc.has_frontmatter:
            yield f"docs/{doc.relative}: no frontmatter block"
        elif not doc.parses:
            yield f"docs/{doc.relative}: frontmatter does not parse; rules skipped"
        elif doc.type not in STATUSES:
            yield f"docs/{doc.relative}: type `{doc.type}` not in {sorted(STATUSES)}"
        elif doc.status not in STATUSES[doc.type]:
            allowed = ", ".join(f"`{s}`" for s in STATUSES[doc.type])
            yield (
                f"docs/{doc.relative}: status `{doc.status}` not allowed for type"
                f" `{doc.type}` ({allowed})"
            )


def form(doc: Document) -> str:
    """The key of `doc` in `REQUIRED_HEADINGS`: its type, or ``design/<kind>``."""
    if doc.type == "design":
        return f"design/{doc.meta.get('kind', '')}"
    return doc.type


def heading_problems(docs: list[Document]) -> Iterator[str]:
    """Rule 2: a required heading that is missing or out of order."""
    for doc in docs:
        required = REQUIRED_HEADINGS.get(form(doc))
        if required is None:
            continue
        present = doc.headings()
        position = 0
        for heading in required:
            try:
                position = present.index(heading, position) + 1
            except ValueError:
                where = "missing" if heading not in present else "out of order"
                yield f"docs/{doc.relative}: required heading `{heading}` {where}"


def link_targets(path: Path) -> set[str]:
    """The relative link targets of the Markdown file `path`, resolved against its
    directory and written relative to ``docs``."""
    targets = set()
    for target in relative_links(markdown.parse(path.read_text(encoding="utf-8"))):
        resolved = (path.parent / target).resolve()
        if resolved.is_relative_to(DOCS.resolve()):
            targets.add(resolved.relative_to(DOCS.resolve()).as_posix())
    return targets


def index_problems(docs: list[Document]) -> Iterator[str]:
    """Rule 3: a document that ``docs/index.md`` does not link."""
    if not docs:
        return
    index = DOCS / "index.md"
    if not index.exists():
        yield "docs/index.md missing"
        return
    linked = link_targets(index)
    for doc in docs:
        if doc.relative != "index.md" and doc.relative not in linked:
            yield f"docs/index.md does not link {doc.relative}"


def ledger_problems(docs: list[Document]) -> Iterator[str]:
    """Rule 4: a Done ledger in ``docs/roadmap.md`` that is missing or not
    chronological."""
    if not docs:
        return
    roadmap = DOCS / "roadmap.md"
    if not roadmap.exists():
        yield "docs/roadmap.md missing"
        return
    sections = markdown.parse(roadmap.read_text(encoding="utf-8")).walk()
    done = next((s for s in sections if s.level == 2 and s.title == "Done"), None)
    if done is None:
        yield "docs/roadmap.md has no `## Done` ledger"
        return
    first_cells = [
        row.cells[0]
        for part in done.walk()
        for table in part.tables
        for row in table.rows
        if row.cells
    ]
    dates = [cell for cell in first_cells if LEDGER_DATE.match(cell)]
    for earlier, later in zip(dates, dates[1:]):
        if later < earlier:
            yield (
                f"docs/roadmap.md Done ledger is not chronological: {later} after"
                f" {earlier} (append at the end)"
            )


def package_directories() -> list[str]:
    """Every directory under the package that holds a module, relative to it."""
    if not PACKAGE.exists():
        return []
    return sorted(
        {
            path.parent.relative_to(PACKAGE).as_posix()
            for path in PACKAGE.rglob("*.py")
            if path.parent != PACKAGE and "__pycache__" not in path.parts
        }
    )


def tree_paths(content: str) -> set[str]:
    """The paths drawn by the fenced tree `content`, each joined to its ancestors.

    A line that starts in the first column opens a tree with the first word as its
    root; a line whose name follows ``├── `` or ``└── `` is a child at the depth that
    its indentation of four-column steps gives. Continuation lines, which have no
    connector, name nothing.
    """
    paths: set[str] = set()
    stack: list[str] = []
    for line in content.splitlines():
        if match := TREE_ENTRY.match(line):
            depth = len(match.group(1)) // TREE_STEP + 1
            stack = [*stack[:depth], match.group(2).rstrip("/")]
        elif line[:1].strip() and line[0] not in TREE_DRAWING:
            stack = [line.split()[0].rstrip("/")]
        else:
            continue
        paths.add("/".join(stack))
    return paths


def architecture_names(text: str) -> set[str]:
    """The paths that the architecture document `text` names, without a trailing
    ``/``: its backticked spans, and the paths of the trees in its fenced blocks."""
    fences = (
        fence for section in markdown.parse(text).walk() for fence in section.fences
    )
    drawn = chain.from_iterable(tree_paths(fence.content) for fence in fences)
    backticked = (token.rstrip("/") for token in BACKTICKED.findall(text))
    return {*backticked, *drawn}


def architecture_problems(docs: list[Document]) -> Iterator[str]:
    """Rule 5: a package directory that ``docs/architecture.md`` does not name, by a
    backticked path or in a fenced tree."""
    if not docs:
        return
    architecture = DOCS / "architecture.md"
    if not architecture.exists():
        yield "docs/architecture.md missing"
        return
    tokens = architecture_names(architecture.read_text(encoding="utf-8"))
    for directory in package_directories():
        if not any(t == directory or t.endswith("/" + directory) for t in tokens):
            yield f"docs/architecture.md does not name the directory {directory}/"


def opening(doc: Document) -> str:
    """The text of an ADR above its *Context* heading, without the headings."""
    sections = markdown.parse(doc.body).walk()
    above = takewhile(lambda section: section.title != "Context", sections)
    return "\n".join(section.text for section in above)


def section_links(index: markdown.Section, title: str) -> set[str]:
    """The ``decisions/`` link targets under the index heading titled `title`,
    relative to ``docs``."""
    section = index.find(title)
    if section is None:
        return set()
    return {
        target.removeprefix("./")
        for target in relative_links(section)
        if target.removeprefix("./").startswith("decisions/")
    }


@dataclass(frozen=True)
class Adr:
    """One ADR under ``docs/decisions``, with what rule 6 judges about it.

    `successor` and `predecessor` are the frontmatter ``superseded_by`` and
    ``supersedes`` as written, or ``""``; `opening` is the text above *Context*; and
    `under_in_force` and `under_superseded` say whether the index links the ADR under
    the table of that title.
    """

    number: str
    relative: str
    status: str
    successor: str
    predecessor: str
    opening: str
    under_in_force: bool
    under_superseded: bool

    @property
    def name(self) -> str:
        """The ADR's path relative to the root, as the messages name it."""
        return f"docs/{self.relative}"

    @property
    def in_force(self) -> bool:
        """Whether the ADR is in force: every status other than ``superseded``,
        ``amended`` included."""
        return self.status != "superseded"


def adr_records(docs: list[Document]) -> dict[str, Adr]:
    """The ADRs among `docs`, keyed by their four-digit number; the index is read only
    when there is one."""
    found = {
        match.group(1): doc
        for doc in docs
        if doc.relative.startswith("decisions/")
        and (match := ADR_FILE.match(doc.path.name))
    }
    if not found:
        return {}
    index = DOCS / "index.md"
    text = index.read_text(encoding="utf-8") if index.exists() else ""
    sections = markdown.parse(text)
    in_force = section_links(sections, "In force")
    superseded = section_links(sections, "Superseded")
    return {
        number: Adr(
            number=number,
            relative=doc.relative,
            status=doc.status,
            successor=str(doc.meta.get("superseded_by", "")),
            predecessor=str(doc.meta.get("supersedes", "")),
            opening=opening(doc),
            under_in_force=doc.relative in in_force,
            under_superseded=doc.relative in superseded,
        )
        for number, doc in found.items()
    }


def superseded_rules(adr: Adr, adrs: dict[str, Adr]) -> Iterator[str]:
    """A superseded ADR names an existing successor that names it back, and opens
    above *Context* with a sentence naming that successor."""
    if adr.in_force:
        return
    match = ADR_ID.match(adr.successor)
    if match is None:
        yield f"{adr.name}: superseded without `superseded_by: ADR-<NNNN>`"
        return
    later = adrs.get(match.group(1))
    if later is None:
        yield f"{adr.name}: `superseded_by` names a missing {adr.successor}"
    elif later.predecessor != f"ADR-{adr.number}":
        yield (
            f"{later.name}: lacks `supersedes: ADR-{adr.number}`,"
            f" which {adr.name} names as its successor"
        )
    if adr.successor not in adr.opening:
        yield f"{adr.name}: does not open, above Context, naming {adr.successor}"


def in_force_rules(adr: Adr) -> Iterator[str]:
    """An ADR in force carries no ``superseded_by`` and does not open with a
    redirect."""
    if not adr.in_force:
        return
    if adr.successor:
        yield f"{adr.name}: carries `superseded_by` with status `{adr.status}`"
    if re.search(r"superseded by", adr.opening, re.I):
        yield f"{adr.name}: opens with a redirect with status `{adr.status}`"


def placement_rules(adr: Adr) -> Iterator[str]:
    """The index links an ADR in force under *In force* only, and a superseded ADR
    under *Superseded* only."""
    if not adr.in_force and not adr.under_superseded:
        yield f"docs/index.md: {adr.name} is not under the Superseded table"
    if not adr.in_force and adr.under_in_force:
        yield f"docs/index.md: superseded {adr.name} is under the In force table"
    if adr.in_force and not adr.under_in_force:
        yield f"docs/index.md: {adr.name} is not under the In force table"
    if adr.in_force and adr.under_superseded:
        yield f"docs/index.md: {adr.name} in force is under the Superseded table"


def predecessor_rules(adr: Adr, adrs: dict[str, Adr]) -> Iterator[str]:
    """An ADR that supersedes another names an existing ADR that names it back."""
    if not adr.predecessor:
        return
    match = ADR_ID.match(adr.predecessor)
    earlier = adrs.get(match.group(1)) if match else None
    if earlier is None:
        yield f"{adr.name}: `supersedes` names a missing {adr.predecessor}"
    elif earlier.successor != f"ADR-{adr.number}":
        yield (
            f"{earlier.name}: lacks `superseded_by: ADR-{adr.number}`,"
            f" which {adr.name} names as its predecessor"
        )


def adr_problems(docs: list[Document]) -> Iterator[str]:
    """Rule 6: an ADR whose status, ``superseded_by``, ``supersedes``, opening line and
    index table disagree, ADR by ADR in the order of their numbers."""
    adrs = adr_records(docs)
    for _, adr in sorted(adrs.items()):
        yield from chain(
            superseded_rules(adr, adrs),
            in_force_rules(adr),
            placement_rules(adr),
            predecessor_rules(adr, adrs),
        )


def unconfirmed_problems(docs: list[Document]) -> Iterator[str]:
    """Rule 7: an ``[unconfirmed]`` marker inside a settled document.

    The marker in code font is a mention of it, as in a sentence about removing
    markers, and does not count.
    """
    for doc in docs:
        if doc.status not in SETTLED:
            continue
        for number, line in enumerate(doc.body.splitlines(), start=doc.offset + 1):
            if UNCONFIRMED in CODE_SPAN.sub("", line):
                yield (
                    f"docs/{doc.relative}:{number}: `{UNCONFIRMED}` in a document with"
                    f" status `{doc.status}`"
                )


def templates() -> list[tuple[str, ...]]:
    """The headings quoted by each fenced ``markdown`` template of the conventions."""
    root = markdown.parse(CONVENTIONS.read_text(encoding="utf-8"))
    fences = chain.from_iterable(section.fences for section in root.walk())
    found = []
    for fence in fences:
        if fence.info != "markdown":
            continue
        quoted = tuple(
            TEMPLATE_GAP.split(line, maxsplit=1)[0]
            for line in fence.content.splitlines()
            if line.startswith("#")
        )
        if quoted:
            found.append(quoted)
    return found


def template_problems(docs: list[Document]) -> Iterator[str]:
    """Rule 8: a form that is not quoted by exactly one template of the conventions,
    or a template that quotes not exactly one form."""
    if not docs:
        return
    if not CONVENTIONS.exists():
        yield "docs/document-conventions.md missing"
        return
    quoted = templates()
    for name, headings in sorted(REQUIRED_HEADINGS.items()):
        count = quoted.count(headings)
        if count != 1:
            yield (
                f"docs/document-conventions.md: {count} template(s) quote the required"
                f" headings of `{name}`: {' / '.join(headings)}"
            )
    for headings in quoted:
        if headings not in REQUIRED_HEADINGS.values():
            yield (
                "docs/document-conventions.md: a template quotes no form of"
                f" REQUIRED_HEADINGS: {' / '.join(headings)}"
            )


def main() -> int:
    """Run the eight rules over ``docs``; return the exit code."""
    docs = documents()
    checked = [doc for doc in docs if doc.parses]
    problems = [
        *vocabulary_problems(docs),
        *heading_problems(checked),
        *index_problems(docs),
        *ledger_problems(docs),
        *architecture_problems(docs),
        *adr_problems(checked),
        *unconfirmed_problems(checked),
        *template_problems(docs),
    ]
    for problem in problems:
        print("PROBLEM:", problem)
    print(f"{len(docs)} document(s) checked, {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
