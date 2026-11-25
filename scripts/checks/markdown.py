# From Jhko725/UniversalForceReconstruction scripts/checks/markdown.py at b9e4cc3.
"""Parse a Markdown document into a tree of sections, each with its tables, links and
subsections, for the check scripts.

`parse` reads a document with `PARSER`, the CommonMark parser of ``markdown_it`` with
tables enabled, and every check script that reads Markdown structure uses that parser.
``markdown_it`` returns a flat list of tokens: its ``SyntaxTreeNode`` nests the rows
and cells of a table, but it leaves the content under a heading beside the heading. The
section tree is built here on top of the tokens, so that a check goes to a section by
its title and reads the tables there, and text inside fenced or inline code is never
taken for structure.

The tree is built in two stages. `_flat_sections` cuts the tokens at every heading into
sections without subsections, and `_nest` then places each section under the nearest
preceding section of a lower level. A section therefore holds what stands between its
heading and its first subsection, and `Section.walk` gives the whole section with its
subsections. The document itself is a root section of level 0, which holds what
precedes the first heading.

A section's links are `Link` records, each with its target and its 1-based line, and
this module is the one place that walks the inline tokens to find them: a check reads
the links of a document from its sections. Its fenced code blocks are `Fence` records
in the same way.

A YAML frontmatter block at the top of the file is not Markdown: read as Markdown, its
closing ``---`` makes its last line a heading. `parse` blanks the block out before
parsing, so that line numbers still count from the top of the file.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from urllib.parse import unquote

from markdown_it import MarkdownIt
from markdown_it.token import Token


PARSER = MarkdownIt("commonmark").enable("table")
"""The parser of every check script that reads Markdown structure."""
FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n", re.S)


@dataclass(frozen=True)
class Link:
    """One inline link: its 1-based line and its target, percent-decoded with its
    ``#anchor`` kept."""

    line: int
    target: str


@dataclass(frozen=True)
class Row:
    """One body row of a table: its 1-based line and its cells, as written."""

    line: int
    cells: tuple[str, ...]


@dataclass(frozen=True)
class Table:
    """One table: the cells of its header row and its body rows."""

    header: tuple[str, ...]
    rows: tuple[Row, ...]


@dataclass(frozen=True)
class Fence:
    """One fenced code block: its info string, its content and the 1-based line of its
    opening fence.

    `info` is the text after the opening fence without surrounding blanks, so ``""``
    for a fence with no language.
    """

    info: str
    content: str
    line: int


@dataclass(frozen=True)
class Section:
    """One heading and what stands under it, down to its first subsection.

    `title` is the heading's text as written, inline markup included, and `line` is the
    heading's 1-based line. `markup` is the heading's own markup: ``"##"`` for an ATX
    heading of level 2, so that ``f"{markup} {title}"`` rebuilds ``## Summary``, and
    ``"="`` or ``"-"`` for a setext heading. `tables`, `links`, `fences` and `text` are
    the section's own: those after its heading and before its first subsection.
    `links` also holds the links of the heading itself. `text` is the source of the
    section's own lines, without the heading. The root section of a document has
    level 0, the markup and the title ``""`` and the line 1.
    """

    title: str
    level: int
    line: int
    markup: str
    tables: tuple[Table, ...]
    links: tuple[Link, ...]
    fences: tuple[Fence, ...]
    text: str
    subsections: tuple[Section, ...] = ()

    @classmethod
    def from_tokens(
        cls, tokens: Sequence[Token], lines: Sequence[str], end_line: int
    ) -> Section:
        """The section, without subsections, whose tokens are `tokens`.

        The slice runs from the section's ``heading_open`` token up to the next
        heading's token, or from the start of the document to its first heading for the
        root, and the root's slice may be empty. `lines` are the lines of the document,
        and `end_line` is the 0-based index of the first line after the section: the
        first line of the next heading, or ``len(lines)`` for the last section.
        """
        heading = tokens[0] if tokens and tokens[0].type == "heading_open" else None
        first = heading.map[1] if heading and heading.map else 0
        return cls(
            title=tokens[1].content if heading else "",
            level=int(heading.tag[1]) if heading else 0,
            line=heading.map[0] + 1 if heading and heading.map else 1,
            markup=heading.markup if heading else "",
            tables=_tables(tokens),
            links=_links(tokens),
            fences=_fences(tokens),
            text="\n".join(lines[first:end_line]),
        )

    def walk(self) -> Iterator[Section]:
        """This section and every section under it, in document order."""
        yield self
        for subsection in self.subsections:
            yield from subsection.walk()

    def find(self, title: str) -> Section | None:
        """The first section of `walk` titled `title`, or ``None``."""
        return next((s for s in self.walk() if s.title == title), None)


def _spans(tokens: Sequence[Token], kind: str) -> list[Sequence[Token]]:
    """The runs of `tokens` from each ``<kind>_open`` token to its ``<kind>_close``,
    for a kind that does not nest, such as ``table`` or ``tr``."""
    opens = [i for i, token in enumerate(tokens) if token.type == f"{kind}_open"]
    closes = [i for i, token in enumerate(tokens) if token.type == f"{kind}_close"]
    return [tokens[o : c + 1] for o, c in zip(opens, closes, strict=True)]


def _cells(row: Sequence[Token]) -> tuple[str, ...]:
    """The cells of the row whose tokens are `row`: one ``inline`` token each."""
    return tuple(token.content for token in row if token.type == "inline")


def _table(tokens: Sequence[Token]) -> Table:
    """The table whose tokens, from ``table_open`` to ``table_close``, are `tokens`.

    The rows before ``tbody_open`` are the header; a table with no body row has no
    ``tbody_open``.
    """
    body = next(
        (i for i, token in enumerate(tokens) if token.type == "tbody_open"),
        len(tokens),
    )
    header = _spans(tokens[:body], "tr")
    rows = _spans(tokens[body:], "tr")
    return Table(
        header=_cells(header[0]) if header else (),
        rows=tuple(Row(row[0].map[0] + 1, _cells(row)) for row in rows if row[0].map),
    )


def _tables(tokens: Sequence[Token]) -> tuple[Table, ...]:
    """The tables among `tokens`, in order."""
    return tuple(_table(table) for table in _spans(tokens, "table"))


def _links(tokens: Sequence[Token]) -> tuple[Link, ...]:
    """The links in the inline tokens among `tokens`, in order.

    A link's line is the first line of its block plus the soft and hard line breaks
    before it in the block. A token without a source map contributes no link.
    """
    found = []
    for token in tokens:
        if token.map is None:
            continue
        line = token.map[0] + 1
        for child in token.children or []:
            if child.type in ("softbreak", "hardbreak"):
                line += 1
            elif child.type == "link_open":
                found.append(Link(line, unquote(str(child.attrGet("href") or ""))))
    return tuple(found)


def _fences(tokens: Sequence[Token]) -> tuple[Fence, ...]:
    """The fenced code blocks among `tokens`, in order."""
    return tuple(
        Fence(token.info.strip(), token.content, token.map[0] + 1)
        for token in tokens
        if token.type == "fence" and token.map
    )


def _flat_sections(tokens: list[Token], lines: list[str]) -> list[Section]:
    """The root section, then one section per heading, none with subsections.

    Each section's tokens run from its heading to the next heading of any level; the
    root's run from the start to the first heading.
    """
    starts = [i for i, token in enumerate(tokens) if token.type == "heading_open"]
    bounds = zip([0, *starts], [*starts, len(tokens)], strict=True)
    sections = []
    for start, end in bounds:
        after = tokens[end].map if end < len(tokens) else None
        end_line = after[0] if after else len(lines)
        sections.append(Section.from_tokens(tokens[start:end], lines, end_line))
    return sections


def _nest(flat: list[Section], index: int = 0) -> tuple[Section, int]:
    """The section `flat[index]` with every later section of a higher level placed
    under it, and the index of the first section after it."""
    parent, index = flat[index], index + 1
    children = []
    while index < len(flat) and flat[index].level > parent.level:
        child, index = _nest(flat, index)
        children.append(child)
    return copy.replace(parent, subsections=tuple(children)), index


def parse(text: str) -> Section:
    """The document `text` as its root section."""
    match = FRONTMATTER.match(text)
    if match:
        text = "\n" * match.group(0).count("\n") + text[match.end() :]
    root, _ = _nest(_flat_sections(PARSER.parse(text), text.splitlines()))
    return root
