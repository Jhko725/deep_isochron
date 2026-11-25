# Adapted from Jhko725/UniversalForceReconstruction scripts/checks/check_handoff.py at
# b9e4cc3.
"""Check a handoff document against the repository, so that a unit's brief names files,
lines, recipes and tests that exist.

A handoff is a plan whose units cite paths, ``path:line`` references, ``just`` recipes
and pytest selectors. This script reads the document with ``markdown.py`` and reports,
as ``path:line: reason`` with the path and line of the handoff document, every mention
that the repository does not bear out:

- a backticked path with a slash and a file suffix that does not exist, unless the same
  sentence or table cell says ``(new)``;
- a ``path:line`` reference (also ``path:a-b`` and ``path:a,b``) whose line is past the
  end of the file;
- a ``just <recipe>`` that ``just --summary`` does not list;
- a ``tests/<file>.py::<name>`` selector, or a ``-k`` expression in a fenced command,
  that ``pytest --collect-only -q`` matches with no test.

A path is looked up relative to the repository root first, then relative to
``src/deep_isochron/``, since a handoff often abbreviates the second. A
path that starts with ``claude/`` names a Project document and is not checked, and a
path with a placeholder (``<date>``) or a glob (``*``) does not match the path pattern,
so it is not checked either.

``--commit <sha>`` reads paths and line counts from that commit through git, and the
recipes from the ``justfile`` of that commit. Selectors and ``-k`` expressions are
collected from the working tree, so they are checked only when the commit is ``HEAD``,
the default; otherwise one line says they were skipped. A ``-k`` expression is matched
as pytest does, by case-insensitive substring against the test's node id, which holds
its file, class, function and parameters.

This script is run by hand, since a handoff lives outside the repository until its
round starts, and it is not a hook.

Usage: ``uv run python scripts/checks/check_handoff.py <document> [--commit <sha>]``.
Exits 1 when anything is flagged, 2 when the commit does not exist.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import markdown


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "src/deep_isochron"
PROJECT_PREFIX = "claude/"
NEW_MARKER = "(new)"

SELECTOR = re.compile(r"(tests/[\w./-]+\.py)::([^\s\"'`]+)")
PATH = re.compile(r"([\w.-]+(?:/[\w.-]+)+\.[A-Za-z]\w*)(?::(\d+(?:[-,]\d+)*))?")
INLINE_RECIPE = re.compile(r"(?:uv run )?just\s+([A-Za-z][\w-]*)")
COMMAND_RECIPE = re.compile(r"\s*(?:\$ )?(?:uv run )?just\s+([A-Za-z][\w-]*)")
KEYWORD_OPTION = re.compile(r"-k\s+(?:\"([^\"]*)\"|'([^']*)'|(\S+))")
TEST_FILE = re.compile(r"tests/[\w./-]+\.py")
CODE_SPAN = re.compile(r"`([^`]+)`")
LIST_ITEM = re.compile(r"\s*(?:[-*+]|\d+[.)])\s")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
KEYWORD_TOKEN = re.compile(r"\(|\)|[^\s()]+")


@dataclass(frozen=True)
class Finding:
    """One flagged mention: its 1-based line in the handoff document and the reason."""

    line: int
    reason: str


@dataclass(frozen=True)
class Unit:
    """A sentence of prose or a table cell, with the 1-based line where it starts.

    A ``(new)`` marker exempts the paths of its own unit only.
    """

    line: int
    text: str

    def line_of(self, offset: int) -> int:
        """The line of the character at `offset` of the text."""
        return self.line + self.text.count("\n", 0, offset)


@dataclass(frozen=True)
class PathRef:
    """A backticked path, with the line numbers it cites and whether its unit says
    ``(new)``."""

    line: int
    path: str
    numbers: tuple[int, ...]
    new: bool


@dataclass(frozen=True)
class Recipe:
    """A ``just`` recipe named at `line`."""

    line: int
    name: str


@dataclass(frozen=True)
class Selector:
    """A pytest selector ``file::name`` written at `line`."""

    line: int
    file: str
    name: str


@dataclass(frozen=True)
class KeywordExpression:
    """A ``-k`` expression of a fenced command at `line`, with the test files that the
    command names (none means every test)."""

    line: int
    expression: str
    files: tuple[str, ...]


@dataclass(frozen=True)
class Mentions:
    """Everything a handoff mentions that the repository can confirm or refute."""

    paths: tuple[PathRef, ...]
    recipes: tuple[Recipe, ...]
    selectors: tuple[Selector, ...]
    expressions: tuple[KeywordExpression, ...]


def body_start(section: markdown.Section) -> int:
    """The 1-based line where the own text of `section` starts: the root starts at line
    1, an ATX heading takes one line and a setext heading two."""
    if section.level == 0:
        return 1
    return section.line + (1 if section.markup.startswith("#") else 2)


def fenced_lines(section: markdown.Section) -> set[int]:
    """The document lines taken by the fences of `section`, opening and closing fence
    included."""
    taken: set[int] = set()
    for fence in section.fences:
        taken.update(range(fence.line, fence.line + fence.content.count("\n") + 2))
    return taken


def split_sentences(paragraph: str, first_line: int) -> list[Unit]:
    """The sentences of `paragraph`, which starts at `first_line`."""
    units, start = [], 0
    for boundary in [*SENTENCE_END.finditer(paragraph), None]:
        end = boundary.start() if boundary else len(paragraph)
        sentence = paragraph[start:end]
        if sentence.strip():
            units.append(Unit(first_line + paragraph.count("\n", 0, start), sentence))
        start = boundary.end() if boundary else end
    return units


def table_cells(line: int, text: str) -> list[Unit]:
    """The cells of the table row `text` at `line`."""
    return [Unit(line, cell) for cell in text.strip().strip("|").split("|")]


def prose_units(section: markdown.Section) -> list[Unit]:
    """The sentences and table cells of the own text of `section`, outside its fences.

    A blank line or a list item starts a new paragraph; a line that starts with ``|``
    is a table row, and each of its cells is a unit.
    """
    skipped = fenced_lines(section)
    units: list[Unit] = []
    paragraph: list[str] = []
    first = 0

    def flush() -> None:
        """Turn the collected paragraph lines into sentences."""
        if paragraph:
            units.extend(split_sentences("\n".join(paragraph), first))
        paragraph.clear()

    for number, text in enumerate(section.text.split("\n"), body_start(section)):
        is_table = text.lstrip().startswith("|")
        if number in skipped or not text.strip() or is_table or LIST_ITEM.match(text):
            flush()
        if number in skipped or not text.strip():
            continue
        if is_table:
            units.extend(table_cells(number, text))
            continue
        if not paragraph:
            first = number
        paragraph.append(text)
    flush()
    return units


def span_mentions(unit: Unit) -> Iterator[PathRef | Recipe | Selector]:
    """The paths, recipes and selectors in the code spans of `unit`."""
    new = NEW_MARKER in unit.text
    for span in CODE_SPAN.finditer(unit.text):
        content, line = span.group(1).strip(), unit.line_of(span.start())
        selector = SELECTOR.fullmatch(content)
        path = PATH.fullmatch(content)
        recipe = INLINE_RECIPE.match(content)
        if selector:
            yield Selector(line, selector.group(1), selector.group(2))
        elif path and not path.group(1).startswith(PROJECT_PREFIX):
            numbers = re.findall(r"\d+", path.group(2) or "")
            yield PathRef(line, path.group(1), tuple(map(int, numbers)), new)
        elif recipe:
            yield Recipe(line, recipe.group(1))


def commands(fence: markdown.Fence) -> list[tuple[int, str]]:
    """The commands of `fence` as ``(line, text)``, with a continued command joined
    into one text that carries the line of its first physical line."""
    found: list[tuple[int, str]] = []
    joined, first = "", 0
    for number, text in enumerate(fence.content.split("\n"), fence.line + 1):
        first = first if joined else number
        if text.endswith("\\"):
            joined += text[:-1] + " "
            continue
        if joined or text.strip():
            found.append((first, joined + text))
        joined = ""
    return found


def command_mentions(
    line: int, text: str
) -> Iterator[Recipe | Selector | KeywordExpression]:
    """The recipe, selectors and ``-k`` expression of one fenced command."""
    recipe = COMMAND_RECIPE.match(text)
    if recipe:
        yield Recipe(line, recipe.group(1))
    for selector in SELECTOR.finditer(text):
        yield Selector(line, selector.group(1), selector.group(2))
    option = KEYWORD_OPTION.search(text)
    if option:
        expression = next(group for group in option.groups() if group is not None)
        files = tuple(dict.fromkeys(TEST_FILE.findall(text)))
        yield KeywordExpression(line, expression, files)


def collect_mentions(root: markdown.Section) -> Mentions:
    """Every path, recipe, selector and ``-k`` expression in the tree under `root`."""
    found: list[PathRef | Recipe | Selector | KeywordExpression] = []
    for section in root.walk():
        for unit in prose_units(section):
            found.extend(span_mentions(unit))
        for fence in section.fences:
            for line, text in commands(fence):
                found.extend(command_mentions(line, text))
    return Mentions(
        paths=tuple(m for m in found if isinstance(m, PathRef)),
        recipes=tuple(m for m in found if isinstance(m, Recipe)),
        selectors=tuple(m for m in found if isinstance(m, Selector)),
        expressions=tuple(m for m in found if isinstance(m, KeywordExpression)),
    )


def git(*arguments: str) -> subprocess.CompletedProcess[bytes]:
    """Run git in the repository root and capture its output."""
    return subprocess.run(["git", "-C", str(ROOT), *arguments], capture_output=True)


@cache
def line_count(commit: str, path: str) -> int | None:
    """The number of lines of the file `path` at `commit`, or ``None`` if the commit
    has no such file."""
    shown = git("cat-file", "blob", f"{commit}:{path}")
    return len(shown.stdout.splitlines()) if shown.returncode == 0 else None


def resolved_line_count(commit: str, path: str) -> int | None:
    """The line count of `path` at `commit`, looked up from the repository root and
    then from the package directory; ``None`` if neither exists."""
    for candidate in (path, f"{PACKAGE}/{path}"):
        count = line_count(commit, candidate)
        if count is not None:
            return count
    return None


def path_findings(ref: PathRef, count: int | None) -> list[Finding]:
    """What is wrong with the path mention `ref`, given the `count` of lines of the
    file it names (``None`` for a file that does not exist)."""
    if count is None:
        return [] if ref.new else [Finding(ref.line, f"`{ref.path}` does not exist")]
    if ref.numbers and max(ref.numbers) > count:
        return [
            Finding(
                ref.line,
                f"`{ref.path}` has {count} lines; line {max(ref.numbers)} is past "
                "the end",
            )
        ]
    return []


def recipe_names(commit: str) -> set[str]:
    """The recipes that ``just --summary`` lists for the ``justfile`` at `commit`."""
    shown = git("show", f"{commit}:justfile")
    with tempfile.TemporaryDirectory() as directory:
        copy = Path(directory) / "justfile"
        copy.write_bytes(shown.stdout)
        summary = subprocess.run(
            ["just", "--justfile", str(copy), "--working-directory", str(ROOT)]
            + ["--summary"],
            capture_output=True,
            text=True,
            check=True,
        )
    return set(summary.stdout.split())


def collected_tests() -> list[str]:
    """The node ids that ``pytest --collect-only -q`` lists in the working tree."""
    collected = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in collected.stdout.splitlines() if "::" in line]


def selector_matches(tests: list[str], selector: Selector) -> bool:
    """Whether some test of `tests` is the selected test or one of its parameter sets
    or members."""
    prefix = f"{selector.file}::{selector.name}"
    return any(
        test == prefix or test.startswith((prefix + "[", prefix + "::"))
        for test in tests
    )


def keyword_matcher(expression: str) -> Callable[[str], bool]:
    """The predicate that tells whether a node id satisfies the ``-k`` `expression`.

    Each word is a case-insensitive substring test; ``and``, ``or``, ``not`` and
    parentheses combine them. Raises ``ValueError`` for an expression that does not
    parse.
    """
    operators = {"and", "or", "not", "(", ")"}
    parts = [
        token if token in operators else f"m({token!r})"
        for token in KEYWORD_TOKEN.findall(expression)
    ]
    try:
        code = compile(" ".join(parts), "<-k expression>", "eval")
    except SyntaxError as error:
        raise ValueError(f"-k expression {expression!r} does not parse") from error

    def matches(node_id: str) -> bool:
        """Whether `node_id` satisfies the expression."""
        lowered = node_id.lower()

        def m(word: str) -> bool:
            """Whether `word` occurs in the node id."""
            return word.lower() in lowered

        return bool(eval(code, {"__builtins__": {}}, {"m": m}))

    return matches


def expression_findings(tests: list[str], found: KeywordExpression) -> list[Finding]:
    """What is wrong with the ``-k`` expression `found`, given the collected `tests`."""
    try:
        matches = keyword_matcher(found.expression)
    except ValueError as error:
        return [Finding(found.line, str(error))]
    pool = [t for t in tests if not found.files or t.split("::")[0] in found.files]
    if any(matches(test) for test in pool):
        return []
    return [Finding(found.line, f"`-k {found.expression}` matches no collected test")]


def recipe_findings(recipes: tuple[Recipe, ...], commit: str) -> list[Finding]:
    """One finding for each recipe of `recipes` that the ``justfile`` at `commit`
    lacks."""
    if not recipes:
        return []
    known = recipe_names(commit)
    return [
        Finding(recipe.line, f"`just {recipe.name}` is not a recipe of the justfile")
        for recipe in recipes
        if recipe.name not in known
    ]


def test_findings(mentions: Mentions) -> list[Finding]:
    """The findings for the selectors and ``-k`` expressions of `mentions`."""
    if not (mentions.selectors or mentions.expressions):
        return []
    tests = collected_tests()
    findings = [
        Finding(s.line, f"`{s.file}::{s.name}` matches no collected test")
        for s in mentions.selectors
        if not selector_matches(tests, s)
    ]
    for found in mentions.expressions:
        findings.extend(expression_findings(tests, found))
    return findings


def check(text: str, commit: str, at_head: bool) -> tuple[list[Finding], bool]:
    """The findings for the handoff `text` against `commit`, and whether the checks of
    tests were skipped because `at_head` is false."""
    mentions = collect_mentions(markdown.parse(text))
    findings = [
        finding
        for ref in mentions.paths
        for finding in path_findings(ref, resolved_line_count(commit, ref.path))
    ]
    findings.extend(recipe_findings(mentions.recipes, commit))
    skipped = not at_head and bool(mentions.selectors or mentions.expressions)
    if at_head:
        findings.extend(test_findings(mentions))
    return sorted(findings, key=lambda finding: finding.line), skipped


def resolve_commit(name: str) -> str | None:
    """The full hash of the commit `name`, or ``None`` if there is none."""
    resolved = git("rev-parse", "--verify", f"{name}^{{commit}}")
    return resolved.stdout.decode().strip() if resolved.returncode == 0 else None


def main(argv: list[str] | None = None) -> int:
    """Check the handoff named in `argv`; return 1 if anything is flagged."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("document", type=Path)
    parser.add_argument("--commit", default="HEAD")
    arguments = parser.parse_args(argv)

    commit, head = resolve_commit(arguments.commit), resolve_commit("HEAD")
    if commit is None:
        print(f"no commit {arguments.commit}")
        return 2
    text = arguments.document.read_text(encoding="utf-8")
    findings, skipped = check(text, commit, commit == head)
    for finding in findings:
        print(f"{arguments.document}:{finding.line}: {finding.reason}")
    if skipped:
        print("selectors and -k expressions skipped: the commit is not HEAD")
    print(f"{len(findings)} mention(s) flagged against {commit[:7]}")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
