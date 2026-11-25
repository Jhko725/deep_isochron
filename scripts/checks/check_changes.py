# Adapted from Jhko725/UniversalForceReconstruction, scripts/checks/check_changes.py,
# at b9e4cc3.
"""Check that each review round of a change document covers exactly the files changed
in that round, so that Joon reviews every change from one list and a verdict always
sits beside the change it was about.

`docs/document-conventions.md` § Change documents fixes the form. *Review notes* is the
only list of changes. It is a log of review rounds, appended at the end: each round has
a subheading ``### Round <n>: from `<commit>` `` and a table with the columns
``Change | Thoughts | Modifications``. A *Change* cell is one backticked path, a dash,
and what changed, on one line. A path that ends in ``/`` covers every changed file
under it; any other path covers only itself. The *Thoughts* and *Modifications* columns
are Joon's.

The stages, in the order of this module:

1. **The rounds of the document**: `rounds` reads the subsections of *Review notes*
   and the rows of their tables, from the section tree of ``markdown.py``. A round is
   reviewed when a *Thoughts* cell of it is filled; a *Modifications* cell does not
   count, since Claude fills that column when asked. A row under
   *Review notes* before its first subsection stands outside every round, and
   `outside_rows` finds it.
2. **The review commits**: `review_commits` finds the commits that review a round. A
   review commit is a commit to the change document that Claude did not author and
   that changes a *Thoughts* or *Modifications* cell. The history alone knows them: a
   round after the first starts from one, and the latest one ends a reviewed last
   round.
3. **The range of each round**: a round runs from the commit of its subheading to the
   commit of the next round's subheading. Round 1 starts from the merge base of
   ``origin/<base>`` and ``HEAD``, where ``<base>`` is the branch that the
   document's frontmatter field ``base`` names, and ``master`` without the field. A
   branch cut from an unmerged branch names that branch, so that its first round
   lists only its own files. Each later round from a review commit after the one
   that the round before it starts from, and an open last round from the latest review
   commit. The last round is the open one unless it is
   reviewed, and it runs to the index: at commit time the index is what is being
   committed, and in CI the index equals ``HEAD``. A reviewed last round runs to the
   latest review commit, and an index that differs from that commit needs a new round
   from it.

   A file other than the change document that a review commit changes needs a row in
   the round that the commit closes or in the round that it opens, and a row in either
   round that names it covers a changed file. While the round that the commit opens is
   not written yet, such a file needs no row: the row comes with the next commit, which
   opens that round.

   The commit being made can itself be a review: its author is not Claude, and the
   index changes a *Thoughts* or *Modifications* cell. The last round then runs to the
   index, a reviewed last round needs no review commit in the history yet, and the
   other files that the index changes need no row.
4. **The problems**: a row outside a round; a subheading that does not name the merge
   base for round 1, or for a later round a review commit after the previous round's,
   and for an open last round the latest one; a round before the last
   that is not reviewed; a reviewed last round with changes beyond it. A round's rows
   must cover exactly the files that ``git diff --name-only --find-renames`` lists for
   its range, and a *Change* cell must have the form above. When Claude is the author
   of the commit being made, a row of a round that is reviewed in ``HEAD`` must be as
   it is there: Claude appends rows and never edits a reviewed one.

A document whose status is ``merged`` is skipped, since its branch is no longer the one
checked out. The documents in `OLD_FORMAT` are skipped before their frontmatter is read:
they were written in the earlier format of a *Files* section and one review table.
Without ``origin/master``, or without the branch that a document's ``base`` names, the
merge base is unknown, and the check fails with a message that says to fetch it. On
``master`` itself the check passes, since no branch is under review.

Usage: `uv run python scripts/checks/check_changes.py [path ...]` (default: every
Markdown file under `docs/changes`). Exits 1 when anything is flagged.
"""

from __future__ import annotations

import re
import subprocess
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import markdown
import yaml


DEFAULT_DIR = Path("docs/changes")
REVIEW_HEADING = "Review notes"
BASE_REF = "origin/master"
"""The base of a document without a ``base`` field, and the ref whose merge base with
``HEAD`` tells whether a branch is under review at all."""
CLAUDE_EMAIL = "noreply@anthropic.com"
"""The author email of every commit that Claude makes."""
CHANGE_CELL = re.compile(r"`(?P<path>[^`]+)` — \S")
"""A *Change* cell: one backticked path, a dash, and the start of what changed."""
ROUND_HEADING = re.compile(r"Round (?P<number>\d+): from `(?P<commit>[0-9a-f]{7,40})`")
"""A round's subheading: its number and the commit that its range starts from."""
FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n", re.S)
OLD_FORMAT = (
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
    "2026-10-29-plotting.md",
    "2026-10-30-deformation-plots.md",
    "2026-10-30-phase-turns.md",
    "2026-11-02-ksiam-animations.md",
    "2026-11-04-conjugacy-schematic.md",
    "2026-11-04-inn-schematics.md",
)
"""The change documents written before the rounds format (2026-11-25), with a *Files*
section and one review table; history, not checked. The tuple never grows."""


# ------------------------------------------------------- the rounds of the document --


@dataclass(frozen=True)
class Row:
    """One body row of a review table: its line and its three cells."""

    line: int
    change: str
    thoughts: str
    modifications: str

    @property
    def path(self) -> str | None:
        """The backticked path of the *Change* cell, or ``None`` if it is malformed."""
        match = CHANGE_CELL.match(self.change)
        return match.group("path") if match else None

    @property
    def cells(self) -> tuple[str, str, str]:
        """The three cells, which is what "the row is unchanged" compares."""
        return (self.change, self.thoughts, self.modifications)

    def covers(self, changed: str) -> bool:
        """Whether the row's path is `changed`, or a directory path above it."""
        if self.path is None:
            return False
        if self.path.endswith("/"):
            return changed.startswith(self.path)
        return changed == self.path


@dataclass(frozen=True)
class Round:
    """One review round of a document: its subheading and its rows."""

    line: int
    heading: str
    rows: tuple[Row, ...]

    @property
    def reviewed(self) -> bool:
        """Whether a *Thoughts* cell of the round is filled; *Modifications* does not
        count, since Claude fills that column when asked."""
        return any(row.thoughts for row in self.rows)


def frontmatter(text: str) -> dict[str, object]:
    """The frontmatter fields of a document, or ``{}`` when it has none."""
    match = FRONTMATTER.match(text)
    meta = yaml.safe_load(match.group(1)) if match else None
    return meta if isinstance(meta, dict) else {}


def status(text: str) -> str:
    """The frontmatter ``status`` of a document, or ``""``."""
    return str(frontmatter(text).get("status", ""))


def base_ref(text: str) -> str:
    """The ref that round 1 of a document starts from the merge base of:
    ``origin/<base>`` when the frontmatter names a ``base`` branch, else `BASE_REF`."""
    branch = frontmatter(text).get("base")
    return BASE_REF if branch is None else f"origin/{branch}"


def review_rows(tables: Iterable[markdown.Table]) -> tuple[Row, ...]:
    """The body rows of `tables` as review rows; a missing cell reads as empty."""
    rows = []
    for table in tables:
        for row in table.rows:
            change, thoughts, modifications = (*row.cells, "", "", "")[:3]
            rows.append(Row(row.line, change, thoughts, modifications))
    return tuple(rows)


def review_section(text: str) -> markdown.Section | None:
    """The section *Review notes* of the document `text`, or ``None``."""
    return markdown.parse(text).find(REVIEW_HEADING)


def rounds(text: str) -> list[Round]:
    """The rounds under the *Review notes* heading, in document order: one per
    subsection, with the rows of the tables in it."""
    review = review_section(text)
    if review is None:
        return []
    return [
        Round(
            subsection.line,
            subsection.title,
            review_rows(table for part in subsection.walk() for table in part.tables),
        )
        for subsection in review.subsections
    ]


def outside_rows(text: str) -> tuple[Row, ...]:
    """The rows under the *Review notes* heading before its first subsection."""
    review = review_section(text)
    return review_rows(review.tables) if review is not None else ()


# -------------------------------------------------------------- the review commits --


def git(*arguments: str) -> str:
    """The standard output of one git command, which must succeed."""
    return subprocess.run(
        ["git", *arguments], capture_output=True, text=True, check=True
    ).stdout


def has_ref(ref: str) -> bool:
    """Whether `ref` names a commit in this clone."""
    command = ["git", "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"]
    return subprocess.run(command, capture_output=True).returncode == 0


def text_at(commit: str, path: Path) -> str:
    """The text of `path` at `commit`, or in the index when `commit` is ``""``; the
    text is ``""`` when the commit, or the index, does not hold the file."""
    shown = subprocess.run(
        ["git", "show", f"{commit}:{path.as_posix()}"], capture_output=True, text=True
    )
    return shown.stdout if shown.returncode == 0 else ""


def reviewer_cells(text: str) -> list[tuple[str, str]]:
    """Every pair of *Thoughts* and *Modifications* cells under the *Review notes*
    heading, sorted; rows outside a round included, since a document from before the
    rounds existed has only those."""
    review = review_section(text)
    if review is None:
        return []
    tables = (table for part in review.walk() for table in part.tables)
    return sorted((row.thoughts, row.modifications) for row in review_rows(tables))


def review_commits(base: str, path: Path) -> list[str]:
    """The commits since `base` that review a round of `path`, oldest first.

    A commit reviews a round when Claude is not its author and it changes a *Thoughts*
    or *Modifications* cell. An edit by Joon elsewhere in the document reviews nothing.
    """
    log = git(
        "log", "--reverse", "--format=%H %ae", f"{base}..HEAD", "--", path.as_posix()
    )
    closing: list[str] = []
    for line in log.splitlines():
        commit, email = line.split(" ", 1)
        if email == CLAUDE_EMAIL:
            continue
        before = reviewer_cells(text_at(f"{commit}^", path))
        if reviewer_cells(text_at(commit, path)) != before:
            closing.append(commit)
    return closing


# ---------------------------------------------------------- the range of each round --


def changed_files(start: str, end: str | None) -> list[str]:
    """The files changed from `start` to `end`, or to the index when `end` is ``None``.

    Rename detection is requested explicitly, so that a moved file appears once, under
    its new path, whatever ``diff.renames`` is set to.
    """
    target = ["--cached", start] if end is None else [start, end]
    names = git("diff", "--name-only", "--find-renames", "-z", *target)
    return [name for name in names.split("\0") if name]


def staged_review(path: Path) -> bool:
    """Whether the commit being made reviews a round of `path`: Claude is not its
    author, and the index changes a *Thoughts* or *Modifications* cell of ``HEAD``."""
    if claude_is_committing():
        return False
    return reviewer_cells(text_at("", path)) != reviewer_cells(text_at("HEAD", path))


def claude_is_committing() -> bool:
    """Whether the commit being made, or checked, would carry Claude as its author.

    A clone with no author identity, such as a CI checkout, makes no commit, so the
    answer there is no.
    """
    identity = subprocess.run(
        ["git", "var", "GIT_AUTHOR_IDENT"], capture_output=True, text=True, check=False
    )
    return identity.returncode == 0 and CLAUDE_EMAIL in identity.stdout


# ------------------------------------------------------------------- the problems --


def reviewer_files(path: Path, commit: str) -> list[str]:
    """The files other than the change document `path` that the review commit
    `commit` changes."""
    return [
        name for name in changed_files(f"{commit}^", commit) if name != path.as_posix()
    ]


def unopened_files(path: Path, last: str | None, reviewing: bool) -> set[str]:
    """The files whose row may come in a round that is not written yet: the files
    other than the change document `path` that the review of the last round changes.
    That review is the index when `reviewing`, else the commit `last`, and there is
    none when the last round is open."""
    if reviewing:
        return {name for name in changed_files("HEAD", None) if name != path.as_posix()}
    return set(reviewer_files(path, last)) if last is not None else set()


def covered(round_: Round, names: Iterable[str]) -> set[str]:
    """The files of `names` that a row of `round_` covers."""
    return {name for name in names if any(row.covers(name) for row in round_.rows)}


def coverage_problems(
    path: Path, round_: Round, required: Iterable[str], allowed: set[str], number: int
) -> Iterator[str]:
    """The rows of one round that are malformed or cover no file of `allowed`, and the
    files of `required` that no row of the round covers."""
    for row in round_.rows:
        if row.path is None:
            yield (
                f"{path}:{row.line}: the Change cell is not a backticked path, "
                "a dash, and what changed"
            )
        elif not any(row.covers(name) for name in allowed):
            yield (
                f"{path}:{row.line}: `{row.path}` covers no file changed in "
                f"round {number}"
            )
    for name in required:
        if not any(row.covers(name) for row in round_.rows):
            yield f"{path}: `{name}` is changed in round {number} but has no row there"


def frozen_problems(path: Path, written: list[Round]) -> Iterator[str]:
    """The rows of the rounds reviewed in ``HEAD`` that differ from ``HEAD``'s.

    A round that ``HEAD``'s document does not have yet is not compared, which is the
    case of the commit that first writes it.
    """
    committed = rounds(text_at("HEAD", path))
    for number, (before, after) in enumerate(zip(committed, written), 1):
        if not before.reviewed or before.heading != after.heading:
            continue
        if [row.cells for row in before.rows] != [row.cells for row in after.rows]:
            yield (
                f"{path}:{after.line}: a row of closed round {number} is "
                "edited; a later change gets a row in the open round"
            )


def heading_problems(
    path: Path, written: list[Round], base: str, reviews: list[str]
) -> Iterator[str]:
    """The subheadings that are not ``### Round <n>: from `<commit>` `` with the
    round's number, or whose commit is not the merge base for round 1; for a later
    round, not a review commit after the one of the round before it; and for an open
    last round, not the latest review commit."""
    # The position in `reviews` of the previous round's commit; -1 for the base.
    previous = -1
    for number, round_ in enumerate(written, 1):
        heading = ROUND_HEADING.fullmatch(round_.heading)
        if heading is None or int(heading.group("number")) != number:
            yield (
                f"{path}:{round_.line}: the subheading is not "
                f"`### Round {number}: from `<commit>``"
            )
            continue
        commit = heading.group("commit")
        position = next(
            (i for i, review in enumerate(reviews) if review.startswith(commit)), None
        )
        if number == 1 and not base.startswith(commit):
            yield (
                f"{path}:{round_.line}: round 1 starts from {base[:7]}, "
                f"not from {commit}"
            )
        if number > 1 and position is None:
            yield (
                f"{path}:{round_.line}: round {number} starts from {commit}, "
                "which is not a review commit"
            )
        if number > 1 and position is not None and position <= previous:
            yield (
                f"{path}:{round_.line}: round {number} starts from {commit}, "
                f"which does not come after the start of round {number - 1}"
            )
        is_open = round_ is written[-1] and not round_.reviewed
        if number > 1 and is_open and position not in (None, len(reviews) - 1):
            yield (
                f"{path}:{round_.line}: the open round {number} starts from "
                f"{commit}, not from the latest review commit {reviews[-1][:7]}"
            )
        previous = position if position is not None else previous


def review_problems(
    path: Path, written: list[Round], reviews: list[str], reviewing: bool
) -> Iterator[str]:
    """A round before the last that is not reviewed, and a reviewed last round that
    no review commit ends or that the index goes beyond. `reviewing` says that the
    commit being made is the review, so the last round has neither problem."""
    for number, round_ in enumerate(written[:-1], 1):
        if not round_.reviewed:
            yield (
                f"{path}:{round_.line}: round {number} is not reviewed, "
                "but a later round follows it"
            )
    last = written[-1]
    if not last.reviewed or reviewing:
        return
    if not reviews:
        yield (
            f"{path}:{last.line}: round {len(written)} is reviewed, "
            "but no review commit ends it"
        )
    elif changed_files(reviews[-1], None):
        yield (
            f"{path}: round {len(written)} is reviewed; "
            f"open round {len(written) + 1} from `{reviews[-1][:7]}`"
        )


def absent_ref(ref: str) -> str:
    """The message for a base ref that this clone does not have."""
    branch = ref.removeprefix("origin/")
    return (
        f"`{ref}` is absent, so the merge base is unknown: "
        f"run `git fetch origin {branch}` and check again"
    )


def problems(path: Path) -> Iterator[str]:
    """Every problem of one change document, round by round."""
    text = path.read_text(encoding="utf-8")
    if status(text) == "merged":
        return
    ref = base_ref(text)
    if not has_ref(ref):
        yield f"{path}: {absent_ref(ref)}"
        return
    base = git("merge-base", ref, "HEAD").strip()
    for row in outside_rows(text):
        yield f"{path}:{row.line}: a review row outside a round"
    written = rounds(text)
    if not written:
        yield f"{path}: no round under Review notes; open round 1 from `{base[:7]}`"
        return
    reviews = review_commits(base, path)
    reviewing = staged_review(path)
    found = [
        *heading_problems(path, written, base, reviews),
        *review_problems(path, written, reviews, reviewing),
    ]
    yield from found
    if claude_is_committing():
        yield from frozen_problems(path, written)
    if found:
        return
    starts = [
        heading.group("commit")
        for round_ in written
        if (heading := ROUND_HEADING.fullmatch(round_.heading))
    ]
    last = reviews[-1] if written[-1].reviewed and not reviewing else None
    # The files that each round's own review commit changes besides the document.
    shared = [[], *(reviewer_files(path, start) for start in starts[1:]), []]
    for number, (round_, start, end) in enumerate(
        zip(written, starts, [*starts[1:], last], strict=True), 1
    ):
        changed = changed_files(start, end)
        if number < len(written):
            later = covered(written[number], shared[number])
        else:
            later = unopened_files(path, last, reviewing)
        required = [name for name in changed if name not in later]
        allowed = {*changed, *shared[number - 1]}
        yield from coverage_problems(path, round_, required, allowed, number)


def main(argv: list[str]) -> int:
    """Run the check over the given paths, or `DEFAULT_DIR`; return the exit code."""
    if not has_ref(BASE_REF):
        print(absent_ref(BASE_REF))
        return 1
    master_base = git("merge-base", BASE_REF, "HEAD").strip()
    if master_base == git("rev-parse", "HEAD").strip():
        # On `master` itself no branch is under review, so no round has a range.
        print(f"`HEAD` is on `{BASE_REF}`: no branch under review")
        return 0
    paths = [Path(a) for a in argv] or sorted(DEFAULT_DIR.glob("*.md"))
    paths = [path for path in paths if path.name not in OLD_FORMAT]
    found = [p for path in paths for p in problems(path)]
    for p in found:
        print(p)
    print(f"{len(found)} problem(s)")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
