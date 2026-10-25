"""Flag British spellings that a word list cannot cover.

`codespell` carries a 535-entry British-to-American list. It catches the irregulars and
the typos, and misses every productive form: this project's vocabulary includes the
British spellings of `discretize`, `parametrize`, `orthogonalize`, `diagonalize`,
`linearize` and `vectorize`, none of which any list ships. The `-ise` family is regular
morphology, so a rule matches it; what is finite and enumerable is the set of words where
`-ise` is also correct in American English, and that is the exception list below.

Run both checks: `codespell` for the irregulars and for typos, this script for the
families.

Usage: `uv run python scripts/check_spelling.py [path ...]` (default: the repository
root). Exits 1 when anything is flagged.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

# Words whose `-ise` ending is correct in American English. Verbs of Latin (not Greek)
# origin, plus every noun and adjective the pattern would otherwise catch. Closed: the
# language does not coin new members.
ISE_IS_CORRECT: frozenset[str] = frozenset(
    """
    advertise advise anise apprise appraise arise braise bruise chaise chastise
    circumcise comprise compromise concise counterpoise cruise demise denoise despise
    devise disenfranchise disguise enfranchise enterprise excise exercise expertise
    franchise guise improvise incise likewise malaise mayonnaise merchandise mise
    moonrise mortise noise otherwise paradise poise porpoise praise precise premise
    prise promise raise reprise revise rise sunrise supervise surmise surprise
    televise tortoise treatise turquoise upraise valise wise
    """.split()
)

# Suffixes with no exceptions at all: no English word ends in -isation or -isable and
# keeps the s in American English.
EXCEPTION_FREE = re.compile(
    r"\b([A-Za-z]{3,}?)(isation|isations|isable|isability|isational)\b"
)
# The verb family, where the exception list applies.
ISE_FAMILY = re.compile(r"\b([A-Za-z]{2,}?)(ise|ises|ised|ising)\b")
# The Greek -lyse verbs (analyze, catalyze, dialyze, electrolyze, hydrolyze, paralyze)
# all take -yze in American English, with no exceptions. The plural suffix is left out of
# the rule on purpose: it is also the plural of -ysis, which is correct, and that plural
# is far more common in prose than the British third-person verb form.
YSE_FAMILY = re.compile(r"\b([A-Za-z]{3,}?)(yse|ysed|ysing)\b")  # codespell:ignore

# A line carrying either marker is skipped, so that a document quoting the spellings it
# describes does not trip the check. `codespell:ignore` is codespell's own directive
# (verified on 2.4.3), so one comment serves both checks.
IGNORE_MARKERS = ("spelling:ignore", "codespell:ignore")

TEXT_SUFFIXES = frozenset({".py", ".md", ".toml", ".yaml", ".yml", ".txt", ".cfg"})
SKIP_PARTS = frozenset(
    {".git", ".venv", "node_modules", "__pycache__", "third_party", "icon", "notebooks"}
)


def _americanize(stem: str, suffix: str) -> str:
    """The American spelling of `stem + suffix`, with `s` replaced by `z`."""
    return stem + suffix.replace("is", "iz", 1).replace("ys", "yz", 1)


def check_text(text: str) -> Iterator[tuple[int, str, str]]:
    """Yield `(line number, found, suggested)` for every flagged word in `text`."""
    for number, line in enumerate(text.splitlines(), start=1):
        if any(marker in line for marker in IGNORE_MARKERS):
            continue
        for pattern, honor_exceptions in (
            (EXCEPTION_FREE, False),
            (ISE_FAMILY, True),
            (YSE_FAMILY, False),
        ):
            for match in pattern.finditer(line):
                stem, suffix = match.group(1), match.group(2)
                word = match.group(0)
                if honor_exceptions:
                    base = stem.lower() + "ise"
                    if base in ISE_IS_CORRECT or stem.lower().endswith("w"):
                        continue
                yield number, word, _americanize(stem, suffix)


def walk(paths: Iterable[Path]) -> Iterator[Path]:
    """Yield every text file under `paths`, skipping vendored and generated trees."""
    for path in paths:
        if path.is_file():
            yield path
            continue
        for child in sorted(path.rglob("*")):
            if child.is_file() and child.suffix in TEXT_SUFFIXES:
                if not SKIP_PARTS.intersection(child.parts):
                    yield child


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, default=[Path(".")])
    arguments = parser.parse_args(argv)

    root = Path.cwd()
    problems = 0
    for file in walk(arguments.paths or [Path(".")]):
        try:
            text = file.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for number, found, suggested in check_text(text):
            try:
                shown = file.relative_to(root)
            except ValueError:
                shown = file
            print(f"{shown}:{number}: {found} ==> {suggested}")
            problems += 1

    print(f"{problems} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
