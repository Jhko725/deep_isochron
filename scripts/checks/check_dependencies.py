# Adapted from Jhko725/UniversalForceReconstruction
# scripts/checks/check_dependencies.py at b9e4cc3.
"""Check two import rules that no linter carries.

1. Every top-level package imported under ``src``, ``tests`` or ``scripts`` is declared
   in ``pyproject.toml``: in ``dependencies``, an optional extra, or a dependency
   group. An import name is mapped to its distribution through the installed
   environment (``yaml`` to ``PyYAML``), so a package that arrives only through another
   one's dependencies fails here. The standard library, the package itself, and the
   modules that ``tests`` and ``scripts/checks`` import from their own directories are
   not packages to declare. `KNOWN_UNDECLARED` lists the exceptions at this commit,
   each with the unit that removes it.
2. No module under ``src`` imports from ``scripts``, ``notebooks`` or ``tests``.

Only absolute imports name a top-level package; a relative import names none.

Usage: ``uv run python scripts/checks/check_dependencies.py`` (no arguments). Exits 1
when anything is flagged.
"""

from __future__ import annotations

import ast
import re
import sys
import tomllib
from collections.abc import Iterator, Mapping
from importlib.metadata import packages_distributions
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "deep_isochron"
SCANNED = ("src", "tests", "scripts")
NOT_FROM_PACKAGE = frozenset({"scripts", "notebooks", "tests"})
KNOWN_UNDECLARED = {}
"""Undeclared imports allowed at this commit, each with the unit that removes it."""
_REQUIREMENT_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def normalize(name: str) -> str:
    """The PEP 503 form of a distribution name: lowercase, runs of ``-_.`` as ``-``."""
    return re.sub(r"[-_.]+", "-", name).lower()


def declared(pyproject: dict) -> set[str]:
    """The normalized names of every distribution `pyproject` declares."""
    project = pyproject.get("project", {})
    requirements = list(project.get("dependencies", []))
    for extra in project.get("optional-dependencies", {}).values():
        requirements += extra
    for group in pyproject.get("dependency-groups", {}).values():
        requirements += [r for r in group if isinstance(r, str)]  # not include-group
    names = set()
    for requirement in requirements:
        match = _REQUIREMENT_NAME.match(requirement)
        if match:
            names.add(normalize(match.group(1)))
    return names


def local_modules() -> set[str]:
    """The module names that ``tests`` and ``scripts/checks`` import from their own
    directories, such as ``helpers``."""
    names = set()
    for directory in (ROOT / "tests", ROOT / "scripts" / "checks"):
        if directory.exists():
            names |= {path.stem for path in directory.glob("*.py")}
    return names


def imports(path: Path) -> list[tuple[int, str, int]]:
    """``(line, dotted name, level)`` for every import of `path`; a ``from`` import
    also lists ``module.name`` for each imported name."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.append((node.lineno, alias.name, 0))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            found.append((node.lineno, module, node.level))
            for alias in node.names:
                name = f"{module}.{alias.name}".lstrip(".")
                found.append((node.lineno, name, node.level))
    return found


def python_files() -> list[Path]:
    """Every ``.py`` file under `SCANNED`."""
    return [
        path
        for directory in SCANNED
        if (ROOT / directory).exists()
        for path in sorted((ROOT / directory).rglob("*.py"))
    ]


def undeclared(
    top: str, known: set[str], distributions: Mapping[str, list[str]]
) -> str | None:
    """The distributions that provide the top-level module `top`, as text for a
    report, when none of them is in `known`; `None` when one is declared."""
    candidates = {normalize(d) for d in distributions.get(top, [])}
    if candidates & known:
        return None
    return ", ".join(sorted(candidates)) or "not installed"


def problems(pyproject: dict) -> Iterator[str]:
    """Every flagged import, as ``file:line: reason``."""
    known = declared(pyproject)
    distributions = packages_distributions()
    # The top-level modules that need no declaration.
    exempt = {
        *sys.stdlib_module_names,
        PACKAGE,
        *local_modules(),
        *NOT_FROM_PACKAGE,
        *KNOWN_UNDECLARED,
    }
    for path in python_files():
        shown = path.relative_to(ROOT).as_posix()
        for line, name, level in imports(path):
            top = name.split(".")[0]
            if level or not top:
                continue
            if shown.startswith("src/") and top in NOT_FROM_PACKAGE:
                yield f"{shown}:{line}: the package imports from `{top}`"
            found = None if top in exempt else undeclared(top, known, distributions)
            if found is not None:
                yield (
                    f"{shown}:{line}: `{top}` ({found}) is not declared in"
                    " pyproject.toml"
                )


def main() -> int:
    """Run the two rules over the repository; return the exit code."""
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    found = sorted(set(problems(pyproject)))
    for problem in found:
        print(problem)
    print(f"{len(found)} problem(s)")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
