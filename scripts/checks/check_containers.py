"""Check the container rule of `docs/coding-standards.md`: which class holds which kind
of value.

A value that a JAX transform sees is an ``eqx.Module``; a value that must stay out of
tracing is a frozen ``@dataclass`` holding NumPy or Python values; a ``NamedTuple`` is
allowed only where the tuple structure is the interface. The review of ``plotting``
found all three kinds mixed in ``analysis.ode``: ``ShootingResult``, ``ShootingArgs``
and ``PeriodicSchur`` were ``NamedTuple``s carried through jitted code, and
``PhaseAlignment``, a frozen dataclass, held a ``jax.Array``. The rule cannot be read
off a type at runtime, so the check walks the syntax tree of every module under
``src`` and flags three patterns:

1. A class decorated ``@dataclass(frozen=True)``, in any spelling, whose field
   annotation mentions ``Array`` (``Float[Array, ...]``, a bare ``Array``,
   ``jax.Array``). A mutable dataclass is a Hydra schema or a stateful service such
   as ``Evaluator``, and a service holds what it needs.
2. A class deriving from ``NamedTuple`` whose name does not end in ``Params``, the
   constrained-parameter bundles of the scalar bijections. A second legitimate use
   edits this script, with its reason in a comment.
3. A field whose default is ``eqx.field(static=True)`` and whose annotation mentions
   ``Array``: a static field decides the traced program's structure, and an array
   there is a new program per value.

In tests 1 and 3, an ``Array`` inside a ``Callable[...]`` subscript is a function's
signature, not an array, and is not flagged.

Usage: `uv run python scripts/checks/check_containers.py` (no arguments: it walks
`src`). Exits 1 when anything is flagged.
"""

from __future__ import annotations

import ast
import sys
from collections.abc import Iterator
from pathlib import Path


ROOT = Path("src")
PARAMS_SUFFIX = "Params"


def mentions_array(annotation: ast.expr) -> bool:
    """Whether `annotation` names ``Array`` as a name or an attribute, also inside a
    string annotation, outside any ``Callable[...]`` subscript."""
    if (
        isinstance(annotation, ast.Subscript)
        and _name_of(annotation.value) == "Callable"
    ):
        return False
    if isinstance(annotation, ast.Name):
        return annotation.id == "Array"
    if isinstance(annotation, ast.Attribute):
        return annotation.attr == "Array"
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        try:
            inner = ast.parse(annotation.value, mode="eval").body
        except SyntaxError:
            return False  # a shape string such as " n dim", not an annotation
        return mentions_array(inner)
    return any(mentions_array(child) for child in ast.iter_child_nodes(annotation))


def _name_of(node: ast.expr) -> str:
    """The last component of a dotted name or call: ``dataclass`` for
    ``dataclasses.dataclass(frozen=True)``, ``""`` for anything else."""
    if isinstance(node, ast.Call):
        node = node.func
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def is_frozen_dataclass(decorator: ast.expr) -> bool:
    """Whether `decorator` is ``dataclass(..., frozen=True, ...)`` in any spelling."""
    return (
        isinstance(decorator, ast.Call)
        and _name_of(decorator) == "dataclass"
        and any(
            k.arg == "frozen" and isinstance(k.value, ast.Constant) and k.value.value
            for k in decorator.keywords
        )
    )


def is_static_field(value: ast.expr | None) -> bool:
    """Whether `value` is a call ``eqx.field(..., static=True, ...)``."""
    if not isinstance(value, ast.Call) or _name_of(value.func) != "field":
        return False
    return any(
        k.arg == "static" and isinstance(k.value, ast.Constant) and k.value.value
        for k in value.keywords
    )


def fields(node: ast.ClassDef) -> Iterator[tuple[str, ast.AnnAssign]]:
    """The annotated fields of the class body `node`, each with its name."""
    for stmt in node.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            yield stmt.target.id, stmt


def class_problems(path: Path, node: ast.ClassDef) -> Iterator[str]:
    """The flagged lines of the class `node` in `path`."""
    is_frozen = any(is_frozen_dataclass(d) for d in node.decorator_list)
    is_named_tuple = any(_name_of(b) == "NamedTuple" for b in node.bases)
    if is_named_tuple and not node.name.endswith(PARAMS_SUFFIX):
        yield (
            f"{path}:{node.lineno}: {node.name}: a NamedTuple outside the *Params "
            "bundles; an eqx.Module if JAX sees it, else a frozen dataclass"
        )
    for field_name, f in fields(node):
        name = f"{node.name}.{field_name}"
        if not mentions_array(f.annotation):
            continue
        if is_frozen:
            yield (
                f"{path}:{f.lineno}: {name}: a frozen dataclass holds a jax.Array; "
                "an eqx.Module, or NumPy in the dataclass"
            )
        if is_static_field(f.value):
            yield (
                f"{path}:{f.lineno}: {name}: a static field holds a jax.Array; "
                "an array leaf instead"
            )


def problems(path: Path) -> Iterator[str]:
    """Every flagged line of `path`, as ``file:line: <class>.<field>: <rule>``."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            yield from class_problems(path, node)


def main() -> int:
    """Run the check over `ROOT`; return the exit code."""
    found = [p for f in sorted(ROOT.rglob("*.py")) for p in problems(f)]
    for p in found:
        print(p)
    print(f"{len(found)} problem(s)")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
