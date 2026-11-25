"""Every module of the package imports."""

import importlib
from pathlib import Path

import deep_isochron as pkg


def _module_names() -> list[str]:
    """Dotted names of every .py file, including directories without __init__.py."""
    root = Path(pkg.__file__).parent
    names = []
    for path in sorted(root.rglob("*.py")):
        parts = path.relative_to(root.parent).with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        names.append(".".join(parts))
    return names


def test_import_every_module() -> None:
    """Every module of the package imports on CPU. `auto` is imported only inside
    `load_auto`, so no module needs AUTO-07p's tree to import."""
    for name in _module_names():
        importlib.import_module(name)
