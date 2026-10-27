"""Finding and loading AUTO-07p, the only place in the package that imports ``auto``.

AUTO has two halves at the same commit: the Python layer, which ``uv.lock`` pins in the
dependency group ``auto``, and the compiled Fortran tree, which
``scripts/install_auto.sh`` builds. The Python layer cannot be imported until the
environment variable ``AUTO_DIR`` names the tree, so `load_auto` resolves the tree,
sets the variable and imports ``auto`` only then. ``docs/auto.md`` describes the
installation; the plan's decision 4 and ADR-0017 (pending) give the reasoning.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import warnings
from importlib import metadata
from pathlib import Path
from types import ModuleType
from typing import Any, NamedTuple


DEFAULT_DIR = Path.home() / ".local" / "opt" / "auto-07p"
"""Where ``scripts/install_auto.sh`` builds the tree unless told otherwise."""

STAMP = ".deep_isochron_commit"
"""The file in the tree where ``scripts/install_auto.sh`` records the built commit."""

ENV_VARS = ("DEEP_ISOCHRON_AUTO_DIR", "AUTO_DIR")
"""The environment variables that name the tree, in the order they are consulted."""


class AutoNotInstalled(RuntimeError):
    """No built AUTO-07p tree was found where `auto_dir` looked."""


class AutoHandle(NamedTuple):
    """What a pipeline stage needs to run AUTO.

    **Attributes:**

    - `directory`: The built tree, also exported as ``AUTO_DIR``.
    - `commands`: The module ``auto.AUTOCommands``, whose ``run`` performs one AUTO
        run.
    - `runner`: The ``auto.runAUTO.runAUTO`` instance to pass as ``runner=``.
    """

    directory: Path
    commands: ModuleType
    runner: Any


def auto_dir() -> Path:
    """The built AUTO-07p tree: the first of the variables in `ENV_VARS` that is set,
    otherwise `DEFAULT_DIR`. The tree is a property of the machine, so it is chosen by
    the environment and never by an argument (``docs/auto.md``).

    **Raises:**

    `AutoNotInstalled` when that directory has no ``cmds/cmds.make`` or no compiled
    objects in ``lib/``; the message names the install script.
    """
    path = next((os.environ[v] for v in ENV_VARS if os.environ.get(v)), DEFAULT_DIR)
    directory = Path(path).expanduser()
    built = (directory / "cmds" / "cmds.make").is_file() and any(
        (directory / "lib").glob("*.o")
    )
    if not built:
        raise AutoNotInstalled(
            f"no built AUTO-07p tree at {directory}; run scripts/install_auto.sh, or "
            f"set {' or '.join(ENV_VARS)} to an existing build (docs/auto.md)."
        )
    return directory


def pinned_commit() -> str | None:
    """The commit of the installed Python layer, from its ``direct_url.json``; `None`
    when the package is not installed from git."""
    try:
        text = metadata.distribution("auto-07p").read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        return None
    if not text:
        return None
    return json.loads(text).get("vcs_info", {}).get("commit_id")


def load_auto() -> AutoHandle:
    """Resolve the tree with `auto_dir`, export it as ``AUTO_DIR``, and import AUTO's
    Python layer.

    The import prints a banner about the plotting modules, which is suppressed. A
    warning is issued when the tree's stamp names another commit than the Python layer,
    since the two halves then disagree on the file formats between them.

    **Returns:**

    An `AutoHandle`.

    **Raises:**

    `AutoNotInstalled`, from `auto_dir`.
    """
    directory = auto_dir()
    os.environ["AUTO_DIR"] = str(directory)
    stamp = directory / STAMP
    built = stamp.read_text().strip() if stamp.is_file() else None
    pinned = pinned_commit()
    if built is not None and pinned is not None and built != pinned:
        warnings.warn(
            f"AUTO-07p tree {directory} is built at {built[:7]}, but the Python layer "
            f"is pinned at {pinned[:7]}; rerun scripts/install_auto.sh.",
            stacklevel=2,
        )
    with contextlib.redirect_stdout(io.StringIO()):
        from auto import AUTOCommands, runAUTO
    return AutoHandle(
        directory, AUTOCommands, runAUTO.runAUTO(auto_dir=str(directory), verbose="no")
    )


__all__ = [
    "DEFAULT_DIR",
    "ENV_VARS",
    "STAMP",
    "AutoHandle",
    "AutoNotInstalled",
    "auto_dir",
    "load_auto",
    "pinned_commit",
]
