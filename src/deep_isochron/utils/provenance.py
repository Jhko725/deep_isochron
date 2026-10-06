"""Where this code came from: the git state of the checkout the package is
imported from, and the installed package version. Recorded in dataset metadata
(``data.generate``) and in run metadata (``experiment.run``)."""

from __future__ import annotations

import importlib.metadata
import subprocess
from pathlib import Path


def git_state() -> tuple[str, bool]:
    """``(sha, dirty)`` of the checkout this package is imported from (not the cwd);
    ``("", False)`` outside a checkout or without git."""
    here = Path(__file__).resolve().parent
    try:
        sha = subprocess.run(
            ["git", "-C", str(here), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(here), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return sha, bool(dirty)
    except (OSError, subprocess.CalledProcessError):
        return "", False


def package_version() -> str:
    try:
        return importlib.metadata.version("deep-isochron")
    except importlib.metadata.PackageNotFoundError:
        return ""
