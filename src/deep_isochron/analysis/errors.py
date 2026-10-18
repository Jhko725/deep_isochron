"""Exceptions shared by both halves of ``analysis``."""

from __future__ import annotations

from typing import Any


class NotConvergedError(RuntimeError):
    """A numerical procedure did not meet its convergence criterion.

    ``stage`` names the step that failed (``"settle"``, ``"return"``, ``"shooting"``,
    ...), ``last`` carries whatever the step had when it gave up (the last iterate, the
    solver's result) so the caller can inspect or restart from it.
    """

    def __init__(self, stage: str, message: str, last: Any = None):
        super().__init__(f"{stage}: {message}")
        self.stage = stage
        self.last = last
