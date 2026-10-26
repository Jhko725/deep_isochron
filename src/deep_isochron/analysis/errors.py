"""Exceptions shared by both halves of ``analysis``."""

from __future__ import annotations

from typing import Any


class NotConvergedError(RuntimeError):
    """A numerical procedure did not meet its convergence criterion.

    **Arguments:**

    - `stage`: The step that failed: ``"settle"``, ``"return"``, ``"shooting"``.
    - `message`: What went wrong; the error reads ``"<stage>: <message>"``.
    - `last`: Whatever the step had when it gave up, such as the last iterate or the
        solver's result, so that the caller can inspect it or restart from it.

    **Attributes:**

    - `stage`, `last`: As above.
    """

    def __init__(self, stage: str, message: str, last: Any = None):
        super().__init__(f"{stage}: {message}")
        self.stage = stage
        self.last = last
