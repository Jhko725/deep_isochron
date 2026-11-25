# From Jhko725/UniversalForceReconstruction .claude/hooks/block_push.py
# at b9e4cc3.
"""Block every ``git push`` made by the `implementer` subagent.

A ``PreToolUse`` hook for the Bash tool, registered in the frontmatter of
``.claude/agents/implementer.md``, so it runs only while that subagent is active. It
reads the hook's JSON on stdin and exits with code 2, which blocks the call and shows
the message on stderr to the subagent, when the command contains a ``git push``. Any
other command exits with code 0.

An implementer commits and reports; the orchestrator pushes after it has reviewed the
diff. The command is split with the parser of ``block_history_rewrite.py``, so that a
push after ``&&``, inside ``bash -c`` or behind git's global options is found too.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent))

from block_history_rewrite import (  # noqa: E402
    FALLBACK,
    git_arguments,
    segments,
    SHELLS,
)


def pushes(command: str) -> bool:
    """Whether `command` runs a ``git push`` in any of its simple commands."""
    try:
        parts = segments(command)
    except ValueError:
        return bool(FALLBACK.search(command)) or "push" in command
    for words in parts:
        if len(words) >= 3 and words[0] in SHELLS and words[1] == "-c":
            if pushes(words[2]):
                return True
        parsed = git_arguments(words)
        if parsed is not None and parsed[0] == "push":
            return True
    return False


def main() -> int:
    """Read the hook's JSON on stdin; return 2 to block the command, 0 to allow it."""
    payload = json.load(sys.stdin)
    if not pushes(payload.get("tool_input", {}).get("command", "")):
        return 0
    print(
        "Blocked by .claude/hooks/block_push.py: an implementer does not push. Commit,"
        " report, and the orchestrator pushes after its review"
        " (.claude/agents/implementer.md).",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
