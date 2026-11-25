# From Jhko725/UniversalForceReconstruction .claude/hooks/block_history_rewrite.py
# at b9e4cc3.
"""Block a shell command that rewrites a pushed git commit.

A ``PreToolUse`` hook for the Bash tool, registered in ``.claude/settings.json``. It
reads the hook's JSON on stdin and exits with code 2, which blocks the call and shows
the message on stderr to Claude, when the command holds a ``git rebase``, a ``git push``
with ``--force``, ``--force-with-lease``, ``-f`` or a ``+`` refspec, or a ``git commit
--amend`` of a pushed HEAD. Any other command exits with code 0 and takes the normal
permission flow.

A pushed commit is never amended or rebased, because Joon may already have merged it
(``CLAUDE.md``). An amend is allowed when ``git branch -r --contains HEAD`` prints
nothing and no branch of any remote, as ``git ls-remote --heads`` reports it, is HEAD
or has HEAD as an ancestor. The second query asks the remote itself, because a shallow
single-branch clone does not update its remote-tracking refs on push
(``docs/findings.md``, finding 10). A repository with no remote has only the first
query. The queries run in the directory the amend would run in: the payload's
``cwd``, joined with each ``git -C <dir>`` of the command in turn. The hook blocks the
amend when it cannot tell: a query fails, a remote cannot be reached, a remote branch
points at a commit this clone does not have, the directory holds a shell variable, the
command changes directory with ``cd``, ``pushd`` or ``popd`` before the amend, or it
names the repository through ``--git-dir``, ``--work-tree``, ``GIT_DIR`` or
``GIT_WORK_TREE``. Every rebase and force-push is blocked, pushed or not.

The command is split into words with `shlex`, so a quoted commit message that mentions
``git rebase`` is one word and blocks nothing. A heredoc body is not read. A command
that `shlex` cannot split falls back to a regular expression over the raw text, which
errs toward blocking.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys


OPERATORS = frozenset({";", "&&", "||", "|", "&", "(", ")", "\n"})
SHELLS = frozenset({"bash", "sh", "zsh"})
GIT_OPTIONS_WITH_VALUE = frozenset({"-C", "-c", "--git-dir", "--work-tree"})
# Words that leave the directory or the repository of a later git command unknown.
DIRECTORY_CHANGES = frozenset({"cd", "pushd", "popd"})
REPOSITORY_OPTIONS = ("--git-dir", "--work-tree")
REPOSITORY_VARIABLES = ("GIT_DIR=", "GIT_WORK_TREE=")
QUERY_TIMEOUT_SECONDS = 5
HEREDOC = re.compile(r"<<-?\s*(['\"]?)(\w+)\1[^\n]*\n.*?^\s*\2\s*$", re.S | re.M)
FALLBACK = re.compile(
    r"\bgit\b[^;&|\n]*\b(?:commit\b[^;&|\n]*--amend|rebase\b"
    r"|push\b[^;&|\n]*(?:--force|\s-[a-zA-Z]*f|\s\+\S))"
)


def without_heredocs(command: str) -> str:
    """`command` with the body of every heredoc removed, its ``<<`` line kept."""
    return HEREDOC.sub(lambda match: match.group(0).split("\n", 1)[0], command)


def segments(command: str) -> list[list[str]]:
    """The words of `command`, one list per simple command between shell operators
    and line breaks."""
    lexer = shlex.shlex(
        without_heredocs(command), posix=True, punctuation_chars=";&|()\n"
    )
    lexer.whitespace = " \t\r"
    lexer.whitespace_split = True
    found: list[list[str]] = [[]]
    for word in lexer:
        if word in OPERATORS or set(word) <= set(";&|()\n"):
            found.append([])
        else:
            found[-1].append(word)
    return [segment for segment in found if segment]


def git_arguments(words: list[str]) -> tuple[str, list[str], list[str]] | None:
    """The git subcommand of `words`, the words after it, and the words before it, or
    `None` when `words` is not a git command. Leading ``NAME=value`` assignments and
    git's global options are skipped to find the subcommand, and kept in the third
    list."""
    position = 0
    while position < len(words) and re.match(r"^[A-Za-z_]\w*=", words[position]):
        position += 1
    if position >= len(words) or words[position].rsplit("/", 1)[-1] != "git":
        return None
    position += 1
    while position < len(words) and words[position].startswith("-"):
        position += 2 if words[position] in GIT_OPTIONS_WITH_VALUE else 1
    if position >= len(words):
        return None
    return words[position], words[position + 1 :], words[:position]


def is_force_push(arguments: list[str]) -> bool:
    """Whether the arguments of a ``git push`` force the update."""
    for word in arguments:
        if word.startswith("--force"):
            return True
        if re.fullmatch(r"-[a-zA-Z]*f[a-zA-Z]*", word):
            return True
        if word.startswith("+") and len(word) > 1:
            return True
    return False


def amend_directory(prefix: list[str], cwd: str) -> str | None:
    """The directory in which the git command whose words before its subcommand are
    `prefix` runs, starting from `cwd`; `None` when `prefix` leaves it unknown."""
    directory = cwd
    for position, word in enumerate(prefix):
        if word.startswith(REPOSITORY_VARIABLES) or word.startswith(REPOSITORY_OPTIONS):
            return None
        if word == "-C" and position + 1 < len(prefix):
            target = prefix[position + 1]
            if "$" in target or "`" in target:
                return None
            directory = os.path.join(directory, os.path.expanduser(target))
    return directory


def query(directory: str, *arguments: str) -> subprocess.CompletedProcess[str] | None:
    """The finished ``git <arguments>`` run in `directory`, or `None` when git cannot
    be started or does not answer within `QUERY_TIMEOUT_SECONDS`."""
    try:
        return subprocess.run(
            ["git", "-C", directory, *arguments],
            capture_output=True,
            text=True,
            timeout=QUERY_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def answer(directory: str, *arguments: str) -> str | None:
    """The standard output of ``git <arguments>`` run in `directory`, stripped, or
    `None` when the query fails."""
    result = query(directory, *arguments)
    if result is None or result.returncode:
        return None
    return result.stdout.strip()


def pushed_branches(directory: str) -> str | None:
    """The remote-tracking branches that contain HEAD in `directory`, as ``git branch
    -r --contains HEAD`` prints them (empty when there are none), or `None` when the
    query fails."""
    return answer(directory, "branch", "-r", "--contains", "HEAD")


def contains(directory: str, commit: str, head: str) -> bool | None:
    """Whether `commit` is `head` or has it as an ancestor in `directory`, or `None`
    when the hook cannot tell, which includes a `commit` that this clone lacks."""
    ancestry = query(directory, "merge-base", "--is-ancestor", head, commit)
    # Exit code 1 is "not an ancestor"; 128 includes a commit this clone lacks.
    if ancestry is None or ancestry.returncode not in (0, 1):
        return None
    return ancestry.returncode == 0


def pushed_remote_branches(directory: str) -> str | None:
    """The branches of the remotes of `directory` that are HEAD or have HEAD as an
    ancestor, one ``<remote> <ref>`` per line (empty when there are none), or `None`
    when the hook cannot tell.

    Each remote is asked with ``git ls-remote --heads``, so the answer does not depend
    on the remote-tracking refs of this clone. The hook cannot tell when a query fails,
    which includes a remote that cannot be reached, or when a remote branch points at a
    commit this clone does not have, since its ancestry is then unknown.
    """
    remotes, head = answer(directory, "remote"), answer(directory, "rev-parse", "HEAD")
    if remotes is None or head is None:
        return None
    found = []
    for remote in remotes.split():
        listed = answer(directory, "ls-remote", "--heads", remote)
        if listed is None:
            return None
        for line in listed.splitlines():
            commit, ref = line.split("\t", 1)
            pushed = contains(directory, commit, head)
            if pushed is None:
                return None
            if pushed:
                found.append(f"{remote} {ref}")
    return "\n".join(found)


def amend_reason(prefix: list[str], cwd: str | None, moved: bool) -> str | None:
    """Why a ``git commit --amend`` with the words `prefix` before its subcommand is
    blocked, or `None` when HEAD is on no remote-tracking branch and no remote branch.
    `moved` says that an earlier command, or this one, changed the directory or named
    the repository."""
    directory = None if moved or cwd is None else amend_directory(prefix, cwd)
    if directory is None:
        return "`git commit --amend` in a directory the hook cannot determine"
    branches = pushed_branches(directory)
    if branches == "":
        branches = pushed_remote_branches(directory)
    if branches is None:
        return "`git commit --amend` where the hook cannot tell whether HEAD is pushed"
    if branches:
        shown = ", ".join(line.strip() for line in branches.splitlines())
        return f"`git commit --amend` rewrites HEAD, which is on {shown}"
    return None


def leaves_directory_unknown(words: list[str]) -> bool:
    """Whether the command `words` changes the directory, or names the repository
    through a variable, so that the directory of a later git command is unknown."""
    return words[0] in DIRECTORY_CHANGES or any(
        word.startswith(REPOSITORY_VARIABLES) for word in words
    )


def shell_reason(words: list[str], cwd: str | None) -> str | None:
    """Why the command that ``<shell> -c <command>`` in `words` would run in `cwd` is
    blocked, or `None` when it is allowed or `words` is no such call."""
    if len(words) >= 3 and words[0] in SHELLS and words[1] == "-c":
        return reason(words[2], cwd)
    return None


def git_reason(words: list[str], cwd: str | None, moved: bool) -> str | None:
    """Why the git command `words` is blocked, or `None` when it is allowed or
    `words` is not a git command. `moved` is as in `amend_reason`."""
    parsed = git_arguments(words)
    if parsed is None:
        return None
    subcommand, arguments, prefix = parsed
    if subcommand == "commit" and "--amend" in arguments:
        return amend_reason(prefix, cwd, moved)
    if subcommand == "rebase":
        return "`git rebase` rewrites commits"
    if subcommand == "push" and is_force_push(arguments):
        return "a force-push overwrites the remote branch"
    return None


def reason(command: str, cwd: str | None = None) -> str | None:
    """Why `command`, run in `cwd`, is blocked, or `None` when it is allowed."""
    try:
        parts = segments(command)
    except ValueError:
        return "it rewrites history" if FALLBACK.search(command) else None
    moved = False
    for words in parts:
        moved = moved or leaves_directory_unknown(words)
        blocked = shell_reason(words, None if moved else cwd) or git_reason(
            words, cwd, moved
        )
        if blocked:
            return blocked
    return None


def main() -> int:
    """Read the hook's JSON on stdin; return 2 to block the command, 0 to allow it."""
    payload = json.load(sys.stdin)
    command = payload.get("tool_input", {}).get("command", "")
    blocked = reason(command, payload.get("cwd"))
    if blocked is None:
        return 0
    print(
        f"Blocked by .claude/hooks/block_history_rewrite.py: {blocked}. A pushed commit"
        " is never amended, rebased or force-pushed; make the change a new commit on"
        " top (CLAUDE.md).",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
