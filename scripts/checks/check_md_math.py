r"""Check that the math in Markdown files survives GitHub's rendering pipeline.

GitHub runs Markdown *before* MathJax, so inside inline ``$...$`` any backslash
followed by ASCII punctuation is a Markdown escape (``\\,`` -> ``,``, ``\\{`` ->
``{``, ``\\\\`` -> ``\\``), ``*`` and pairable ``_`` open emphasis, and ``<`` can
start an HTML tag; display math written inline (``$$...$$`` inside a paragraph)
suffers the same, and a continuation line starting with ``- `` becomes a list item.
Standalone ``$$`` blocks (a line that is exactly ``$$``, content, a line that is
exactly ``$$``) are treated as raw math and are safe. VS Code's KaTeX preview is
tolerant of all of this, so a document can look right in VS Code and break on GitHub.

This script emulates GitHub: it renders each paragraph with markdown-it (CommonMark,
HTML on) and reports every inline span whose text did not come through verbatim, and
every display formula not in a standalone block. Fixes that render in both: ``\\,`` ->
``\\thinspace``, ``\;`` -> ``\\medspace``, ``\\{`` ``\\}`` -> ``\\lbrace``
``\\rbrace``, ``\\|`` -> ``\\Vert``, ``*`` -> ``\\ast``, matrices (``\\\\``) in ``$$``
blocks, ``|_{..}`` -> ``\\vert_{..}``, ``\\mathbf{e}_r`` -> ``\\mathbf e_r``,
``C^{k}_{\\rm loc}`` -> ``C_{\\rm loc}^{k}`` (an ``_`` preceded by a letter cannot
open emphasis), spaces around ``<`` and ``>``.

The frontmatter block is checked too, since GitHub renders it as a table only when it
is valid YAML: it has to parse, carry ``type``, ``status`` and ``updated``, and write
``sources`` as a block list of quoted items. A flow list ``[a (b, c), d]`` parses, but
splits at every comma, so ``a (b`` and ``c)`` become two sources
(``docs/document-conventions.md`` § Frontmatter). A file with no block is not checked
for one: the change documents that predate the block are history.

Usage: ``uv run python scripts/checks/check_md_math.py docs/**/*.md`` (exit code 1 on
problems). Requires ``markdown-it-py`` and ``PyYAML`` (dev dependencies).
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path

import yaml
from markdown_it import MarkdownIt


CODE = re.compile(r"`[^`\n]+`")
INLINE = re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)", flags=re.S)
DISPLAY_INLINE = re.compile(r"\$\$(.+?)\$\$", flags=re.S)
TAG = "␟"  # marks where a tag stood in the rendered output


def _split_blocks(text: str) -> list[tuple[str, str]]:
    """``(kind, chunk)`` with standalone ``$$`` blocks and fenced code separated out."""
    out: list[tuple[str, str]] = []
    lines = text.split("\n")
    buf: list[str] = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped == "$$" or stripped.startswith("```"):
            closer = "$$" if stripped == "$$" else "```"
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith(closer):
                j += 1
            out.append(("text", "\n".join(buf)))
            buf = []
            out.append(("raw", "\n".join(lines[i : j + 1])))
            i = j + 1
        else:
            buf.append(lines[i])
            i += 1
    out.append(("text", "\n".join(buf)))
    return out


REQUIRED_KEYS = ("type", "status", "updated")


def frontmatter_problems(block: str) -> list[str]:
    """The problems of a frontmatter `block` (the text between the ``---`` lines)."""
    try:
        data = yaml.safe_load(block)
    except yaml.YAMLError as exc:
        return [f"frontmatter is not valid YAML: {str(exc).splitlines()[0]}"]
    if not isinstance(data, dict):
        return ["frontmatter is not a mapping"]
    problems = [f"frontmatter lacks `{k}`" for k in REQUIRED_KEYS if k not in data]
    if "sources" in data:
        if not isinstance(data["sources"], list):
            problems.append("frontmatter `sources` is not a list")
        elif re.search(r"^sources: *\[", block, flags=re.M):
            problems.append(
                "frontmatter `sources` is a flow list; write a block list of quoted"
                " items"
            )
    return problems


def check(path: Path) -> list[str]:
    """The math problems of the Markdown file at `path`, and its frontmatter's."""
    text = path.read_text()
    problems: list[str] = []
    if text.startswith("---"):
        end = text.index("\n---", 3)
        problems += frontmatter_problems(text[4:end])
        text = text[end + 4 :]
    md = MarkdownIt("commonmark", {"html": True})
    for kind, chunk in _split_blocks(text):
        if kind != "text":
            continue
        for para in re.split(r"\n\s*\n", chunk):
            para = CODE.sub("`code`", para)  # inline code is not Markdown-processed
            spans = INLINE.findall(DISPLAY_INLINE.sub("", para))
            if DISPLAY_INLINE.search(para):
                head = para.strip().replace("\n", " ")[:80]
                problems.append(f"display math inside a paragraph: {head}…")
            if not spans:
                continue
            rendered = html.unescape(re.sub(r"<[^>]+>", TAG, md.render(para)))
            got = INLINE.findall(rendered)
            if len(got) != len(spans):
                head = para.strip().replace("\n", " ")[:80]
                problems.append(
                    f"{len(spans)} inline spans became {len(got)} after Markdown: "
                    f"{head}…"
                )
                continue
            for src, out in zip(spans, got):
                if src != out:
                    problems.append(
                        f"inline span changed by Markdown: ${src}$ -> ${out}$"
                    )
    return problems


def main(argv: list[str]) -> int:
    """Check `argv`, or every Markdown file under ``docs``; return the exit code."""
    paths = [Path(a) for a in argv] or sorted(Path("docs").rglob("*.md"))
    total = 0
    for path in paths:
        problems = check(path)
        total += len(problems)
        if problems:
            print(f"{path}: {len(problems)} problem(s)")
            for p in problems:
                print(f"  {p[:200]}")
    print(f"{len(paths)} file(s) checked, {total} problem(s)")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
