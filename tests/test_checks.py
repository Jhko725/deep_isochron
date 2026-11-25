# Adapted from Jhko725/UniversalForceReconstruction tests/test_checks.py at b9e4cc3.
"""Directed tests of the pure functions of ``scripts/checks/check_handoff.py`` and
``scripts/docs/generate_references.py``.

The handoff script reads a handoff document and checks it against git, ``just`` and
pytest; these tests cover only the parts that read text, so they need no repository
state. The generator's tests render BibTeX strings written in the test.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import check_handoff
import markdown
import pytest


HANDOFF = """# Plan

Edit `a/b.py:3-5`. Add `c/d.md` (new). Then `just check`.

| File | Note |
|---|---|
| `e/f.py` (new) | `g/h.py` |

```bash
uv run just test dev tests/test_x.py::test_y -k "alpha and not beta" \\
    --maxfail=1
```
"""


def test_mentions_carry_their_lines_and_new_markers() -> None:
    """Each mention keeps its document line, and `(new)` exempts its own unit."""
    mentions = check_handoff.collect_mentions(markdown.parse(HANDOFF))
    assert [(p.line, p.path, p.numbers, p.new) for p in mentions.paths] == [
        (3, "a/b.py", (3, 5), False),
        (3, "c/d.md", (), True),
        (7, "e/f.py", (), True),
        (7, "g/h.py", (), False),
    ]
    assert [(r.line, r.name) for r in mentions.recipes] == [(3, "check"), (10, "test")]
    assert [(s.file, s.name) for s in mentions.selectors] == [
        ("tests/test_x.py", "test_y")
    ]
    assert [(e.expression, e.files) for e in mentions.expressions] == [
        ("alpha and not beta", ("tests/test_x.py",))
    ]


@pytest.mark.parametrize(
    ("expression", "node_id", "expected"),
    [
        ("alpha and not beta", "tests/t.py::test_Alpha", True),
        ("alpha and not beta", "tests/t.py::test_alpha_beta", False),
        ("(alpha or beta) and t.py", "tests/t.py::test_beta[1]", True),
    ],
)
def test_keyword_matcher_follows_pytest(
    expression: str, node_id: str, expected: bool
) -> None:
    """The matcher agrees with pytest's `-k` semantics on and, or, not and groups."""
    assert check_handoff.keyword_matcher(expression)(node_id) is expected


def test_keyword_matcher_rejects_an_unparsable_expression() -> None:
    """An expression that ends in an operator raises `ValueError`."""
    with pytest.raises(ValueError, match="does not parse"):
        check_handoff.keyword_matcher("alpha and")


def test_path_findings_flag_a_missing_file_and_a_line_past_the_end() -> None:
    """A missing file and a line past the end are each one finding."""
    ref = check_handoff.PathRef(4, "a/b.py", (2, 9), False)
    assert check_handoff.path_findings(ref, None) == [
        check_handoff.Finding(4, "`a/b.py` does not exist")
    ]
    assert len(check_handoff.path_findings(ref, 5)) == 1
    assert check_handoff.path_findings(ref, 9) == []


@pytest.fixture(scope="module")
def generator() -> ModuleType:
    """``scripts/docs/generate_references.py``, loaded from its file, since
    ``scripts/docs`` is not on the pytest path."""
    path = Path(__file__).parents[1] / "scripts" / "docs" / "generate_references.py"
    spec = importlib.util.spec_from_file_location("generate_references", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # A dataclass looks its module up in sys.modules while it is being defined.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_references_list_an_entry_with_a_key_field_under_that_key(
    generator: ModuleType,
) -> None:
    """An entry with a `key` field and no author is listed under the field's value,
    and its reference opens with the title."""
    text = """@misc{auto07p,
  key = {AUTO-07p},
  title = {AUTO-07p: continuation and bifurcation software},
  url = {https://github.com/auto-07p/auto-07p},
}
"""
    assert generator.references(text) == [
        generator.Reference(
            "AUTO-07p",
            "AUTO-07p: continuation and bifurcation software. "
            "<https://github.com/auto-07p/auto-07p>",
        )
    ]


def test_references_reject_an_entry_with_neither_key_nor_author(
    generator: ModuleType,
) -> None:
    """An entry with no `key` field and no author raises `ValueError`."""
    text = """@misc{anonymous,
  title = {A page with no author},
  year = {2020},
}
"""
    with pytest.raises(
        ValueError, match="entry .anonymous. has neither a `key` field nor"
    ):
        generator.references(text)
