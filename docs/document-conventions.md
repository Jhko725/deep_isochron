---
type: reference
status: current
updated: 2026-11-25
---

# Document conventions

This document says what each kind of document under `docs/` is for, the form each one
takes, and which check enforces what. The prose rules are the `clear-prose` skill's;
the conventions for code, tests and docstrings are in
[`coding-standards.md`](coding-standards.md).

## What goes where

| Place | Holds | Read it when |
|---|---|---|
| [`architecture.md`](architecture.md) | the module map and the data flow | starting any task |
| [`roadmap.md`](roadmap.md) | the single current plan, phase by phase, and the append-only Done ledger | choosing what to do next |
| `design/` | the mathematics behind the code: the survey, the derivation, the oracles and the limits, one document per method | implementing or checking a method |
| `decisions/` | numbered ADRs: why the structure is what it is, each describing the current design with its history | touching the area an ADR covers |
| `guides/` | how to do something: install AUTO, compute isochrons, run an experiment from a notebook | doing that thing |
| `changes/` | one document per branch, the reviewer's map of the change | reviewing or continuing a branch |
| `plans/` | the handoff of each implementation round, frozen when committed | reading why a branch was built the way it was |
| [`references.md`](references.md) | the one bibliography | citing |
| [`index.md`](index.md) | every document above, with its status | finding a document |

`isochron2pbvp/` at the repository's root is the coworker's project, under his
conventions: these conventions and the checks that enforce them stop at its edge, and
its `README.md` is its one document.

A guide is named for what the reader wants to do, not for the tool
(`guides/auto07p.md` installs AUTO-07p; `guides/isochrons.md` computes isochrons with
it). A design document holds a method's reasoning and a guide holds its commands; when
one document starts to carry both, it is split.

## Frontmatter and status

Every document opens with a frontmatter block: `type`, `status`, `updated` (the date of the last edit, changed in the same commit as the text), and where relevant `verified_by` (who or what has checked the claims), `sources` (what the document rests on), `base` (§ Change documents) and `id`. A reader learns a document's standing from the block, without reading the document.

`type` and `status` are closed vocabularies, and `scripts/checks/check_docs.py` fails on a value outside them. Each type allows these statuses:

| Type | Allowed status | Meaning |
|---|---|---|
| architecture, roadmap, glossary, findings, verification, reference, index | `current` | kept true at all times |
| design | `draft for review`, `tentative`, `agreed` | written and awaiting Joon; written by Claude and unchecked; settled with Joon |
| decision | `accepted`, `amended`, `superseded` | in force; in force and changed in place; replaced by a later ADR, which it names |
| change | `planned`, `in-progress`, `landed`, `merged` | written before the work; work under way; work complete on the branch; branch merged |
| guide | `tentative`, `current` | unchecked; checked against the tool |
| plan | `planned`, `executed` | committed at the start of its round; the round's documents are written |

A `merged` change document is history. Its text is not rewritten, and a correction is appended as a dated line.

The block is valid YAML, since GitHub renders it as a table only then, and `sources` is a block list with every item quoted:

```yaml
sources:
  - "ADR-0012 (analysis layout)"
  - "Wilson & Moehlis 2016"
```

A flow list `[ADR-0012 (analysis layout), Wilson & Moehlis 2016]` parses, but YAML splits it at every comma, so a parenthesized remark becomes two sources. `scripts/checks/check_md_math.py` fails on a block that does not parse, lacks `type`, `status` or `updated`, or writes `sources` as a flow list.

## Required sections

Each type has required headings, in a fixed order, so that a heading means the same thing in every document of the type. A section with nothing to say is one sentence, never left out. The required headings live in one table in `scripts/checks/check_docs.py`, which fails on a missing or reordered heading. The same check asserts that each form of that table is quoted by exactly one template of this document, so the two cannot drift.

## Change documents

Every branch has a companion document `changes/<YYYY-MM-DD>-<branch>.md`, committed with the code and updated in the same commit as any later change on the branch. It is the reviewer's map of the change, not a changelog.

```markdown
## Summary        what the branch does and why
## Design         the decisions not obvious from the code, each with the rejected alternative, and one line on whether the change can be undone
## Bugs fixed     with the symptom
## Tests          what they pin down and how to run them, with exact counts
## Open issues    with the reason each is open
## Review notes   a log of review rounds, each a table Change | Thoughts | Modifications; the only list of changes
```

- *Tests* is the baseline the next session reads. A test's outcome is stated only after that test has run on the code the document describes.
- The change document has no *Files* section. *Review notes* is the only list of changes. It is a log of review rounds, appended at the end: each round has a subheading ``### Round <n>: from `<commit>` `` and a table with the columns `Change | Thoughts | Modifications`.
- A round lists the files changed since the commit it starts from. Round 1 starts from the merge base of `HEAD` with `origin/master`, or with the branch that the frontmatter names (see the next bullet). Each later round starts from a review commit that comes after the start of the round before it, and the open round starts from the latest review commit. A file other than the change document that a review commit changes is listed in the round that the commit closes or in the round that it opens.
- A change document's frontmatter may carry `base: <branch>`. Round 1 of that document starts from the merge base of `origin/<branch>` and `HEAD`. Without the field the base is `master`. A branch cut from an unmerged branch names that branch, so that its first round lists only its own files.
- A review commit is a commit to the change document that Claude did not author and that changes a *Thoughts* or *Modifications* cell.
- A round is reviewed, and so closed, once one of its *Thoughts* cells is filled. A filled *Modifications* cell does not close a round, because Claude fills that column when asked. The open round is the last subsection of *Review notes*, and every round before it is reviewed.
- Joon reviews the open round only. The check takes his latest review commit as the start of the next round, so an edit of his to a *Thoughts* cell of an older round, made while a later round is open, would move that start; the check does not support it.
- The rows of a closed round are never edited, so a verdict always sits beside the change it was about. A later change to the same file gets a row in the open round, and a change after the last round is reviewed opens a new round.
- A *Change* cell is one backticked path, a dash, and what changed, on one line. A path that ends in `/` covers every changed file under it. A later commit of the same round that changes the file again extends the cell's text.
- Every file in `git diff --name-only` over a round's range is covered by a row of that round, and every row covers at least one such file. `scripts/checks/check_changes.py` fails on each of the following:
  - a changed file without a row, or a row that covers no changed file;
  - a row outside a round;
  - a subheading that does not name a review commit, or that names one no later than the start of the round before it;
  - an open round that does not start from the latest review commit;
  - a round before the last that is not reviewed;
  - a staged change after the last round is reviewed, which needs a new round;
  - in a commit by Claude, an edited row of a closed round.
- A review commit passes the check as it is made. The last round then runs to the staged index. A file other than the change document that the review changes needs no row yet: its row comes with the next commit, which opens the next round.
- The *Thoughts* and *Modifications* cells are Joon's, and Claude edits them only when asked.
- When a handoff document planned the branch, the frontmatter's `sources` links its committed copy under `plans/` by relative path.
- A `merged` change document keeps Joon's words verbatim, and the spelling checks skip `changes/`.
- The 24 change documents dated before 2026-11-25 are in the old format: a *Files* section and one review table with a row per *Files* bullet. They keep that form, because a reviewed table is not rewritten. `check_changes.py` lists them by name in `OLD_FORMAT` and skips them. The tuple never grows.

## Plans

`plans/<YYYY-MM-DD>-<branch>[-<suffix>].md` is the handoff of one implementation round. A handoff is committed once and then frozen. The planning session writes and revises a handoff in the Project. The implementing session commits it verbatim under `plans/` at the start of the round, and nobody edits its body afterward. Only its frontmatter `status` changes, from `planned` to `executed`, when the round's documents are written. The change document's `sources` links to it by relative path.

```markdown
## 1. Header                     goal, date, who settled the design, repository, branch and tip, earlier documents
## 2. Out of scope and assumptions   what the round leaves alone, and the writer's assumptions, each confirmed or not
## 3. What the implementing session does   a table of the units and who implements each
## 4. Before anything            access, the tip, the environment, and what to read first
## 5. Rule text                  the rules not yet in the repository, verbatim
## 6. Units                      one subsection per unit: goal, model, what it may change, items, evidence, decisions
## 7. Baseline                   the commands and their outputs before the first unit
## 8. The documents unit         the documents written last, after the code
## 9. Gates                      what is run between units, and the one condition that stops the run
## 10. Decisions and where they are recorded   each decision and the document that records it
```

- The frontmatter block stands above the verbatim body, and its `sources` names the handoff's Project path.
- `check_md_math.py` checks the frontmatter of a plan and skips its body, and `check_references.py` skips `plans/` as it skips `changes/`: a plan is a record of what was said, and its line numbers refer to the commit it names. The spelling checks read a plan like any other document.

## Decision records

An ADR describes the current design, and its history follows it
([ADR-0019](decisions/0019-adrs-describe-the-current-design.md)). An ADR earns its place
when the decision is hard to reverse, surprising without context, and came from a real
trade-off; otherwise the change document's *Design* section is enough. Its sections, in
order:

```markdown
## Context
### Motivation    one bullet per point, each with its evidence, or *[not recorded]*
### Forces        prose: what any solution had to respect
## Decision       the current design
## Consequences
## Alternatives rejected   superseded designs too, marked *Superseded* with date and reason
## History        one dated line per change, with its branch
```

A change to a decision rewrites *Decision* and *Consequences* in the same commit, and the status becomes `amended`. A change that reverses the decision is a new ADR. The first mention of an ADR, design document or code file in each section is a relative link, and `scripts/checks/check_references.py` fails on a link whose target does not exist.

Every ADR keeps one path for its whole life, `decisions/<NNNN>-<slug>.md`, and a number is never reused. A superseded ADR is not moved, because other ADRs, docstrings and merged change documents link to it, and a merged change document is not rewritten. Its standing is shown in three places:

- **The index groups decisions.** `index.md` lists them in two tables, *In force* and *Superseded*.
- **The frontmatter links both ways.** The superseded ADR carries `superseded_by: ADR-<NNNN>`, and its successor carries `supersedes: ADR-<NNNN>`.
- **The first line redirects.** A superseded ADR opens, above *Context*, with one sentence naming its successor.

`scripts/checks/check_docs.py` fails when the status, the two frontmatter fields, the opening line and the index table disagree.

## Design documents

A design document under `design/` is one of three kinds, named in its frontmatter as `kind`. A **method** document takes a numerical method from
the literature to tested code and follows the `numerical-method` skill; a **derivation**
(`normal-forms.md`, `cubic-bspline.md`) fixes conventions, derives the quantities in one
convention stated once, and ends with the API and the tests it implies; a **note**
(`validation-split.md`, `training-step-performance.md`) states a problem, what was
measured, and the rule adopted. One document covers one method, one roadmap item.

A method document has six sections, in this order and with these numbers, so that a
section number means the same thing in every one; a section with nothing to say is one
sentence, never left out:

```markdown
## 1. The problem        what is computed, from what, to what accuracy; the conventions fixed up front
## 2. What exists        the survey: the routes, what each costs and assumes, where each fails
## 3. The method         the derivation in one convention; one subheading per stage, in the order the code runs them
## 4. Exact and approximated   what is exact, what is approximated, and the bound on each approximation
## 5. Oracles            the independent routes to the same number, each with the tolerance it justifies, its entry in `tests/helpers.py` and the test that uses it
## 6. Limits             what the method does not do, and which later item does
```

The subsections of §2 and §3 follow the method; a tutorial the reader needs (AUTO from
the ground up) or an algorithm the method was built on (the coworker's run sequence) is a
subsection of §2. The module docstring of the implementation names the stages of §3 in
the same order, so the two can be read side by side. A **Status:** line opens a section
only while that section's state differs from the document's `status` (the survey
delivered but the method not yet implemented, say); it is removed when they agree.

A *Decisions* section exists while the document is a draft for Joon, each decision
phrased so that "yes" accepts the recommendation. Once taken, a decision lives in the
ADR or the change document, and the section is deleted: the ADR's History is the
record, and a reader of the current design does not meet questions that are closed.

Every claim carries a provenance label, in square brackets: **[cited]** from a named
source, **[cited via X]** read in X rather than in the source it credits, **[code]**
verified in the tree at the moment of writing, **[deduced]** Claude's inference. The
labels stay when the text is edited, and a `[code]` claim is re-read rather than
recalled: ADR-0004's *Consequences* once described a `# ty: ignore` three branches
after it was removed. The italic forms `*cited*`, `*deduced*` and `*secondary*` were
the older spelling; `scripts/checks/check_references.py` refuses them under `design/`.

## Unconfirmed decisions

A decision that Claude took and Joon has not confirmed is marked **[unconfirmed]** where it is stated. `just pending` lists every marker with its file and line. `check_docs.py` fails on a marker inside a document whose status is `agreed`, `accepted`, `amended` or `merged`, so that nothing unconfirmed is recorded as settled. The marker written in code font is a mention of it and does not count.

## Paragraphs

A paragraph carries one theme, and a short paragraph is fine. Review of ADR-0017 found
"unrelated ideas grouped into a conjoined paragraph", and the fix was line breaks, not
more words. The `clear-prose` skill has the rules; the skim test (read only the first and
last sentence of every paragraph) is the check.

## References

The repository has one bibliography, `references.md`, and one in-text form. The key is
`Author Year` for one author (`Lust 2001`), `Author & Author Year` for two
(`Wilson & Moehlis 2016`), `Author et al. Year` for three or more (`Dieci et al. 1997`),
with a letter suffix on a collision (`Langfield et al. 2025a`). <!-- references:ignore: the example --> A locator follows the key after a comma, in one of
the forms `Eq. 7`, `Eqs. 9–14`, `§3`, `Prop. 5`, `Theorem 2`, `Table 5`, `Fig. 9`,
`pp. 210–214`, `Ch. 8`, `App. D`, `Remark 4`, and several citations are separated by
semicolons inside one pair of parentheses. Every key resolves to an entry in
`references.md`, which `scripts/checks/check_references.py` enforces, and the entry
carries the full reference, a DOI or arXiv identifier, and the locators the repository
uses. Design documents and ADRs carry no reference list of their own; their frontmatter
`sources:` lists keys. `README.md` is in the check's scope too: its acknowledgment once
named a paper without a citation.

## Markdown that renders in both previews

GitHub runs Markdown before MathJax, and VS Code's preview parses `$…$` first. These are
the rules `scripts/checks/check_md_math.py` enforces, with the reason each exists:

- There is no line-length limit, since ruff formats Python only, but an inline formula
  `$…$` never crosses a line.
- Display math only as a standalone block: `$$` alone on a line, the formula, `$$` alone
  on a line. Never `$$…$$` inside a paragraph.
- Inside inline math, no backslash followed by ASCII punctuation: `\,` `\;` `\{` `\|`
  `\\` become `\thinspace` `\medspace` `\lbrace` `\Vert`, and matrices move into blocks.
- No `*` inside inline math: use `\ast`, since `x^*` loses its star and italicizes the
  rest of the paragraph.
- Spaces around `<` and `>`.
- No `_` immediately after `}` `)` `|`: write `\mathbf e_r`, `C_{\rm loc}^{k}`,
  `\vert_{E}`, because such an `_` opens Markdown emphasis.

Every span has also been checked against KaTeX as VS Code configures it, so the two
previews agree.

## Checks

| Rule | Check | Runs in |
|---|---|---|
| both Markdown previews; the frontmatter parses, carries `type`, `status`, `updated`, and lists `sources` in block form | `check_md_math.py` | `prek`, CI |
| `type` and `status` are in the closed vocabularies | `check_docs.py` | `prek`, CI |
| required headings present and in order | `check_docs.py` | `prek`, CI |
| `index.md` links every document | `check_docs.py` | `prek`, CI |
| the Done ledger is chronological | `check_docs.py` | `prek`, CI |
| every package directory appears in `architecture.md` | `check_docs.py` | `prek`, CI |
| a superseded ADR's status, frontmatter, opening line and index table agree | `check_docs.py` | `prek`, CI |
| no `[unconfirmed]` in a settled document | `check_docs.py`; `just pending` lists all | `prek`, CI |
| the templates in this document match the check's table | `check_docs.py` | `prek`, CI |
| every citation resolves to `references.md`; every relative Markdown link resolves; no italic provenance label under `design/` | `check_references.py` | `prek`, CI |
| each round of *Review notes* covers exactly the files changed in its range; closed rounds are not edited | `check_changes.py` | `prek`, CI |
| American English | `codespell` and `check_spelling.py`, described in [`coding-standards.md`](coding-standards.md) § Deterministic checks | `prek`, CI |

The check scripts are in `scripts/checks/`.
