---
type: plan
status: planned
updated: 2026-11-25
sources:
  - "Project document claude/conventions-sync-handoff-2026-11-25.md"
---

# Handoff: `conventions-sync`

## 1. Header

**Goal.** Bring `deep_isochron` up to the conventions of `UniversalForceReconstruction` (UFR): review rounds in change documents, rule files and hooks under `.claude/`, the document checks, committed plans, the retrospective's checks, and a generated bibliography.

Implement with `orchestrate-implementation`.

- **Date:** 2026-11-25.
- **Design settled by:** Joon, in the planning chat of 2026-11-25, including the nine points A1 to A9 of § 2.
- **Repository:** `Jhko725/deep_isochron`. **Branch:** `conventions-sync` (new), cut from the tip of `inn-schematics`.
- **Tip and line numbers:** `8ef91130af138c3d2f0a8ae1d7eeb1c94830d51a`. Every `file:line` below refers to this commit. Confirm each by grep before editing.
- **Source of the ported files:** `Jhko725/UniversalForceReconstruction`, branch `measurement`, commit `b9e4cc3596c00155d00c58b552c322b7c331f8f8`. This document calls it "UFR".
- **Earlier documents:** the Project documents `claude/phase-turns-handoff-2026-10-30.md` and `claude/inn-schematics-handoff-2026-11-04.md` show the earlier handoff format.

A reason that this document does not give is a gap in this document. Stop and ask.

## 2. Out of scope and assumptions

**Out of scope.**

- The prose cleanup of documents and docstrings. It is the next branch, `prose-2`, stacked on this one, with its own handoff written after this branch is reviewed. Its settled decisions are rows 15 to 18 of § 10.
- The four UFR document types with no user here: paper records, experiment reports, dataset records and format notes.
- UFR's unit rules (R1, R9), its clean and legacy scopes, its archive check and its dissertation hook.
- `isochron2pbvp/`, the notebooks, and the bodies of the 24 existing change documents.
- The Project instructions. Joon pastes the two changes of § 8 himself.

**Assumptions.** None remain. The planning session proposed the nine points below, and Joon confirmed all of them on 2026-11-25. They are decisions, kept under their labels A1 to A9 because the units cite them.

- A1. The old status sentence of each change document is kept as the last line of the document, in the form `Status before 2026-11-25: <sentence>`.
- A2. ADR-0016, ADR-0017 and ADR-0020 carry the status `accepted (review pending)`. They become `accepted`, and the three pending reviews are listed in the change document's *Open issues*.
- A3. The design documents get these kinds: `method` for `cycle-estimation.md`, `floquet.md`, `isochrons.md`, `limit-cycle.md` and `phase-field.md`. `derivation` for `normal-forms.md` and `cubic-bspline.md`. `note` for `training-step-performance.md` and `validation-split.md`.
- A4. `docs/design/normal-forms.md` becomes `agreed`, and one line under its title names the sections that are still tentative. The status vocabulary has no value for a document that is partly agreed.
- A5. The misordered row at `docs/roadmap.md:224` (dated 2026-10-27, after two rows dated 2026-10-28) moves above them. It is the one reordering the ledger ever gets.
- A6. The architecture rule accepts a directory named inside the fenced module tree of `docs/architecture.md`, since that tree is this repository's module map. UFR's rule reads backticked paths outside fences only.
- A7. The glossary's fourth column is *Unit or range* (for example "turns, in [0, 1)"), since this repository has no named-dimension table.
- A8. Every undeclared import is declared in `pyproject.toml`, and none is removed from the code.
- A9. `verification.md` is seeded from § 5 of the five `method` design documents only. A claim outside them gets its row when its code is next touched.

## 3. What the implementing session does

| # | Unit | Who |
|---|---|---|
| O1 | Cut the branch and commit this handoff under `docs/plans/` | orchestrator |
| 1 | The Markdown reader and the rounds check | implementer, `opus` |
| O2 | The change document in the rounds format, the conventions text for change documents and plans, and the agent file's row rule | orchestrator |
| 2 | Rule files, settings and hooks | implementer, `sonnet` |
| — | **Gate 1** | Joon |
| 3 | The math and reference checks on the Markdown reader, and the plan-label rule | implementer, `opus` |
| 4 | The complexity rule at 8, and the seven functions it flags | implementer, `opus` |
| 5 | The dependency check and the import test | implementer, `sonnet` |
| — | **Gate 2** | Joon |
| 6 | The document check, not yet a hook | implementer, `opus` |
| O3 | The documents made to pass it, the check wired as a hook, and `just pending` | orchestrator |
| 7 | The handoff check and its tests | implementer, `sonnet` |
| — | **Gate 3** | Joon |
| 8 | The bibliography generator, not yet used | implementer, `opus` |
| O4 | `references.bib` from the 118 entries, verified against CrossRef, and the page regenerated | orchestrator |
| 9 | The reference check reads the `.bib` file | implementer, `sonnet` |
| — | **Gate 4** | Joon |
| O5 | The documents unit (§ 8) | orchestrator |

## 4. Before anything

- **Access.** Attach `Jhko725/deep_isochron` with push access and `Jhko725/UniversalForceReconstruction` with read access. Check out UFR at the commit of § 1 and read the ported files from that checkout.
- **Save this document.** Read the Project document `claude/conventions-sync-handoff-2026-11-25.md` and save it to disk. Unit O1 commits it.
- **Branch.** `git fetch origin master inn-schematics`, then `git switch -c conventions-sync origin/inn-schematics`. Confirm that `git rev-parse HEAD` prints the tip of § 1. If `inn-schematics` has moved, stop and report.
- **History.** The rounds check needs the merge base with `origin/inn-schematics` and with `origin/master`, so the clone is not shallow at depth 1: fetch both branches with `--depth=400` or more.
- **Environment.** `JAX_PLATFORMS=cpu uv sync`, about five minutes. The cloud session has 2 cores and 7 GB. AUTO-07p is not needed: no unit touches a test marked `auto`.
- **Network.** The shell reaches the package registries and GitHub only: `curl` to `api.crossref.org` fails with `CONNECT tunnel failed, response 403`. The web fetch tool reaches CrossRef, which unit O4 relies on.
- **Skills.** The subagents need `code-conventions` and `clear-prose`. `.claude/agents/implementer.md` exists at the tip (55 lines).
- **Read first.** `CLAUDE.md`, `docs/document-conventions.md`, `docs/coding-standards.md`, and UFR's `CLAUDE.md`, document conventions and coding standards.

## 5. Rule text

The rules below are not in either repository. Every other rule this round applies is in UFR's `docs/document-conventions.md` and `docs/coding-standards.md` at the commit of § 1, and is copied from there with `deep_isochron`'s names.

**R-a. The base of a stacked branch.** A change document's frontmatter may carry `base: <branch>`. Round 1 of that document starts from the merge base of `origin/<branch>` and `HEAD`. Without the field the base is `master`. A branch cut from an unmerged branch names that branch, so that its first round lists only its own files.

**R-b. The old format.** The 24 change documents dated before 2026-11-25 keep their *Files* section and their single review table. `check_changes.py` lists them by name in `OLD_FORMAT` and skips them. The tuple never grows.

**R-c. Status of the old change documents.** The 20 documents that `origin/master` holds become `merged`. `2026-10-30-deformation-plots.md`, `2026-11-02-ksiam-animations.md` and `2026-11-04-inn-schematics.md` become `landed`. `2026-11-04-conjugacy-schematic.md` becomes `in-progress`. The six documents with no frontmatter block get one: `type: change`, the status above, and `updated: 2026-11-25`.

**R-d. Explicit bibliography keys.** An entry of `references.bib` may carry a `key` field. The generator uses it as the in-text key and then requires neither `author` nor `year`. It is for software and documentation pages, such as `AUTO-07p` and `JAX docs: profiling`.

**R-e. The lock file.** The session that changes a dependency runs `uv lock` and commits `uv.lock` with the change. It never passes `--upgrade`.

**R-f. Semicolons, labels and history lines.** These three belong to `prose-2` and are recorded here so that they are not lost. A plan label is restated in a phrase at its first use in a document and removed from a docstring. An ADR's *History* lines stay as written and are exempt from the semicolon check. A `Raises` or argument entry that chains conditions with semicolons is split into one bullet per condition.

## 6. Units

### Unit O1. The branch and the plan

The orchestrator cuts the branch (§ 4) and commits this document, unchanged, as `docs/plans/2026-11-25-conventions-sync.md` (new), under a frontmatter block: `type: plan`, `status: planned`, `updated: 2026-11-25`, and `sources` naming the Project path. Evidence: `uv run just check` ends with `All checks passed!`.

### Unit 1. The Markdown reader and the rounds check

**Goal.** Replace the rows check with UFR's rounds check, extended with rules R-a and R-b.

**Model.** `opus`: the base rule is new design.

**May change.** `scripts/checks/markdown.py` (new), `scripts/checks/check_changes.py`, `prek.toml`, `pyproject.toml`, `.github/workflows/ci.yml`. **Must not change:** any file under `docs/`.

**Items.**

1. `scripts/checks/markdown.py` (new): copy UFR's `scripts/checks/markdown.py`. Its first line names the source repository, path and commit.
2. `scripts/checks/check_changes.py`: replace the file with UFR's. Keep the tuple at `scripts/checks/check_changes.py:39`, renamed `OLD_FORMAT`, with the four plotting documents of R-c added, 24 names in all. A document in `OLD_FORMAT` is skipped before its status is read.
3. `check_changes.py`: implement R-a. `BASE_REF` becomes the default. A document's `base` field, read from its frontmatter, replaces it for that document.
4. `prek.toml:43`: the hook line becomes UFR's (`pass_filenames = false`, `always_run = true`, the new name).
5. `pyproject.toml:103`: add `pythonpath = ["tests", "scripts/checks"]` to the pytest options and a `[tool.ty.environment]` table with `extra-paths = ["scripts/checks"]`. Add `pyyaml` to the `dev` group, which the checks already import (`scripts/checks/check_md_math.py:39`), and run `uv lock` (R-e).
6. `.github/workflows/ci.yml:26`: the `checks` job's checkout gets `fetch-depth: 0`, since the rounds check reads history.

**Evidence.**

- `uv run python scripts/checks/check_changes.py; echo $?` prints `0`: every existing document is in `OLD_FORMAT`.
- `diff <(git -C <UFR> show b9e4cc3:scripts/checks/markdown.py) scripts/checks/markdown.py` shows the source line only.
- `uv run just check` ends with `All checks passed!` and no `ty: ignore` line.

**Decisions.** R-a and R-b. No test file is added for the rounds check here: unit O2 gives the red-capable evidence.

### Unit O2. The change document and its conventions

The orchestrator writes four things in one commit.

1. `docs/changes/2026-11-25-conventions-sync.md` (new), in the rounds format, with `status: in-progress`, `base: inn-schematics`, and `sources` linking the plan. *Review notes* opens with ``### Round 1: from `<merge base>` `` and one row per file changed so far. *Design* lists A1 to A9 as decisions that Joon confirmed on 2026-11-25, with no marker.
2. `docs/document-conventions.md:65`, § Change documents: replace the section with UFR's, add R-a and R-b, and add a § Plans after it, copied from UFR.
3. `.claude/agents/implementer.md:18`: the *Files* bullet rule becomes UFR's rule for rows of the open round.
4. `docs/index.md`: one line for the plan and one for the change document.

Evidence, run by the orchestrator: `uv run python scripts/checks/check_changes.py` exits 0. Then delete one row of Round 1 and run it again: it exits 1 and names the uncovered file. Restore the row.

### Unit 2. Rule files, settings and hooks

**Goal.** A cloud session loads Joon's rules from the repository, and a hook blocks history rewrites.

**Model.** `sonnet`: the files are copied and names are substituted.

**May change.** `.claude/rules/working-rules.md` (new), `.claude/rules/python.md` (new), `.claude/rules/notebooks.md` (new), `.claude/settings.json` (new), `.claude/hooks/block_history_rewrite.py` (new), `.claude/hooks/block_push.py` (new), `.claude/agents/implementer.md`, the change document's rows. **Must not change:** `CLAUDE.md`, which unit O5 rewrites.

**Items.**

1. The two rule files `working-rules.md` and `python.md`: copy UFR's, with `deep_isochron` as the repository.
2. `notebooks.md`: the frontmatter `paths` is `"*.ipynb"`, since the notebooks sit at the root. The body is UFR's first paragraph. The three bullets about UFR's own notebooks are left out.
3. The two hooks: copy UFR's files unchanged, each with its source line.
4. `settings.json`: UFR's file without the `Edit|Write|NotebookEdit` matcher, which serves the dissertation hook.
5. `implementer.md:1-7`: add `disallowedTools: Agent` and the `PreToolUse` hook that runs `block_push.py`. In the body, take UFR's lines on precedence of the repository's files, on not starting subagents, on not pushing, and the *Skills* line of the report. `implementer.md:16` loses "notebook cells": a notebook that must follow an API change is the orchestrator's.

**Evidence.**

- `echo '{"tool_input":{"command":"git rebase -i HEAD~2"},"cwd":"."}' | python3 .claude/hooks/block_history_rewrite.py; echo $?` prints `2`.
- `echo '{"tool_input":{"command":"git status"},"cwd":"."}' | python3 .claude/hooks/block_history_rewrite.py; echo $?` prints `0`.
- `python3 -c "import json; json.load(open('.claude/settings.json'))"` prints nothing.
- `uv run just check` ends with `All checks passed!`.

**Decisions.** The rule files are copies of Joon's global `CLAUDE.md` of 2026-11-21, as in UFR, because a cloud session does not load the global file.

### Unit 3. The math and reference checks on the Markdown reader

**Goal.** Both checks read documents through `markdown.py`, and a docstring cannot open with a plan label.

**Model.** `opus`: two checks are merged with local behavior that must survive.

**May change.** `scripts/checks/check_md_math.py`, `scripts/checks/check_references.py`, `scripts/checks/check_docstrings.py`, `prek.toml`, the change document's rows.

**Items.**

1. `check_md_math.py`: take UFR's version.
2. `check_references.py`: take UFR's version without its `references.bib` rules, which unit 9 adds. Keep the italic-label rule (`check_references.py:58`, `OLD_LABEL`, and `check_references.py:57`, `LABEL_SCOPE`). `SKIPPED_DIRS` at `check_references.py:55` gains `docs/plans`.
3. `check_docstrings.py`: add UFR's `PLAN_LABEL` rule. It flags a docstring whose first line opens with `Item`, `Unit`, `Pass`, `Round`, `Phase` or `Q` and a number.
4. `prek.toml:40-42`: the three hook lines become UFR's, without the "clean scope" wording.

**Evidence.**

- `uv run python scripts/checks/check_docstrings.py $(git ls-files '*.py' | grep -v isochron2pbvp); echo $?` prints `0`. The planning session found no docstring that opens with a plan label at the tip.
- `uv run just check` ends with `All checks passed!`.

**Decisions.** The two checks keep their current scope. `docs/plans/` is skipped by the spelling, math and reference checks, as `docs/changes/` is, because a committed plan is a frozen record.

### Unit 4. The complexity rule

**Goal.** Ruff's `C901` runs at 8, and no function exceeds it.

**Model.** `opus`: five of the splits are in the training and data path.

**May change.** `pyproject.toml`, the seven files below, the change document's rows. **Must not change:** any test's expected value.

**Items.**

1. `pyproject.toml:56`: add `"C901"` to `select`, and a `[tool.ruff.lint.mccabe]` table with `max-complexity = 8`, with UFR's comment.
2. Split each function below so that it scores 8 or less. Each split moves one question into a named function. The output is unchanged.

| Function | Site | Score |
|---|---|---|
| `train` | `src/deep_isochron/training/trainer.py:209` | 16 |
| `__call__` | `src/deep_isochron/training/evaluation.py:372` | 10 |
| `__init__` | `src/deep_isochron/data/windows/batched.py:68` | 11 |
| `schematic_spline_coupling` | `src/deep_isochron/plotting/schematic_layers.py:132` | 13 |
| `hlo_launch_estimate` | `scripts/diagnostics/bench_dataloader.py:252` | 13 |
| `main` | `scripts/diagnostics/bench_dataloader.py:358` | 20 |
| `main` | `scripts/diagnostics/find_nan.py:69` | 10 |

**Evidence.**

- `uv run ruff check --no-fix --select C901 src tests scripts` prints `All checks passed!`. The command uses the `ruff` on the session's `PATH`, version 0.16.9 in the planning session. At the tip, with the limit at 8, it reports nine functions: these seven and two in the check scripts that unit 3 replaces.
- `uv run just test dev tests/test_windows.py tests/test_training.py tests/test_plotting.py` passes with the baseline's count (§ 7).
- `uv run python scripts/diagnostics/find_nan.py --help` and `uv run python scripts/diagnostics/bench_dataloader.py --help` both exit 0.

**Decisions.** The limit is 8, UFR's value: ruff's default of 10 would pass two of the seven. The two diagnostics scripts have no tests, so their evidence is that they still start. If a split of `train` would change the order of side effects (logging, evaluation, checkpointing), stop and report.

### Unit 5. The dependency check and the import test

**Goal.** Every imported package is declared, and every module imports on CPU.

**Model.** `sonnet`.

**May change.** `scripts/checks/check_dependencies.py` (new), `tests/test_imports.py` (new), `pyproject.toml`, `uv.lock`, `prek.toml`, the change document's rows.

**Items.**

1. `check_dependencies.py`: copy UFR's with its rule 2 (the unit libraries) removed. Rules 1 and 3 stay: every imported package is declared, and the package imports nothing from `scripts`, `notebooks` or `tests`.
2. `pyproject.toml:10`: declare `omegaconf` and `optimistix` in `dependencies`. Declare `pillow`, `scipy` and `absl-py` in the `dev` group. Then `uv lock` (R-e).
3. `tests/test_imports.py`: copy UFR's, for the package `deep_isochron`. A module that imports `auto` is skipped when AUTO-07p's tree is absent.
4. `prek.toml`: add UFR's `check-dependencies` hook line, with its name shortened to the two rules kept.

**Evidence.**

- `uv run python scripts/checks/check_dependencies.py; echo $?` prints `0`. On the tip, UFR's check reports 25 undeclared imports of six packages: `omegaconf` (14), `optimistix` (5), `pillow` (2), `pyyaml` (2), `scipy` (1), `absl-py` (1).
- `uv run just test dev tests/test_imports.py` passes. The file is new (new).
- `git diff --stat HEAD~1 -- uv.lock` shows a change, and `grep -c 'auto-07p?rev=44cc1c42' uv.lock` still prints `2`.

**Decisions.** A8. `pyyaml` is declared in unit 1.

### Unit 6. The document check

**Goal.** `check_docs.py` exists and is tested by hand. It becomes a hook in unit O3, once the documents pass it.

**Model.** `opus`.

**May change.** `scripts/checks/check_docs.py` (new), the change document's rows. **Must not change:** `prek.toml`, and any other file under `docs/`.

**Items.**

1. Copy UFR's `check_docs.py`.
2. `STATUSES` and `REQUIRED_HEADINGS` lose the types `paper`, `experiment`, `dataset` and `format`.
3. The architecture rule also reads directory names inside fenced blocks (A6).
4. A change document in `OLD_FORMAT` of `check_changes.py` is exempt from nothing here: it has the six required headings already.

**Evidence.**

- `uv run python scripts/checks/check_docs.py | tail -1` prints a count of problems, and the list holds only these kinds: a missing frontmatter block (6 change documents), a status outside the vocabulary (32: 18 change documents, 11 ADRs, 3 design documents), an ADR outside the index's *In force* table (20), the ledger row (1). No line reports a missing template, since unit O2 added the `change` and `plan` templates. No line names a missing directory of `docs/architecture.md`.
- `uv run just check` ends with `All checks passed!`, since the check is not a hook yet.

**Decisions.** The four dropped types have no document here. A type is added when its first document is. If the list holds a kind not named above, stop and report it: unit O3 is scoped to these kinds.

### Unit O3. The documents pass the check

The orchestrator makes the documents pass `check_docs.py` and wires it, in two commits.

*Commit 1, the documents.*

- Statuses of the 24 change documents by R-c and A1.
- Statuses of the 11 ADRs: `accepted; amended …` becomes `amended` (ADR-0005, 0007, 0008, 0009, 0011, 0012, 0014, 0015), and `accepted (review pending)` becomes `accepted` (A2).
- `docs/design/isochrons.md` becomes `draft for review`, `phase-field.md` becomes `tentative`, `normal-forms.md` becomes `agreed` (A4).
- Each of the nine design documents gets `kind` (A3). A `method` document's six headings become exactly the template's: the words after the title, such as the module path in `docs/design/floquet.md`'s third heading, move to the first line of the section.
- `docs/index.md:48`: the decision table is split into *In force* and *Superseded*. The second holds one sentence, since no ADR is superseded.
- `docs/roadmap.md:224` moves (A5).

*Commit 2, the wiring.*

- `prek.toml`: UFR's `check-docs` hook line.
- `justfile`: UFR's `pending` recipe.
- `docs/document-conventions.md`: § Frontmatter and status with the closed vocabularies, § Required sections, § Unconfirmed decisions, the superseded-ADR rules, the design `kind` paragraph, and the *Checks* table. Each template is quoted once.

Evidence: `uv run python scripts/checks/check_docs.py; echo $?` prints `0` after commit 1 except for template problems, and `0` with none after commit 2. `uv run just pending` prints nothing, since no document carries a marker yet. `uv run just check` ends with `All checks passed!`.

### Unit 7. The handoff check

**Goal.** A handoff can be checked against the repository before its round starts.

**Model.** `sonnet`.

**May change.** `scripts/checks/check_handoff.py` (new), `tests/test_checks.py` (new), the change document's rows.

**Items.**

1. `check_handoff.py`: copy UFR's, with `PACKAGE = "src/deep_isochron"`.
2. `tests/test_checks.py` (new): copy UFR's four test functions.

**Evidence.**

- `uv run just test dev tests/test_checks.py` reports 6 passed, from four test functions of which one has three cases. The file is new (new).
- `uv run python scripts/checks/check_handoff.py docs/plans/2026-11-25-conventions-sync.md --commit 8ef9113; echo $?` prints `0`. The planning session ran UFR's script on this document at that commit.

**Decisions.** The script is run by hand and is not a hook, as in UFR.

### Unit 8. The bibliography generator

**Goal.** `just references` writes the bibliography page from a `.bib` file. Nothing uses it yet.

**Model.** `opus`: R-d is new design.

**May change.** `scripts/docs/generate_references.py` (new), `justfile`, `pyproject.toml`, `uv.lock`, `tests/test_checks.py`, the change document's rows. **Must not change:** `docs/references.md`.

**Items.**

1. `generate_references.py`: copy UFR's. `_base_key` returns the `key` field when the entry has one (R-d), and such an entry is not grouped for collision suffixes.
2. `justfile`: UFR's `references` recipe.
3. `pyproject.toml:86`: add `bibtexparser>=2.1.0` to the `dev` group, then `uv lock` (R-e).
4. `tests/test_checks.py`: one test that an entry with `key` and no `author` renders under that key, and one that an entry with neither `key` nor `author` raises `ValueError`.

**Evidence.**

- `uv run just test dev tests/test_checks.py` reports 8 passed.
- `uv run just --summary` lists `references`.

**Decisions.** R-d. The alternative was to rename the 23 software and documentation keys to `Author Year` and update every citation. Joon chose the field because `AUTO-07p` reads better in a docstring.

### Unit O4. The bibliography file

The orchestrator writes `docs/references.bib` (new) from the 118 entries of `docs/references.md`, then runs `just references`.

- The "Used:" trail of each entry is dropped. The reference check proves that every citation resolves, and `grep` for a key finds its uses.
- The 23 entries whose key is not `Author Year` carry `key` (R-d). `de Boor 2001` and `Huguet & de la Llave 2013` follow the grammar through BibTeX's name parts and need none.
- Each of the 25 entries marked "doi: not verified" and the 15 marked "[details unverified]" is looked up with the web fetch tool at `https://api.crossref.org/works/<doi>`, or by title when no DOI is recorded. Title, year, volume and pages are compared with the current text.
- A match removes the flag. A mismatch is not overwritten: it is listed in the change document's *Open issues* for Joon. An entry that CrossRef does not have keeps its flag in `note`.

Evidence: the set of keys of the generated page equals the set at the tip (118 keys, compared with `grep -o '^- \*\*[^*]*\*\*'` on both). `uv run just check` ends with `All checks passed!`, because the generated page keeps the entry form that the current reference check reads.

### Unit 9. The reference check reads the `.bib` file

**Goal.** A citation resolves against `references.bib`, and a hand edit of the page fails.

**Model.** `sonnet`.

**May change.** `scripts/checks/check_references.py`, `prek.toml`, the change document's rows.

**Items.**

1. `check_references.py:46`: add UFR's `SOURCE` and `GENERATOR`, and UFR's rules that every key has an entry in the `.bib` file and that the page equals the generator's output.
2. `prek.toml:41`: the hook's `types_or` gains `"bib"`.

**Evidence.**

- `uv run python scripts/checks/check_references.py; echo $?` prints `0`.
- Append a line to `docs/references.md` and run it again: it exits 1 and says that the page differs from the generated one. Restore the page.

**Decisions.** None beyond R-d.

## 7. Baseline

All commands ran on the tip of § 1, on CPU, in the planning session.

- `uv run just check` ends with `All checks passed!` and finds no `ty: ignore` under `src`.
- `uv run pytest --collect-only -q -m "not slow"` collects 742 of 758 tests.
- `uv run pytest --collect-only -q -m "not slow" tests/test_windows.py tests/test_training.py tests/test_plotting.py` collects 89 tests. The planning session did not run them. Unit 4's subagent runs them before its first edit and records the count that passes.
- The change document of `inn-schematics` records 52 passed for `tests/test_plotting.py` at `ac36289`, in its *Tests* section.
- Nothing is recorded as red. Three earlier branches are unreviewed: `ksiam-animations`, `conjugacy-schematic` and `inn-schematics`. A failure in their code belongs to them, not to this branch.

## 8. The documents unit

Before it, the full quick tier runs once: `uv run just test default`. It takes about 15 minutes on two cores with the compilation cache and over an hour without.

The orchestrator then writes, in order:

1. **The change document.** *Summary*, *Design* (the decisions of § 10 with their rejected alternatives and how each is undone, and A1 to A9), *Bugs fixed* (the 25 undeclared imports), *Tests* with exact counts, *Open issues* (the three ADR reviews of A2, any CrossRef mismatch). Status `landed`.
2. **`CLAUDE.md`.** Line 3 points to `.claude/rules/` and states that the repository's files take precedence over a skill. The reading list is scoped to a planning session. The change-document bullet describes rounds. A bullet on `[unconfirmed]` and `just pending` is added. The float64 bullet at line 44 is removed. The lock-file sentence at line 56 becomes R-e. A bullet states the amend, rebase and force-push rule and names the hook.
3. **`docs/coding-standards.md`.** The *Deterministic checks* table gains the new hooks. Its second half of the float64 rule, that notebooks and scripts enable it at the top, is stated there. UFR's rules R10 (a check reads, and a test executes) and R11 ("a test pins a property of our code that could be wrong") are added to § Contracts this repository fixes.
4. **`docs/glossary.md` (new), `docs/verification.md` (new), `docs/findings.md` (new)**, in UFR's forms, with A7 and A9. The glossary is seeded from § 1 of `docs/design/normal-forms.md` and from ADR-0020. A symbol that the design documents leave open is marked `[unconfirmed]`. `findings.md` opens with the dependency finding of unit 5.
5. **`docs/roadmap.md`**, one ledger row per unit, appended. **`docs/index.md`**, every new document. **`docs/architecture.md`**, the `scripts/` entries. **`README.md`**, one line that `just setup` installs the git hooks.
6. **The readers of the old format.** Six files describe the *Files* section and its rows: `scripts/checks/check_changes.py` (unit 1), `docs/document-conventions.md` and `.claude/agents/implementer.md` (unit O2), and `CLAUDE.md:23`, `docs/coding-standards.md:104` and `scripts/README.md`, which this unit rewrites.
7. **A skill proposal.** The `orchestrate-implementation` and `code-conventions` skills describe *Files* bullets. Propose the edit to Joon, who saves skill edits to his account.
8. **The plan.** Its status becomes `executed`.

No ADR is written. For Joon, outside the repository: add to the Project instructions "The review and the merge are Joon's. Claude does not propose skipping a review round or merging ahead of one.", and drop "the `UNTESTED` entries" from the first bullet of *Before and after an edit*.

Offer `retro` at the end, and do not run it unasked.

## 9. Gates

The orchestrator stops for Joon's review at the four gates of § 3, each after about three commits, and pushes the branch before each. It also stops, whatever the batch, when:

- a unit's evidence does not come out as written,
- `inn-schematics` or `master` has moved since the tip of § 1,
- unit 6 reports a kind of problem that § 6 does not name, or
- CrossRef disagrees with an entry on more than a formatting detail.

The one condition that ends the run without a gate is a failing `uv run just check` that a unit cannot explain.

## 10. Decisions and where they are recorded

No decision here meets the test for an ADR: each is undone by reverting a check or a file.

| # | Decision | Reason | Recorded in |
|---|---|---|---|
| 1 | The branch is cut from `inn-schematics`, not `master` | Joon's choice. The latest conventions files are there | change document, *Design* |
| 2 | A stacked branch names its base (R-a) | UFR's round 1 starts at the merge base with `master`, which would list 42 commits of other branches | `document-conventions.md`, the check |
| 3 | The 24 old change documents keep their bodies (R-b) | A reviewed table is not rewritten | `document-conventions.md`, `OLD_FORMAT` |
| 4 | Their statuses are normalized (R-c) | Joon's choice, so that one vocabulary holds everywhere | the documents, change document |
| 5 | Rule files copy the global `CLAUDE.md` | A cloud session does not load the global file | `.claude/rules/` |
| 6 | The project follows UFR's working rules | Joon's choice | `.claude/rules/working-rules.md` (new) |
| 7 | The session that changes a dependency regenerates `uv.lock` (R-e) | `pyproject.toml` and the lock never disagree in a commit. AUTO-07p's commit is pinned by `rev` in `pyproject.toml:101`, so `uv lock` does not move it | `CLAUDE.md` |
| 8 | The float64 bullet leaves `CLAUDE.md` | Every library driver already raises, and Joon agreed with UFR's cut | `coding-standards.md` |
| 9 | `C901` at 8 | UFR's retrospective value. A limit of 10 passes two of the seven functions | `pyproject.toml`, change document |
| 10 | Four document types are not ported | No document of those types exists here | change document |
| 11 | The bibliography is generated from `references.bib` | One source, and a hand edit of the page fails | `document-conventions.md`, the check |
| 12 | The "Used:" trail is dropped | The check proves resolution, `grep` finds uses, and a hand-kept list goes stale | change document |
| 13 | Unverified flags stay in `note` and are checked against CrossRef | The flags are facts about the entry, and the fetch tool reaches CrossRef | `references.bib`, change document |
| 14 | Software and documentation entries carry `key` (R-d) | `AUTO-07p` reads better than a derived author-year key | the generator, `document-conventions.md` |
| 15 | `prose-2` is a separate branch stacked on this one | Review size, order, and a different risk | this plan |
| 16 | A plan label is restated in documents and removed from docstrings | A reader cannot resolve `C7` or `D5` from the document | the `prose-2` handoff |
| 17 | ADR *History* lines stay as written | They are dated records | the `prose-2` handoff |
| 18 | `Raises` and argument entries are split at semicolons | One condition per bullet | the `prose-2` handoff |
| 19 | A1 to A9 of § 2 | Proposed by the planning session and confirmed by Joon on 2026-11-25. Each point states its own reason | change document, *Design* |
