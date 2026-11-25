---
type: change
status: landed
updated: 2026-11-25
base: inn-schematics
sources:
  - "../plans/2026-11-25-conventions-sync.md"
---

# `conventions-sync`

## Summary

The branch brings `deep_isochron` up to the conventions of `UniversalForceReconstruction` (UFR) at its commit `b9e4cc3`. The plan of the round is [`../plans/2026-11-25-conventions-sync.md`](../plans/2026-11-25-conventions-sync.md), and its units landed in the plan's order.

- **Review rounds.** A change document lists its changes in *Review notes* only, as a log of rounds. `check_changes.py` fails when a round does not cover exactly the files changed in its range. A stacked branch names its base in the frontmatter, and the 24 earlier change documents keep their form.
- **Rules and hooks in the repository.** `.claude/rules/` holds Joon's working rules, so that a cloud session reads them. A hook blocks a rebase, a force-push and an amend of a pushed commit, and the implementer agent can neither push nor start a subagent.
- **Document checks.** `check_docs.py` enforces closed vocabularies of `type` and `status`, required headings, the index, the ledger's order, the module map and the marker of an unconfirmed decision. Every document was brought to pass it.
- **Committed plans.** A handoff is committed under `docs/plans/` and frozen, and `check_handoff.py` checks one against a commit by hand.
- **Code checks.** Ruff's `C901` runs at 8, with eight functions split. `check_dependencies.py` requires every imported package to be declared, and `tests/test_imports.py` imports every module.
- **A generated bibliography.** `docs/references.md` is generated from `docs/references.bib`, and the reference check fails on a hand edit of the page.
- **Three new documents.** `glossary.md`, `verification.md` and `findings.md`, in UFR's forms.

The session stopped at Gate 1 after the first four commits. Joon then asked, on 2026-11-25, for the rest of the plan to run without the three later gates.

## Design

No decision here meets the test for an ADR: each is undone by reverting a check or a file.

1. **The branch is cut from `inn-schematics`.** The latest conventions files are there. The rejected alternative is `master`, which lacks the implementer agent file and three branches of documents. Undone by rebasing the branch's commits onto `master` once those branches merge.
2. **A stacked branch names its base.** The frontmatter field `base` makes round 1 start from the merge base with that branch. Without it, round 1 of this document would start at the merge base with `master` and list 42 commits of other branches. Undone by deleting `base_ref` from `check_changes.py`.
3. **The 24 earlier change documents keep their bodies.** `check_changes.py` skips them by name in `OLD_FORMAT`. The rejected alternative is rewriting their *Files* sections and tables into rounds, which would rewrite reviewed tables. The tuple never grows.
4. **Their statuses are normalized.** One vocabulary then holds in every document. The old status sentence of each is kept as its last line (A1), so nothing is lost. Undone by restoring the sentences from those lines.
5. **The rule files copy Joon's global `CLAUDE.md`.** A cloud session does not load the global file. The rejected alternative is the Project instructions alone, which a subagent does not receive.
6. **The session that changes a dependency runs `uv lock`.** `pyproject.toml` and the lock file then never disagree in a commit. AUTO-07p's commit is pinned by `rev` in `pyproject.toml`, and the lock file still names it three times. The rejected alternative is the earlier rule, that Joon regenerates the lock after pulling.
7. **The float64 bullet leaves `CLAUDE.md`.** Every library driver already raises, and the rule for notebooks and scripts is a row of `coding-standards.md` § Contracts this repository fixes.
8. **`C901` at 8.** It is UFR's value. Ruff's default of 10 would pass `Evaluator.__call__` and `find_nan.main`. Each split moves one question into a named function, and the output is unchanged: `Trainer.train` keeps its order of side effects per step. Undone by removing `C901` from `select`.
9. **Four document types are not ported.** No paper record, experiment report, dataset record or format note exists here. A type is added to `check_docs.py` with its first document.
10. **The bibliography is generated from `references.bib`.** There is one source, and a hand edit of the page fails the check. The rejected alternative is the hand-kept page. Undone by reverting `check_references.py` to read the page.
11. **The "Used:" trail of each entry is dropped.** The check proves that every citation resolves, `grep` for a key finds its uses, and a hand-kept list goes stale.
12. **Software and documentation entries carry `key`.** The rejected alternative is renaming the 23 keys to `Author Year` and updating every citation. Joon chose the field because `AUTO-07p` reads better in a docstring.
13. **Unverified flags stay in `note`.** The flags are facts about the entry. A CrossRef lookup that matched removed the flag; a mismatch was not overwritten and is listed under *Open issues*.

Joon confirmed the nine points below on 2026-11-25, in the planning chat. The plan labels them A1 to A9, and the labels are kept here because the commits cite them.

- **A1. An old status sentence is kept.** Each change document whose status is normalized keeps its old status sentence as its last line, in the form `Status before 2026-11-25: <sentence>`.
- **A2. Three ADRs lose "review pending".** ADR-0016, ADR-0017 and ADR-0020 become `accepted`, and their three pending reviews are listed under *Open issues*.
- **A3. Each design document has a kind.** `method` for `cycle-estimation.md`, `floquet.md`, `isochrons.md`, `limit-cycle.md` and `phase-field.md`. `derivation` for `normal-forms.md` and `cubic-bspline.md`. `note` for `training-step-performance.md` and `validation-split.md`.
- **A4. `normal-forms.md` becomes `agreed`.** One line under its title names the sections that are still tentative, since the status vocabulary has no value for a document that is partly agreed.
- **A5. One ledger row moves.** The row of `docs/roadmap.md` dated 2026-10-27 that followed two rows dated 2026-10-28 moves above them. It is the one reordering the ledger ever gets.
- **A6. The architecture rule reads the fenced module tree.** A directory named inside the fenced tree of `docs/architecture.md` counts as listed, since that tree is this repository's module map.
- **A7. The glossary's fourth column is *Unit or range*.** This repository has no named-dimension table.
- **A8. Every undeclared import is declared.** None is removed from the code.
- **A9. `verification.md` is seeded from the method documents.** Its first rows come from § 5 of the five `method` design documents. A claim outside them gets its row when its code is next touched.

The round departed from the plan in these places.

- **`docs/plans/` stays under the spelling checks.** The plan's unit 3 skips it there. Joon decided on 2026-11-25 to leave it, and `document-conventions.md` § Plans says so.
- **Unit 4 split an eighth function.** The `base` rule of unit 1 took `problems` in `check_changes.py` from 8 decision points to 9.
- **The plan's body was typed out by hand.** The Project tool returned the handoff inline and not as a file, so the committed copy has not been compared with the original by a program.
- **The implementers ran as general-purpose agents.** The `implementer` agent type was not registered in the session. Each subagent read `.claude/agents/implementer.md` first, and the `block_push.py` hook did not guard them. None pushed.
- **`tests/test_imports.py` skips nothing.** No module imports `auto` when it is imported: the one import is inside `load_auto`.
- **The commit message of `fff080e` miscounts.** It says that nine entries keep a flag. Ten do; the tenth is `Stoer & Bulirsch 2002`, whose flag is about the edition.

## Bugs fixed

- **Six packages were imported and not declared.** `omegaconf`, `optimistix`, `pillow`, `pyyaml`, `scipy` and `absl-py` were installed only as dependencies of other packages. `check_dependencies.py` reported 24 undeclared import lines at `2d24c68`, after `pyyaml` had been declared. All six are declared now, and [`../findings.md`](../findings.md) row 1 records the finding.

## Tests

The checks are the product of this branch, so most of its evidence is a check that fails on a planted violation. Each was planted in a copy-backed file and restored.

- **The full quick tier.** `uv run just test default` on the tree of `7964fcc`, the last commit that changes code, on CPU with two cores: 740 passed, 10 skipped, 1 expected failure, of 751 collected, in 17 min 57 s. The ten skips are the tests that need an AUTO-07p tree, which the session did not have. The expected failure is `tests/test_analytic.py::test_sinh_second_derivative_near_zero_is_finite`, pinned before this branch. The tip of `inn-schematics` collects 742 tests in this tier; the nine new ones are below.
- **`tests/test_imports.py`** (new, 1 test) imports every module of the package on CPU.
- **`tests/test_checks.py`** (new, 8 tests). Six pin `check_handoff.py`: a mention carries its line and its `(new)` marker, the `-k` matcher follows pytest's on three expressions and rejects one that does not parse, and a missing file and a line past a file's end are each reported. Two pin the generator: an entry with `key` and no author renders under that key, and an entry with neither raises `ValueError`. Run them with `uv run just test dev tests/test_checks.py`.
- **The split functions.** `uv run just test dev tests/test_windows.py tests/test_training.py tests/test_plotting.py` gave 89 passed before unit 4's first edit and 89 passed after it. `find_nan.py --help` and `bench_dataloader.py --help` exit 0, and `uv run ruff check --no-fix --select C901 src tests scripts .claude` passes.
- **`check_changes.py`.** It passes on this document, exits 1 naming the file when a row is removed, and exits 1 with "round 1 starts from 913c52e, not from 8ef9113" when the `base` field is removed.
- **`check_docs.py`.** It reported 59 problems when it arrived, of the four kinds that the plan names, and reports 0 now. It fails on a directory removed from the fenced module tree, on a document missing from the index, and on the marker of an unconfirmed decision in an accepted ADR, while the marker in code font passes.
- **`check_md_math.py` and `check_references.py`.** Both give the verdicts they gave before the port. They fail on a pair of starred formulas, an unknown citation, a broken link and an italic label under `docs/design/`, and pass the same citation inside a plan. The reference check fails on a line appended to `docs/references.md` and on an entry removed from `docs/references.bib`.
- **`check_dependencies.py`.** It reported 24 undeclared import lines before the declarations and 0 after. It fails when `optimistix` is removed from `pyproject.toml` and when a module of the package imports `tests`.
- **The hooks.** `block_history_rewrite.py` exits 2 on `git rebase -i HEAD~2` and 0 on `git status`, and `block_push.py` exits 2 on `git push origin x`.
- **The bibliography.** The generated page has the 118 keys of the page at the tip.
- **`uv run just check`** passes at every commit, with no `ty: ignore` under `src`.

## Open issues

- **Three ADR reviews are pending (A2).** ADR-0016 (symbolic systems), ADR-0017 (isochrons by AUTO continuation) and ADR-0020 (the phase in turns) carried `accepted (review pending)` and are now `accepted`. Their reviews are still Joon's to do.
- **The index carried five more "review pending" notes.** Its status cells for ADR-0008 (Decision 3), ADR-0009 (review round 2), ADR-0013, ADR-0014 and ADR-0015 said so, and the ADRs' own frontmatter did not. The cells now repeat the frontmatter, so these notes are recorded here only.
- **Three entries disagree with CrossRef and were not overwritten.**
  - `Izhikevich 2007`: CrossRef has doi 10.7551/mitpress/2526.001.0001 with the year 2006 and an electronic ISBN. The entry says 2007.
  - `de Boor 2001`: CrossRef has the book only as doi 10.1007/978-1-4612-6333-3, issued 1978, the first edition. The entry is the revised edition of 2001.
  - `Ascher et al. 1988`: CrossRef has only the SIAM reprint of 1995, doi 10.1137/1.9781611971231. The entry is the Prentice-Hall edition.
- **Six entries are not in CrossRef and keep their flags:** `Andronov et al. 1973`, `Deuflhard 1980`, `Doedel 1981`, `Hale 1980`, `Keller 1968` and `Mees 1981`. Title searches at CrossRef went through its `prefixes`, `types` and `members` routes, because the fetch tool answered every title query on `/works` from one cached response.
- **Questions that the old bibliography page listed are still open.** The page's *Open questions* section is not generated. `Zielinski et al. 2014` is settled, since CrossRef confirms the author list. These remain:
  - `Aprille & Trick 1972` and `Andronov et al. 1973`: the repository cites neither by title, and the entries name the likely works.
  - `Blinn 2007`: the cubic series has two 2007 parts, and the entry is Part 5, the one that Peters 2016 modifies.
  - `Perko 2001` and `Stoer & Bulirsch 2002`: the editions are inferred from the sections cited.
  - `Morrison et al. 1962`: the issue number is omitted, because the sources disagree.
- **`check_handoff.py` flags this branch's own plan.** The plan's unit 7 expects exit 0 for the plan at `8ef9113`. The script exits 1 with seven findings: `just pending` (three lines) and `just references` (two) are not recipes at that commit, and `tests/test_imports.py` and `tests/test_checks.py` do not exist there. The port equals UFR's script except for the package path, so the plan's expectation is what is off. Whether a recipe or a test path that the plan itself creates should be exempt, as a path marked `(new)` is, is Joon's to decide.
- **No rule requires `kind`.** A design document without `kind` has no required headings, so `check_docs.py` would not notice a missing one. UFR's check behaves the same way.
- **Two glossary rows carry the marker.** The code names of the phase angle and of the isostable coordinate are Claude's reading. `just pending` lists them.
- **`KNOWN_UNDECLARED` in `check_dependencies.py` is empty.** It is UFR's allowance for imports awaiting a later unit, and nothing here uses it.
- **The auto tier did not run.** The session has no AUTO-07p tree, so the three isochron rows of [`../verification.md`](../verification.md) are not marked green.
- **Two edits of the Project instructions are Joon's.** Add "The review and the merge are Joon's. Claude does not propose skipping a review round or merging ahead of one.", and drop "the `UNTESTED` entries" from the first bullet of *Before and after an edit*.
- **No skill edit is proposed.** The plan expected the `orchestrate-implementation` and `code-conventions` skills to describe *Files* bullets. As loaded on 2026-11-25, both already describe rows of the open round and defer to the repository's document conventions.
- **The prose cleanup is the next branch.** `prose-2` is stacked on this one, and its handoff is written after this branch is reviewed.

## Review notes

### Round 1: from `8ef9113`

| Change | Thoughts | Modifications |
|---|---|---|
| `docs/plans/2026-11-25-conventions-sync.md` — the handoff of this round, committed verbatim under a frontmatter block. Its status becomes `executed` with the documents unit. | | |
| `scripts/checks/markdown.py` — UFR's Markdown reader, copied with a source line. | | |
| `scripts/checks/check_changes.py` — UFR's rounds check in place of the rows check. A document's `base` field names the branch that round 1 starts from (`base_ref`), and the 24 old-format documents are skipped by name (`OLD_FORMAT`). The coverage of each round moves out of `problems` into `round_coverage_problems`, for `C901`. | | |
| `prek.toml` — the `check-changes` hook always runs, and the `check-md-math` and `check-references` hooks take the files that `prek` passes, as in UFR; the `check-references` hook also takes `.bib` files, and its name says that `references.md` is generated from `references.bib`. The `check-dependencies` hook runs the new dependency check. The `check-docs` hook is added. | | |
| `pyproject.toml` — pytest and ty find `scripts/checks`, and `pyyaml` joins the `dev` group. Ruff's `C901` runs with `max-complexity = 8`. `omegaconf` and `optimistix` join `dependencies`, and `absl-py`, `pillow` and `scipy` join the `dev` group, as every import under `src`, `tests` and `scripts` must be declared. `bibtexparser` joins the `dev` group for the bibliography generator. codespell accepts the surname Troup, which the bibliography's `.bib` file cannot mark on its own line. | | |
| `uv.lock` — `pyyaml` in the `dev` group. The five packages above are locked, from `uv lock` without `--upgrade`. `bibtexparser` and its dependency `pylatexenc` are locked the same way. | | |
| `.github/workflows/ci.yml` — the `checks` job fetches the full history. | | |
| `docs/changes/2026-11-25-conventions-sync.md` — this document. | | |
| `docs/document-conventions.md` — § Change documents in the rounds form, with the `base` field and the old-format rule, and a new § Plans. § Frontmatter and status gives the closed vocabularies; § Required sections, § Unconfirmed decisions and the rules for a superseded ADR are added, and the *Checks* table lists the new checks. § References names `references.bib` as the source, the `key` field and the flags kept in `note`. *What goes where* lists the three new documents, and § Glossary and notation and § Findings and verification give their forms. | | |
| `.claude/agents/implementer.md` — the implementer fills rows of the open round, in place of *Files* bullets; it gains `disallowedTools: Agent`, the `block_push.py` hook, the precedence and no-subagent lines, a *Skills* report line, and loses notebook cells. | | |
| `docs/index.md` — the plan and this document listed. The decision table is split into *In force* and *Superseded*, and the status cells of the ADRs and design documents repeat the frontmatter. The glossary, the findings ledger and the verification table are listed. | | |
| `.claude/rules/working-rules.md` — UFR's working rules at b9e4cc3, copied. | | |
| `.claude/rules/python.md` — UFR's design, Python and JAX rules at b9e4cc3, copied. | | |
| `.claude/rules/notebooks.md` — UFR's notebook rule for `*.ipynb`, without the three bullets on UFR's own notebooks. | | |
| `.claude/settings.json` — UFR's Bash hook registration for `block_history_rewrite.py`, without the dissertation matcher. | | |
| `.claude/hooks/block_history_rewrite.py` — UFR's hook that blocks history rewrites, unchanged under a source line. | | |
| `.claude/hooks/block_push.py` — UFR's hook that blocks `git push` for the implementer, unchanged under a source line. | | |
| `scripts/checks/check_md_math.py` — UFR's version under a source line: paragraphs render through `markdown.PARSER`, and under `docs/plans/` only the frontmatter is checked; with no path it still reads every Markdown file under `docs`. | | |
| `scripts/checks/check_references.py` — UFR's version under a source line: links are read through `markdown.py`, `docs/plans/` is skipped, and the italic-label rule stays. Citations resolve against the keys that the generator lists from `docs/references.bib`, which replace the keys read from the lines of `docs/references.md`, and a page that differs from the generator's output fails (`SOURCE`, `GENERATOR`, `load_generator`, `stale_page`). | | |
| `scripts/checks/check_docstrings.py` — UFR's `PLAN_LABEL` rule: a docstring cannot open with a numbered plan label. | | |
| `src/deep_isochron/training/trainer.py` — `Trainer.train` is split for `C901`: `_initial_state` takes the state or builds it, `_check_run_lengths` refuses a bad `eval_every` or `num_steps`, `_evaluate_and_save` replaces the nested closure and returns the step of the last checkpoint, and `_next_batch` reads the loader and warns when it ends early. | | |
| `src/deep_isochron/training/evaluation.py` — `Evaluator.__call__` takes the split metrics from `_split_metrics` and the `latent/` metrics from `_latent_metrics`, for `C901`. | | |
| `src/deep_isochron/data/windows/batched.py` — `WindowBatchSource.__init__` takes the run's epochs from `_run_epochs`, for `C901`. | | |
| `src/deep_isochron/plotting/schematic_layers.py` — the argument checks of `schematic_spline_coupling` move into `_check_spline_layout`, for `C901`. | | |
| `scripts/diagnostics/bench_dataloader.py` — split for `C901`: `hlo_computations` parses the HLO text for `hlo_launch_estimate`, and `main`'s closures become `training_pipeline`, `per_element_windows`, `mp_prefetch_pipeline`, `mp_prefetch_works`, `trivial_step`, `time_floor`, `cell` and `print_table`. | | |
| `scripts/diagnostics/find_nan.py` — `main` loads its starting checkpoint through `load_finite_run`, which walks back past a non-finite one, for `C901`. | | |
| `scripts/checks/check_dependencies.py` — new, adapted from UFR's: every imported package is declared in `pyproject.toml`, and `src` imports nothing from `scripts`, `notebooks` or `tests`. | | |
| `tests/test_imports.py` — new, from UFR's: every module of `deep_isochron` imports on CPU; no module imports `auto` at import time, so none is skipped. | | |
| `scripts/checks/check_docs.py` — new, adapted from UFR's: the eight document rules without the `paper`, `experiment`, `dataset` and `format` types, and the architecture rule also reads the fenced module tree (A6); run by hand, not a hook yet. | | |
| `docs/changes/` — the 24 earlier change documents take a status from the closed vocabulary: `merged` for the 20 that `master` holds, `landed` for `deformation-plots`, `ksiam-animations` and `inn-schematics`, `in-progress` for `conjugacy-schematic`. The six without a frontmatter block get one, and each replaced status sentence is kept as the last line (A1). | | |
| `docs/decisions/` — eleven ADRs take a status from the closed vocabulary: `amended` for 0005, 0007, 0008, 0009, 0011, 0012, 0014 and 0015, and `accepted` for 0016, 0017 and 0020 (A2). | | |
| `docs/design/` — each design document names its `kind` (A3). `isochrons.md`, `phase-field.md` and `normal-forms.md` take a status from the closed vocabulary, `normal-forms.md` names its tentative sections under the title (A4), and the headings of four method documents become the template's, with the removed words as the first line of the section. | | |
| `docs/roadmap.md` — the ledger row of 2026-10-27 for Steps C–D of `auto-isochrons` moves above the two rows of 2026-10-28 (A5). Fourteen rows, one per unit of this branch, are appended to the ledger. | | |
| `justfile` — the `pending` recipe lists every marker of an unconfirmed decision under `docs/`. The `references` recipe, from UFR's, writes `docs/references.md` from `docs/references.bib`. | | |
| `scripts/checks/check_handoff.py` — new, adapted from UFR's: a handoff's paths, `path:line` references, `just` recipes and test selectors are checked against the repository before its round starts; run by hand, not a hook. | | |
| `tests/test_checks.py` — new, from UFR's: the four tests of `check_handoff` (six cases); UFR's file holds no tests of other checks. Two tests of the bibliography generator: an entry with a `key` field and no author is listed under that key, and an entry with neither raises `ValueError`. | | |
| `scripts/docs/generate_references.py` — new, adapted from UFR's: an entry's `key` field, when present, is its in-text key, needs no author or year and takes no collision suffix; a repeated explicit key, or one equal to a derived key, raises `ValueError` (R-d). | | |
| `docs/references.bib` — the 118 entries of the bibliography as BibTeX, without the "Used:" trails. The 23 software and documentation entries carry `key`, and the flags that CrossRef did not clear stay in `note`. | | |
| `docs/references.md` — generated from `references.bib`. The page loses its *Open questions* section, which moves to this document's *Open issues*. | | |
| `CLAUDE.md` — the rules are in `.claude/rules/`, and the repository's files take precedence over a skill. The reading list is for a planning session, the change-document bullet describes rounds, and bullets on unconfirmed decisions, on the lock file and on history rewrites are added. The float64 bullet is removed. | | |
| `docs/coding-standards.md` — the *Deterministic checks* table gains `check_docs.py`, `C901`, `check_dependencies.py` and `tests/test_imports.py`, and its rows for the document checks, the docstring check and the rounds check are updated. § Contracts this repository fixes gains the float64 rule and UFR's two rules on checks and tests. | | |
| `docs/glossary.md` — new: sixteen quantities from § 1 of `design/normal-forms.md` and from ADR-0020, with *Unit or range* as the fourth column (A7). | | |
| `docs/verification.md` — new: fourteen claims from § 5 of the five method documents (A9), each with its test, oracle, tolerance, tier and state. | | |
| `docs/findings.md` — new: the ledger, opened with the six undeclared packages. | | |
| `docs/architecture.md` — the `scripts/` tree lists `scripts/docs/` and the new check scripts. | | |
| `README.md` — a new clone needs `just setup` before its first commit, since the git hooks are installed once in every clone. | | |
| `scripts/README.md` — the checks as a table, with the new scripts, and `scripts/docs/`. | | |
