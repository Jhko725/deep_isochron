---
type: change
status: in-progress
updated: 2026-11-25
base: inn-schematics
sources:
  - "../plans/2026-11-25-conventions-sync.md"
---

# `conventions-sync`

## Summary

The branch brings `deep_isochron` up to the conventions of `UniversalForceReconstruction` (UFR) at its commit `b9e4cc3`. It ports review rounds in change documents, rule files and hooks under `.claude/`, the document checks, committed plans, the complexity and dependency checks, and a bibliography generated from `references.bib`. The plan of the round is [`../plans/2026-11-25-conventions-sync.md`](../plans/2026-11-25-conventions-sync.md).

The round is under way. This section is completed when the last unit lands.

## Design

Joon confirmed the nine points below on 2026-11-25, in the planning chat. The plan labels them A1 to A9, and the labels are kept here because the commits cite them.

- **A1. An old status sentence is kept.** Each change document whose status is normalized keeps its old status sentence as its last line, in the form `Status before 2026-11-25: <sentence>`.
- **A2. Three ADRs lose "review pending".** ADR-0016, ADR-0017 and ADR-0020 become `accepted`, and their three pending reviews are listed under *Open issues*.
- **A3. Each design document has a kind.** `method` for `cycle-estimation.md`, `floquet.md`, `isochrons.md`, `limit-cycle.md` and `phase-field.md`. `derivation` for `normal-forms.md` and `cubic-bspline.md`. `note` for `training-step-performance.md` and `validation-split.md`.
- **A4. `normal-forms.md` becomes `agreed`.** One line under its title names the sections that are still tentative, since the status vocabulary has no value for a document that is partly agreed.
- **A5. One ledger row moves.** The row of `docs/roadmap.md` dated 2026-10-27 that follows two rows dated 2026-10-28 moves above them. It is the one reordering the ledger ever gets.
- **A6. The architecture rule reads the fenced module tree.** A directory named inside the fenced tree of `docs/architecture.md` counts as listed, since that tree is this repository's module map.
- **A7. The glossary's fourth column is *Unit or range*.** This repository has no named-dimension table.
- **A8. Every undeclared import is declared.** None is removed from the code.
- **A9. `verification.md` is seeded from the method documents.** Its first rows come from § 5 of the five `method` design documents. A claim outside them gets its row when its code is next touched.

The remaining decisions of the plan (§ 10), each with its rejected alternative and how it is undone, are written here when the last unit lands.

## Bugs fixed

None so far.

## Tests

No test file has changed so far. `uv run just check` passes at each commit. The full quick tier runs once, after the last code commit, and its counts are recorded here then.

## Open issues

- The round is in progress. The plan's table of units (§ 3) lists what is left.

## Review notes

### Round 1: from `8ef9113`

| Change | Thoughts | Modifications |
|---|---|---|
| `docs/plans/2026-11-25-conventions-sync.md` — the handoff of this round, committed verbatim under a frontmatter block. | | |
| `scripts/checks/markdown.py` — UFR's Markdown reader, copied with a source line. | | |
| `scripts/checks/check_changes.py` — UFR's rounds check in place of the rows check. A document's `base` field names the branch that round 1 starts from (`base_ref`), and the 24 old-format documents are skipped by name (`OLD_FORMAT`). The coverage of each round moves out of `problems` into `round_coverage_problems`, for `C901`. | | |
| `prek.toml` — the `check-changes` hook always runs, and the `check-md-math` and `check-references` hooks take the files that `prek` passes, as in UFR. The `check-dependencies` hook runs the new dependency check. The `check-docs` hook is added. | | |
| `pyproject.toml` — pytest and ty find `scripts/checks`, and `pyyaml` joins the `dev` group. Ruff's `C901` runs with `max-complexity = 8`. `omegaconf` and `optimistix` join `dependencies`, and `absl-py`, `pillow` and `scipy` join the `dev` group, as every import under `src`, `tests` and `scripts` must be declared. | | |
| `uv.lock` — `pyyaml` in the `dev` group. The five packages above are locked, from `uv lock` without `--upgrade`. | | |
| `.github/workflows/ci.yml` — the `checks` job fetches the full history. | | |
| `docs/changes/2026-11-25-conventions-sync.md` — this document. | | |
| `docs/document-conventions.md` — § Change documents in the rounds form, with the `base` field and the old-format rule, and a new § Plans. § Frontmatter and status gives the closed vocabularies; § Required sections, § Unconfirmed decisions and the rules for a superseded ADR are added, and the *Checks* table lists the new checks. | | |
| `.claude/agents/implementer.md` — the implementer fills rows of the open round, in place of *Files* bullets; it gains `disallowedTools: Agent`, the `block_push.py` hook, the precedence and no-subagent lines, a *Skills* report line, and loses notebook cells. | | |
| `docs/index.md` — the plan and this document listed. The decision table is split into *In force* and *Superseded*, and the status cells of the ADRs and design documents repeat the frontmatter. | | |
| `.claude/rules/working-rules.md` — UFR's working rules at b9e4cc3, copied. | | |
| `.claude/rules/python.md` — UFR's design, Python and JAX rules at b9e4cc3, copied. | | |
| `.claude/rules/notebooks.md` — UFR's notebook rule for `*.ipynb`, without the three bullets on UFR's own notebooks. | | |
| `.claude/settings.json` — UFR's Bash hook registration for `block_history_rewrite.py`, without the dissertation matcher. | | |
| `.claude/hooks/block_history_rewrite.py` — UFR's hook that blocks history rewrites, unchanged under a source line. | | |
| `.claude/hooks/block_push.py` — UFR's hook that blocks `git push` for the implementer, unchanged under a source line. | | |
| `scripts/checks/check_md_math.py` — UFR's version under a source line: paragraphs render through `markdown.PARSER`, and under `docs/plans/` only the frontmatter is checked; with no path it still reads every Markdown file under `docs`. | | |
| `scripts/checks/check_references.py` — UFR's version under a source line, without its `references.bib` rules: links are read through `markdown.py`, `docs/plans/` is skipped, and the italic-label rule and the `docs/references.md` keys stay. | | |
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
| `docs/roadmap.md` — the ledger row of 2026-10-27 for Steps C–D of `auto-isochrons` moves above the two rows of 2026-10-28 (A5). | | |
| `justfile` — the `pending` recipe lists every marker of an unconfirmed decision under `docs/`. | | |
| `scripts/checks/check_handoff.py` — new, adapted from UFR's: a handoff's paths, `path:line` references, `just` recipes and test selectors are checked against the repository before its round starts; run by hand, not a hook. | | |
| `tests/test_checks.py` — new, from UFR's: the four tests of `check_handoff` (six cases); UFR's file holds no tests of other checks. | | |
