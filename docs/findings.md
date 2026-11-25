---
type: findings
status: current
updated: 2026-11-25
---

# Findings

The append-only ledger of anything discovered that bears on a reported result or constrains a later branch. Each row names its evidence and the action taken. Cleanup goes in [`roadmap.md`](roadmap.md).

| # | Date | Finding | Evidence | Affects | Action and state |
|---|---|---|---|---|---|
| 1 | 2026-11-25 | The code imported six packages that `pyproject.toml` did not declare: `omegaconf`, `optimistix`, `pillow`, `pyyaml`, `scipy` and `absl-py`. They were installed only as dependencies of other packages, so an upgrade of one of those could have removed them. | `scripts/checks/check_dependencies.py` run on the tree of `2d24c68`: 24 undeclared import lines of five packages, after `7425646` had declared `pyyaml` | every environment built from `pyproject.toml` | closed: all six are declared (`7425646`, `f7d8507`), and the check runs in `prek` and CI |
