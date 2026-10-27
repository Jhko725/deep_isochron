---
type: guide
status: current
updated: 2026-10-27
sources: [the plan "Isochrons from AUTO-07p" (Project document, decisions 4 and 7), AUTO-07p at commit 44cc1c4 (configure, Makefile, python/auto/interactiveBindings.py), scripts/install_auto.sh, src/deep_isochron/analysis/ode/auto/_env.py]
---

# Installing AUTO-07p for the isochron pipeline

The isochron pipeline of roadmap O7 runs AUTO-07p, which has two halves that must come
from the same commit. Its Python layer is an ordinary dependency, locked in `uv.lock`;
its Fortran library is a source tree that you build once with
`scripts/install_auto.sh`. Nothing else in `deep_isochron` needs AUTO, so the rest of
the package and its tests work without it.

## Install

1. **Install a Fortran compiler, `make` and `git`.** On Debian or Ubuntu:
   `apt-get update && apt-get install -y gfortran make git`. In a Claude cloud session
   the package lists start empty, so the `apt-get update` is required there.
2. **Sync the environment with the development group**: `uv sync --group dev`. The group
   `dev` includes the group `auto`, which installs AUTO's Python layer from GitHub at the
   pinned commit. A plain `uv sync` without a group uninstalls it.
3. **Build the Fortran tree**: `scripts/install_auto.sh`. It clones AUTO-07p, checks out
   the commit that `uv.lock` pins, configures it without the plotting programs, and runs
   a serial `make`; this takes about 40 s. It writes the commit to
   `.deep_isochron_commit` in the tree, and a second run at the same commit does
   nothing.

## Where the tree lives

The tree goes to `~/.local/opt/auto-07p` unless `DEEP_ISOCHRON_AUTO_DIR` names another
directory when the script runs. The package looks for it in this order: a path passed
to `load_auto`, then `DEEP_ISOCHRON_AUTO_DIR`, then `AUTO_DIR`, then
`~/.local/opt/auto-07p`. An existing AUTO installation can therefore be used by
pointing `AUTO_DIR` at it, provided it is built at the pinned commit.

`deep_isochron.analysis.ode.auto._env.load_auto` is the only code that imports AUTO. It
sets `AUTO_DIR` before the import, because AUTO's Python layer reads that variable at
import time (`python/auto/interactiveBindings.py`), and it suppresses the banner about
plotting that the import prints.

## The pinned commit

The commit lives in one place: the `[tool.uv.sources]` entry for `auto-07p` in
`pyproject.toml`, resolved into `uv.lock`. `scripts/install_auto.sh --commit` prints
it. To move to another AUTO commit, change the `rev` in `pyproject.toml`, run `uv lock`
and `uv sync --group dev`, and rerun `scripts/install_auto.sh`. CI's `auto` job then
builds the new commit, because its cache key is the commit. `load_auto` warns when the
tree's stamp and the Python layer's commit differ.

## Tests

Tests that run AUTO carry the marker `auto` and skip when no built tree is found:
`uv run pytest -m auto`. The tests of the generated Fortran in
`tests/test_auto_codegen.py` need only gfortran; they are `slow` and skip without it.

## Troubleshooting

- **`AutoNotInstalled`.** No built tree was found where `load_auto` looked; the message
  names the directory. Run `scripts/install_auto.sh`, or set `DEEP_ISOCHRON_AUTO_DIR`.
- **`make` fails with `Cannot open module file 'support.mod'`.** The build ran in
  parallel; AUTO's `src/Makefile` reads that module before writing it. The install
  script runs a serial `make`.
- **`FileNotFoundError: .../cmds` on `import auto`.** `AUTO_DIR` was not set when `auto`
  was imported. Import it through `load_auto`.
- **`ModuleNotFoundError: No module named 'auto'`.** The environment was synced without
  the group; run `uv sync --group dev`.
- **The configure log reports that COIN3D, OI or SoQt is not installed.** These belong
  to the 3D plotting program `plaut04`, which the script disables; the messages are
  harmless.
