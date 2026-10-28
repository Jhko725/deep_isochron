---
type: guide
status: current
updated: 2026-10-29
sources: [the plan "Isochrons from AUTO-07p" (Project document, decisions 4, 7, R1–R3, G1, H1–H4), AUTO-07p at commit 44cc1c4 (configure, Makefile, python/auto/interactiveBindings.py), scripts/install_auto.sh, src/deep_isochron/analysis/ode/auto/_env.py]
---

# Installing AUTO-07p

The isochron pipeline of roadmap O7 runs AUTO-07p, which has two halves that must come
from the same commit. Its Python layer is an ordinary dependency, locked in `uv.lock`;
its Fortran library is a source tree that you build once with
`scripts/install_auto.sh`. Nothing else in `deep_isochron` needs AUTO, so the rest of
the package and its tests work without it. Computing isochrons once it is installed is
[`isochrons.md`](isochrons.md).

The supported platforms are Linux, including a cluster, and Windows through WSL2. macOS
should work with Homebrew's `gfortran` but is untested, and CI runs Linux only.

## Install

`uv run just setup` does all of it on a machine that has the system tools. On a fresh
clone, before the environment exists, `uvx --from rust-just just setup` does the same.

1. **Install a Fortran compiler, `make` and `git`.** These are system tools, outside uv.
   `just setup` checks for them and names three routes when one is missing:
   `sudo apt-get update && sudo apt-get install -y git gfortran make` on Debian, Ubuntu
   or WSL2; `module load gcc`, or whichever module provides `gfortran`, on a cluster;
   `conda install -c conda-forge gfortran make git` elsewhere. Nothing in the
   installation needs `sudo` beyond that first route. In a Claude cloud session the
   package lists start empty, so the `apt-get update` is required there.
2. **Sync the environment**: `uv sync`. uv syncs the group `dev` by default, and `dev`
   includes the group `auto`, which installs AUTO's Python layer from GitHub at the
   pinned commit. Only `uv sync --no-dev` leaves it out.
3. **Build the Fortran tree**: `uv run just install-auto`, which runs
   `scripts/install_auto.sh`. It clones AUTO-07p, checks out the commit that `uv.lock`
   pins, configures it without the plotting programs, and runs a serial `make`; this
   takes about 40 s. It writes the commit to `.deep_isochron_commit` in the tree, and a
   second run at the same commit does nothing.

## Where the tree lives

The tree goes to `~/.local/opt/auto-07p` unless `DEEP_ISOCHRON_AUTO_DIR` names another
directory when the script runs. The package looks for it in the same place:
`DEEP_ISOCHRON_AUTO_DIR`, then `AUTO_DIR`, then `~/.local/opt/auto-07p`. The tree is a
property of the machine, so the environment chooses it, never an argument or a
setting. An existing AUTO installation can be used by pointing `AUTO_DIR` at it,
provided it is built at the pinned commit.

`deep_isochron.analysis.ode.auto._env.load_auto` is the only code that imports AUTO. It
sets `AUTO_DIR` before the import, because AUTO's Python layer reads that variable at
import time (`python/auto/interactiveBindings.py`), and it suppresses the banner about
plotting that the import prints.

## The pinned commit

The commit lives in one place: the `[tool.uv.sources]` entry for `auto-07p` in
`pyproject.toml`, resolved into `uv.lock`. `scripts/install_auto.sh --commit` prints
it. To move to another AUTO commit, change the `rev` in `pyproject.toml`, run `uv lock`
and `uv sync`, and rerun `uv run just install-auto`. CI's `auto` job then
builds the new commit, because its cache key is the commit. `load_auto` warns when the
tree's stamp and the Python layer's commit differ.

## On a cluster

AUTO compiles each equation file when it runs, so `gfortran` must be available on the
compute nodes as well as on the login node where AUTO is installed.

- **Install on the login node, onto a filesystem the compute nodes see.** The default,
  `~/.local/opt/auto-07p`, is in the home directory, which clusters usually share. The
  installation needs the network; nothing does at run time, since AUTO, the generated
  files and the temporary run directories are all local.
- **Load the compiler in every job script**, with the same `module load` line used for
  the installation, before Python starts.
- **Set `DEEP_ISOCHRON_AUTO_DIR` in the job script** if the tree is not at the default.
- **The pool will use the cores the job was given.** `IsochronSettings.max_workers`
  defaults to `os.process_cpu_count()`, which counts only the cores a SLURM allocation
  grants the process. Every AUTO run is single-threaded (`OMP_NUM_THREADS=1`), so a
  job of 24 cores traces 24 phases at once.

## Tests

Tests that run AUTO carry the marker `auto` and skip when no built tree is found:
`uv run just test-auto`. The tests of the generated Fortran in
`tests/test_auto_codegen.py` need only gfortran; they are `slow` and skip without it.

## Troubleshooting

- **`AutoNotInstalled`.** No built tree was found where `load_auto` looked; the message
  names the directory. Run `scripts/install_auto.sh`, or set `DEEP_ISOCHRON_AUTO_DIR`.
- **`make` fails with `Cannot open module file 'support.mod'`.** The build ran in
  parallel; AUTO's `src/Makefile` reads that module before writing it. The install
  script runs a serial `make`.
- **`FileNotFoundError: .../cmds` on `import auto`.** `AUTO_DIR` was not set when `auto`
  was imported. Import it through `load_auto`.
- **`ModuleNotFoundError: No module named 'auto'`.** The environment was synced with
  `--no-dev`; run `uv sync`.
- **A job fails to compile on a compute node.** The compiler module is loaded on the
  login node but not in the job script; see *On a cluster*.
- **The configure log reports that COIN3D, OI or SoQt is not installed.** These belong
  to the 3D plotting program `plaut04`, which the script disables; the messages are
  harmless.
