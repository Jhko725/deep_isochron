# deep_isochron

`deep_isochron` learns smooth conjugacies between planar limit-cycle systems with
invertible neural networks in JAX and Equinox, mapping an observed oscillator such as
FitzHugh–Nagumo onto a normal form such as Bautin's, whose phase and amplitude are known
in closed form. The learned map carries the normal form's isochrons, the curves of equal
asymptotic phase, back to the observed system. To check them, the package also computes
the true isochrons of the observed system by numerical continuation in AUTO-07p, along
with the limit cycle and its Floquet multipliers.

## Setup

The package needs Python 3.13 and [uv](https://docs.astral.sh/uv/). The isochron
computation also needs AUTO-07p, which in turn needs `gfortran`, `make` and `git` on the
machine; `docs/auto.md` covers Linux, clusters and WSL2.

```bash
git clone https://github.com/Jhko725/deep_isochron
cd deep_isochron
uvx --from rust-just just setup
```

`just setup` checks for the system tools and says how to install a missing one, syncs
the environment, builds AUTO's Fortran library at the pinned commit, and installs the
pre-commit hooks. AUTO's tree goes to `~/.local/opt/auto-07p`; to put it elsewhere, or
to use a tree you already have, set `DEEP_ISOCHRON_AUTO_DIR` (or AUTO's own `AUTO_DIR`)
to its path before `just setup` and whenever the package runs. Afterward the commands
are recipes in the `justfile`:

```bash
uv run just --list        # every recipe, with what it does
uv run just check         # the hooks of prek.toml and the type check
uv run just test          # the quick test tier
uv run just test-auto     # the tests that run AUTO
```

## Documentation

[`docs/index.md`](docs/index.md) lists every document: the architecture, the roadmap,
the design documents with the mathematics behind the code, and the decision records.

## Acknowledgments

The AUTO-07p computation of isochrons builds on Andrus Giraldo's reproduction of the
FitzHugh–Nagumo isochrons (Department of Computational Sciences, Korea Institute for
Advanced Study), following P. Langfield, B. Krauskopf and H. M. Osinga, "Solving
Winfree's puzzle: the isochrons in the FitzHugh–Nagumo model," *Chaos* 24, 013131
(2014), and J. Hannam, B. Krauskopf and H. M. Osinga, "Isochron foliations and global
bifurcations: a case study," *Trans. Math. Appl.* 6(2), tnac002 (2022).
