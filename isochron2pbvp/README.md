# `isochron2pbvp` — isochrons of FitzHugh–Nagumo by AUTO-07p (Andrus Giraldo)

Andrus Giraldo's reproduction of the isochrons of the FitzHugh–Nagumo model by
continuation of a multi-segment two-point boundary value problem in AUTO-07p
(Langfield et al. 2014, in the rotating set-up of Hannam et al. 2022). It is his working
copy, kept in this repository beside the `deep_isochron` package so that the two can be
compared, and it follows his conventions: the package's checks, formatters and tests
stop at this directory's edge.

- `fncMultiSeg/` — the current version: the equations file `fncMultiSegIso.f90`, the
  constants file `c.fncMultiSeg`, the driver `mainIsochronMultiSegFNC.py`, and
  `dataIso/`, one CSV per isochron.
- `isoFunctions.py` — the helpers the driver imports (the multi-segment start from a
  periodic orbit).
- `plotIsochron.m` — the MATLAB plot of `dataIso/`.
- `deprecated/` — the single-segment version that preceded it.

The driver runs under AUTO's own Python environment, from inside `fncMultiSeg/`, with
`AUTO_DIR` set to a built AUTO-07p tree (the one `uv run just install-auto` builds at
`~/.local/opt/auto-07p`, see `docs/guides/auto07p.md`):

```bash
cd isochron2pbvp/fncMultiSeg
AUTO_DIR=~/.local/opt/auto-07p $AUTO_DIR/bin/auto mainIsochronMultiSegFNC.py
```

The `deep_isochron` pipeline (`src/deep_isochron/analysis/ode/auto/`) was built from this
code and is checked against three of its isochrons, files 26, 51 and 100 of `dataIso/`
as of commit `3af99f2`, copied to `tests/golden/`; `docs/design/isochrons.md` § 2.3
reads the driver run by run and links that commit.
