---
type: guide
status: current
updated: 2026-10-30
sources:
  - "ADR-0017"
  - "ADR-0011 §1"
  - "src/deep_isochron/experiment/isochrons.py"
  - "scripts/compute_isochrons.py"
  - "configs/isochrons/"
  - "configs/hydra/launcher/slurm.yaml"
---

# Computing and reading the reference isochrons

The isochrons of a system's limit cycle are traced by AUTO-07p continuation
([ADR-0017](../decisions/0017-isochrons-by-auto-continuation.md); the method is
[`design/isochrons.md`](../design/isochrons.md)), once per system, and saved as one netCDF
file that notebooks and analyses read back. AUTO has to be installed first
([`auto07p.md`](auto07p.md)); what the pipeline hands AUTO, slot by slot and run by
run, is [`auto-pipeline.md`](auto-pipeline.md).

## Computing a foliation

`scripts/compute_isochrons.py` traces the isochrons that a config in
`configs/isochrons/` describes and saves them as one netCDF file:

```bash
uv run python scripts/compute_isochrons.py --config-name isochrons/fhn
uv run python scripts/compute_isochrons.py --config-name isochrons/fhn-easy phases.count=20
```

Each config takes its system from the data config of the same name, so the isochrons
are those of the system a model is trained on, and its cycle from `configs/cycle/`
(`x0`, `t_settle`, `m`, and the `anchor` that sets phase 0 and the Poincaré section),
shared with every analysis that uses the same phase 0; it adds the phases (`count`,
evenly spaced, or a list of `values`) and the `IsochronSettings`. The schema
`experiment.IsochronConfig` validates the file and every override at composition.

The file is `data/isochrons/<data name>-<hash>.nc`, the hash taken over everything that
decides the result; when it exists, the script says so and exits, so delete it to
recompute. A run with failed phases still saves the others, prints the failures and
exits with status 1.

## Reading it back

`experiment.load_isochrons(cfg)` reads the file of a config with the cycle rebuilt from
the same config, and raises `FileNotFoundError` naming the command to run when the file
does not exist: a reader never computes. The file holds the points as AUTO traced them;
`Isochron.resample(spacing)` gives evenly spaced points when a reader wants them.

```python
from deep_isochron.experiment import compose, load_isochrons

foliation = load_isochrons(compose(config_name="isochrons/fhn"))
for isochron in foliation.isochrons:
    points = isochron.resample(0.01).points
```

## As a SLURM job

Through Hydra's submitit launcher (the `slurm` dependency group, part of `dev`) and
`configs/hydra/launcher/slurm.yaml`, with the partition on the command line and `-m`,
since the launcher submits multiruns only:

```bash
uv run python scripts/compute_isochrons.py --config-name isochrons/fhn -m \
    hydra/launcher=slurm hydra.launcher.partition=<name> \
    'hydra.launcher.setup=[module load gcc]'
```

`setup` runs in the job before Python starts; the `module load` line makes `gfortran`
available to AUTO's run-time compiles. The launcher config asks for 24 cores, 32 GB and
four hours; override `hydra.launcher.cpus_per_task` and the others as the cluster
needs. The job's log, with one line per finished phase, is under
`data/isochrons/logs/multirun/`. What a cluster needs from the AUTO installation is in
[`auto07p.md`](auto07p.md) § On a cluster.
