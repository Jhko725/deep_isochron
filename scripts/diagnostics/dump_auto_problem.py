"""Write the AUTO problem files of an isochron config into a directory, to read them.

    uv run python scripts/diagnostics/dump_auto_problem.py isochrons/fhn /tmp/fhn-auto
    uv run python scripts/diagnostics/dump_auto_problem.py isochrons/fhn-easy \\
        /tmp/easy settings.epss=1e-12

The first argument is the config name under ``configs/``, the second the directory
(created if needed), and the rest are Hydra overrides. The four files are what
``auto_session`` writes into its temporary directory before the stages run
(``docs/guides/auto-pipeline.md``): the equation file, ``rhs.inc``, ``indices.inc``
and the constants file ``c.isochron``, with the config's tolerances. The script prints
the ``PAR`` slots of the instance, system parameters and pipeline slots alike, and
needs no AUTO installation.
"""

import sys
from pathlib import Path

import jax


jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_enable_x64", True)

from deep_isochron.analysis.ode.auto import write_problem
from deep_isochron.experiment import isochron_config
from deep_isochron.experiment.instantiate import instantiate
from hydra import compose, initialize_config_dir


CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def main(argv: list[str]) -> None:
    """Write the files for ``argv[0]`` into ``argv[1]`` with the overrides ``argv[2:]``
    and print the slot table."""
    if len(argv) < 2:
        sys.exit(__doc__)
    name, out, overrides = argv[0], Path(argv[1]), argv[2:]
    with initialize_config_dir(config_dir=str(CONFIGS), version_base=None):
        config = isochron_config(compose(config_name=name, overrides=overrides))
    system = instantiate(config.data["system"])
    problem = write_problem(system, out, tolerances=config.settings.tolerances)
    sha = problem.fortran_sha[:12]
    print(f"{type(system).__name__} -> {problem.workdir}  (fortran_sha {sha})")
    for filename in (
        f"{problem.name}.f90",
        "rhs.inc",
        "indices.inc",
        f"c.{problem.name}",
    ):
        print(f"  {filename}")
    print("\nslot  name                   value")
    for slot, parameter in problem.parameters.items():
        print(f"{slot:4d}  {parameter:22} {problem.pars[slot]:g}")
    for fortran, slot in problem.idx.fortran_constants().items():
        print(f"{slot:4d}  {fortran:22} set by the stages")


if __name__ == "__main__":
    main(sys.argv[1:])
