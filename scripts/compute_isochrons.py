"""Trace the isochrons of an isochron config by AUTO-07p and save them (ADR-0017).

    uv run python scripts/compute_isochrons.py --config-name isochrons/fhn
    uv run python scripts/compute_isochrons.py --config-name isochrons/fhn-easy \\
        phases.count=20
    uv run python scripts/compute_isochrons.py --config-name isochrons/fhn \\
        -m hydra/launcher=slurm hydra.launcher.partition=<name>   # one SLURM job

The foliation lands at ``<out_dir>/<name>-<hash>.nc`` (``IsochronConfig.file``);
``experiment.load_isochrons(cfg)`` reads it back with the same config. When the file
exists, the script prints its path and exits, since the result would be the same; delete
it to recompute. When some phases fail, the file is saved with them recorded, they are
printed, and the exit status is 1. JAX runs on the CPU: nothing in the pipeline gains
from a GPU, and a GPU backend would preallocate most of the device's memory.
"""

import logging
import sys

import jax


jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_enable_x64", True)

import hydra
from deep_isochron.experiment import compute_isochrons, isochron_config
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="isochrons/fhn")
def main(cfg: DictConfig) -> None:
    """Trace and save the foliation of `cfg`, unless its file exists."""
    for name in ("absl", "jax"):
        logging.getLogger(name).setLevel(logging.WARNING)
    config = isochron_config(cfg)
    path = config.file()
    if path.exists():
        print(f"{path} exists; nothing to do (delete it to recompute)")
        return
    foliation = compute_isochrons(config)
    print(f"wrote {path}: {len(foliation.isochrons)} isochrons")
    if foliation.failures:
        for failure in foliation.failures:
            print(
                f"failed phase {failure.phase:+.6f}: {failure.message}"
                + (f" (files kept in {failure.workdir})" if failure.workdir else "")
            )
        sys.exit(1)


if __name__ == "__main__":
    main()
