"""Generate a trajectory dataset from a Hydra config and save it as one netCDF4 file.

    uv run python scripts/generate_data.py --config-name fhn
    uv run python scripts/generate_data.py --config-name bautin n_trajectories=64 seed=3

The file lands at ``<out_dir>/<name>-<config_hash>.nc``; the hash is over the
generation-defining fields, so re-running an unchanged config overwrites the same file
and a changed one gets a new name, and a training run with the same ``data`` config
finds it (``experiment.build_source``). The resolved config is stored in the file's
``extra`` metadata.
"""

import hydra
import jax
from deep_isochron.experiment import generate_dataset
from omegaconf import DictConfig


jax.config.update("jax_enable_x64", True)


@hydra.main(version_base=None, config_path="../configs/data", config_name="fhn")
def main(cfg: DictConfig) -> None:
    path = generate_dataset(cfg)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
