"""Generate a trajectory dataset from a Hydra config and save it as one netCDF4 file.

    uv run python scripts/generate_data.py --config-name fhn
    uv run python scripts/generate_data.py --config-name bautin n_trajectories=64 seed=3

The file lands at ``<out_dir>/<name>-<config_hash>.nc``; the hash is over the
generation-defining fields, so re-running an unchanged config overwrites the same file
and a changed one gets a new name. The resolved config is stored in the file's
``extra`` metadata.
"""

import hydra
import jax
import jax.numpy as jnp
from deep_isochron.data import dataset_path, generate
from omegaconf import DictConfig, OmegaConf


jax.config.update("jax_enable_x64", True)


@hydra.main(version_base=None, config_path="../configs/data", config_name="fhn")
def main(cfg: DictConfig) -> None:
    system = hydra.utils.instantiate(cfg.system)
    ic_sampler = hydra.utils.instantiate(cfg.ic_sampler)
    config = hydra.utils.instantiate(cfg.solver)
    ts = jnp.linspace(cfg.time.t0, cfg.time.t1, cfg.time.n)
    source = generate(
        system,
        ic_sampler,
        ts,
        cfg.n_trajectories,
        seed=cfg.seed,
        config=config,
        integration=cfg.integration,
        window_size=cfg.window_size,
        extra={"config": OmegaConf.to_container(cfg, resolve=True)},
    )
    assert source.metadata is not None
    path = source.save(dataset_path(cfg.out_dir, cfg.name, source.metadata))
    print(f"wrote {path}  ({source})")


if __name__ == "__main__":
    main()
