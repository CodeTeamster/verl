"""Entrypoint for standalone rollout evaluation."""

from __future__ import annotations

import ray
import hydra
from omegaconf import OmegaConf

from verl.evaluator.evaluator import RolloutEvaluator, prepare_evaluator_config
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env


def _init_ray(config) -> bool:
    """Initialize an evaluator-owned local Ray runtime when needed."""
    if ray.is_initialized():
        return False

    ray_init = config.ray_kwargs.get("ray_init", {})
    runtime_env = OmegaConf.merge(
        get_ppo_ray_runtime_env(),
        ray_init.get("runtime_env", {}),
    )
    ray_init = OmegaConf.create({**ray_init, "runtime_env": runtime_env})
    ray.init(**OmegaConf.to_container(ray_init, resolve=True))
    return True


@hydra.main(config_path="config", config_name="alfworld", version_base=None)
def main(config) -> None:
    config = prepare_evaluator_config(config)
    owns_ray_runtime = _init_ray(config)
    try:
        RolloutEvaluator(config).run()
    finally:
        if owns_ray_runtime:
            ray.shutdown()


if __name__ == "__main__":
    main()
