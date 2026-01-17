from __future__ import annotations

import hydra
import ray
from omegaconf import OmegaConf

from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.main_ppo import TaskRunner, run_ppo
from verl.experimental.simple_rl_distill.ray_trainer import SimpleRLDistillRayPPOTrainer


@hydra.main(config_path="../../trainer/config", config_name="ppo_trainer", version_base=None)
def main(config):
    from verl.utils.config import validate_config
    from verl.utils.device import auto_set_device

    auto_set_device(config)
    validate_config(config)

    # Patch trainer class used inside TaskRunner by monkeypatching import target.
    # This keeps all dataset/reward wiring identical to verl.trainer.main_ppo.
    import verl.trainer.main_ppo as main_ppo_mod

    main_ppo_mod.RayPPOTrainer = SimpleRLDistillRayPPOTrainer  # type: ignore[attr-defined]

    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    task_runner_class = ray.remote(num_cpus=1)(TaskRunner)
    runner = task_runner_class.remote()
    ray.get(runner.run.remote(config))


if __name__ == "__main__":
    main()

