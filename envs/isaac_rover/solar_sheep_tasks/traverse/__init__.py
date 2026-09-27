"""envs/isaac_rover/solar_sheep_tasks/traverse -- the rover's traverse skill: go to (x, y, yaw), avoid, don't tip.

Registers the two Gym ids Isaac Lab's train/play CLIs take as --task. Entry points are strings, so
registering imports no Isaac Lab module; the configs load when a CLI resolves the task.
"""

import gymnasium as gym

from . import agents

for _task, _env_cfg, _agent_cfg in (
    ("SolarSheep-Traverse-Flat-Rover", "TraverseFlatEnvCfg", "TraverseFlatPPORunnerCfg"),
    ("SolarSheep-Traverse-Rough-Rover", "TraverseRoughEnvCfg", "TraverseRoughPPORunnerCfg"),
):
    if _task not in gym.registry:          # importing twice (e.g. by a second launcher) must not raise
        gym.register(
            id=_task,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.traverse_env_cfg:{_env_cfg}",
                "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:{_agent_cfg}",
                "default_agent": "rsl_rl",
            },
        )
