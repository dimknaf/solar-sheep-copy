import torch
from isaaclab.app import AppLauncher
from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab_tasks.utils.parse_cfg import parse_env_cfg

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

env_cfg = parse_env_cfg("Isaac-Velocity-Flat-Anymal-C-v0", num_envs=1)
env = ManagerBasedRLEnv(cfg=env_cfg)

policy = torch.load("policy.pt")
for _ in range(100):
    action = policy(env.observation_manager.compute())
    env.step(action)

simulation_app.close()
