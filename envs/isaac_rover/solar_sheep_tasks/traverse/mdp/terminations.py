"""envs/isaac_rover/solar_sheep_tasks/traverse/mdp/terminations.py -- the prototype's NaN guard.

Time-out and tip-over come from isaaclab.envs.mdp (time_out, bad_orientation). There is deliberately
no collision termination: the prototype penalises contact instead (see RewardsCfg.collide).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


def root_state_nonfinite(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """End the episode if the root pose or velocity is no longer finite (a blown-up solve)."""
    asset: Articulation = env.scene[asset_cfg.name]
    data = asset.data
    state = torch.cat(
        (data.root_pos_w.torch, data.root_quat_w.torch, data.root_lin_vel_w.torch, data.root_ang_vel_w.torch), dim=1
    )
    return ~torch.isfinite(state).all(dim=1)
