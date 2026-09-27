"""envs/isaac_rover/solar_sheep_tasks/traverse/mdp/observations.py -- observation terms Isaac Lab lacks.

Each one reproduces a block of observe() in envs/traverse/train.py; the rest of the 32 floats come
straight from isaaclab.envs.mdp (see ObservationsCfg in traverse_env_cfg.py for the full order).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import RayCaster


def slope_b(
    env: ManagerBasedRLEnv, max_slope_deg: float = 12.0, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Downhill direction in the body frame, sin(slope) * unit vector, / sin(12 deg). Shape (N, 2).

    What an IMU reads on a slope: the xy part of gravity in the body frame. The prototype fed the same
    quantity for its simulated slope force (slope_b / sin(SLOPE_MAX_DEG)); here the slope is real terrain.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.projected_gravity_b.torch[:, :2] / math.sin(math.radians(max_slope_deg))


def target_offset_b(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Target dx, dy / 2 m, clipped to +-2. Shape (N, 2).

    The pose command's own body-frame offset, which Isaac Lab rotates by yaw only; the prototype used the
    full attitude. Identical on level ground, and a few percent apart at the 12 deg slope limit.
    """
    command = env.command_manager.get_command(command_name)
    return torch.clamp(command[:, :2] / 2.0, -2.0, 2.0)


def heading_error_sin_cos(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """sin and cos of wrap(target heading - heading). Shape (N, 2)."""
    heading_error = env.command_manager.get_command(command_name)[:, 3]
    return torch.stack((torch.sin(heading_error), torch.cos(heading_error)), dim=1)


def rangefinders(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, max_distance: float = 3.0) -> torch.Tensor:
    """Range along each ray / max_distance; 1.0 = nothing within range. Shape (N, rays).

    The RayCaster reports hit points (inf on a miss) and the sensor pose, not ranges. For
    ray_alignment="base" every ray starts at pos_w + quat_w * ray_starts (the pattern offset lives in
    ray_starts), so the range is the distance from that start to the hit.
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    data = sensor.data                                    # refreshes the sensor if outdated
    local_starts = sensor.ray_starts.torch                # (N, B, 3), sensor frame
    quat = data.quat_w.torch.unsqueeze(1).expand(-1, local_starts.shape[1], -1)
    starts = data.pos_w.torch.unsqueeze(1) + quat_apply(quat, local_starts)
    ranges = torch.linalg.norm(data.ray_hits_w.torch - starts, dim=-1)
    ranges = torch.nan_to_num(ranges, nan=max_distance, posinf=max_distance).clamp(0.0, max_distance)
    return ranges / max_distance
