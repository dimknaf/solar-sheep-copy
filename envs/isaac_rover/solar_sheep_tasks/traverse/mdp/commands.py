"""envs/isaac_rover/solar_sheep_tasks/traverse/mdp/commands.py -- the rover's pose command.

Port of sample_target() and the reach/resample logic of envs/traverse/train.py onto Isaac Lab's 2-D
pose command (x, y, z, heading; the observation reads its body-frame form).

Targets are drawn 0.8-2.5 m from the ROBOT (so one episode chains several targets), within 3.4 m of the
env origin, with a uniformly random heading. Distances are xy only.
    rough task: from the tile's "target" flat patches (terrain_cfg.py). The patch test already rejects
                any spot on or next to an obstacle, so it stands in for the prototype's "best of 8 by
                obstacle clearance": pick uniformly among the patches in the distance ring, or, if the
                ring holds none, the patch closest to it.
    flat task:  a plane has no patches and no obstacles: the first of 8 uniform draws in the ring that
                lies within 3.4 m of the origin, or, if none does, a point in the ring toward the origin.
A target is reached at < 0.35 m and |heading error| < 0.5 rad; _update_metrics then zeroes time_left,
so CommandTerm.compute() resamples it on the same step. The resampling timer itself is set huge in the
config. Per-episode metric "targets_reached" (Metrics/pose_command/targets_reached) is the success
count the curriculum reads; the inherited "error_pos" / "error_heading" give the final distance.

Isaac Lab steps rewards BEFORE command_manager.compute(), so the reach reward (rewards.target_reached)
evaluates the same condition itself, on the same state, with the thresholds from this term's config.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.commands.pose_2d_command import UniformPose2dCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import RoverPose2dCommandCfg


class RoverPose2dCommand(UniformPose2dCommand):
    """Pose command that resamples on reach, around the robot. Terrain patches when present.

    Built on :class:`UniformPose2dCommand` rather than :class:`TerrainBasedPose2dCommand` because the
    latter raises without "target" patches, and the flat task's plane has none.
    """

    cfg: RoverPose2dCommandCfg

    def __init__(self, cfg: RoverPose2dCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.terrain = env.scene.terrain
        patches = self.terrain.flat_patches if self.terrain is not None else {}
        # (levels, types, patches, 3) world-frame points, or None on a plane
        self.valid_targets: torch.Tensor | None = patches.get("target")
        self.metrics["targets_reached"] = torch.zeros(self.num_envs, device=self.device)

    def _update_metrics(self):
        super()._update_metrics()          # error_pos (xy), error_heading (wrapped, absolute)
        reached = (self.metrics["error_pos"] < self.cfg.reach_radius) & (
            self.metrics["error_heading"] < self.cfg.reach_heading
        )
        self.metrics["targets_reached"] += reached.float()
        self.time_left[reached] = 0.0      # compute() subtracts dt and resamples everything <= 0

    def _resample_command(self, env_ids: Sequence[int]):
        n = len(env_ids)
        d_min, d_max = self.cfg.distance_range
        robot_xy = self.robot.data.root_pos_w.torch[env_ids, :2]
        origin = self._env.scene.env_origins[env_ids]
        if self.valid_targets is None:
            # 8 draws in the ring around the robot; keep the first inside max_target_radius of the origin.
            # (Pulling an outside draw back onto the rim could land it on the robot: a free reach.)
            k = 8
            r = torch.empty(n, k, device=self.device).uniform_(d_min, d_max)
            th = torch.empty(n, k, device=self.device).uniform_(-math.pi, math.pi)
            cand = robot_xy.unsqueeze(1) + torch.stack((r * torch.cos(th), r * torch.sin(th)), dim=-1)
            inside = torch.linalg.norm(cand - origin[:, None, :2], dim=-1) <= self.cfg.max_target_radius
            xy = cand[torch.arange(n, device=self.device), inside.float().argmax(dim=1)]
            # none inside (robot far out): a point in the ring straight toward the origin
            to_origin = origin[:, :2] - robot_xy
            toward = to_origin / torch.linalg.norm(to_origin, dim=1, keepdim=True).clamp(min=1e-6)
            xy = torch.where(inside.any(dim=1, keepdim=True), xy, robot_xy + toward * r[:, :1])
            z = origin[:, 2]
        else:
            cand = self.valid_targets[self.terrain.terrain_levels[env_ids], self.terrain.terrain_types[env_ids]]
            d = torch.linalg.norm(cand[..., :2] - robot_xy.unsqueeze(1), dim=-1)
            r_org = torch.linalg.norm(cand[..., :2] - origin[:, None, :2], dim=-1)
            miss = (d - d.clamp(d_min, d_max)).abs() + (r_org - self.cfg.max_target_radius).clamp(min=0.0)
            score = torch.where(miss == 0.0, 1.0 + torch.rand_like(d), -miss)
            best = cand[torch.arange(n, device=self.device), score.argmax(dim=1)]
            xy, z = best[:, :2], best[:, 2]
        self.pos_command_w[env_ids, :2] = xy
        self.pos_command_w[env_ids, 2] = z + self.robot.data.default_root_pose.torch[env_ids, 2]
        self.heading_command_w[env_ids] = torch.empty(n, device=self.device).uniform_(*self.cfg.ranges.heading)
