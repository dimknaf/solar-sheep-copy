"""envs/isaac_rover/solar_sheep_tasks/traverse/mdp/curriculums.py -- navigation terrain curriculum.

The prototype ramped one difficulty scalar with each env's own step count. Here difficulty is the
terrain row, moved per env on success, the way Isaac Lab's terrain_levels_vel does for locomotion:
at the end of an episode, >= 2 targets reached -> one row harder, 0 reached -> one row easier.
Runs before the command manager resets its metrics, so it sees the finished episode's count.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.terrains import TerrainImporter


def terrain_levels_nav(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], command_name: str, promote_at: int = 2, demote_at: int = 0
) -> torch.Tensor:
    """Promote/demote the resetting envs by targets reached; returns the mean terrain level."""
    terrain: TerrainImporter = env.scene.terrain
    reached = env.command_manager.get_term(command_name).metrics["targets_reached"][env_ids]
    ran = env.episode_length_buf[env_ids] > 0          # the very first reset has no episode behind it
    move_up = (reached >= promote_at) & ran
    move_down = (reached <= demote_at) & ran
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())
