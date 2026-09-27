"""envs/isaac_rover/solar_sheep_tasks/traverse/mdp/rewards.py -- the prototype's reward terms, unweighted.

Port of the reward block of envs/traverse/train.py step(). Weights live in RewardsCfg; Isaac Lab's
reward manager multiplies every term by weight * step_dt, so REWARD_W carries over as relative weights.
Terms Isaac Lab already has (alive, action_l2, action_rate_l2, is_terminated_term, undesired_contacts)
are used from isaaclab.envs.mdp.

Target terms read the command's WORLD-frame target and the robot's CURRENT pose. Rewards run before
command_manager.compute(), so the command's body-frame buffer is one step old at this point; the
world-frame target is not (it only changes on resample, which happens after the rewards).

Every term here returns through _finite(): rewards are computed on the step's state BEFORE the
terminated envs reset, so an env whose solve blew up (terminations.root_state_nonfinite) would put NaN
into the reward buffer and the whole PPO update. The prototype did the same (nan_to_num on the reward).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv

    from .commands import RoverPose2dCommand


def _finite(x: torch.Tensor) -> torch.Tensor:
    """NaN / +-inf -> 0: a blown-up env earns nothing on its last step instead of poisoning the batch."""
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _target_error(env: ManagerBasedRLEnv, command_name: str):
    """(command term, xy offset to the target, xy distance, wrapped heading error)."""
    term: RoverPose2dCommand = env.command_manager.get_term(command_name)
    offset = term.pos_command_w[:, :2] - term.robot.data.root_pos_w.torch[:, :2]
    heading_error = wrap_to_pi(term.heading_command_w - term.robot.data.heading_w.torch)
    return term, offset, torch.linalg.norm(offset, dim=1), heading_error


def progress_to_target(env: ManagerBasedRLEnv, command_name: str, max_speed: float = 0.5) -> torch.Tensor:
    """Closing speed on the target [m/s], clipped to +-max_speed. THE driving signal (MJX progress).

    Stateless form of (prev_dist - dist) / dt: the xy velocity projected on the unit vector to the target.
    """
    term, offset, dist, _ = _target_error(env, command_name)
    velocity = term.robot.data.root_lin_vel_w.torch[:, :2]
    closing = (velocity * offset).sum(dim=1) / dist.clamp(min=1e-6)
    return _finite(closing.clamp(-max_speed, max_speed))


def position_tracking(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """exp(-(d / std)^2) on the xy distance to the target (MJX pos, std 0.6)."""
    _, _, dist, _ = _target_error(env, command_name)
    return _finite(torch.exp(-((dist / std) ** 2)))


def heading_tracking(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """0.5 (1 + cos heading error), gated by exp(-(d / std)^2) so it only pays near the target (MJX head)."""
    _, _, dist, heading_error = _target_error(env, command_name)
    return _finite(0.5 * (1.0 + torch.cos(heading_error)) * torch.exp(-((dist / std) ** 2)))


class target_potential(ManagerTermBase):
    """Potential-based form of the prototype's pos + head terms: gamma * Phi(s') - Phi(s).

    Phi = w_pos exp(-(d/std_pos)^2) + w_head 0.5 (1 + cos he) exp(-(d/std_head)^2), i.e. exactly the
    prototype's pos + head. Paid as a DIFFERENCE, parking near the target earns nothing, so the policy
    cannot farm it by stopping short or staying misaligned. As a per-step value, pos + head made hovering
    at 0.1 m worth ~478 against ~251 for completing (G3 review, 27 Sep). Potential-based shaping
    (Ng, Harada & Russell 1999) leaves the optimal policy unchanged. The term is 0 on the first step of
    every target (reset or resample), and the value is divided by step_dt to cancel the reward manager's
    weight * dt.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._prev = torch.zeros(env.num_envs, device=env.device)
        self._counter = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)

    def reset(self, env_ids=None):
        self._counter[slice(None) if env_ids is None else env_ids] = -1

    def __call__(
        self, env: ManagerBasedRLEnv, command_name: str, gamma: float = 0.995,
        w_pos: float = 1.0, std_pos: float = 0.6, w_head: float = 1.5, std_head: float = 0.8,
    ) -> torch.Tensor:
        term, _, dist, heading_error = _target_error(env, command_name)
        phi = (w_pos * torch.exp(-((dist / std_pos) ** 2))
               + w_head * 0.5 * (1.0 + torch.cos(heading_error)) * torch.exp(-((dist / std_head) ** 2)))
        counter = term.command_counter.to(self._counter.dtype)
        fresh = counter != self._counter                    # reset or new target since the last call
        out = torch.where(fresh, torch.zeros_like(phi), gamma * phi - self._prev)
        self._prev = phi.detach().clone()
        self._counter = counter.clone()
        return _finite(out / env.step_dt)


def target_reached(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """1 on the step the target is reached (MJX reach). Same test and thresholds as the command's resample."""
    term, _, dist, heading_error = _target_error(env, command_name)
    return _finite(((dist < term.cfg.reach_radius) & (heading_error.abs() < term.cfg.reach_heading)).float())


def wheel_power(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """mean(|action| * |wheel speed| / pi): normalised mechanical power (MJX energy).

    ``asset_cfg`` must list the wheel joints in action order (fl, fr, rl, rr, preserve_order=True).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    wheel_speed = asset.data.joint_vel.torch[:, asset_cfg.joint_ids]
    return _finite(torch.mean(env.action_manager.action.abs() * wheel_speed.abs() / torch.pi, dim=1))


def tilt(env: ManagerBasedRLEnv, soft_limit: float = 0.9, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """max(0, soft_limit - up_z), up_z = body up-vector z = -projected gravity z (MJX tip)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return _finite((soft_limit + asset.data.projected_gravity_b.torch[:, 2]).clamp(min=0.0))


def spin(env: ManagerBasedRLEnv, max_rate: float = 0.5, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """max(0, |body yaw rate| - max_rate)^2 (MJX spin). The measured turn rate is 0.571 rad/s."""
    asset: Articulation = env.scene[asset_cfg.name]
    return _finite(((asset.data.root_ang_vel_b.torch[:, 2].abs() - max_rate).clamp(min=0.0)) ** 2)


def idle_far_from_target(
    env: ManagerBasedRLEnv, command_name: str, min_speed: float = 0.05, min_distance: float = 0.8
) -> torch.Tensor:
    """1 when parked (planar speed < min_speed) while still > min_distance from the target (MJX idle)."""
    term, _, dist, _ = _target_error(env, command_name)
    speed = torch.linalg.norm(term.robot.data.root_lin_vel_b.torch[:, :2], dim=1)
    return _finite(((speed < min_speed) & (dist > min_distance)).float())


def out_of_bounds(env: ManagerBasedRLEnv, radius: float = 4.0, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Metres beyond ``radius`` from the env origin, xy (MJX bounds)."""
    asset: Articulation = env.scene[asset_cfg.name]
    offset = asset.data.root_pos_w.torch[:, :2] - env.scene.env_origins[:, :2]
    return _finite((torch.linalg.norm(offset, dim=1) - radius).clamp(min=0.0))
