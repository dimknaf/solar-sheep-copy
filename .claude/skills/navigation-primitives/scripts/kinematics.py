# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Differential-drive and holonomic kinematics helpers for path following.

Provides pure-geometry transforms (body twist ↔ wheel velocities) and a
pure-pursuit path-following controller suitable for Carter and Kaya robots.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class DifferentialParams:
    """Parameters for a two-wheel differential-drive robot (e.g. Nova Carter)."""

    wheel_radius: float = 0.14
    track_width: float = 0.499
    max_linear_speed: float = 1.5
    max_angular_speed: float = 1.5


@dataclass
class HolonomicParams:
    """Parameters for a 3-wheel omnidirectional robot (e.g. Kaya)."""

    wheel_radius: float = 0.04
    wheel_base: float = 0.10
    max_linear_speed: float = 0.5
    max_angular_speed: float = 2.0


def differential_forward(linear: float, angular: float, params: DifferentialParams) -> tuple:
    """Convert body twist (v, ω) to left/right wheel angular velocities.

    Returns:
        (vL, vR) in rad/s.
    """
    linear = np.clip(linear, -params.max_linear_speed, params.max_linear_speed)
    angular = np.clip(angular, -params.max_angular_speed, params.max_angular_speed)

    v_left = (linear - angular * params.track_width / 2.0) / params.wheel_radius
    v_right = (linear + angular * params.track_width / 2.0) / params.wheel_radius
    return (float(v_left), float(v_right))


def differential_inverse(v_left: float, v_right: float, params: DifferentialParams) -> tuple:
    """Convert wheel angular velocities to body twist (v, ω).

    Returns:
        (linear, angular) — linear in m/s, angular in rad/s.
    """
    linear = params.wheel_radius * (v_left + v_right) / 2.0
    angular = params.wheel_radius * (v_right - v_left) / params.track_width
    return (float(linear), float(angular))


def holonomic_forward(vx: float, vy: float, omega: float, params: HolonomicParams) -> np.ndarray:
    """Convert body twist (vx, vy, ω) to 3 mecanum/omni wheel velocities.

    Assumes 120°-spaced wheels (Kaya layout). Wheel 0 at front, 1 at back-left,
    2 at back-right.

    Returns:
        Array of shape (3,) — wheel angular velocities in rad/s.
    """
    vx = np.clip(vx, -params.max_linear_speed, params.max_linear_speed)
    vy = np.clip(vy, -params.max_linear_speed, params.max_linear_speed)
    omega = np.clip(omega, -params.max_angular_speed, params.max_angular_speed)

    angles = np.array([np.pi / 2, np.pi / 2 + 2 * np.pi / 3, np.pi / 2 + 4 * np.pi / 3])
    wheel_vels = np.zeros(3)
    for i, theta in enumerate(angles):
        wheel_vels[i] = -np.sin(theta) * vx + np.cos(theta) * vy + params.wheel_base * omega
    return wheel_vels / params.wheel_radius


def holonomic_inverse(wheel_vels: np.ndarray, params: HolonomicParams) -> tuple:
    """Convert 3 wheel angular velocities back to body twist (vx, vy, ω).

    Returns:
        (vx, vy, omega).
    """
    angles = np.array([np.pi / 2, np.pi / 2 + 2 * np.pi / 3, np.pi / 2 + 4 * np.pi / 3])
    J = np.zeros((3, 3))
    for i, theta in enumerate(angles):
        J[i] = [-np.sin(theta), np.cos(theta), params.wheel_base]
    J /= params.wheel_radius

    body_vel = np.linalg.lstsq(J, wheel_vels, rcond=None)[0]
    return (float(body_vel[0]), float(body_vel[1]), float(body_vel[2]))


def normalize_angle(angle: float) -> float:
    """Wrap angle to [-π, π]."""
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


@dataclass
class PurePursuitConfig:
    """Pure-pursuit path-following controller configuration."""

    lookahead_distance: float = 1.0
    linear_speed: float = 1.0
    goal_tolerance: float = 0.5
    slowdown_distance: float = 2.0
    max_angular_speed: float = 1.5
    kp_angular: float = 2.5
    kd_angular: float = 1.2


def pure_pursuit_step(
    robot_pos: np.ndarray,
    robot_yaw: float,
    path: np.ndarray,
    config: PurePursuitConfig,
    prev_heading_error: float = 0.0,
) -> dict:
    """Compute one step of pure-pursuit for a differential-drive robot.

    Args:
        robot_pos: Current robot (x, y) position.
        robot_yaw: Current heading in radians.
        path: Array of shape (N, 2) — waypoints in world frame.
        config: Controller parameters.
        prev_heading_error: Previous heading error for derivative term.

    Returns:
        Dict with keys: linear, angular, heading_error, target_idx, done.
    """
    if len(path) == 0:
        return {"linear": 0.0, "angular": 0.0, "heading_error": 0.0, "target_idx": 0, "done": True}

    dists = np.linalg.norm(path - robot_pos[:2], axis=1)
    goal_dist = dists[-1]

    if goal_dist < config.goal_tolerance:
        return {"linear": 0.0, "angular": 0.0, "heading_error": 0.0, "target_idx": len(path) - 1, "done": True}

    # Find lookahead point: first waypoint beyond lookahead distance from robot
    target_idx = 0
    for i in range(len(path)):
        if dists[i] >= config.lookahead_distance:
            target_idx = i
            break
    else:
        target_idx = len(path) - 1

    target = path[target_idx]
    dx = target[0] - robot_pos[0]
    dy = target[1] - robot_pos[1]
    desired_yaw = np.arctan2(dy, dx)

    heading_error = normalize_angle(desired_yaw - robot_yaw)

    # PD angular control
    angular = config.kp_angular * heading_error + config.kd_angular * (heading_error - prev_heading_error)
    angular = np.clip(angular, -config.max_angular_speed, config.max_angular_speed)

    # Linear speed with slowdown near goal and during large heading errors
    speed = config.linear_speed
    if goal_dist < config.slowdown_distance:
        speed *= goal_dist / config.slowdown_distance
    heading_factor = max(0.0, 1.0 - abs(heading_error) / (np.pi / 2))
    speed *= heading_factor

    return {
        "linear": float(speed),
        "angular": float(angular),
        "heading_error": float(heading_error),
        "target_idx": int(target_idx),
        "done": False,
    }


def holonomic_path_step(
    robot_pos: np.ndarray,
    robot_yaw: float,
    path: np.ndarray,
    config: PurePursuitConfig,
) -> dict:
    """Compute one step for a holonomic robot — moves directly toward target.

    Unlike differential drive, the holonomic robot can strafe, so linear velocity
    is decomposed into body-frame vx/vy and heading tracks the path tangent.

    Returns:
        Dict with keys: vx, vy, omega, target_idx, done.
    """
    if len(path) == 0:
        return {"vx": 0.0, "vy": 0.0, "omega": 0.0, "target_idx": 0, "done": True}

    dists = np.linalg.norm(path - robot_pos[:2], axis=1)
    goal_dist = dists[-1]

    if goal_dist < config.goal_tolerance:
        return {"vx": 0.0, "vy": 0.0, "omega": 0.0, "target_idx": len(path) - 1, "done": True}

    target_idx = 0
    for i in range(len(path)):
        if dists[i] >= config.lookahead_distance:
            target_idx = i
            break
    else:
        target_idx = len(path) - 1

    target = path[target_idx]
    dx = target[0] - robot_pos[0]
    dy = target[1] - robot_pos[1]
    dist_to_target = np.hypot(dx, dy)

    # World-frame unit direction to target
    world_vx = dx / max(dist_to_target, 1e-6)
    world_vy = dy / max(dist_to_target, 1e-6)

    # Transform to body frame
    cos_yaw = np.cos(robot_yaw)
    sin_yaw = np.sin(robot_yaw)
    body_vx = cos_yaw * world_vx + sin_yaw * world_vy
    body_vy = -sin_yaw * world_vx + cos_yaw * world_vy

    speed = config.linear_speed
    if goal_dist < config.slowdown_distance:
        speed *= goal_dist / config.slowdown_distance

    body_vx *= speed
    body_vy *= speed

    # Align heading to path tangent
    desired_yaw = np.arctan2(dy, dx)
    heading_error = normalize_angle(desired_yaw - robot_yaw)
    omega = np.clip(config.kp_angular * heading_error, -config.max_angular_speed, config.max_angular_speed)

    return {
        "vx": float(body_vx),
        "vy": float(body_vy),
        "omega": float(omega),
        "target_idx": int(target_idx),
        "done": False,
    }
