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

"""Reusable RobotState builders and one motion-generation controller step."""

import isaacsim.robot_motion.experimental.motion_generation as mg
import warp as wp


def estimated_state(robot):
    """Build a name-addressed state from an experimental Articulation."""
    names = robot.dof_names
    return mg.RobotState(
        joints=mg.JointState.from_name(
            robot_joint_space=names,
            positions=(names, robot.get_dof_positions()),
            velocities=(names, robot.get_dof_velocities()),
        )
    )


def setpoint_state(site_space, tool_frame, position, orientation_wxyz):
    """Build a single-site setpoint; orientation must use WXYZ ordering."""
    return mg.RobotState(
        sites=mg.SpatialState.from_name(
            spatial_space=site_space,
            positions=([tool_frame], wp.array([position.tolist()], dtype=wp.float32)),
            orientations=([tool_frame], wp.array([orientation_wxyz.tolist()], dtype=wp.float32)),
        )
    )


def apply_desired_joint_state(robot, desired):
    """Apply every joint target supplied by a controller result."""
    if desired is None or desired.joints is None:
        return
    joints = desired.joints
    if joints.positions is not None:
        robot.set_dof_position_targets(joints.positions, dof_indices=joints.position_indices)
    if joints.velocities is not None:
        robot.set_dof_velocity_targets(joints.velocities, dof_indices=joints.velocity_indices)
    if joints.efforts is not None:
        robot.set_dof_efforts(joints.efforts, dof_indices=joints.effort_indices)
