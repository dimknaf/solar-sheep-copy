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

"""Construct a cuMotion RMPflow controller from a bound world."""

from isaacsim.robot_motion.cumotion import RmpFlowController, load_cumotion_supported_robot


def build_rmpflow_controller(robot, world_binding, robot_name="ur10"):
    """Return ``(controller, tool_frames, tool_frame)`` for a supported robot."""
    cumotion_robot = load_cumotion_supported_robot(robot_name)
    tool_frames = cumotion_robot.robot_description.tool_frame_names()
    if not tool_frames:
        raise RuntimeError("No cuMotion tool frames found.")
    tool_frame = tool_frames[0]
    controller = RmpFlowController(
        cumotion_robot=cumotion_robot,
        cumotion_world_interface=world_binding.get_world_interface(),
        robot_joint_space=robot.dof_names,
        robot_site_space=tool_frames,
        tool_frame=tool_frame,
    )
    controller.get_rmp_flow_config().set_param("cspace_target_rmp/metric_scalar", 1.0)
    return controller, tool_frames, tool_frame
