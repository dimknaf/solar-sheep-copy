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

"""Build and synchronize a cuMotion obstacle world binding."""

import isaacsim.robot_motion.experimental.motion_generation as mg
from isaacsim.core.experimental.objects import Cone, Cylinder, Mesh
from isaacsim.robot_motion.cumotion import CumotionWorldInterface


def build_world_binding(robot, robot_path, excluded_prim_paths=()):
    """Track nearby collision prims, excluding the robot and grasped objects."""
    robot_pos, robot_orientation = robot.get_world_poses()
    tracked_prims = mg.SceneQuery().get_prims_in_aabb(
        search_box_origin=robot_pos.numpy()[0],
        search_box_minimum=[-10.0, -10.0, -10.0],
        search_box_maximum=[10.0, 10.0, 10.0],
        tracked_api=mg.TrackableApi.PHYSICS_COLLISION,
        exclude_prim_paths=[robot_path, *excluded_prim_paths],
    )
    strategy = mg.ObstacleStrategy()
    for prim_type in (Mesh, Cone, Cylinder):
        strategy.set_default_configuration(prim_type, mg.ObstacleConfiguration("obb", 0.01))
    binding = mg.WorldBinding(
        world_interface=CumotionWorldInterface(),
        obstacle_strategy=strategy,
        tracked_prims=tracked_prims,
        tracked_collision_api=mg.TrackableApi.PHYSICS_COLLISION,
    )
    binding.initialize()
    binding.get_world_interface().update_world_to_robot_root_transforms((robot_pos, robot_orientation))
    binding.synchronize_transforms()
    return binding


def synchronize_world_binding(binding, robot):
    """Synchronize robot-root transforms once per controller frame."""
    binding.get_world_interface().update_world_to_robot_root_transforms(robot.get_world_poses())
    binding.synchronize_transforms()
