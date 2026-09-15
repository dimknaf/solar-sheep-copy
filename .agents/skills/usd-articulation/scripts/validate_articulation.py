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

"""Validate multi-link robot USD articulation structure.

Checks: exactly one ArticulationRootAPI, connected joint hierarchy under that
root, at least one chassis-to-child attachment before flatten-before-deploy,
and Robot Schema overlay presence.

Run with: $ISAAC_SIM_DIR/python.sh validate_articulation.py <path/to/robot.usd>
"""

import sys
from collections import defaultdict, deque


def _ancestor_paths(path):
    """Yield a prim path and all ancestors as strings."""
    current = path
    while current and str(current) != ".":
        yield str(current)
        current = current.GetParentPath()


def _build_joint_graph(stage, usd_physics):
    """Build undirected connectivity data from physics joints."""
    graph = defaultdict(set)
    edges = []
    joints = [prim for prim in stage.Traverse() if prim.IsA(usd_physics.Joint)]
    for joint_prim in joints:
        joint = usd_physics.Joint(joint_prim)
        body0_targets = joint.GetBody0Rel().GetTargets()
        body1_targets = joint.GetBody1Rel().GetTargets()
        if not body0_targets or not body1_targets:
            continue
        body0_path = str(body0_targets[0])
        body1_path = str(body1_targets[0])
        graph[body0_path].add(body1_path)
        graph[body1_path].add(body0_path)
        edges.append((joint_prim, body0_path, body1_path))
    return joints, graph, edges


def _find_connected_bodies(graph, root_candidates):
    """Return the set of bodies reachable from the articulation root branch."""
    visited = set()
    queue = deque(node for node in root_candidates if node in graph)
    visited.update(queue)
    while queue:
        current = queue.popleft()
        for neighbor in graph[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return visited


def validate_articulation(usd_path: str) -> bool:
    """Validate articulation root count, hierarchy connectivity, and Robot Schema.

    Args:
        usd_path: Path to the robot USD file.

    Returns:
        True if all checks pass.

    Raises:
        AssertionError: If the articulated hierarchy is not deployment-safe.
    """
    from pxr import Usd, UsdPhysics
    from usd.schema.isaac.robot_schema import Attributes, Classes, GetAllNamedPoses

    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise ValueError(f"failed to open USD stage: {usd_path}")

    art_roots = [p.GetPath() for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    print(f"ArticulationRootAPI count: {len(art_roots)} -- {art_roots}")
    assert len(art_roots) == 1, "must be exactly 1 articulation root"

    articulation_root = art_roots[0]
    root_candidates = set(_ancestor_paths(articulation_root))
    joints, graph, edges = _build_joint_graph(stage, UsdPhysics)
    connected_bodies = _find_connected_bodies(graph, root_candidates)
    all_bodies = set(graph)
    disconnected_bodies = sorted(all_bodies - connected_bodies)

    direct_fixed_children = set()
    for joint_prim, body0_path, body1_path in edges:
        if not joint_prim.IsA(UsdPhysics.FixedJoint):
            continue
        body0_is_root_side = any(
            body0_path == candidate or body0_path.startswith(f"{candidate}/") for candidate in root_candidates
        )
        body1_is_root_side = any(
            body1_path == candidate or body1_path.startswith(f"{candidate}/") for candidate in root_candidates
        )
        if body0_is_root_side and not body1_is_root_side:
            direct_fixed_children.add(body1_path)
        if body1_is_root_side and not body0_is_root_side:
            direct_fixed_children.add(body0_path)

    print(f"Joints: {len(joints)} | Bodies in joint graph: {len(all_bodies)}")
    print(f"Connected bodies from articulation root: {sorted(connected_bodies)}")
    print(f"Direct FixedJoint children from chassis/root branch: {sorted(direct_fixed_children)}")

    assert joints, "must define at least one physics joint before deployment"
    assert connected_bodies, "articulation root is not connected to any joint bodies"
    assert not disconnected_bodies, f"disconnected joint bodies found: {disconnected_bodies}"
    assert direct_fixed_children, "missing root-to-child FixedJoint attachment before flatten-before-deploy"

    # Robot Schema overlay
    robots = [p for p in stage.Traverse() if p.HasAPI(Classes.ROBOT_API)]
    print(f"IsaacRobotAPI count: {len(robots)}")
    for r in robots:
        rt = r.GetAttribute(Attributes.ROBOT_TYPE).Get()
        poses = GetAllNamedPoses(stage, r)
        print(f"  {r.GetPath()}: robot_type={rt!r}  named_poses={list(poses)}")

    if robots:
        robot_links = robots[0].GetRelationship(Attributes.ROBOT_LINKS).GetTargets()
        robot_joints = robots[0].GetRelationship(Attributes.ROBOT_JOINTS).GetTargets()
        print(f"  robot_links={list(robot_links)}")
        print(f"  robot_joints={list(robot_joints)}")
        assert robot_links, "Robot Schema missing ROBOT_LINKS relation targets"
        assert robot_joints, "Robot Schema missing ROBOT_JOINTS relation targets"

    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_articulation.py <robot.usd>")
        sys.exit(1)
    validate_articulation(sys.argv[1])
