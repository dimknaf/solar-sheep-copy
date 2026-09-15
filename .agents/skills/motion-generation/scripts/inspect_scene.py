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

"""Inspect motion-generation scene state in a running Isaac Sim Python server.

Generic scene inspection (robot DOFs/links/poses, object AABB/schemas/velocities,
prim matching, markers, camera). The optional ``supported_robot`` arg adds a
cuMotion-specific probe (tool frames, controlled joints, config dir).

Injected globals via ``isaacsim_send.py --arg``:
    usd_path: optional USD path to open before inspection.
    root_path: root prim to traverse, default "/".
    robot_path: optional articulation path to inspect.
    object_path: optional manipulated object path to inspect.
    supported_robot: optional cuMotion supported robot name, e.g. "ur10" or "franka".
    prim_path: optional exact prim path to inspect.
    match: optional substring or semicolon/comma-separated substrings to match prim paths.
    max_results: maximum matched prims to print, default 80.
    create_markers: add debug spheres at matched/object AABB centers, default False.
    marker_points: optional "name:x,y,z;name2:x,y,z" marker list.
    marker_scope: marker root, default "/World/CumotionProbeMarkers".
    camera_eye: optional "x,y,z".
    camera_target: optional "x,y,z".
"""

if "usd_path" not in dir():
    usd_path = None
if "root_path" not in dir():
    root_path = "/"
if "robot_path" not in dir():
    robot_path = None
if "object_path" not in dir():
    object_path = None
if "supported_robot" not in dir():
    supported_robot = None
if "prim_path" not in dir():
    prim_path = None
if "match" not in dir():
    match = None
if "max_results" not in dir():
    max_results = 80
if "create_markers" not in dir():
    create_markers = False
if "marker_points" not in dir():
    marker_points = ""
if "marker_scope" not in dir():
    marker_scope = "/World/CumotionProbeMarkers"
if "marker_radius" not in dir():
    marker_radius = 0.025
if "camera" not in dir():
    camera = "/OmniverseKit_Persp"
if "camera_eye" not in dir():
    camera_eye = None
if "camera_target" not in dir():
    camera_target = None

import re

import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import numpy as np
from isaacsim.core.experimental.prims import Articulation, RigidPrim
from isaacsim.core.rendering_manager import ViewportManager
from isaacsim.robot_motion.cumotion import load_cumotion_supported_robot
from pxr import Gf, Sdf, Usd, UsdGeom


def _as_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "on"}


def _vector(value):
    if value is None or value == "":
        return None
    if isinstance(value, str):
        return [float(item.strip()) for item in value.split(",")]
    return [float(item) for item in value]


def _tokens(value):
    if value is None:
        return []
    if isinstance(value, str):
        parts = re.split(r"[;,]", value)
        return [part.strip().lower() for part in parts if part.strip()]
    return [str(item).lower() for item in value]


def _matches(prim, tokens):
    if prim_path:
        return str(prim.GetPath()) == prim_path
    if not tokens:
        return True
    text = str(prim.GetPath()).lower()
    return any(token in text for token in tokens)


def _bbox_info(cache, prim):
    bbox = cache.ComputeWorldBound(prim).ComputeAlignedBox()
    center = (bbox.GetMin() + bbox.GetMax()) * 0.5
    size = bbox.GetMax() - bbox.GetMin()
    return center, size


def _physics_schemas(prim):
    return [schema for schema in prim.GetAppliedSchemas() if "Physics" in schema or "Physx" in schema]


def _ensure_scope(stage, scope_path):
    path = Sdf.Path(scope_path)
    current = Sdf.Path.absoluteRootPath
    for element in path.pathString.strip("/").split("/"):
        if not element:
            continue
        current = current.AppendChild(element)
        UsdGeom.Xform.Define(stage, current)


def _safe_name(name):
    return re.sub(r"[^A-Za-z0-9_]+", "_", name.strip("/")) or "marker"


def _create_marker(stage, name, position, color=(1.0, 0.45, 0.0)):
    _ensure_scope(stage, marker_scope)
    marker_path = Sdf.Path(marker_scope).AppendChild(_safe_name(name))
    if stage.GetPrimAtPath(marker_path).IsValid():
        stage.RemovePrim(marker_path)
    sphere = UsdGeom.Sphere.Define(stage, marker_path)
    sphere.CreateRadiusAttr(float(marker_radius))
    sphere.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdGeom.Xformable(sphere.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in position]))
    print(f"marker {name}: {marker_path} at {list(position)}")


def _parse_marker_points():
    points = []
    if not marker_points:
        return points
    for chunk in str(marker_points).split(";"):
        if not chunk.strip():
            continue
        name, raw = chunk.split(":", 1) if ":" in chunk else (f"marker_{len(points)}", chunk)
        points.append((name.strip(), _vector(raw)))
    return points


def _print_robot(stage):
    if not robot_path:
        return
    prim = stage.GetPrimAtPath(robot_path)
    print(f"\nRobot: {robot_path} valid={prim.IsValid()} type={prim.GetTypeName() if prim else ''}")
    if not prim or not prim.IsValid():
        return
    try:
        articulation = Articulation(robot_path)
        print(f"  dof_names={articulation.dof_names}")
        print(f"  link_names={articulation.link_names}")
        print(f"  link_paths={articulation.link_paths}")
        try:
            pos, orn = articulation.get_world_poses()
            print(f"  world_position={pos.numpy()[0].tolist()}")
            print(f"  world_orientation_wxyz={orn.numpy()[0].tolist()}")
        except Exception as exc:
            print(f"  world_pose_error={type(exc).__name__}: {exc}")
    except Exception as exc:
        print(f"  articulation_error={type(exc).__name__}: {exc}")


def _print_supported_robot():
    if not supported_robot:
        return
    print(f"\ncuMotion supported robot: {supported_robot}")
    try:
        robot = load_cumotion_supported_robot(str(supported_robot))
        print(f"  controlled_joint_names={robot.controlled_joint_names}")
        print(f"  tool_frames={robot.robot_description.tool_frame_names()}")
        print(f"  config_directory={robot.directory}")
    except Exception as exc:
        print(f"  cumotion_load_error={type(exc).__name__}: {exc}")


def _print_object(stage, bbox_cache):
    if not object_path:
        return
    prim = stage.GetPrimAtPath(object_path)
    print(f"\nObject: {object_path} valid={prim.IsValid()} type={prim.GetTypeName() if prim else ''}")
    if not prim or not prim.IsValid():
        return
    center, size = _bbox_info(bbox_cache, prim)
    print(f"  bbox_center={list(center)}")
    print(f"  bbox_size={list(size)}")
    print(f"  physics_schemas={_physics_schemas(prim)}")
    if _as_bool(create_markers):
        _create_marker(stage, f"{object_path}_bbox_center", center, color=(0.1, 0.8, 1.0))
    try:
        rigid = RigidPrim(object_path)
        pos, orn = rigid.get_world_poses()
        print(f"  rigid_world_position={pos.numpy()[0].tolist()}")
        print(f"  rigid_world_orientation_wxyz={orn.numpy()[0].tolist()}")
        try:
            lin, ang = rigid.get_velocities()
            print(f"  linear_velocity={lin.numpy()[0].tolist()}")
            print(f"  angular_velocity={ang.numpy()[0].tolist()}")
        except Exception as exc:
            print(f"  velocity_error={type(exc).__name__}: {exc}")
    except Exception as exc:
        print(f"  rigidprim_error={type(exc).__name__}: {exc}")


def _run():
    if usd_path:
        result, _ = stage_utils.open_stage(usd_path)
        app_utils.update_app(steps=10)
        if not result:
            raise RuntimeError(f"Failed to open USD: {usd_path}")

    stage = stage_utils.get_current_stage()
    if not stage:
        raise RuntimeError("No stage is open.")

    eye = _vector(camera_eye)
    target = _vector(camera_target)
    if eye is not None or target is not None:
        ViewportManager.set_camera_view(camera, eye=eye, target=target)
        app_utils.update_app(steps=10)

    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        raise RuntimeError(f"Root prim not found: {root_path}")

    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
    )

    print(f"Stage: {stage.GetRootLayer().identifier}")
    print(f"Root: {root_path}")
    _print_supported_robot()
    _print_robot(stage)
    _print_object(stage, bbox_cache)

    for name, point in _parse_marker_points():
        _create_marker(stage, name, point, color=(0.2, 1.0, 0.2))

    tokens = _tokens(match)
    results = []
    if tokens or prim_path:
        for prim in Usd.PrimRange(root):
            if _matches(prim, tokens):
                results.append(prim)
                if len(results) >= int(max_results):
                    break

    if results:
        print(f"\nMatched prims: {len(results)} shown")
    for index, prim in enumerate(results):
        path = prim.GetPath()
        print(f"[{index:03d}] {path} type={prim.GetTypeName()} active={prim.IsActive()}")
        if prim.IsA(UsdGeom.Xformable):
            matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            print(f"      world_translate={list(matrix.ExtractTranslation())}")
        center, size = _bbox_info(bbox_cache, prim)
        print(f"      bbox_center={list(center)} bbox_size={list(size)}")
        schemas = _physics_schemas(prim)
        if schemas:
            print(f"      physics_schemas={schemas}")
        if _as_bool(create_markers):
            _create_marker(stage, str(path), center)


_run()
