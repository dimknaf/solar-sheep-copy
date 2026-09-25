"""Compute robot footprint dimensions and Z-offset from USD collision geometry.

Returns footprint size, Z-offset (how far the articulation origin sits above
the lowest collider), and inscribed / circumscribed radii.
"""

import numpy as np


def compute_robot_footprint(stage, robot_root: str) -> dict:
    """Return footprint dims + Z-offset + inscribed/circumscribed radii.

    Uses prims tagged with UsdPhysics.CollisionAPI under `robot_root`. Falls back
    to UsdGeom.Imageable if no colliders are authored.

    Args:
        stage: Open USD stage.
        robot_root: USD path to the robot root prim (e.g. "/World/Robot").

    Returns:
        Dict with keys: size, z_offset, inscribed_radius, circumscribed_radius,
        aabb_min, aabb_max.
    """
    import isaacsim.core.experimental.utils.bounds as bounds_utils
    from pxr import Usd, UsdPhysics

    collider_paths = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath(robot_root)):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            collider_paths.append(prim.GetPath())
    if not collider_paths:
        collider_paths = [stage.GetPrimAtPath(robot_root).GetPath()]

    aabb = bounds_utils.compute_combined_aabb(collider_paths)  # [xmin,ymin,zmin, xmax,ymax,zmax]
    mn, mx = aabb[:3], aabb[3:]
    size = mx - mn

    origin_z = stage.GetPrimAtPath(robot_root).GetAttribute("xformOp:translate").Get()[2]
    z_offset = max(0.0, origin_z - mn[2])  # how far origin sits above lowest collider

    half_w, half_d = size[0] / 2.0, size[1] / 2.0
    return {
        "size": tuple(size),  # full footprint extents (m)
        "z_offset": float(z_offset),  # origin → lowest collider (m)
        "inscribed_radius": float(min(half_w, half_d)),  # safe for ANY yaw
        "circumscribed_radius": float(np.hypot(half_w, half_d)),  # worst-case yaw
        "aabb_min": tuple(mn),
        "aabb_max": tuple(mx),
    }
