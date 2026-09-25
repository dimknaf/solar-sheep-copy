"""Rasterize a 2D occupancy grid directly from USD geometry (no PhysX required).

Use when the scene lacks colliders, or for fast offline prototyping.
For the canonical workflow that uses PhysX raycasts, see the occupancy-map skill.
"""

import numpy as np


def occupancy_map_from_usd(
    stage,
    x_range: tuple = (-15, 175),
    y_range: tuple = (-25, 155),
    resolution: float = 0.25,
    z_cutoff: float = 4.5,
    colliders_only: bool = True,
) -> np.ndarray:
    """Rasterize USD geometry into a 2D occupancy grid (0=free, 255=occupied).

    Args:
        stage: Open USD stage.
        x_range: World X extent (min, max) in meters.
        y_range: World Y extent (min, max) in meters.
        resolution: Meters per grid cell.
        z_cutoff: Ignore obstacles with Z_min above this height.
        colliders_only: If True, only include prims with UsdPhysics.CollisionAPI
            (recommended; excludes visual-only geometry automatically).

    Returns:
        numpy uint8 array of shape (grid_h, grid_w) with 0=free, 255=occupied.
    """
    from pxr import Usd, UsdGeom, UsdPhysics

    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])

    grid_w = int((x_range[1] - x_range[0]) / resolution)
    grid_h = int((y_range[1] - y_range[0]) / resolution)
    grid = np.zeros((grid_h, grid_w), dtype=np.uint8)  # 0=free, 255=occupied

    for prim in Usd.PrimRange(stage.GetPrimAtPath("/World")):
        if colliders_only:
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            enabled = prim.GetAttribute("physics:collisionEnabled")
            if enabled and enabled.Get() is False:
                continue

        bb = bbox_cache.ComputeWorldBound(prim)
        r = bb.ComputeAlignedRange()
        if r.IsEmpty():
            continue
        mn, mx = r.GetMin(), r.GetMax()
        if mn[2] > z_cutoff:
            continue

        x0 = max(0, int((mn[0] - x_range[0]) / resolution))
        x1 = min(grid_w, int((mx[0] - x_range[0]) / resolution))
        y0 = max(0, int((mn[1] - y_range[0]) / resolution))
        y1 = min(grid_h, int((mx[1] - y_range[0]) / resolution))
        grid[y0:y1, x0:x1] = 255

    return grid
