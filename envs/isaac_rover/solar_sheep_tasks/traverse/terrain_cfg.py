"""envs/isaac_rover/solar_sheep_tasks/traverse/terrain_cfg.py -- the rough-task ground, at rover scale.

Rows are difficulty levels (the navigation curriculum moves each env up or down a row), columns are
terrain types in proportion. Every tile is 8 x 8 m with the spawn platform at its centre. Numbers are
the MJX prototype's (envs/traverse/train.py) turned into geometry, capped by robot/SPEC.md:

    flat      plane
    rough     random uniform bumps 0-3 cm, 5 mm steps
    waves     amplitude 0 -> 5 cm (two crossed sines, 2 m wavelength: at most ~6.3 deg, inside the
              rangefinder tilt's 7.4 deg margin, so the fan never reads a crest as an obstacle; at
              8 cm, ~10 deg, a fifth of the poses had a ray striking the ground within 3 m)
    stairs    pyramid steps 1 -> 4.5 cm, 0.6 m treads
    grid      random box grid, heights up to 2.25 cm
    slope     pyramid slope 0 -> 0.17 rise/run. The generator's pyramid is bilinear, so the grade is
              0.17 (9.6 deg) along the axes and up to 0.17*sqrt(2) (13.5 deg) at the diagonals: under
              the rover's 14 deg climb limit, never the 16 deg stall
    boxes     1 -> 6 boxes, 0.6 -> 0.2 m wide }  1.0 m tall, centred on the ground: 0.5 m proud, above
    posts     1 -> 6 posts, 0.30 -> 0.10 m r  }  the whole rangefinder fan. Big-and-few first (easy to
              see with a 22.5 deg fan), small-and-many last. 3.2 m spawn platform, 1 cm high.

Rays hit only this terrain mesh (PhysX RayCaster supports one static mesh), which is why obstacles are
terrain tiles and not separate rigid bodies.

Every tile samples flat "target" patches: the pose command draws its goals from them, and the patch
test is its clearance test. A patch is accepted when rings of radius 0.1/0.3/0.5/0.7 m around it
differ in height by at most 0.3 m: drivable slopes, steps and waves pass (worst case ~0.25 m across
1.4 m), a 0.5 m obstacle anywhere on a ring fails. Patches stay within +-3.4 m of the tile centre.
"""

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import FlatPatchSamplingCfg, TerrainGeneratorCfg

TARGET_EXTENT = 3.4    # m, half-width of the square the target patches are sampled in (0.85 x 4 m bounds)


def _targets() -> dict[str, FlatPatchSamplingCfg]:
    """A fresh "target" flat-patch spec for one sub-terrain (the command reads the "target" key)."""
    return {"target": FlatPatchSamplingCfg(
        num_patches=256, patch_radius=[0.1, 0.3, 0.5, 0.7], x_range=(-TARGET_EXTENT, TARGET_EXTENT),
        y_range=(-TARGET_EXTENT, TARGET_EXTENT), max_height_diff=0.3)}


_Boxes = terrain_gen.MeshRepeatedBoxesTerrainCfg
_Posts = terrain_gen.MeshRepeatedCylindersTerrainCfg

ROVER_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.1, flat_patch_sampling=_targets()),
        "rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.1, noise_range=(0.0, 0.03), noise_step=0.005, border_width=0.25,
            flat_patch_sampling=_targets()),
        "waves": terrain_gen.HfWaveTerrainCfg(
            proportion=0.1, amplitude_range=(0.0, 0.05), num_waves=4, border_width=0.25,
            flat_patch_sampling=_targets()),
        "stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.1, step_height_range=(0.01, 0.045), step_width=0.6, platform_width=2.0,
            border_width=1.0, flat_patch_sampling=_targets()),
        "grid": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.1, grid_width=0.45, grid_height_range=(0.005, 0.0225), platform_width=2.0,
            flat_patch_sampling=_targets()),
        "slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.17), platform_width=2.0, border_width=0.25,
            flat_patch_sampling=_targets()),
        # object_type is given as the short name repeated_objects_terrain looks up (make_<name>): EA's
        # configclass turns the class default "{DIR}.utils:make_box" into a dotted ResolvableString,
        # which that lookup cannot resolve (it raises during terrain generation).
        "boxes": _Boxes(
            proportion=0.2,
            object_type="box",
            object_params_start=_Boxes.ObjectCfg(num_objects=1, height=1.0, size=(0.6, 0.6)),
            object_params_end=_Boxes.ObjectCfg(num_objects=6, height=1.0, size=(0.2, 0.2)),
            platform_width=3.2, platform_height=0.02, flat_patch_sampling=_targets()),
        "posts": _Posts(
            proportion=0.2,
            object_type="cylinder",
            object_params_start=_Posts.ObjectCfg(num_objects=1, height=1.0, radius=0.30),
            object_params_end=_Posts.ObjectCfg(num_objects=6, height=1.0, radius=0.10),
            platform_width=3.2, platform_height=0.02, flat_patch_sampling=_targets()),
    },
)
"""Rough-task terrain generator (curriculum over rows)."""
