"""robot/usd/convert_rover.py -- rover MJCF -> USD with the official Isaac Sim importer (gate G2, step 1).

Runs INSIDE NVIDIA's Isaac Lab container on the GPU box (it needs Isaac Sim), from the box:

    bash scripts/gpu/isaac.sh robot/usd/convert_rover.py            # -> /data/runs/usd

Inputs are the two rover-only MJCFs from robot/make_import_variants.py:
    robot/import/rover_train.xml  -> <out>/rover_train/rover_train.usda  (one 29.4 kg articulation)
    robot/import/rover_swap.xml   -> <out>/rover_swap/rover_swap.usda    (pack + latch FixedJoint)

Same path as Isaac Lab's scripts/tools/convert_mjcf.py (MjcfConverter -> Isaac Sim 6.1
isaacsim.asset.importer.mjcf), plus the two settings that script does not expose:
robot_type="Wheeled" and fix_base=False. The "Physics" variant is set to PhysX; the
MuJoCo variant stays in the file for the Newton MuJoCo-Warp check (docs/decisions.md D5).

After converting, it prints a physics report (bodies, mass, joints, drives, colliders,
friction) so the asset can be checked against robot/SPEC.md before any training.
Drive gains in the USD are NOT used for training: Isaac Lab's actuator config sets them
in SI units (docs/decisions.md D6), so the report only shows what the importer wrote.
"""

import argparse
import os

from isaaclab.app import AppLauncher, add_launcher_args, launch_simulation
from isaaclab.utils.version import standalone_importers_available

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

parser = argparse.ArgumentParser(description="Convert the rover MJCFs to USD with the official importer.")
parser.add_argument("--out", default="/data/runs/usd", help="output directory (the repo is mounted read-only)")
parser.add_argument("--variants", nargs="+", default=["rover_train", "rover_swap"])
add_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.require_kit = not standalone_importers_available()
args_cli.physics = "isaacsim_physx" if args_cli.require_kit else "newton_mjwarp"

from isaaclab.physics import PhysicsCfg  # noqa: E402
from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg  # noqa: E402


def report(usd_path: str) -> None:
    """Print what the importer produced, in the terms robot/SPEC.md uses."""
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

    stage = Usd.Stage.Open(usd_path)
    roots = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    print(f"  articulation roots: {[str(p.GetPath()) for p in roots]}")

    total = 0.0
    for p in stage.Traverse():
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            m = UsdPhysics.MassAPI(p).GetMassAttr().Get() if p.HasAPI(UsdPhysics.MassAPI) else None
            total += m or 0.0
            print(f"  body   {p.GetName():18s} mass={m}")
    print(f"  total mass: {total:.3f} kg   (SPEC: 29.4 for rover_train)")

    for p in stage.Traverse():
        if not p.IsA(UsdPhysics.Joint):
            continue
        j = UsdPhysics.Joint(p)
        kind = p.GetTypeName()
        axis = p.GetAttribute("physics:axis").Get() if p.HasAttribute("physics:axis") else ""
        line = f"  joint  {p.GetName():18s} {kind:22s} axis={axis}"
        if p.HasAttribute("physics:excludeFromArticulation"):
            line += f" excludeFromArticulation={p.GetAttribute('physics:excludeFromArticulation').Get()}"
        if p.HasAttribute("physics:jointEnabled"):
            line += f" jointEnabled={p.GetAttribute('physics:jointEnabled').Get()}"
        drive = UsdPhysics.DriveAPI.Get(p, "angular")
        if drive:
            line += (f" drive(stiffness={drive.GetStiffnessAttr().Get()}, damping={drive.GetDampingAttr().Get()},"
                     f" maxForce={drive.GetMaxForceAttr().Get()})")
        for name in ("physxJoint:jointFriction", "physxJoint:armature"):
            if p.HasAttribute(name):
                line += f" {name.split(':')[1]}={p.GetAttribute(name).Get()}"
        print(line + f"  body0={j.GetBody0Rel().GetTargets()} body1={j.GetBody1Rel().GetTargets()}")

    for p in stage.Traverse():
        if p.HasAPI(UsdPhysics.CollisionAPI):
            size = ""
            if p.IsA(UsdGeom.Cylinder):
                c = UsdGeom.Cylinder(p)
                size = f"r={c.GetRadiusAttr().Get()} h={c.GetHeightAttr().Get()} axis={c.GetAxisAttr().Get()}"
            elif p.IsA(UsdGeom.Cube):
                size = f"size={UsdGeom.Cube(p).GetSizeAttr().Get()}"
            enabled = UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()
            print(f"  collider {p.GetName():16s} {p.GetTypeName():10s} enabled={enabled} {size}")

    for p in stage.Traverse():
        if p.HasAPI(UsdPhysics.MaterialAPI):
            mat = UsdPhysics.MaterialAPI(p)
            print(f"  material {p.GetName():16s} static={mat.GetStaticFrictionAttr().Get()}"
                  f" dynamic={mat.GetDynamicFrictionAttr().Get()} restitution={mat.GetRestitutionAttr().Get()}")
    bound = [p for p in stage.Traverse()
             if p.HasAPI(UsdPhysics.CollisionAPI)
             and UsdShade.MaterialBindingAPI(p).GetDirectBinding("physics").GetMaterial()]
    print(f"  colliders with a physics material bound: {len(bound)}")


def main() -> None:
    os.makedirs(args_cli.out, exist_ok=True)
    with launch_simulation(cfg=PhysicsCfg(), launcher_args=args_cli):
        for name in args_cli.variants:
            cfg = MjcfConverterCfg(
                asset_path=os.path.join(REPO, "robot", "import", f"{name}.xml"),
                usd_dir=os.path.join(args_cli.out, name),
                usd_file_name=f"{name}.usda",
                force_usd_conversion=True,
                robot_type="Wheeled",
                fix_base=False,
                physics_variant="physx",
            )
            usd_path = MjcfConverter(cfg).usd_path
            print("=" * 80)
            print(f"{name}: {usd_path}")
            report(usd_path)
        print("=" * 80)
        print("CONVERSION DONE")


if __name__ == "__main__":
    main()
