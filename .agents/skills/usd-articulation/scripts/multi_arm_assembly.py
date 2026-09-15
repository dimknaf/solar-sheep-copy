"""Assemble a multi-arm robot in USD using FixedJoints.

Pattern: spawn chassis USD, spawn each arm as a child, then connect with a
FixedJoint and remove the arm's ArticulationRootAPI so only the chassis root
remains.
"""


def assemble_multi_arm_robot(
    stage,
    chassis_usd: str,
    arm_usd: str,
    robot_root: str = "/World/Robot",
    arm_offset: tuple = (0.0, 0.6, 0.5),
    chassis_body_path: str = "/World/Robot/.../Chassis",
    arm_base_path: str = "/World/Robot/LeftArm/.../BaseMount",
) -> None:
    """Spawn a chassis + arm and connect them with a FixedJoint.

    Requires: pxr (Usd, UsdGeom, UsdPhysics, Sdf, Gf) from Isaac Sim.

    Args:
        stage: Open USD stage.
        chassis_usd: Path to the chassis USD asset.
        arm_usd: Path to the arm USD asset.
        robot_root: Prim path for the assembled robot root.
        arm_offset: XYZ translation of the arm relative to the robot root.
        chassis_body_path: Body0 target for the FixedJoint (chassis link).
        arm_base_path: Body1 target for the FixedJoint (arm base mount link).
    """
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    # Spawn chassis
    chassis = stage.DefinePrim(robot_root, "Xform")
    chassis.GetReferences().AddReference(chassis_usd)

    # Spawn arm as child
    left = stage.DefinePrim(f"{robot_root}/LeftArm", "Xform")
    left.GetReferences().AddReference(arm_usd)
    UsdGeom.Xformable(left).AddTranslateOp().Set(Gf.Vec3d(*arm_offset))

    # REQUIRED: FixedJoint connecting arm to chassis
    joint = UsdPhysics.FixedJoint.Define(stage, f"{robot_root}/LeftArmAttachment")
    joint.CreateBody0Rel().SetTargets([Sdf.Path(chassis_body_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(arm_base_path)])

    # REQUIRED: Remove ArticulationRootAPI from arm — only chassis keeps it
    arm_prim = stage.GetPrimAtPath(f"{robot_root}/LeftArm/...")
    if arm_prim and arm_prim.IsValid():
        arm_prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
