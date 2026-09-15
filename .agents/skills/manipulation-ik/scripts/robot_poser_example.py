"""Schema-native IK + named-pose workflow using isaacsim.robot.poser."""


def solve_and_store_pose(stage, robot_prim, start_prim, end_prim, target_pos, target_orient, pose_name):
    """Solve IK for target and store as a named pose on the robot.

    robot_prim must carry IsaacRobotAPI (applied by URDF/MJCF importers).
    Returns True if IK succeeded.
    """
    from isaacsim.robot.poser import (
        RobotPoser,
        Transform,
        store_named_pose,
        validate_robot_schema,
    )

    validate_robot_schema(stage, robot_prim)
    poser = RobotPoser(stage, robot_prim, start_prim, end_prim)
    target = Transform(position=target_pos, orientation=target_orient)
    result = poser.solve_ik(target)
    if result.success:
        poser.apply_pose(result.joints)
        store_named_pose(stage, robot_prim, pose_name, result)
    return result.success


def apply_stored_pose(stage, robot_prim, pose_name):
    """Apply a previously stored named pose."""
    from isaacsim.robot.poser import apply_pose_by_name

    apply_pose_by_name(stage, robot_prim, pose_name)


def export_all_poses(stage, robot_prim, json_path):
    """Export all named poses on the robot to a JSON file."""
    from isaacsim.robot.poser import export_poses

    export_poses(stage, robot_prim, json_path)
