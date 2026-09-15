"""Apply Isaac Robot Schema overlay to an existing USD articulation."""


def apply_robot_schema(stage, robot_prim, link_prims, joint_prims, site_prims=None, robot_type="Default"):
    """Apply IsaacRobotAPI, IsaacLinkAPI, IsaacJointAPI, and IsaacSiteAPI.

    robot_type: one of get_allowed_tokens(Attributes.ROBOT_TYPE) —
      Default, End Effector, Manipulator, Humanoid,
      Wheeled, Holonomic, Quadruped, Mobile Manipulators, Aerial.
    Call PopulateRobotSchemaFromArticulation afterward to fill ROBOT_LINKS
    and ROBOT_JOINTS relations from the physics traversal.
    """
    from usd.schema.isaac.robot_schema import (
        ApplyJointAPI,
        ApplyLinkAPI,
        ApplyRobotAPI,
        ApplySiteAPI,
        Attributes,
        PopulateRobotSchemaFromArticulation,
    )

    ApplyRobotAPI(robot_prim)
    robot_prim.GetAttribute(Attributes.ROBOT_TYPE).Set(robot_type)

    for link in link_prims:
        ApplyLinkAPI(link)
    for joint in joint_prims:
        ApplyJointAPI(joint)
    if site_prims:
        for site in site_prims:
            ApplySiteAPI(site)

    PopulateRobotSchemaFromArticulation(stage, robot_prim)
