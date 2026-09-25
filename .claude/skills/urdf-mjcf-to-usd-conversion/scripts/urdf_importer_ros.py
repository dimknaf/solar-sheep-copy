"""Import URDF from a live ROS 2 robot_state_publisher via URDFImporter."""

from functools import partial


def import_urdf_from_ros(usd_out_path, merge_fixed_joints=True, fix_base=False, robot_type="Manipulator"):
    """Subscribe to robot_state_publisher and import URDF when received.

    usd_out_path: destination USD path (no $ISAAC_SIM_DIR prefix).
    Blocks until a description is received on the topic.
    """
    from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig
    from isaacsim.ros2.urdf import RobotDefinitionReader

    config = URDFImporterConfig(
        usd_path=usd_out_path,
        merge_fixed_joints=merge_fixed_joints,
        fix_base=fix_base,
        robot_type=robot_type,
    )
    importer = URDFImporter()

    def _on_description(urdf_abs_path, package_found):
        config.urdf_path = urdf_abs_path
        importer.config = config
        importer.import_urdf()

    reader = RobotDefinitionReader()
    reader.description_received_fn = partial(_on_description)
    reader.start_get_robot_description("robot_state_publisher")
