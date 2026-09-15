# cuMotion (RMPflow controller)

cuMotion-specific reference for `isaacsim.robot_motion.cumotion`. The generic substrate
it plugs into lives in sibling references: obstacles / world binding in
`world-binding.md`, the `RobotState` control loop in `control-loop.md`, tool-frame
discipline in `frames-and-grasping.md`.

## Imports

In standalone scripts, import the controller after `SimulationApp` is constructed.
Use [`scripts/cumotion_setup.py`](../scripts/cumotion_setup.py) for the supported-robot
loader and controller construction.

`CumotionWorldInterface` is the cuMotion implementation of the `world_interface` that
`mg.WorldBinding` requires (see `world-binding.md`).

## Controller construction

Call `build_rmpflow_controller(robot, world_binding, robot_name)` from
[`scripts/cumotion_setup.py`](../scripts/cumotion_setup.py). It returns the controller,
available tool frames, and selected tool frame.

Do not hard-code a tool frame until inspection proves it. Common UR10 configs expose
`tool0` or `wrist_3_link`, while the visible physical grasp point may be elsewhere (see
`frames-and-grasping.md`).

Once constructed, drive it with the generic loop in `control-loop.md`.

## Common RMPflow failures

- `forward()` never converges: wrong tool frame, wrong quaternion, stale target, or no
  reset after a target jump.
- robot moves but the object target is offset: controlling the flange/tool while
  measuring the fingertip or suction point (see `frames-and-grasping.md`).
- object blocks the path unexpectedly: the grasped object is still in the world binding
  as an obstacle (see `world-binding.md`).
- erratic motion after a phase transition: target changed discontinuously without
  `reset(...)`.
- gripper joint command ignored: gripper DOF index is wrong or resolved before
  physics / articulation initialization.
- FK rotation assumption fails: `CumotionRobot.kinematics.pose(...).rotation` can expose
  `matrix()` without a `.quaternion` property. Derive WXYZ from `rotation.matrix()` or
  inspect the current build's rotation API before using it.
