# Frames and gripper targeting

Use this reference for motion-generation tool-frame targeting (any controller).
Generic physical grasp validation lives in `manipulation-ik`; generic
dynamic-object collision rules live in `physics-simulation`.

## Name the points

Never use "end effector" without defining the point:

- `tool_frame`: the controller-controlled frame name (e.g. the cuMotion tool frame).
- `tool_pose`: world pose of that frame from controller FK or a stage prim.
- `finger_midpoint`: midpoint between visible finger centers.
- `pinch_point` or `suction_tip`: physical point that should contact the object.
- `object_origin`: authored prim origin.
- `object_center`: object AABB center or center of mass.
- `grasp_point`: point on/in the object used for close/contact validation.

Log these at setup and at close/contact.

## Axis contract

Before phase targets, define the relationship between robot motion and object
motion:

- `tool_z_axis_world = tool_rotation @ [0, 0, 1]`: common UR10/Robotiq approach
  axis.
- `grasp_axis_world`: direction from which the gripper approaches the grasp
  point.
- `object_axis_world`: object semantic axis, such as cylinder local `+Z`.
- `target_object_axis_world`: expected axis after manipulation.

Do not derive a gripper pose from desired object orientation alone. Build a WXYZ
tool quaternion from the desired tool-local `+Z` axis with the robot stack's
quaternion utility, then log desired and measured axes at every critical phase.

Then log desired and measured axes at every critical phase.

## Tool-local offset

If cuMotion controls `tool0` but the gripper contacts at a fingertip midpoint or
suction tip, compute the offset in a known approach pose:

Use `tool_local_contact_offset` in
[`scripts/frames_and_grasping.py`](../scripts/frames_and_grasping.py) to calibrate it.

Convert task-space contact targets to cuMotion tool targets:

Use `tool_target_from_contact_target` in
[`scripts/frames_and_grasping.py`](../scripts/frames_and_grasping.py) to convert the
physical contact target to a controller tool target.

Freeze the offset while closing and lifting. Recomputing it during finger motion
can make the target drift into or away from the object. Do not calibrate in the
arbitrary tucked/start pose and reuse it after the arm reorients.

## Contact/closure checks

Before lifting, validate at least:

- measured contact/grasp point is within threshold of the object grasp point
- gripper joint moved toward the closed target
- object pose is still near the expected pre-lift pose
- contact sensors or contact reports indicate contact, when available

If contact reporting is not wired, use conservative geometry: visible contact
marker overlap with the object AABB/grasp region, plus gripper closure progress.
Do not require a gripper joint to exactly reach the commanded closed target when
an object is between the fingers; stopping short can be contact evidence.

## Task feasibility

For tabletop flip or placement tasks, fail fast if the final pose requires the
gripper to point upward or pass below the support surface unless that strategy is
deliberate and visually validated. A top grasp of a vertical object can make
"same XY but upside down" infeasible near the table; prefer a side grasp or
pivot/roll sequence when the wrist would otherwise need to pass through the
support.

For ground-level UR10/Robotiq side grasps, a purely horizontal tool `+Z` can put
the pads above a short object because the `tool0` to pinch-point offset has a
vertical component. Measure the pad midpoint in a Python-server diagnostic before
closing; tune tool orientation/offset before changing controller gains.

## Source-specific note

When adapting the bin-filling sample, inspect the source stage before choosing
object dimensions, masses, collision, or camera framing. Use the physical KLT
asset for physics demos, not a visual-only bin plus an invented solid proxy.
