"""Conceptual sketch for a custom Jacobian-based differential IK controller.

This is a pattern template, not a runnable or maintained example controller.
Use ``isaacsim.robot_motion.examples`` for supported manipulation examples.
"""

import numpy as np


def _quaternion_multiply(a, b):
    """Multiply batched WXYZ quaternions."""
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        axis=-1,
    )


def differential_ik_step(
    arm,
    end_effector,
    end_effector_link_index,
    target_pos,
    target_quat,
    arm_dofs,
    method="damped-least-squares",
    damping=0.05,
    scale=1.0,
    min_singular_value=1e-5,
):
    """Compute and apply one IK step toward (target_pos, target_quat).

    arm: ``isaacsim.core.experimental.prims.Articulation``.
    end_effector: articulation link used to measure the current pose.
    Returns joint delta (dq) applied this step.
    """
    jacobian = arm.get_jacobian_matrices().numpy()[:, end_effector_link_index - 1, :, :arm_dofs]
    current_pos, current_quat = end_effector.get_world_poses()
    current_pos = current_pos.numpy()
    current_quat = current_quat.numpy()
    target_pos = np.broadcast_to(np.asarray(target_pos), current_pos.shape)
    target_quat = np.broadcast_to(np.asarray(target_quat), current_quat.shape)

    conjugate = current_quat * np.array([1.0, -1.0, -1.0, -1.0])
    rotation_error = _quaternion_multiply(target_quat, conjugate)
    error = np.concatenate((target_pos - current_pos, rotation_error[:, 1:] * np.sign(rotation_error[:, :1])), axis=-1)[
        ..., None
    ]

    transpose = np.swapaxes(jacobian, 1, 2)
    if method == "damped-least-squares":
        inverse = transpose @ np.linalg.inv(jacobian @ transpose + damping**2 * np.eye(6))
    elif method == "pseudoinverse":
        inverse = np.linalg.pinv(jacobian)
    elif method == "transpose":
        inverse = transpose
    elif method == "singular-value-decomposition":
        u, singular_values, vh = np.linalg.svd(jacobian)
        reciprocal = np.where(singular_values > min_singular_value, 1.0 / singular_values, 0.0)
        v = np.swapaxes(vh, 1, 2)[:, :, :6]
        inverse = (v * reciprocal[:, None, :]) @ np.swapaxes(u, 1, 2)
    else:
        raise ValueError(f"Unsupported differential IK method: {method}")

    dq = (scale * inverse @ error).squeeze(-1)
    cur_dofs = arm.get_dof_position_targets()
    arm.set_dof_position_targets(cur_dofs[:, :arm_dofs] + dq, dof_indices=list(range(arm_dofs)))
    return dq
