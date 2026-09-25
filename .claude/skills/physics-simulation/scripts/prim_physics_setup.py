"""Per-prim physics setup helpers for Isaac Sim / USD (Kit 110).

Provides setup_dynamic_body, setup_static_collider, and setup_kinematic_body
as the three standard physics configurations for scene prims.
"""


def setup_dynamic_body(stage, prim_path: str, mass_kg: float = 1.0, com_offset=None):
    """Apply RigidBodyAPI + CollisionAPI to make a prim fully simulated.

    Args:
        stage: Open USD stage.
        prim_path: USD path of the target prim.
        mass_kg: Mass in kilograms.
        com_offset: Optional center-of-mass offset as (x, y, z) in local space.

    Returns:
        The USD prim.
    """
    from pxr import Gf, UsdPhysics

    prim = stage.GetPrimAtPath(prim_path)
    UsdPhysics.RigidBodyAPI.Apply(prim)
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr().Set(mass_kg)
    if com_offset:
        mass_api.CreateCenterOfMassAttr().Set(Gf.Vec3f(*com_offset))
    UsdPhysics.CollisionAPI.Apply(prim)
    return prim


def setup_static_collider(stage, prim_path: str):
    """Apply CollisionAPI to make a prim an immovable static obstacle.

    Args:
        stage: Open USD stage.
        prim_path: USD path of the target prim.

    Returns:
        The USD prim.
    """
    from pxr import UsdPhysics

    prim = stage.GetPrimAtPath(prim_path)
    UsdPhysics.CollisionAPI.Apply(prim)
    return prim


def setup_kinematic_body(stage, prim_path: str):
    """Apply RigidBodyAPI (kinematic) + CollisionAPI for scripted motion.

    Use for conveyors, elevators, vibrating bowls, and escape wheels that
    are moved by the script rather than the physics solver.

    Args:
        stage: Open USD stage.
        prim_path: USD path of the target prim.

    Returns:
        The USD prim.
    """
    from pxr import UsdPhysics

    prim = stage.GetPrimAtPath(prim_path)
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr().Set(True)
    UsdPhysics.CollisionAPI.Apply(prim)
    return prim
