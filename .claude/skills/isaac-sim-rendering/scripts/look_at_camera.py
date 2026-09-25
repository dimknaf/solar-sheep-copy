"""Look-at camera math for USD cameras (Z-up, USD -Z forward convention)."""


def look_at_matrix(eye, target, up=None):
    """Return Gf.Matrix4d for a USD camera at `eye` looking at `target`.

    Handles degenerate up-vector (camera looking straight down/up).
    USD cameras: -Z is forward, +Y is up in camera space.
    """
    from pxr import Gf

    if up is None:
        up = Gf.Vec3d(0, 0, 1)
    eye = Gf.Vec3d(*eye)
    target = Gf.Vec3d(*target)
    fwd = (target - eye).GetNormalized()

    if abs(fwd * up) > 0.99:
        up = Gf.Vec3d(0, 1, 0)

    right = (fwd ^ up).GetNormalized()
    cam_up = (right ^ fwd).GetNormalized()

    m = Gf.Matrix4d()
    m[0] = [right[0], right[1], right[2], 0]
    m[1] = [cam_up[0], cam_up[1], cam_up[2], 0]
    m[2] = [-fwd[0], -fwd[1], -fwd[2], 0]
    m[3] = [eye[0], eye[1], eye[2], 1]
    return m
