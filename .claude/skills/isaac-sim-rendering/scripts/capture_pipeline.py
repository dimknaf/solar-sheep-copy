"""Standard Kit 110 / Isaac Sim 6.0+ headless capture pipeline."""

from isaacsim import SimulationApp


def setup_capture_pipeline(
    stage_path, width=1920, height=1080, renderer="RayTracedLighting", settle_frames=200, film_iso=200.0
):
    """Open stage and return (app, rgb_annotator, render_product).

    Caller is responsible for calling app.close() when done.
    Use settle_frames=500 and film_iso=600 for deeply-occluded indoor aisles.
    """
    app = SimulationApp({"headless": True, "width": width, "height": height, "renderer": renderer})

    import carb
    import omni.replicator.core as rep
    from pxr import Gf, UsdGeom

    stage = app.context.open_stage(stage_path)

    s = carb.settings.get_settings()
    s.set("/rtx/post/tonemap/op", 4)
    s.set("/rtx/post/tonemap/filmIso", film_iso)
    s.set("/rtx/post/tonemap/whitepoint", 6500.0)
    s.set("/rtx/post/tonemap/enabled", True)
    s.set("/rtx/post/aa/op", 3)

    cam = UsdGeom.Camera.Define(stage, "/World/RenderCam")
    cam_xf = UsdGeom.Xformable(cam.GetPrim())
    cam_xf.AddTranslateOp().Set(Gf.Vec3d(0, 0, 10))
    cam_xf.AddRotateXYZOp().Set(Gf.Vec3f(-90, 0, 0))
    cam.CreateFocalLengthAttr().Set(20.0)

    render_product = rep.create.render_product("/World/RenderCam", (width, height))
    rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb_annot.attach([render_product])

    for _ in range(settle_frames):
        app.update()

    return app, rgb_annot, render_product


def capture_frame(rgb_annot):
    """Step replicator and return (H, W, 3) uint8 RGB array."""
    import omni.replicator.core as rep

    rep.orchestrator.step()
    data = rgb_annot.get_data()
    return data[:, :, :3]
