"""Physics-only CI nightly wrapper for Kit 110.

Launches SimulationApp with the renderer fully disabled and runs a physics
step loop.  Intended for unattended CI nightly pipelines where no GPU
rendering subsystem should initialize.

Run via ``python.sh`` (renderer-disable flags passed through to Carbonite):

    ./python.sh evals/files/physics_nightly_candidate.py \
        --/renderer/enabled=false \
        --/app/window/enabled=false \
        --/app/livestream/enabled=false

Or via the CI helper (which supplies these flags automatically):

    ./scripts/ci_physics_nightly.sh evals/files/physics_nightly_candidate.py

Set ``CI_KIT_MODE=1`` to use Kit CLI with
``apps/isaacsim.exp.base.python.physics.headless.kit`` instead.
"""

import signal
import sys

PHYSICS_STEPS = 120
TIMEOUT_SECONDS = 300

CONFIG = {
    "headless": True,
    "renderer": "disabled",
    "width": 1,
    "height": 1,
    "anti_aliasing": 0,
    "open_usd": None,
}

CARB_SETTINGS = {
    "/renderer/enabled": False,
    "/app/window/enabled": False,
    "/app/livestream/enabled": False,
    "/app/renderer/resolution/width": 1,
    "/app/renderer/resolution/height": 1,
}


def _timeout_handler(signum, frame):
    print(f"TIMEOUT: physics nightly exceeded {TIMEOUT_SECONDS}s", file=sys.stderr)
    sys.exit(2)


def main() -> int:
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(TIMEOUT_SECONDS)

    from isaacsim import SimulationApp

    simulation_app = SimulationApp(CONFIG)

    import carb.settings
    import omni.physx
    import omni.timeline

    settings = carb.settings.get_settings()
    for key, value in CARB_SETTINGS.items():
        settings.set(key, value)

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()

    try:
        for step in range(PHYSICS_STEPS):
            simulation_app.update()

        print(f"OK: {PHYSICS_STEPS} physics steps completed")
        return 0
    except Exception as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1
    finally:
        timeline.stop()
        simulation_app.close()
        signal.alarm(0)


if __name__ == "__main__":
    sys.exit(main())
