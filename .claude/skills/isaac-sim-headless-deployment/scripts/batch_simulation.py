"""Headless batch simulation loop for Isaac Sim (Kit 110).

Runs N episodes, each loading a USD stage, stepping physics, and closing cleanly.
Run with $ISAAC_SIM_DIR/python.sh.
"""

from isaacsim import SimulationApp


def run_batch_simulation(
    scene_path: str,
    num_episodes: int = 100,
    steps_per_episode: int = 1000,
    dt: float = 1.0 / 60.0,
    device: str = "cpu",
) -> None:
    """Run a headless batch simulation loop.

    Args:
        scene_path: Path to the USD scene to load each episode.
        num_episodes: Number of simulation episodes to run.
        steps_per_episode: Physics steps per episode.
        dt: Physics timestep in seconds.
        device: Physics device ("cpu" or "cuda").
    """
    simulation_app = SimulationApp({"headless": True, "renderer": "RayTracedLighting"})

    import isaacsim.core.experimental.utils.app as app_utils
    import isaacsim.core.experimental.utils.stage as stage_utils
    from isaacsim.core.simulation_manager import SimulationManager

    for episode in range(num_episodes):
        stage_utils.open_stage(scene_path)
        while stage_utils.is_stage_loading():
            simulation_app.update()
        SimulationManager.setup_simulation(dt=dt, device=device)
        app_utils.play()
        for _ in range(steps_per_episode):
            simulation_app.update()
        app_utils.stop()

    simulation_app.close()
