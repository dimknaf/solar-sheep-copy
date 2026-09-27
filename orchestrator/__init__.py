"""orchestrator -- the Solar Sheep fleet brain (Contract 1). Pure stdlib, no Isaac/torch/numpy.

WHAT IT IS
    Nemotron (an NVIDIA open model on Nebius Token Factory) dispatches the herd with strict
    JSON commands; a rule-based fallback -- a port of the browser sim's state machine
    (simulation.js) plus recovery rules -- has a plan ready every tick and takes over,
    visibly, per rover, whenever the LLM is off, slow, wrong, stale or over budget.

FILES
    telemetry.py  dataclasses up (RoverTelemetry, DockTelemetry, FleetSnapshot) and down
                  (Command, Plan), constants (RESERVE 0.05, FULL 0.95, BROWNOUT 0.15, ...)
    layout.py     site geometry: dock/berth/approach/queue/park + the (row, slot) pasture
    economics.py  sun vector, 20-deg tilted panel power, weather, SoC on a time-lapse clock
    schema.py     strict JSON schema for the LLM + the validator that enforces the real rules
    rules.py      RuleEngine: the fallback brain
    client.py     Token Factory chat client (urllib), key redaction, price estimates
    brain.py      FleetBrain: non-blocking tick, one background call at a time, merge
    smoke.py      ONE live request against Token Factory (python -m orchestrator.smoke)
    tests/        offline unittest suite with a fake LLM transport

USE (inside the factory loop)
    from orchestrator import Layout, Economics, FleetBrain, FleetSnapshot, RoverTelemetry, DockTelemetry
    layout, econ = Layout.default(5), Economics(timelapse=120.0, start_hour=8.0)
    brain = FleetBrain(ids, layout, use_llm=True, log_path="/data/runs/factory/orchestrator.jsonl")
    every step:  t = econ.day_seconds(t_phys)
                 snap = FleetSnapshot(t, econ.sun_yaw(t), econ.sun_el(t), econ.weather, rovers, dock)
                 brain.tick(snap, time.monotonic())        # never blocks, never raises
                 for rid, cmd in brain.plan.commands.items(): executor[rid].follow(cmd)
    brain.status -> calls / in_flight / cost / last_error / source;  brain.close() at the end.
    The executor restarts a rover's task when ``cmd.key()`` changes and reports RUNNING, then
    SUCCESS/FAILURE (+ reason) in RoverTelemetry.last_status / last_reason.

ENV
    NEBIUS_TOKEN_FACTORY_KEY (only source of the key; missing -> rules only, source says so),
    NEBIUS_TOKEN_FACTORY_BASE_URL, SOLAR_TF_MODEL (default deepseek-ai/DeepSeek-V4.1-Flash).

TEST
    .venv\\Scripts\\python.exe -m unittest orchestrator.tests.test_orchestrator -v
    .venv\\Scripts\\python.exe -m orchestrator.smoke --env-file .env     (one live call)
"""

from .brain import FleetBrain
from .economics import Economics
from .layout import Layout
from .telemetry import (ACTIONS, BROWNOUT, CAPACITY_WH, FACES, FULL, PANEL_PEAK_W, RESERVE,
                        ROVER_STATES, Command, DockTelemetry, FleetSnapshot, Plan,
                        RoverTelemetry)

__all__ = ["Layout", "Economics", "FleetBrain", "FleetSnapshot", "RoverTelemetry",
           "DockTelemetry", "Plan", "Command", "ROVER_STATES", "ACTIONS", "FACES",
           "RESERVE", "FULL", "BROWNOUT", "CAPACITY_WH", "PANEL_PEAK_W"]
