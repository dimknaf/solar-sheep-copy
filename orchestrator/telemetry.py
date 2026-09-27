"""orchestrator/telemetry.py -- the data that crosses the brain boundary (Contract 1).

Telemetry goes UP (RoverTelemetry, DockTelemetry, FleetSnapshot, filled by the factory
every tick); commands come DOWN (Command, grouped in a Plan). Plain dataclasses, stdlib
only, so the same objects work inside Isaac Sim's python and on the laptop.

Not named ``types.py`` on purpose: a module called ``types`` next to a script shadows the
standard library the moment someone runs ``python orchestrator/smoke.py`` directly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

# ---- constants (browser sim config.js + robot/SPEC.md; Contract 1) --------------------
RESERVE = 0.05            # SoC floor: a pack is "empty" here (config.js battery.reserveSoc)
FULL = 0.95               # harvest until this, then dock (config.js fullThreshold)
BROWNOUT = 0.15           # SoC at/below which an active rover away from the dock docks now
CAPACITY_WH = 500.0       # pack capacity (config.js battery.capacityWh)
PANEL_PEAK_W = 195.6      # rover panel, measured (robot/SPEC.md; config.js says 200)
MOTOR_W = 30.0            # motor envelope, W electrical (robot/SPEC.md 30.2 W)

DAY_S = 86400.0
SUNRISE_S = 6 * 3600.0                   # config.js sunrise
SUNSET_S = 18 * 3600.0                   # config.js sunset
RECALL_S = 17 * 3600.0 + 20 * 60.0       # config.js recallTime (17:20)
WAKE_SPREAD_S = 45 * 60.0                # per-rover stagger after sunrise
RECALL_JITTER_S = 15 * 60.0              # per-rover recall, +/- this

ROVER_STATES = ("resting", "to_field", "harvesting", "to_dock", "queuing", "swapping", "fault")
ACTIONS = ("harvest", "dock", "swap", "rest", "hold")
FACES = ("sun", "dock", "keep")
STATUSES = ("RUNNING", "SUCCESS", "FAILURE", "")
# swap_battery.py FAILURE codes (envs/swap/SPEC.md "FAILURE paths"); anything else is
# treated as a traverse failure.
SWAP_REASONS = ("misaligned", "no_empty_pack", "busy")


@dataclass
class RoverTelemetry:
    rover_id: str
    x: float
    y: float
    yaw: float
    soc: float                 # fraction 0..1 of CAPACITY_WH
    state: str                 # one of ROVER_STATES
    pv_w: float = 0.0          # panel power right now (W)
    last_status: str = ""      # "RUNNING" | "SUCCESS" | "FAILURE" | "" of the current task
    last_reason: str = ""      # FAILURE code / note ("stuck", "misaligned", ...)


@dataclass
class DockTelemetry:
    busy: bool = False
    current: Optional[str] = None               # rover on the berth / swapping
    queue: List[str] = field(default_factory=list)   # waiting rovers, next first
    empty_packs: int = 0                        # empty packs on the ready rack


@dataclass
class FleetSnapshot:
    t_day_s: float             # sim clock, seconds since midnight
    sun_yaw: float             # rad, rover yaw that faces the sun (Economics.sun_yaw)
    sun_el: float              # rad, sun elevation (<= 0 at night)
    weather: str               # "clear" | "cloudy" | "overcast"
    rovers: List[RoverTelemetry]
    dock: DockTelemetry

    def rover(self, rover_id: str) -> Optional[RoverTelemetry]:
        for r in self.rovers:
            if r.rover_id == rover_id:
                return r
        return None


@dataclass
class Command:
    rover_id: str
    action: str                # one of ACTIONS
    row: int = -1              # harvest cell; -1 = n/a
    slot: int = -1
    face: str = "keep"         # "sun" | "dock" | "keep"
    dock_order: int = -1       # FIFO position at the dock (0 = next onto the berth); -1 = n/a
    reason: str = ""
    # --- extras beyond the LLM schema (defaults keep Contract 1 construction working) ---
    source: str = "rules"      # "rules" or "llm:<model id>" -- show it on screen
    via_xy: Optional[Tuple[float, float]] = None   # recovery detour waypoint (rules only)

    def key(self) -> tuple:
        """What the executor should compare to decide whether the order CHANGED."""
        return (self.action, self.row, self.slot, self.face, self.dock_order, self.via_xy)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Plan:
    commands: Dict[str, Command]
    source: str                # "llm:<model id>" | "rules" | "rules (llm: <short error>)"
    summary: str = ""
    t_day_s: float = 0.0
    errors: List[str] = field(default_factory=list)   # why LLM commands were dropped

    def to_dict(self) -> dict:
        return {"source": self.source, "summary": self.summary, "t_day_s": self.t_day_s,
                "errors": list(self.errors),
                "commands": {k: c.to_dict() for k, c in self.commands.items()}}


def hhmm(t_day_s: float) -> str:
    t = int(t_day_s) % int(DAY_S)
    return f"{t // 3600:02d}:{(t % 3600) // 60:02d}"
