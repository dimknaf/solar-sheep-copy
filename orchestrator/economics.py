"""orchestrator/economics.py -- sun, panel power, weather and pack SoC for the 3D factory.

A Python port of the browser sim's economics (config.js / simulation.js) with a real sun
vector instead of a screen point, so turning the rover to face the sun genuinely matters.

Clock. Physics runs at 1x; the sun, harvest and SoC clock run at K x (``timelapse``,
default 120): ``day_seconds(t_phys) = start_hour*3600 + K*t_phys`` (mod 24 h). Motor
energy is billed on PHYSICS time (the rover really drove for dt seconds); charging on the
time-lapse clock. Label the time-lapse on screen. At K = 120 a pack fills 0.05 -> 0.95 in
about 2.8 sim-hours = ~85 s real.

Sun (London near the equinox). p = (d - 6 h) / 12 h; azimuth (compass) az = 90 + 180 p deg
(east at sunrise, south at noon, west at sunset); elevation el = elMax * sin(pi p) with
elMax = 38 deg. The same formula continues through the night (p outside [0, 1]), where
sin(pi p) < 0, so the sun is below the horizon: sz <= 0.
    s = (cos el sin az, cos el cos az, sin el)            X east, Y north, Z up

Panel. Tilted tau = 20 deg toward the rover's +X (the panel leans forward):
    n = (sin tau cos yaw, sin tau sin yaw, cos tau)
    P = 195.6 W * weather * max(0, n . s)                  (0 when the sun is down)
Facing the sun n.s = sin(el + tau): at el = 38 deg 0.85 facing vs 0.31 facing away.

Weather (config.js): clear 1.0 / cloudy 0.8 / overcast 0.45, each lasting 1-4 sim-hours,
picked by weight 0.45 / 0.35 / 0.20 from a random.Random(1337) (deterministic run).
"""

from __future__ import annotations

import math
import random
from typing import Optional, Sequence, Tuple

from .telemetry import (CAPACITY_WH, DAY_S, FULL, PANEL_PEAK_W, RESERVE, SUNRISE_S,
                        SUNSET_S)

Vec3 = Tuple[float, float, float]

CHARGE_EFF = 0.95         # config.js battery.chargeEff
DISCHARGE_EFF = 0.95      # config.js battery.dischargeEff
EL_MAX_DEG = 38.0
TILT_DEG = 20.0

WEATHER = {               # name -> (solar factor, pick weight)   config.js weather.states
    "clear": (1.0, 0.45),
    "cloudy": (0.8, 0.35),
    "overcast": (0.45, 0.20),
}
WEATHER_MIN_H, WEATHER_MAX_H = 1.0, 4.0


class Economics:
    def __init__(self, timelapse: float = 120.0, start_hour: float = 8.0,
                 weather: str = "clear", seed: int = 1337,
                 el_max_deg: float = EL_MAX_DEG, tilt_deg: float = TILT_DEG):
        self.timelapse = float(timelapse)
        self.start_hour = float(start_hour)
        self.el_max = math.radians(el_max_deg)
        self.tilt = math.radians(tilt_deg)
        self._rng = random.Random(seed)
        if weather not in WEATHER:          # e.g. "random": seeded pick
            weather = self._pick_weather()
        self.weather = weather
        self._weather_left_s = self._weather_duration()

    # ---- clock ----------------------------------------------------------------
    def day_seconds(self, t_phys_s: float) -> float:
        return (self.start_hour * 3600.0 + self.timelapse * t_phys_s) % DAY_S

    # ---- sun ------------------------------------------------------------------
    def _az_el(self, t_day_s: float) -> Tuple[float, float]:
        p = (t_day_s % DAY_S - SUNRISE_S) / (SUNSET_S - SUNRISE_S)
        az = math.radians(90.0 + 180.0 * p)
        el = self.el_max * math.sin(math.pi * p)
        return az, el

    def sun_dir(self, t_day_s: float) -> Vec3:
        """Unit vector TOWARD the sun, world frame (X east, Y north, Z up)."""
        az, el = self._az_el(t_day_s)
        ce = math.cos(el)
        return (ce * math.sin(az), ce * math.cos(az), math.sin(el))

    def sun_yaw(self, t_day_s: float) -> float:
        sx, sy, _ = self.sun_dir(t_day_s)
        return math.atan2(sy, sx)

    def sun_el(self, t_day_s: float) -> float:
        return self._az_el(t_day_s)[1]

    # ---- panel ----------------------------------------------------------------
    def panel_normal(self, yaw: float) -> Vec3:
        """World-frame panel normal of a level rover at ``yaw`` (panel leans toward +X)."""
        st = math.sin(self.tilt)
        return (st * math.cos(yaw), st * math.sin(yaw), math.cos(self.tilt))

    @property
    def weather_f(self) -> float:
        return WEATHER[self.weather][0]

    def panel_power_w(self, normal_w: Sequence[float], sun_dir: Sequence[float],
                      weather_f: Optional[float] = None) -> float:
        if sun_dir[2] <= 0.0:
            return 0.0
        wf = self.weather_f if weather_f is None else weather_f
        dot = sum(n * s for n, s in zip(normal_w, sun_dir))
        return PANEL_PEAK_W * wf * max(0.0, dot)

    # ---- battery --------------------------------------------------------------
    def step_soc(self, soc: float, pv_w: float, motor_w: float, dt_phys_s: float,
                 harvesting: bool) -> float:
        """Charge on the time-lapse clock (only while harvesting), motor on physics time."""
        wh = soc * CAPACITY_WH
        if harvesting and pv_w > 0.0:
            wh += pv_w * CHARGE_EFF * (dt_phys_s * self.timelapse) / 3600.0
        if motor_w > 0.0:
            wh -= motor_w / DISCHARGE_EFF * dt_phys_s / 3600.0
        return min(1.0, max(RESERVE, wh / CAPACITY_WH))

    # ---- weather --------------------------------------------------------------
    def _pick_weather(self) -> str:
        r, acc = self._rng.random(), 0.0
        for name, (_, w) in WEATHER.items():
            acc += w
            if r <= acc:
                return name
        return "clear"

    def _weather_duration(self) -> float:
        return (WEATHER_MIN_H + self._rng.random() * (WEATHER_MAX_H - WEATHER_MIN_H)) * 3600.0

    def set_weather(self, name: str) -> None:
        if name not in WEATHER:
            raise ValueError(f"unknown weather {name!r}; one of {sorted(WEATHER)}")
        self.weather = name
        self._weather_left_s = self._weather_duration()

    def step_weather(self, dt_phys_s: float) -> bool:
        """Advance the weather state machine on the time-lapse clock; True if it changed."""
        self._weather_left_s -= dt_phys_s * self.timelapse
        if self._weather_left_s > 0.0:
            return False
        old = self.weather
        self.weather = self._pick_weather()
        self._weather_left_s = self._weather_duration()
        return self.weather != old


__all__ = ["Economics", "WEATHER", "CHARGE_EFF", "DISCHARGE_EFF", "RESERVE", "FULL",
           "CAPACITY_WH", "PANEL_PEAK_W"]
