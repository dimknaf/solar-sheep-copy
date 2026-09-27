"""orchestrator/rules.py -- the rule-based fallback brain (always computed, every tick).

A port of the browser sim's sheep state machine (simulation.js stepSim) to per-rover
commands, plus the recovery rules from the design brief. Level-triggered: given the
snapshot it says what each rover should be doing NOW, so re-issuing it every tick is
harmless; the only memory is what a state machine needs (cell assignment, failure counts,
"just swapped", brownout latch, FIFO arrival order).

Per rover, first match wins:
  fault                          -> hold
  swapping on the berth          -> swap                   (never abandon a swap mid-cycle)
  FAILURE no_empty_pack          -> rest
  FAILURE misaligned / busy      -> dock                   (re-drive to approach / queue)
  FAILURE timeout at the dock    -> dock                   (clear the berth, re-queue)
  FAILURE on a traverse          -> 1st: same cell via an offset waypoint (Command.via_xy)
                                    2nd: nearest free cell (the failed one is blocked)
  just swapped                   -> harvest (day) / rest (night)
  on the berth, in the dock line -> swap  (or rest when the rack has no empty pack)
  brownout soc <= 0.15, active,
    away from the dock           -> dock now  (latched: re-armed once soc > 0.25)
  soc >= FULL                    -> dock      (rest if the rack has no empty pack; once in
                                    the dock line it stays docking down to FULL - 0.05)
  after recall / sun down        -> dock if it carries harvest, else rest  (simulation.js)
  awake (sunrise + wake .. recall) -> harvest own cell, face the sun
  otherwise                      -> rest
Per-rover wake (0-45 min after sunrise) and recall (17:20 +/- 15 min) come from
random.Random(1337), as mulberry32(1337) seeds them in the browser sim. dock_order is FIFO:
the current berth occupant, then dock.queue order, then new dockers by arrival.
"""

from __future__ import annotations

import math
import random
from dataclasses import replace
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .layout import Layout
from .telemetry import (BROWNOUT, FULL, RECALL_JITTER_S, RECALL_S, RESERVE, SUNRISE_S,
                        SWAP_REASONS, WAKE_SPREAD_S, Command, FleetSnapshot, Plan,
                        RoverTelemetry)

Cell = Tuple[int, int]
REARM_SOC = BROWNOUT + 0.10
FULL_HYST = 0.05          # a rover already in the dock line stays "full" down to FULL - this


class RuleEngine:
    def __init__(self, rover_ids: Sequence[str], layout: Layout, seed: int = 1337):
        self.ids = list(rover_ids)
        self.layout = layout
        rng = random.Random(seed)
        self.wake: Dict[str, float] = {}
        self.recall: Dict[str, float] = {}
        for rid in self.ids:
            self.wake[rid] = rng.random() * WAKE_SPREAD_S
            self.recall[rid] = RECALL_S + (rng.random() * 2 - 1) * RECALL_JITTER_S
        cells = layout.spread_cells()
        self.cell: Dict[str, Cell] = {rid: cells[i % len(cells)] for i, rid in enumerate(self.ids)}
        self.via: Dict[str, Tuple[float, float]] = {}
        self.note: Dict[str, str] = {}
        self.fails: Dict[str, int] = {rid: 0 for rid in self.ids}
        self.blocked: Set[Cell] = set()
        self.just_swapped: Set[str] = set()
        self.brownout_armed: Dict[str, bool] = {rid: True for rid in self.ids}
        self._prev: Dict[str, Tuple[str, str, str]] = {}
        self._dock_since: Dict[str, int] = {}
        self._tick = 0

    # ---- memory ---------------------------------------------------------------
    def observe(self, snap: FleetSnapshot) -> None:
        self._tick += 1
        bx, by = self.layout.berth_xy
        for r in snap.rovers:
            rid = r.rover_id
            if rid not in self.fails:
                continue
            p = self._prev.get(rid)
            st, stat, why = r.state, r.last_status, r.last_reason
            done = ((st == "swapping" and stat == "SUCCESS" and why != "idle")
                    or (p is not None and p[0] == "swapping" and st != "swapping"
                        and stat != "FAILURE"))
            if done:
                self.just_swapped.add(rid)
                self.brownout_armed[rid] = False
                self.fails[rid] = 0
                self.via.pop(rid, None)
                self.note.pop(rid, None)
                self._dock_since.pop(rid, None)
            elif rid in self.just_swapped and math.hypot(r.x - bx, r.y - by) > 1.0:
                self.just_swapped.discard(rid)
            new_fail = stat == "FAILURE" and (p is None or p[1] != "FAILURE" or p[2] != why)
            dockside = (why in SWAP_REASONS or st in ("queuing", "swapping")
                        or (why == "timeout" and self.layout.near_dock(r.x, r.y)))
            if new_fail and not dockside:
                self._traverse_failed(r, snap)
            if st == "harvesting" and stat != "FAILURE":
                self.fails[rid] = 0
                self.via.pop(rid, None)
                self.note.pop(rid, None)
            if r.soc > REARM_SOC:
                self.brownout_armed[rid] = True
            self._prev[rid] = (st, stat, why)

    def _traverse_failed(self, r: RoverTelemetry, snap: FleetSnapshot) -> None:
        rid, why = r.rover_id, r.last_reason or "traverse"
        self.fails[rid] += 1
        if r.state == "to_dock":                          # on the way to the dock
            ax, ay = self.layout.approach_xy
            side = 1 if self.fails[rid] % 2 else -1
            self.via[rid] = self.layout.detour_xy(r.x, r.y, ax, ay, side)
            self.note[rid] = f"recover: traverse FAILURE ({why}) -> dock via offset waypoint"
            return
        row, slot = self.cell[rid]
        if self.fails[rid] == 1:
            tx, ty = self.layout.slot_xy(row, slot)
            self.via[rid] = self.layout.detour_xy(r.x, r.y, tx, ty, 1)
            self.note[rid] = f"recover: traverse FAILURE ({why}) -> retry ({row},{slot}) via offset"
            return
        self.blocked.add((row, slot))
        new = self.free_cell(rid, r.x, r.y)
        self.via.pop(rid, None)
        self.fails[rid] = 0
        if new is not None and new != (row, slot):
            self.cell[rid] = new
            self.note[rid] = (f"recover: 2nd traverse FAILURE at ({row},{slot}) -> "
                              f"nearest free cell ({new[0]},{new[1]})")
        else:
            self.note[rid] = f"recover: 2nd traverse FAILURE ({why}), no free cell -> retry"

    def free_cell(self, rid: str, x: float, y: float,
                  extra_taken: Iterable[Cell] = ()) -> Optional[Cell]:
        taken = {c for o, c in self.cell.items() if o != rid} | set(extra_taken)
        c = self.layout.nearest_free_cell(x, y, taken, exclude=tuple(self.blocked))
        return c if c is not None else self.layout.nearest_free_cell(x, y, taken)

    def adopt(self, commands: Dict[str, Command]) -> None:
        """Take over the cells / dock arrivals a (validated) LLM plan decided."""
        for rid, c in commands.items():
            if rid not in self.cell or c.source == "rules":
                continue
            if c.action == "harvest":
                want = (c.row, c.slot)
                if self.cell[rid] != want:
                    holder = next((o for o, cc in self.cell.items() if cc == want and o != rid), None)
                    if holder is not None:
                        self.cell[holder] = self.cell[rid]
                    self.cell[rid] = want
                    self.via.pop(rid, None)
            elif c.action == "dock":
                self._dock_since.setdefault(rid, self._tick)

    # ---- decisions ------------------------------------------------------------
    def plan(self, snap: FleetSnapshot) -> Plan:
        self.observe(snap)
        cmds: Dict[str, Command] = {}
        for rid in self.ids:
            r = snap.rover(rid)
            cmds[rid] = (Command(rid, "hold", reason="no telemetry") if r is None
                         else self._decide(r, snap))
        self.number_dock_line(cmds, snap)
        n_dock = sum(c.action in ("dock", "swap") for c in cmds.values())
        n_harv = sum(c.action == "harvest" for c in cmds.values())
        summary = f"rules: {n_harv} harvesting, {n_dock} at/to the dock"
        return Plan(cmds, "rules", summary, snap.t_day_s)

    def number_dock_line(self, cmds: Dict[str, Command], snap: FleetSnapshot) -> None:
        """FIFO dock_order 0..k-1 over every 'dock' command (0 = next onto the berth)."""
        dock = snap.dock
        dockers = [rid for rid, c in cmds.items() if c.action == "dock"]
        for rid in list(self._dock_since):
            if cmds.get(rid) is None or cmds[rid].action not in ("dock", "swap"):
                self._dock_since.pop(rid, None)
        ax, ay = self.layout.approach_xy
        for rid in dockers:                  # FIFO key frozen at arrival: (tick, distance)
            if rid not in self._dock_since or not isinstance(self._dock_since[rid], tuple):
                r = snap.rover(rid)
                dist = math.hypot(r.x - ax, r.y - ay) if r else 0.0
                self._dock_since[rid] = (self._dock_since.get(rid, self._tick), round(dist, 2))

        def key(rid: str):
            if rid == dock.current:
                return (0, 0, 0.0, rid)
            if rid in dock.queue:
                return (1, dock.queue.index(rid), 0.0, rid)
            since, dist = self._dock_since[rid]
            return (2, since, dist, rid)

        for i, rid in enumerate(sorted(dockers, key=key)):
            cmds[rid] = replace(cmds[rid], dock_order=i)

    def _decide(self, r: RoverTelemetry, snap: FleetSnapshot) -> Command:
        rid, soc, dock, d = r.rover_id, r.soc, snap.dock, snap.t_day_s
        sun_up = snap.sun_el > 0.0
        awake = sun_up and SUNRISE_S + self.wake[rid] <= d < self.recall[rid]
        day_over = (not sun_up) or d >= self.recall[rid]
        at_berth = dock.current == rid or self.layout.at_berth(r.x, r.y)
        in_line = rid in dock.queue or dock.current == rid or r.state in ("to_dock", "queuing")
        near = at_berth or in_line or self.layout.near_dock(r.x, r.y)
        active = r.state in ("to_field", "harvesting", "to_dock", "queuing")

        def harvest(reason: str) -> Command:
            row, slot = self.cell[rid]
            return Command(rid, "harvest", row, slot, "sun", -1, reason, via_xy=self.via.get(rid))

        def dock_cmd(reason: str) -> Command:
            return Command(rid, "dock", -1, -1, "dock", 0, reason, via_xy=self.via.get(rid))

        def rest(reason: str) -> Command:
            return Command(rid, "rest", -1, -1, "keep", -1, reason)

        if r.state == "fault":
            return Command(rid, "hold", reason="fault: hold until cleared")
        if (r.state == "swapping" and r.last_status != "FAILURE" and at_berth
                and rid not in self.just_swapped):
            return Command(rid, "swap", -1, -1, "dock", -1, "swap in progress")
        if r.last_status == "FAILURE":
            why = r.last_reason
            if why == "no_empty_pack":
                return rest("recover: no_empty_pack -> rest until the plant frees a pack")
            if why == "misaligned":
                return dock_cmd("recover: misaligned -> re-drive to the approach, then berth")
            if why == "busy":
                return dock_cmd("recover: dock busy -> queue")
            if why == "timeout" and (near or r.state in ("queuing", "swapping")):
                return dock_cmd("recover: swap timeout -> clear the berth, re-queue")
        if rid in self.just_swapped:
            return harvest("swap done -> back to the pasture") if awake else rest("swap done -> park")
        if at_berth and r.state in ("queuing", "to_dock", "swapping"):
            if dock.empty_packs <= 0:
                return rest("no empty pack on the rack -> rest")
            if dock.current in (None, rid):
                return Command(rid, "swap", -1, -1, "dock", -1, f"on the berth -> swap (soc {soc:.2f})")
        if soc <= BROWNOUT and self.brownout_armed[rid] and active and (in_line or not near):
            return dock_cmd(self.note.get(rid) or f"brownout: soc {soc:.2f} <= {BROWNOUT} -> dock now")
        if soc >= FULL or (in_line and soc >= FULL - FULL_HYST):   # driving burns a little
            if dock.empty_packs <= 0:
                return rest(f"pack full (soc {soc:.2f}) but no empty pack -> rest")
            return dock_cmd(self.note.get(rid) or f"pack full (soc {soc:.2f}) -> dock")
        if day_over:
            if soc > RESERVE + 0.02 and r.state != "resting" and dock.empty_packs > 0:
                return dock_cmd(f"recall: deliver the day's harvest (soc {soc:.2f})")
            return rest("night -> rest" if not sun_up else "after recall -> rest")
        if awake:
            return harvest(self.note.get(rid) or f"harvest (soc {soc:.2f} < {FULL})")
        return rest("before wake-up -> rest")

    def cells_in_use(self) -> List[Cell]:
        return list(self.cell.values())
