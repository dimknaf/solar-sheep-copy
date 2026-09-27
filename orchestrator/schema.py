"""orchestrator/schema.py -- the strict JSON command schema and the validator behind it.

The schema is what the LLM must emit (``response_format`` json_schema, strict). Token
Factory's strict mode needs ``additionalProperties: false`` on every object and EVERY
property listed in ``required`` (else 400 invalid_json_schema), so there are no optional
fields: -1 means "n/a" and the real rules live in ``validate_plan`` below, not in the schema.

``validate_plan`` keeps the good commands and drops the bad ones PER ROVER, saying why;
the brain then fills the dropped rovers from the rule plan. Rules enforced:
  * well-formed command, known rover, no rover twice, action/face in their enums;
  * harvest: cell on the grid, not taken by another rover, sun up, pack not already full;
  * swap: only the rover on the berth, berth not held by another rover, an empty pack on
    the rack, not a rover that has just swapped;
  * dock: only if soc >= FULL, or brownout (soc <= 0.15), or recovering from a FAILURE /
    fault, or already in the dock line, or the day is over (recall: deliver the harvest,
    as simulation.js does); dock_order >= 0 and unique;
  * a rover mid-swap (swapping + RUNNING) may only keep swapping or hold.
"""

from __future__ import annotations

import json
import re
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .layout import Layout
from .telemetry import (ACTIONS, BROWNOUT, FACES, FULL, RECALL_JITTER_S, RECALL_S, Command,
                        FleetSnapshot)

FIELDS = ("rover_id", "action", "row", "slot", "face", "dock_order", "reason")
MAX_REASON = 160


def plan_schema(rover_ids: Sequence[str]) -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "required": ["summary", "commands"],
        "properties": {
            "summary": {"type": "string"},
            "commands": {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "required": list(FIELDS),
                "properties": {
                    "rover_id": {"type": "string", "enum": list(rover_ids)},
                    "action": {"type": "string", "enum": list(ACTIONS)},
                    "row": {"type": "integer"},
                    "slot": {"type": "integer"},
                    "face": {"type": "string", "enum": list(FACES)},
                    "dock_order": {"type": "integer"},
                    "reason": {"type": "string"},
                }}},
        }}


def response_format(rover_ids: Sequence[str], mode: str = "json_schema") -> dict:
    if mode == "json_object":
        return {"type": "json_object"}
    return {"type": "json_schema",
            "json_schema": {"name": "fleet_plan", "strict": True,
                            "schema": plan_schema(rover_ids)}}


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.I)


def parse_json_text(text: Optional[str]) -> Tuple[Optional[dict], str]:
    """Model text -> dict. Tolerates ``` fences and prose around ONE JSON object."""
    if not text or not text.strip():
        return None, "empty response"
    s = _FENCE.sub("", text.strip())
    try:
        obj = json.loads(s)
    except ValueError:
        i, j = s.find("{"), s.rfind("}")
        if i < 0 or j <= i:
            return None, "no JSON object in response"
        try:
            obj = json.loads(s[i:j + 1])
        except ValueError as e:
            return None, f"bad JSON: {e.msg} at {e.pos}"
    if not isinstance(obj, dict):
        return None, "top level is not an object"
    return obj, ""


def _as_int(v) -> Optional[int]:
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return None


def validate_plan(obj: Mapping, snapshot: FleetSnapshot, layout: Layout,
                  rover_ids: Sequence[str], *, model: str = "",
                  reserved_cells: Optional[Mapping[str, Tuple[int, int]]] = None,
                  just_swapped: Iterable[str] = (),
                  brownout_armed: Optional[Iterable[str]] = None
                  ) -> Tuple[Dict[str, Command], List[str], str]:
    """-> (valid commands by rover, errors, summary). Never raises on bad input.

    ``reserved_cells``: harvest cells the RULE plan gives each rover; a rover the LLM does
    not command keeps its rule cell, so the LLM may not send anyone else there.
    ``brownout_armed``: rovers whose brownout exception is live (the rules' latch: a rover
    that has just swapped to an empty pack is not a brownout); None = all rovers.
    """
    errors: List[str] = []
    if not isinstance(obj, Mapping):
        return {}, ["plan is not a JSON object"], ""
    summary = obj.get("summary")
    summary = summary[:300] if isinstance(summary, str) else ""
    cmds = obj.get("commands")
    if not isinstance(cmds, list):
        return {}, ["'commands' is missing or not a list"], summary
    extra = set(obj) - {"summary", "commands"}
    if extra:
        errors.append(f"ignored extra top-level keys {sorted(extra)}")

    known = set(rover_ids)
    source = f"llm:{model}" if model else "llm"
    # ---- pass 1: shape, known rover, duplicates --------------------------------
    parsed: List[dict] = []
    seen: Dict[str, int] = {}
    for i, c in enumerate(cmds):
        if not isinstance(c, Mapping):
            errors.append(f"commands[{i}]: not an object")
            continue
        missing = [f for f in FIELDS if f not in c]
        if missing:
            errors.append(f"commands[{i}]: missing {missing}")
            continue
        rid = c["rover_id"]
        if not isinstance(rid, str) or rid not in known:
            errors.append(f"commands[{i}]: unknown rover {rid!r}")
            continue
        seen[rid] = seen.get(rid, 0) + 1
        parsed.append(dict(c))
    dups = {r for r, n in seen.items() if n > 1}
    for r in sorted(dups):
        errors.append(f"{r}: commanded {seen[r]} times -- all dropped")
    parsed = [c for c in parsed if c["rover_id"] not in dups]
    commanded = {c["rover_id"] for c in parsed}

    taken = {cell for r, cell in (reserved_cells or {}).items()
             if r not in commanded and cell is not None}
    orders: set = set()
    swapped = set(just_swapped)
    armed = None if brownout_armed is None else set(brownout_armed)
    dock = snapshot.dock
    sun_up = snapshot.sun_el > 0.0
    day_over = (not sun_up) or snapshot.t_day_s >= RECALL_S - RECALL_JITTER_S
    out: Dict[str, Command] = {}

    # ---- pass 2: the rules -----------------------------------------------------
    for c in parsed:
        rid = c["rover_id"]
        rv = snapshot.rover(rid)
        act, face, reason = c["action"], c["face"], c["reason"]
        row, slot, order = _as_int(c["row"]), _as_int(c["slot"]), _as_int(c["dock_order"])

        def bad(msg: str) -> None:
            errors.append(f"{rid}: {act} rejected -- {msg}")

        if act not in ACTIONS:
            errors.append(f"{rid}: unknown action {act!r}")
            continue
        if face not in FACES:
            bad(f"unknown face {face!r}")
            continue
        if row is None or slot is None or order is None:
            bad("row/slot/dock_order must be integers")
            continue
        if not isinstance(reason, str):
            reason = ""
        if rv is None:
            bad("no telemetry for this rover")
            continue
        at_berth = dock.current == rid or layout.at_berth(rv.x, rv.y)
        if rv.state == "swapping" and rv.last_status == "RUNNING" and act not in ("swap", "hold"):
            bad("rover is mid-swap; it must keep swapping")
            continue

        if act == "harvest":
            if not layout.in_grid(row, slot):
                bad(f"cell ({row},{slot}) is off the {layout.rows}x{layout.slots} grid")
                continue
            if (row, slot) in taken:
                bad(f"cell ({row},{slot}) is taken")
                continue
            if not sun_up:
                bad("sun is down")
                continue
            if rv.soc >= FULL:
                bad(f"pack full (soc {rv.soc:.2f}); harvesting would clip")
                continue
            taken.add((row, slot))
            order = -1
        elif act == "swap":
            if not at_berth:
                bad("rover is not on the berth")
                continue
            if dock.current not in (None, rid):
                bad(f"berth is held by {dock.current}")
                continue
            if dock.empty_packs <= 0:
                bad("no empty pack on the rack")
                continue
            if rid in swapped:
                bad("rover has just swapped")
                continue
            row, slot, order = -1, -1, -1
        elif act == "dock":
            in_line = (rid in dock.queue or dock.current == rid
                       or rv.state in ("to_dock", "queuing", "swapping"))
            recovering = rv.last_status == "FAILURE" or rv.state == "fault"
            brownout = rv.soc <= BROWNOUT and (armed is None or rid in armed)
            if not (rv.soc >= FULL or brownout or recovering or in_line or day_over):
                bad(f"soc {rv.soc:.2f} < {FULL} and no brownout/recovery/recall")
                continue
            if order < 0:
                bad("dock_order must be >= 0")
                continue
            if order in orders:
                bad(f"dock_order {order} is not unique")
                continue
            if rid in swapped and at_berth:
                bad("rover has just swapped")
                continue
            orders.add(order)
            row, slot = -1, -1
        else:                       # rest / hold
            row, slot, order = -1, -1, -1
        out[rid] = Command(rid, act, row, slot, face, order, reason[:MAX_REASON], source=source)

    for rid in rover_ids:
        if rid not in commanded and rid not in dups:
            errors.append(f"{rid}: no command")
    return out, errors, summary
