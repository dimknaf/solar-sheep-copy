"""orchestrator/brain.py -- FleetBrain: Nemotron re-plans, the rules always have a plan ready.

``tick(snapshot, now_real_s)`` is non-blocking (a few hundred microseconds): it recomputes
the rule plan, notices events, and when one fires it hands ONE request to a background
worker thread (at most one in flight, >= min_interval_s apart, at most max_calls per run).
``plan`` is the latest merged plan: each rover takes the LLM's command when that command is
valid against the CURRENT snapshot, the plan is younger than 2 sim-hours and nothing has
happened to that rover since the request was sent; otherwise the rule command. Every
Command carries ``source`` ("llm:<model>" or "rules") and a ``reason`` for the overlay.

Triggers: start; a pack reaching FULL; any FAILURE; a finished swap; dock busy/current/
queue change; a brownout (soc <= 0.15 away from the dock); a weather change; sunrise /
recall / sunset; heartbeat every 30 sim-minutes.

Log (``log_path``, JSONL, one line per call): time, model, schema mode, latency, tokens,
cost estimate, events, request messages, raw response text, parsed plan and validation
errors -- serialised, then redacted (the key and anything like ``v1.<20+ chars>``).
"""

from __future__ import annotations

import json
import math
import os
import queue
import threading
import time
from dataclasses import replace
from typing import Dict, List, Optional, Sequence

from .client import KEY_ENV, MODEL_ENV, DEFAULT_MODEL, MissingKey, TokenFactoryClient, redact
from .layout import Layout
from .rules import RuleEngine
from .schema import parse_json_text, validate_plan
from .telemetry import (BROWNOUT, DAY_S, FULL, MOTOR_W, PANEL_PEAK_W, RECALL_S, SUNRISE_S,
                        SUNSET_S, Command, DockTelemetry, FleetSnapshot, Plan, RoverTelemetry,
                        hhmm)

HEARTBEAT_SIM_S = 30 * 60.0
PLAN_TTL_SIM_S = 2 * 3600.0


def system_prompt(layout: Layout, rover_ids: Sequence[str]) -> str:
    R, S = layout.rows, layout.slots
    return (
        f"You dispatch Solar Sheep: {len(rover_ids)} solar rovers that harvest sunlight on a "
        f"pasture and bring full battery packs to ONE swap dock. Each rover has a "
        f"{PANEL_PEAK_W:.0f} W panel (x weather x sun angle) and a 500 Wh pack; driving costs "
        f"~{MOTOR_W:.0f} W. soc = pack fraction 0..1.\n"
        "Actions (exactly one command for every rover):\n"
        f"- harvest: sit on pasture cell (row 0..{R - 1}, slot 0..{S - 1}) facing the sun "
        "(face \"sun\"). One rover per cell; keep a rover on its current cell unless there "
        f"is a reason to move. Only while the sun is up and soc < {FULL}.\n"
        "- dock: go to the dock and join its line; dock_order = place in line (0 = next onto "
        f"the berth), unique. Allowed only if soc >= {FULL} (pack full), soc <= {BROWNOUT} "
        "away from the dock (brownout: dock now), recovering from a FAILURE, already in the "
        "dock line, or after recall (~17:20: deliver the day's harvest). You may reorder the "
        "line, e.g. a brownout rover first.\n"
        "- swap: only the rover on the berth (dock.current or at_berth) and only if "
        "dock.empty_packs > 0. One rover on the berth at a time. A rover mid-swap keeps \"swap\".\n"
        "- rest: park (night, before its wake time, or no empty pack on the rack).\n"
        "- hold: stop where it is (fault).\n"
        "Recovery when last_status is FAILURE: traverse failure -> retry the cell or the "
        "nearest free cell; misaligned -> dock (re-drive to the approach); busy -> dock "
        "(queue); no_empty_pack -> rest; timeout at the dock -> dock.\n"
        "Use -1 for row/slot/dock_order when not applicable. face: \"sun\" for harvest, "
        "\"dock\" for dock/swap, else \"keep\". reason: at most 12 words. summary: one sentence.\n"
        "Output ONLY one JSON object, no prose, no markdown: {\"summary\": string, \"commands\": "
        "[{\"rover_id\", \"action\", \"row\", \"slot\", \"face\", \"dock_order\", \"reason\"}]}")


def telemetry_payload(snap: FleetSnapshot, layout: Layout, rules: RuleEngine,
                      events: Sequence[str]) -> dict:
    cells = rules.cell
    used = set(cells.values())
    rovers = []
    for r in snap.rovers:
        rid = r.rover_id
        rovers.append({
            "id": rid, "x": round(r.x, 2), "y": round(r.y, 2), "soc": round(r.soc, 3),
            "state": r.state, "cell": list(cells.get(rid, (-1, -1))),
            "at_berth": snap.dock.current == rid or layout.at_berth(r.x, r.y),
            "pv_w": round(r.pv_w, 1), "last_status": r.last_status,
            "last_reason": r.last_reason,
            "wake": hhmm(SUNRISE_S + rules.wake.get(rid, 0.0)),
            "recall": hhmm(rules.recall.get(rid, RECALL_S))})
    d = snap.dock
    return {
        "time": hhmm(snap.t_day_s), "t_day_s": round(snap.t_day_s),
        "sun": {"el_deg": round(math.degrees(snap.sun_el), 1),
                "facing_yaw_deg": round(math.degrees(snap.sun_yaw), 1)},
        "weather": snap.weather, "events": list(events),
        "dock": {"busy": d.busy, "current": d.current, "queue": list(d.queue),
                 "empty_packs": d.empty_packs},
        "free_cells": [list(c) for c in layout.cells() if c not in used and c not in rules.blocked],
        "rovers": rovers}


def _crossed(t0: float, t1: float, x: float) -> bool:
    if t1 >= t0:
        return t0 < x <= t1
    return x > t0 or x <= t1           # wrapped past midnight


class FleetBrain:
    def __init__(self, rover_ids: Sequence[str], layout: Layout, use_llm: bool,
                 model: Optional[str] = None, log_path: Optional[str] = None,
                 max_calls: int = 300, min_interval_s: float = 5.0, *,
                 transport=None, timeout_s: float = 8.0, base_url: Optional[str] = None,
                 heartbeat_sim_s: float = HEARTBEAT_SIM_S,
                 plan_ttl_sim_s: float = PLAN_TTL_SIM_S, seed: int = 1337):
        self.ids = list(rover_ids)
        self.layout = layout
        self.rules = RuleEngine(self.ids, layout, seed)
        self.model = model or os.environ.get(MODEL_ENV, "").strip() or DEFAULT_MODEL
        self.max_calls = int(max_calls)
        self.min_interval_s = float(min_interval_s)
        self.heartbeat_sim_s = heartbeat_sim_s
        self.plan_ttl_sim_s = plan_ttl_sim_s
        self._log_path = log_path
        self._log_lock = threading.Lock()
        self._lock = threading.Lock()
        self._client: Optional[TokenFactoryClient] = None
        self.use_llm = bool(use_llm)
        self._off_reason = "" if use_llm else "llm off"
        if self.use_llm:
            try:
                self._client = TokenFactoryClient(self.model, base_url, timeout_s, transport)
            except MissingKey:
                self.use_llm = False
                self._off_reason = f"llm off: no {KEY_ENV}"
                self._log({"event": "llm_disabled", "reason": self._off_reason})
        self._system = system_prompt(layout, self.ids)
        self._jobs: "queue.Queue" = queue.Queue(maxsize=1)
        self._thread: Optional[threading.Thread] = None
        self._in_flight = False
        self._result: Optional[dict] = None
        self._llm: Optional[dict] = None        # latest usable LLM plan
        self._calls = 0
        self._cost = 0.0
        self._tok_in = self._tok_out = 0
        self._last_latency = 0.0
        self._last_error = ""
        self._last_call_real: Optional[float] = None
        self._last_req_t: Optional[float] = None
        self._pending: List[str] = []
        self._rover_seq: Dict[str, int] = {rid: 0 for rid in self.ids}
        self._prev: Optional[FleetSnapshot] = None
        self._cap_logged = False
        self._last_exchange: dict = {}
        self._closed = False
        self._plan = Plan({rid: Command(rid, "hold", reason="no telemetry yet") for rid in self.ids},
                          "rules", "waiting for the first snapshot")
        if self.use_llm:                  # started here: Thread.start() can take a scheduler tick
            self._thread = threading.Thread(target=self._worker, name="fleet-brain-llm",
                                            daemon=True)
            self._thread.start()

    # ---- public ---------------------------------------------------------------
    @property
    def plan(self) -> Plan:
        return self._plan

    @property
    def status(self) -> dict:
        return {"calls": self._calls, "in_flight": self._in_flight,
                "cost_usd_est": round(self._cost, 6), "last_error": self._last_error,
                "model": self.model if self.use_llm else None, "source": self._plan.source,
                "schema_mode": self._client.schema_mode if self._client else None,
                "use_llm": self.use_llm, "off_reason": self._off_reason,
                "tokens_in": self._tok_in, "tokens_out": self._tok_out,
                "last_latency_s": self._last_latency, "max_calls": self.max_calls}

    @property
    def last_exchange(self) -> dict:
        """Last request telemetry + raw response, for the 'JSON on the left' overlay."""
        return dict(self._last_exchange)

    def tick(self, snapshot: FleetSnapshot, now_real_s: float) -> None:
        try:
            self._tick(snapshot, now_real_s)
        except Exception as e:                      # never raise into the sim loop
            self._last_error = redact(f"tick: {type(e).__name__}: {e}")[:200]
            try:
                self._plan = self.rules.plan(snapshot)
            except Exception:
                pass

    def close(self) -> None:
        self._closed = True
        if self._thread is not None:
            try:
                self._jobs.put_nowait(None)
            except queue.Full:
                pass
            self._thread.join(timeout=0.2)

    # ---- internals ------------------------------------------------------------
    def _tick(self, snap: FleetSnapshot, now: float) -> None:
        rule_plan = self.rules.plan(snap)
        self._collect()
        events = self._events(snap)
        self._prev = _copy_snap(snap)
        if self.use_llm:
            for e in events:
                if e not in self._pending:
                    self._pending.append(e)
            self._maybe_call(snap, now)
        self._plan = self._merge(snap, rule_plan)

    def _events(self, snap: FleetSnapshot) -> List[str]:
        p = self._prev
        if p is None:
            return ["start"]
        ev: List[str] = []

        def rover_event(rid: str, what: str) -> None:
            self._rover_seq[rid] = self._rover_seq.get(rid, 0) + 1
            ev.append(f"{what}:{rid}")

        for r in snap.rovers:
            q = p.rover(r.rover_id)
            if q is None:
                continue
            if r.soc >= FULL > q.soc:
                rover_event(r.rover_id, "pack_full")
            if r.last_status == "FAILURE" and (q.last_status != "FAILURE"
                                               or q.last_reason != r.last_reason):
                rover_event(r.rover_id, f"failure({r.last_reason or '?'})")
            if (r.soc <= BROWNOUT < q.soc) and not self.layout.near_dock(r.x, r.y):
                rover_event(r.rover_id, "brownout")
            swap_done = ((q.state == "swapping" and r.state != "swapping" and r.last_status != "FAILURE")
                         or (r.state == "swapping" and r.last_status == "SUCCESS"
                             and r.last_reason != "idle" and q.last_status != "SUCCESS"))
            if swap_done:
                rover_event(r.rover_id, "swap_done")
        a, b = p.dock, snap.dock
        if (a.busy, a.current, list(a.queue)) != (b.busy, b.current, list(b.queue)):
            ev.append("dock_line")
        if p.weather != snap.weather:
            ev.append(f"weather({snap.weather})")
        for name, x in (("sunrise", SUNRISE_S), ("recall", RECALL_S), ("sunset", SUNSET_S)):
            if _crossed(p.t_day_s, snap.t_day_s, x):
                ev.append(name)
        if self._last_req_t is not None and not ev:
            if (snap.t_day_s - self._last_req_t) % DAY_S >= self.heartbeat_sim_s:
                ev.append("heartbeat")
        return ev

    def _maybe_call(self, snap: FleetSnapshot, now: float) -> None:
        if not self._pending or self._in_flight or self._closed:
            return
        if self._calls >= self.max_calls:
            if not self._cap_logged:
                self._cap_logged = True
                self._last_error = f"call cap {self.max_calls} reached -> rules only"
                self._log({"event": "cap_reached", "max_calls": self.max_calls,
                           "t_day_s": snap.t_day_s})
            self._pending.clear()
            return
        if self._last_call_real is not None and now - self._last_call_real < self.min_interval_s:
            return
        events, self._pending = self._pending, []
        payload = telemetry_payload(snap, self.layout, self.rules, events)
        user = json.dumps(payload, separators=(",", ":"))
        messages = [{"role": "system", "content": self._system},
                    {"role": "user", "content": user}]
        reserved = {rid: tuple(c) for rid, c in self.rules.cell.items()}
        job = {"messages": messages, "snap": _copy_snap(snap), "events": events,
               "seq": dict(self._rover_seq), "reserved": reserved,
               "just_swapped": set(self.rules.just_swapped), "armed": self._armed()}
        self._in_flight = True                     # before put: the worker may finish first
        try:
            self._jobs.put_nowait(job)
        except queue.Full:
            self._in_flight = False
            return
        self._calls += 1
        self._last_call_real = now
        self._last_req_t = snap.t_day_s
        self._last_exchange = {"telemetry": user, "events": events, "response_text": "",
                               "summary": "", "t_day_s": snap.t_day_s}

    def _worker(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return
            try:
                out = self._run(job)
            except Exception as e:
                out = {"ok": False, "error": redact(f"{type(e).__name__}: {e}")[:200]}
            with self._lock:
                self._result = out
                self._in_flight = False

    def _run(self, job: dict) -> dict:
        client = self._client
        res = client.chat(job["messages"], self.ids)
        obj, perr = (parse_json_text(res.text) if res.ok else (None, res.error))
        cmds, verrs, summary = ({}, [], "")
        if obj is not None:
            snap = job["snap"]
            cmds, verrs, summary = validate_plan(
                obj, snap, self.layout, self.ids, model=client.model,
                reserved_cells=job["reserved"], just_swapped=job["just_swapped"],
                brownout_armed=job["armed"])
        error = "" if obj is not None else (perr or res.error or "no plan")
        self._log({"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "model": client.model, "schema_mode": res.schema_mode,
                   "status": res.status, "attempts": res.attempts,
                   "latency_s": res.latency_s, "tokens_in": res.tokens_in,
                   "tokens_out": res.tokens_out, "cost_est_usd": round(res.cost_usd, 6),
                   "finish_reason": res.finish_reason, "events": job["events"],
                   "t_day_s": job["snap"].t_day_s, "request": job["messages"],
                   "response_text": res.text, "plan": obj, "valid_rovers": sorted(cmds),
                   "errors": verrs, "error": error})
        return {"ok": obj is not None and bool(cmds), "obj": obj, "summary": summary,
                "t_day_s": job["snap"].t_day_s, "seq": job["seq"], "model": client.model,
                "latency": res.latency_s, "tin": res.tokens_in, "tout": res.tokens_out,
                "cost": res.cost_usd, "text": res.text,
                "error": error or ("" if cmds else "no valid command")}

    def _collect(self) -> None:
        with self._lock:
            out, self._result = self._result, None
        if out is None:
            return
        self._cost += out.get("cost", 0.0)
        self._tok_in += out.get("tin", 0)
        self._tok_out += out.get("tout", 0)
        self._last_latency = out.get("latency", 0.0)
        self._last_exchange.update({"response_text": out.get("text", ""),
                                    "summary": out.get("summary", "")})
        if out.get("ok"):
            self._llm = out
            self._last_error = ""
        else:
            self._last_error = _short(out.get("error", "llm error"))

    def _merge(self, snap: FleetSnapshot, rule_plan: Plan) -> Plan:
        cmds: Dict[str, Command] = dict(rule_plan.commands)
        llm, errors, used_llm = self._llm, [], False
        if llm is not None and (snap.t_day_s - llm["t_day_s"]) % DAY_S > self.plan_ttl_sim_s:
            errors.append("llm plan older than 2 sim-hours -> rules")
            llm = self._llm = None
        if llm is not None:
            reserved = {rid: (c.row, c.slot) for rid, c in rule_plan.commands.items()
                        if c.action == "harvest"}
            valid, verrs, _ = validate_plan(llm["obj"], snap, self.layout, self.ids,
                                            model=llm["model"], reserved_cells=reserved,
                                            just_swapped=self.rules.just_swapped,
                                            brownout_armed=self._armed())
            errors.extend(verrs)
            for rid, c in valid.items():
                if self._rover_seq.get(rid, 0) != llm["seq"].get(rid, 0):
                    errors.append(f"{rid}: event since the request -> rules")
                    continue
                cmds[rid] = c
                used_llm = True
            if used_llm:
                self._fix_conflicts(cmds, snap)
                self.rules.adopt(cmds)
        if used_llm:
            n = sum(c.source != "rules" for c in cmds.values())
            source = f"llm:{llm['model']}"
            summary = llm.get("summary") or ""
            if n < len(cmds):
                summary = (summary + f" [rules for {len(cmds) - n}]").strip()
        else:
            source = "rules"
            why = self._off_reason if not self.use_llm else _short(self._last_error)
            if why:
                source = f"rules ({why})" if why.startswith("llm off") else f"rules (llm: {why})"
            summary = rule_plan.summary
        return Plan(cmds, source, summary, snap.t_day_s, errors)

    def _fix_conflicts(self, cmds: Dict[str, Command], snap: FleetSnapshot) -> None:
        """Rule-fallback rovers must not sit on a cell an LLM command took; renumber the line."""
        llm_cells = {(c.row, c.slot) for c in cmds.values()
                     if c.source != "rules" and c.action == "harvest"}
        for rid, c in list(cmds.items()):
            if c.source == "rules" and c.action == "harvest" and (c.row, c.slot) in llm_cells:
                r = snap.rover(rid)
                taken = {(o.row, o.slot) for k, o in cmds.items() if k != rid and o.action == "harvest"}
                new = self.rules.free_cell(rid, r.x, r.y, taken) if r else None
                if new is not None:
                    self.rules.cell[rid] = new
                    cmds[rid] = replace(c, row=new[0], slot=new[1],
                                        reason=c.reason + " (cell moved)")
        dockers = sorted((c.dock_order, c.source == "rules", rid) for rid, c in cmds.items()
                         if c.action == "dock")
        for i, (_, _, rid) in enumerate(dockers):
            if cmds[rid].dock_order != i:
                cmds[rid] = replace(cmds[rid], dock_order=i)

    def _armed(self) -> set:
        return {rid for rid, a in self.rules.brownout_armed.items() if a}

    def _log(self, record: dict) -> None:
        if not self._log_path:
            return
        key = os.environ.get(KEY_ENV, "").strip() or None
        try:
            line = redact(json.dumps(record, default=_jsonable, ensure_ascii=False), key)
            with self._log_lock:
                d = os.path.dirname(self._log_path)
                if d:
                    os.makedirs(d, exist_ok=True)
                with open(self._log_path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except Exception as e:
            self._last_error = _short(redact(f"log: {type(e).__name__}", key))


def _copy_snap(s: FleetSnapshot) -> FleetSnapshot:
    """Cheap private copy (the caller may mutate its objects after tick returns)."""
    d = s.dock
    return FleetSnapshot(s.t_day_s, s.sun_yaw, s.sun_el, s.weather,
                         [RoverTelemetry(r.rover_id, r.x, r.y, r.yaw, r.soc, r.state, r.pv_w,
                                         r.last_status, r.last_reason) for r in s.rovers],
                         DockTelemetry(d.busy, d.current, list(d.queue), d.empty_packs))


def _short(s: str, n: int = 60) -> str:
    s = (s or "").replace("\n", " ")
    return s if len(s) <= n else s[: n - 3] + "..."


def _jsonable(o):
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    if hasattr(o, "__dict__"):
        return o.__dict__
    return str(o)
