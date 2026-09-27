"""Offline checks for the fleet brain -- no network, a fake LLM transport.

Run from the repo root:
  .venv\\Scripts\\python.exe -m unittest orchestrator.tests.test_orchestrator -v
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import threading
import time
import unittest

from orchestrator import (BROWNOUT, FULL, RESERVE, Command, DockTelemetry, Economics,
                          FleetBrain, FleetSnapshot, Layout, RoverTelemetry)
from orchestrator.client import BASE_ENV, KEY_ENV, MODEL_ENV
from orchestrator.rules import RuleEngine
from orchestrator.schema import parse_json_text, plan_schema, validate_plan

IDS = [f"sheep{i}" for i in range(5)]
FAKE_KEY = "v1.FAKEKEYFAKEKEYFAKEKEYFAKE"
LAYOUT = Layout.default(5)
ECON = Economics(timelapse=120.0, start_hour=8.0)
CELLS = RuleEngine(IDS, LAYOUT).cell            # the rules' initial assignment


def snapshot(socs, t=9 * 3600.0, states=None, dock=None, pos=None, status=None):
    rovers = []
    for i, rid in enumerate(IDS):
        x, y = (pos or {}).get(rid, LAYOUT.slot_xy(*CELLS[rid]))
        st = (states or {}).get(rid, "harvesting")
        ls, lr = (status or {}).get(rid, ("RUNNING", ""))
        rovers.append(RoverTelemetry(rid, x, y, 0.0, socs[i], st, 100.0, ls, lr))
    return FleetSnapshot(t, ECON.sun_yaw(t), ECON.sun_el(t), "clear", rovers,
                         dock or DockTelemetry(False, None, [], 3))


STAGED = [0.90, 0.75, 0.55, 0.35, 0.15]


def good_plan():
    cmds = []
    for rid in IDS[:4]:
        r, s = CELLS[rid]
        cmds.append({"rover_id": rid, "action": "harvest", "row": r, "slot": s, "face": "sun",
                     "dock_order": -1, "reason": "keep harvesting"})
    cmds.append({"rover_id": "sheep4", "action": "dock", "row": -1, "slot": -1, "face": "dock",
                 "dock_order": 0, "reason": "brownout"})
    return {"summary": "four harvest, sheep4 docks", "commands": cmds}


def chat_response(content: str) -> str:
    return json.dumps({"choices": [{"message": {"role": "assistant", "content": content},
                                    "finish_reason": "stop"}],
                       "usage": {"prompt_tokens": 900, "completion_tokens": 200}})


class FakeTransport:
    """Scripted transport: ``fn(body_dict) -> (status, text)`` or raises."""

    def __init__(self, fn, delay=0.0):
        self.fn, self.delay, self.calls = fn, delay, []

    def __call__(self, url, headers, body, timeout):
        self.calls.append((url, dict(headers), json.loads(body)))
        if self.delay:
            time.sleep(self.delay)
        return self.fn(json.loads(body))


def wait_idle(brain, snap, t0=100.0, timeout=3.0):
    end = time.time() + timeout
    while time.time() < end:
        if not brain.status["in_flight"]:
            break
        time.sleep(0.005)
    brain.tick(snap, t0)


class EnvCase(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in (KEY_ENV, MODEL_ENV, BASE_ENV)}
        os.environ[KEY_ENV] = FAKE_KEY
        os.environ.pop(MODEL_ENV, None)
        os.environ.pop(BASE_ENV, None)
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)          # cleanups run LIFO: after brain.close()
        self.log = os.path.join(self.tmp.name, "orchestrator.jsonl")

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def brain(self, transport, **kw):
        kw.setdefault("min_interval_s", 0.0)
        b = FleetBrain(IDS, LAYOUT, True, model=kw.pop("model", "nvidia/Nemotron-3_5-Lightning"),
                       log_path=self.log, transport=transport, **kw)
        self.addCleanup(b.close)
        return b


# ---------------------------------------------------------------------------------------
class TestSchema(unittest.TestCase):
    def test_strict_every_object(self):
        def walk(node, path="$"):
            if isinstance(node, dict):
                if node.get("type") == "object":
                    self.assertIs(node.get("additionalProperties"), False, path)
                    self.assertEqual(sorted(node["required"]), sorted(node["properties"]), path)
                for k, v in node.items():
                    walk(v, f"{path}.{k}")
            elif isinstance(node, list):
                for i, v in enumerate(node):
                    walk(v, f"{path}[{i}]")
        s = plan_schema(IDS)
        walk(s)
        item = s["properties"]["commands"]["items"]
        self.assertEqual(item["properties"]["rover_id"]["enum"], IDS)
        json.dumps(s)

    def test_parse_fenced(self):
        obj, err = parse_json_text("```json\n" + json.dumps(good_plan()) + "\n```")
        self.assertEqual(err, "")
        self.assertEqual(len(obj["commands"]), 5)
        self.assertIsNone(parse_json_text("not json {{")[0])


class TestValidator(unittest.TestCase):
    def check(self, plan, snap=None, **kw):
        return validate_plan(plan, snap or snapshot(STAGED), LAYOUT, IDS, model="m", **kw)

    def test_accepts_good_plan(self):
        cmds, errs, summary = self.check(good_plan())
        self.assertEqual(sorted(cmds), IDS)
        self.assertEqual(errs, [])
        self.assertEqual(cmds["sheep4"].action, "dock")
        self.assertEqual(cmds["sheep0"].source, "llm:m")

    def _bad(self, mutate, rover, snap=None, **kw):
        p = good_plan()
        mutate(p["commands"])
        cmds, errs, _ = self.check(p, snap, **kw)
        self.assertNotIn(rover, cmds, errs)
        self.assertTrue(any(rover in e for e in errs), errs)
        return errs

    def test_rejects_unknown_rover(self):
        p = good_plan()
        p["commands"][0]["rover_id"] = "sheep9"
        cmds, errs, _ = self.check(p)
        self.assertTrue(any("unknown rover" in e for e in errs))
        self.assertNotIn("sheep0", cmds)

    def test_rejects_duplicate_rover(self):
        self._bad(lambda c: c.append(dict(c[1])), "sheep1")

    def test_rejects_off_grid(self):
        self._bad(lambda c: c[1].update(row=7), "sheep1")

    def test_rejects_taken_cell(self):
        self._bad(lambda c: c[1].update(row=c[0]["row"], slot=c[0]["slot"]), "sheep1")

    def test_rejects_reserved_cell_of_uncommanded_rover(self):
        def m(c):
            r, s = CELLS["sheep3"]
            del c[3]                                   # sheep3 stays on rules
            c[1].update(row=r, slot=s)
        self._bad(m, "sheep1", reserved_cells=dict(CELLS))

    def test_rejects_swap_off_berth(self):
        self._bad(lambda c: c[1].update(action="swap", row=-1, slot=-1), "sheep1")

    def test_rejects_dock_below_full(self):
        errs = self._bad(lambda c: c[2].update(action="dock", row=-1, slot=-1, dock_order=1), "sheep2")
        self.assertTrue(any("soc 0.55" in e for e in errs))

    def test_accepts_dock_when_full_or_recovering(self):
        p = good_plan()
        p["commands"][0].update(action="dock", row=-1, slot=-1, dock_order=1)
        p["commands"][2].update(action="dock", row=-1, slot=-1, dock_order=2)
        snap = snapshot([0.96, 0.75, 0.55, 0.35, 0.15],
                        status={"sheep2": ("FAILURE", "misaligned")})
        cmds, errs, _ = self.check(p, snap)
        self.assertEqual(cmds["sheep0"].action, "dock")
        self.assertEqual(cmds["sheep2"].action, "dock")

    def test_brownout_exception_respects_latch(self):
        p = good_plan()                                 # sheep4 (0.15) docks
        cmds, _, _ = self.check(p, brownout_armed=set(IDS))
        self.assertIn("sheep4", cmds)
        self._bad(lambda c: None, "sheep4", brownout_armed={"sheep0"})

    def test_rejects_duplicate_dock_order(self):
        def m(c):
            c[0].update(action="dock", row=-1, slot=-1, dock_order=0)
        self._bad(m, "sheep4", snap=snapshot([0.96, 0.75, 0.55, 0.35, 0.15]))

    def test_rejects_missing_field_and_bad_enum(self):
        self._bad(lambda c: c[1].pop("face"), "sheep1")
        self._bad(lambda c: c[1].update(action="fly"), "sheep1")

    def test_rejects_harvest_when_full(self):
        self._bad(lambda c: None, "sheep0", snap=snapshot([0.97, 0.75, 0.55, 0.35, 0.15]))

    def test_rejects_leaving_mid_swap(self):
        bx, by = LAYOUT.berth_xy
        snap = snapshot(STAGED, states={"sheep1": "swapping"}, pos={"sheep1": (bx, by)},
                        dock=DockTelemetry(True, "sheep1", [], 3))
        self._bad(lambda c: None, "sheep1", snap=snap)

    def test_accepts_swap_on_berth(self):
        bx, by = LAYOUT.berth_xy
        snap = snapshot([0.96, 0.75, 0.55, 0.35, 0.15], states={"sheep0": "queuing"},
                        pos={"sheep0": (bx, by)})
        p = good_plan()
        p["commands"][0].update(action="swap", row=-1, slot=-1, face="dock")
        cmds, errs, _ = self.check(p, snap)
        self.assertEqual(cmds["sheep0"].action, "swap", errs)
        snap.dock.empty_packs = 0
        cmds, errs, _ = self.check(p, snap)
        self.assertNotIn("sheep0", cmds)


# ---------------------------------------------------------------------------------------
class TestRules(unittest.TestCase):
    def test_staged_socs(self):
        eng = RuleEngine(IDS, LAYOUT)
        p = eng.plan(snapshot(STAGED))
        c = p.commands
        for rid in IDS[:4]:
            self.assertEqual(c[rid].action, "harvest", c[rid])
            self.assertEqual(c[rid].face, "sun")
        cells = [(c[r].row, c[r].slot) for r in IDS[:4]]
        self.assertEqual(len(set(cells)), 4)
        self.assertEqual(c["sheep4"].action, "dock")            # brownout at 0.15
        self.assertEqual(c["sheep4"].dock_order, 0)
        self.assertIn("brownout", c["sheep4"].reason)
        # sheep0 fills up: it docks too, behind the brownout rover (FIFO)
        p = eng.plan(snapshot([0.96, 0.75, 0.55, 0.35, 0.15],
                              states={"sheep4": "to_dock"}))
        self.assertEqual(p.commands["sheep0"].action, "dock")
        self.assertEqual(p.commands["sheep0"].dock_order, 1)
        self.assertEqual(p.commands["sheep4"].dock_order, 0)
        self.assertEqual(p.source, "rules")

    def test_night_and_morning(self):
        eng = RuleEngine(IDS, LAYOUT)
        p = eng.plan(snapshot(STAGED, t=5 * 3600.0, states={r: "resting" for r in IDS}))
        self.assertTrue(all(c.action == "rest" for c in p.commands.values()))
        p = eng.plan(snapshot([0.6, 0.05, 0.05, 0.05, 0.05], t=22 * 3600.0,
                              states={"sheep1": "resting", "sheep2": "resting",
                                      "sheep3": "resting", "sheep4": "resting"}))
        self.assertEqual(p.commands["sheep0"].action, "dock")   # deliver the day's harvest
        self.assertEqual(p.commands["sheep1"].action, "rest")

    def test_traverse_failure_retry_then_new_cell(self):
        eng = RuleEngine(IDS, LAYOUT)
        eng.plan(snapshot(STAGED, states={"sheep2": "to_field"}))
        fail = {"sheep2": ("FAILURE", "stuck")}
        c = eng.plan(snapshot(STAGED, states={"sheep2": "to_field"}, status=fail)).commands["sheep2"]
        self.assertEqual((c.action, (c.row, c.slot)), ("harvest", CELLS["sheep2"]))
        self.assertIsNotNone(c.via_xy)
        self.assertIn("retry", c.reason)
        eng.plan(snapshot(STAGED, states={"sheep2": "to_field"}))       # executor restarted
        c = eng.plan(snapshot(STAGED, states={"sheep2": "to_field"}, status=fail)).commands["sheep2"]
        self.assertEqual(c.action, "harvest")
        self.assertNotEqual((c.row, c.slot), CELLS["sheep2"])
        self.assertNotIn((c.row, c.slot), [CELLS[r] for r in IDS if r != "sheep2"])
        self.assertIn("nearest free cell", c.reason)

    def test_dock_failures(self):
        eng = RuleEngine(IDS, LAYOUT)
        ax, ay = LAYOUT.approach_xy
        for why, act in (("misaligned", "dock"), ("busy", "dock"), ("no_empty_pack", "rest")):
            snap = snapshot([0.96, 0.75, 0.55, 0.35, 0.5], states={"sheep0": "queuing"},
                            pos={"sheep0": (ax, ay)}, status={"sheep0": ("FAILURE", why)})
            c = eng.plan(snap).commands["sheep0"]
            self.assertEqual(c.action, act, why)
            self.assertIn(why, c.reason)

    def test_berth_swap_then_leave(self):
        eng = RuleEngine(IDS, LAYOUT)
        bx, by = LAYOUT.berth_xy
        socs = [0.96, 0.75, 0.55, 0.35, 0.5]
        d = DockTelemetry(True, "sheep0", [], 3)
        c = eng.plan(snapshot(socs, states={"sheep0": "queuing"}, pos={"sheep0": (bx, by)},
                              dock=d)).commands["sheep0"]
        self.assertEqual(c.action, "swap")
        c = eng.plan(snapshot(socs, states={"sheep0": "swapping"}, pos={"sheep0": (bx, by)},
                              dock=d)).commands["sheep0"]
        self.assertEqual(c.action, "swap")
        socs[0] = RESERVE
        c = eng.plan(snapshot(socs, states={"sheep0": "swapping"}, pos={"sheep0": (bx, by)},
                              status={"sheep0": ("SUCCESS", "")}, dock=d)).commands["sheep0"]
        self.assertEqual(c.action, "harvest")
        self.assertIn("swap done", c.reason)
        # a freshly swapped (empty) rover is not a brownout: no dock loop
        c = eng.plan(snapshot(socs, states={"sheep0": "to_field"},
                              pos={"sheep0": (2.0, 2.0)})).commands["sheep0"]
        self.assertEqual(c.action, "harvest")


# ---------------------------------------------------------------------------------------
class TestBrain(EnvCase):
    def test_llm_plan_used(self):
        tr = FakeTransport(lambda b: (200, chat_response(json.dumps(good_plan()))))
        brain = self.brain(tr)
        snap = snapshot(STAGED)
        brain.tick(snap, 0.0)
        wait_idle(brain, snap)
        p = brain.plan
        self.assertTrue(p.source.startswith("llm:nvidia/Nemotron-3_5-Lightning"), p.source)
        self.assertTrue(all(c.source.startswith("llm:") for c in p.commands.values()))
        self.assertEqual(p.commands["sheep4"].action, "dock")
        body = tr.calls[0][2]
        self.assertEqual(body["response_format"]["type"], "json_schema")
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
        st = brain.status
        self.assertEqual(st["calls"], 1)
        self.assertGreater(st["cost_usd_est"], 0.0)
        with open(self.log, encoding="utf-8") as f:
            rec = json.loads(f.readline())
        self.assertEqual(rec["tokens_in"], 900)
        self.assertEqual(rec["valid_rovers"], IDS)

    def test_fallback_on_timeout(self):
        def boom(b):
            raise TimeoutError("slow")
        tr = FakeTransport(boom)
        brain = self.brain(tr)
        snap = snapshot(STAGED)
        brain.tick(snap, 0.0)
        wait_idle(brain, snap)
        self.assertEqual(len(tr.calls), 2)                    # one retry
        self.assertEqual(brain.plan.source, "rules (llm: timeout)")
        self.assertTrue(all(c.source == "rules" for c in brain.plan.commands.values()))

    def test_fallback_on_bad_json(self):
        brain = self.brain(FakeTransport(lambda b: (200, chat_response("sure! here you go {{"))))
        snap = snapshot(STAGED)
        brain.tick(snap, 0.0)
        wait_idle(brain, snap)
        self.assertTrue(brain.plan.source.startswith("rules (llm:"), brain.plan.source)

    def test_invalid_commands_fall_back_per_rover(self):
        p = good_plan()
        p["commands"][2].update(action="swap", row=-1, slot=-1)      # sheep2 not on the berth
        brain = self.brain(FakeTransport(lambda b: (200, chat_response(json.dumps(p)))))
        snap = snapshot(STAGED)
        brain.tick(snap, 0.0)
        wait_idle(brain, snap)
        cmds = brain.plan.commands
        self.assertEqual(cmds["sheep2"].source, "rules")
        self.assertEqual(cmds["sheep2"].action, "harvest")
        self.assertTrue(cmds["sheep1"].source.startswith("llm:"))
        self.assertTrue(any("sheep2" in e for e in brain.plan.errors))

    def test_schema_rejection_switches_to_json_object(self):
        def fn(b):
            if b["response_format"]["type"] == "json_schema":
                return 400, json.dumps({"error": {"message": "invalid_json_schema: "
                                                  "response_format not supported"}})
            return 200, chat_response(json.dumps(good_plan()))
        tr = FakeTransport(fn)
        brain = self.brain(tr, model="deepseek-ai/DeepSeek-V4.1-Flash")
        snap = snapshot(STAGED)
        brain.tick(snap, 0.0)
        wait_idle(brain, snap)
        self.assertEqual(brain.status["schema_mode"], "json_object")
        self.assertTrue(brain.plan.source.startswith("llm:deepseek"), brain.plan.source)
        self.assertEqual(tr.calls[0][2]["chat_template_kwargs"], {"thinking": False})
        brain.tick(snapshot(STAGED, dock=DockTelemetry(True, None, ["sheep4"], 3)), 50.0)
        wait_idle(brain, snap, 60.0)
        self.assertEqual(tr.calls[-1][2]["response_format"], {"type": "json_object"})

    def test_call_cap(self):
        tr = FakeTransport(lambda b: (200, chat_response(json.dumps(good_plan()))))
        brain = self.brain(tr, max_calls=1)
        snap = snapshot(STAGED)
        brain.tick(snap, 0.0)
        wait_idle(brain, snap)
        snap2 = snapshot(STAGED)
        snap2.weather = "overcast"                               # a trigger
        brain.tick(snap2, 200.0)
        wait_idle(brain, snap2, 300.0)
        self.assertEqual(len(tr.calls), 1)
        self.assertIn("cap", brain.status["last_error"])
        with open(self.log, encoding="utf-8") as f:
            self.assertIn("cap_reached", f.read())

    def test_min_interval_and_one_in_flight(self):
        tr = FakeTransport(lambda b: (200, chat_response(json.dumps(good_plan()))), delay=0.2)
        brain = self.brain(tr, min_interval_s=5.0)
        brain.tick(snapshot(STAGED), 0.0)
        for i in range(5):                                       # events while in flight
            brain.tick(snapshot(STAGED, dock=DockTelemetry(i % 2 == 0, None, [], 3)), 0.1 * i)
        self.assertEqual(len(tr.calls) + (1 if brain.status["in_flight"] else 0) >= 1, True)
        wait_idle(brain, snapshot(STAGED), 1.0)
        self.assertEqual(brain.status["calls"], 1)               # pending events wait 5 s
        brain.tick(snapshot(STAGED), 6.0)
        self.assertEqual(brain.status["calls"], 2)

    def test_tick_never_blocks(self):
        # The fake call hangs until released, so a tick that waited on it would take >= 2 s.
        # On a busy laptop the OS can preempt any single call for a scheduler quantum
        # (~15 ms; seen here with no GC and no worker activity), so the 5 ms budget is
        # asserted on the 95th percentile and the mean, the hard cap (never waits on the
        # network) on the maximum.
        release = threading.Event()

        def hang(b):
            release.wait(2.0)
            return 200, chat_response(json.dumps(good_plan()))
        tr = FakeTransport(hang)
        brain = self.brain(tr)
        times = []
        for i in range(300):
            socs = list(STAGED)
            socs[0] = 0.90 + 0.06 * (i % 2)                      # pack_full events
            snap = snapshot(socs, dock=DockTelemetry(i % 3 == 0, None, [], 3))
            t0 = time.perf_counter()
            brain.tick(snap, i * 0.01)
            times.append(time.perf_counter() - t0)
        self.assertTrue(brain.status["in_flight"])
        release.set()
        times.sort()
        p95 = times[int(0.95 * len(times))]
        self.assertLess(p95, 0.005, f"p95 {p95 * 1e3:.2f} ms")
        self.assertLess(sum(times) / len(times), 0.002)
        self.assertLess(times[-1], 0.050, f"worst tick {times[-1] * 1e3:.2f} ms")
        self.assertEqual(len(tr.calls), 1)                       # one in flight, no pile-up

    def test_log_never_contains_key(self):
        state = {"n": 0}

        def fn(b):
            state["n"] += 1
            if state["n"] == 1:
                return 500, f"upstream error, your key {FAKE_KEY} and v1.ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            content = json.dumps(dict(good_plan(), summary=f"echo {FAKE_KEY}"))
            return 200, chat_response(content)
        tr = FakeTransport(fn)
        brain = self.brain(tr)
        snap = snapshot(STAGED)
        brain.tick(snap, 0.0)
        wait_idle(brain, snap)
        self.assertEqual(tr.calls[0][1]["Authorization"], f"Bearer {FAKE_KEY}")
        with open(self.log, encoding="utf-8") as f:
            text = f.read()
        self.assertTrue(text)
        self.assertNotIn(FAKE_KEY, text)
        self.assertNotIn("FAKEKEYFAKEKEY", text)
        self.assertNotIn("ABCDEFGHIJKLMNOPQRSTUVWXYZ", text)
        self.assertNotIn(FAKE_KEY, json.dumps(brain.status))
        self.assertNotIn(FAKE_KEY, json.dumps(brain.plan.to_dict()))

    def test_missing_key_means_rules(self):
        os.environ.pop(KEY_ENV, None)
        brain = FleetBrain(IDS, LAYOUT, True, log_path=self.log)
        self.addCleanup(brain.close)
        brain.tick(snapshot(STAGED), 0.0)
        self.assertFalse(brain.use_llm)
        self.assertIn("no NEBIUS_TOKEN_FACTORY_KEY", brain.plan.source)
        self.assertEqual(brain.plan.commands["sheep4"].action, "dock")

    def test_tick_never_raises(self):
        brain = FleetBrain(IDS, LAYOUT, False)
        self.addCleanup(brain.close)
        brain.tick(None, 0.0)                                    # garbage in
        self.assertIn("tick", brain.status["last_error"])


# ---------------------------------------------------------------------------------------
class TestEconomics(unittest.TestCase):
    def test_sun_unit_and_night(self):
        for h in range(24):
            s = ECON.sun_dir(h * 3600.0)
            self.assertAlmostEqual(math.sqrt(sum(v * v for v in s)), 1.0, places=9)
        for h in (0, 3, 5.9, 18.1, 20, 23.5):
            self.assertLessEqual(ECON.sun_dir(h * 3600.0)[2], 0.0, h)
        self.assertAlmostEqual(math.degrees(ECON.sun_el(12 * 3600.0)), 38.0, places=6)
        sx, sy, _ = ECON.sun_dir(12 * 3600.0)
        self.assertAlmostEqual(sx, 0.0, places=9)
        self.assertLess(sy, 0.0)                                 # noon sun due south

    def test_facing_sun_beats_facing_away(self):
        for h in (8, 10, 12, 15):
            t = h * 3600.0
            s, yaw = ECON.sun_dir(t), ECON.sun_yaw(t)
            face = ECON.panel_power_w(ECON.panel_normal(yaw), s)
            away = ECON.panel_power_w(ECON.panel_normal(yaw + math.pi), s)
            self.assertGreater(face, away, h)
            el = ECON.sun_el(t)
            self.assertAlmostEqual(face, 195.6 * math.sin(el + math.radians(20)), places=6)
        self.assertEqual(ECON.panel_power_w((0, 0, 1), ECON.sun_dir(2 * 3600.0)), 0.0)
        self.assertAlmostEqual(ECON.panel_power_w((0, 0, 1), (0, 0, 1), weather_f=0.45),
                               195.6 * 0.45)

    def test_soc_bounds_and_clocks(self):
        e = Economics(timelapse=120.0, start_hour=8.0)
        self.assertEqual(e.step_soc(0.99, 5000.0, 0.0, 60.0, True), 1.0)
        self.assertEqual(e.step_soc(0.06, 0.0, 5000.0, 60.0, False), RESERVE)
        self.assertEqual(e.step_soc(0.50, 150.0, 0.0, 1.0, False), 0.50)   # not harvesting
        up = e.step_soc(0.50, 150.0, 0.0, 1.0, True)
        self.assertAlmostEqual(up - 0.50, 150 * 0.95 * 120 / 3600 / 500)  # time-lapse charge
        down = e.step_soc(0.50, 0.0, 30.0, 1.0, False)
        self.assertAlmostEqual(0.50 - down, 30 / 0.95 / 3600 / 500)        # physics-time motor
        self.assertAlmostEqual(e.day_seconds(0.0), 8 * 3600.0)
        self.assertAlmostEqual(e.day_seconds(30.0), 8 * 3600.0 + 3600.0)
        self.assertAlmostEqual(e.day_seconds(16.5 * 3600 / 120), 0.5 * 3600, places=6)

    def test_weather(self):
        e = Economics(weather="overcast")
        self.assertEqual(e.weather_f, 0.45)
        changed = sum(e.step_weather(60.0) for _ in range(200))   # 200 min real = 400 h sim
        self.assertGreater(changed, 10)
        self.assertEqual(Economics(weather="random").weather, Economics(weather="random").weather)


class TestLayout(unittest.TestCase):
    def test_geometry(self):
        L = LAYOUT
        self.assertEqual((L.rows, L.slots), (2, 4))
        self.assertEqual(L.approach_xy, (-1.5, 0.0))
        self.assertEqual(L.berth_xy, L.dock_xy)
        cells = [L.slot_xy(*c) for c in L.cells()]
        for i, a in enumerate(cells):
            for b in cells[i + 1:]:
                self.assertGreaterEqual(math.dist(a, b), 2.0 - 1e-9)
            self.assertLessEqual(math.dist(a, L.approach_xy), 6.0 + 1e-9)
        qs = [L.queue_xy(i) for i in range(6)]
        for a, b in zip(qs, qs[1:]):
            self.assertGreaterEqual(math.dist(a, b), 1.2)
        self.assertGreaterEqual(math.dist(qs[0], L.approach_xy), 1.2)
        pts = cells + qs[:5] + [L.park_xy(i) for i in range(5)] + [L.exit_xy]
        self.assertTrue(all(abs(x) <= 7.0 and abs(y) <= 7.0 for x, y in pts))


if __name__ == "__main__":
    unittest.main()
