"""orchestrator/smoke.py -- ONE live Token Factory request for a staged 5-rover snapshot.

Run from the repo root:
    .venv\\Scripts\\python.exe -m orchestrator.smoke --env-file .env
    .venv\\Scripts\\python.exe -m orchestrator.smoke --env-file .env --model nvidia/Nemotron-3_5-Lightning

The key comes from NEBIUS_TOKEN_FACTORY_KEY in the environment, or ``--env-file`` parses a
dotenv file itself (only the Token Factory variables are taken; nothing is echoed). Prints
model, latency, tokens, cost estimate, the schema mode that worked and the validated plan
-- never the key. Staged scene at 09:00, clear sky: sheep0 full (0.96), sheep1 0.75,
sheep2 0.55 with a traverse FAILURE, sheep3 0.35, sheep4 0.15 (brownout); dock free,
3 empty packs. Exit code 0 when the reply parses and at least one command validates.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .brain import system_prompt, telemetry_payload
from .client import BASE_ENV, KEY_ENV, MODEL_ENV, MissingKey, TokenFactoryClient
from .economics import Economics
from .layout import Layout
from .rules import RuleEngine
from .schema import parse_json_text, validate_plan
from .telemetry import DockTelemetry, FleetSnapshot, RoverTelemetry

IDS = [f"sheep{i}" for i in range(5)]


def load_env_file(path: str) -> list:
    """Set the Token Factory variables from a dotenv file (existing env wins). Silent."""
    wanted, got = {KEY_ENV, BASE_ENV, MODEL_ENV}, []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            if k.startswith("export "):
                k = k[7:].strip()
            v = v.strip().strip('"').strip("'")
            if k in wanted and v and not os.environ.get(k):
                os.environ[k] = v
                got.append(k)
    return got


def staged_snapshot(layout: Layout, econ: Economics, t_day_s: float = 9 * 3600.0) -> FleetSnapshot:
    rules = RuleEngine(IDS, layout)
    yaw_sun = econ.sun_yaw(t_day_s)
    pv = econ.panel_power_w(econ.panel_normal(yaw_sun), econ.sun_dir(t_day_s))
    rovers = []
    socs = [0.96, 0.75, 0.55, 0.35, 0.15]
    for i, rid in enumerate(IDS):
        x, y = layout.slot_xy(*rules.cell[rid])
        state, status, reason, p = "harvesting", "RUNNING", "", pv
        if rid == "sheep2":                       # stuck half way to its cell
            ex, ey = layout.exit_xy
            x, y = (x + ex) / 2, (y + ey) / 2
            state, status, reason, p = "to_field", "FAILURE", "stuck", 0.0
        if rid == "sheep4":
            state, p = "to_field", 0.0
            x, y = x - 0.8, y - 0.8
        rovers.append(RoverTelemetry(rid, x, y, yaw_sun, socs[i], state, p, status, reason))
    return FleetSnapshot(t_day_s, yaw_sun, econ.sun_el(t_day_s), econ.weather, rovers,
                         DockTelemetry(busy=False, current=None, queue=[], empty_packs=3))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--env-file", default=None)
    ap.add_argument("--model", default=None)
    args = ap.parse_args(argv)
    if args.env_file:
        got = load_env_file(args.env_file)
        print(f"env-file: set {', '.join(got) if got else 'nothing new'} (values not shown)")
    layout, econ = Layout.default(5), Economics(timelapse=120.0, start_hour=8.0)
    snap = staged_snapshot(layout, econ)
    rules = RuleEngine(IDS, layout)
    rule_plan = rules.plan(snap)
    payload = telemetry_payload(snap, layout, rules, ["start", "pack_full:sheep0",
                                                      "failure(stuck):sheep2"])
    messages = [{"role": "system", "content": system_prompt(layout, IDS)},
                {"role": "user", "content": json.dumps(payload, separators=(",", ":"))}]
    try:
        client = TokenFactoryClient(args.model)
    except MissingKey as e:
        print(f"no key: {e}")
        return 2
    print(f"model: {client.model}\nendpoint: {client.url}")
    res = client.chat(messages, IDS)
    print(f"http: {res.status} attempts={res.attempts} schema_mode={res.schema_mode} "
          f"finish={res.finish_reason or '-'} reasoning_chars={res.reasoning_chars}")
    print(f"latency: {res.latency_s:.2f} s  tokens in/out: {res.tokens_in}/{res.tokens_out}  "
          f"cost est: ${res.cost_usd:.6f}")
    if not res.ok:
        print(f"FAILED: {res.error}")
        return 1
    obj, err = parse_json_text(res.text)
    if obj is None:
        print(f"FAILED to parse: {err}\nraw: {res.text[:600]}")
        return 1
    reserved = {rid: (c.row, c.slot) for rid, c in rule_plan.commands.items() if c.action == "harvest"}
    cmds, errors, summary = validate_plan(obj, snap, layout, IDS, model=client.model,
                                          reserved_cells=reserved,
                                          just_swapped=rules.just_swapped)
    print(f"summary: {summary}")
    for rid in IDS:
        c = cmds.get(rid)
        r = rule_plan.commands[rid]
        mine = (f"{c.action:7s} cell=({c.row},{c.slot}) face={c.face:4s} order={c.dock_order:2d}  {c.reason}"
                if c else "-- dropped --")
        print(f"  {rid} LLM: {mine}\n         rules: {r.action} ({r.row},{r.slot}) order={r.dock_order}  {r.reason}")
    print(f"validation: {len(cmds)}/{len(IDS)} valid; errors: {errors or 'none'}")
    return 0 if cmds else 1


if __name__ == "__main__":
    sys.exit(main())
