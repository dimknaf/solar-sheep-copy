# CONTRACTS — frozen interfaces

Three people building one Omniverse scene will collide within an hour. This file is how we avoid
that. **No code yet** — this is the agreement. Whoever writes the code implements exactly this.

**Rule: nobody changes this file alone.** All three agree, or it doesn't change.

---

## Why this exists

Training runs take days. If C waits for Dimitris's locomotion policy and B waits for the scene,
nothing ships. So:

> **Everyone codes against a stub of the other two workstreams.**

A stub is a working fake with the real signature — teleport the vehicle, wait a timer, return
`SUCCESS`. C builds and demos the *entire* orchestration loop against stubs on day one. Each person
then replaces only their own stub with the real thing. No one is ever blocked.

This also means the **Isaac Sim ↔ MuJoCo decision costs us nothing**. The interface is the same
either way; only the implementation behind it changes.

---

## Interface 1 — the robot

**Owner: Dimitris.** One USD articulation, published in `robot/SPEC.md`.

Must specify: joint names and order · the action space (what a policy outputs) · the observation
space (what a policy sees) · the frame where the battery pack attaches.

B and C import it **read-only**. If B models the battery mount against a different robot, the swap
will never line up. This is the tightest coupling in the project — settle it first.

---

## Interface 2 — the world

**Owner: C.** Published in `scene/SPEC.md`.

- Panel array addressed as **`(row, slot)`** — integer grid, origin and spacing stated explicitly.
- The dock has a **fixed named pose**.
- One stated **up-axis and unit scale** for all USD assets (Isaac Sim defaults are Z-up, metres —
  say so explicitly and do not let anyone quietly author in centimetres).

Everyone positions geometry against this. Nobody invents their own coordinates.

---

## Interface 3 — the policies

Both policies expose the same shape, so the orchestrator never knows or cares whether it is talking
to a stub, a scripted controller, or a trained network:

```
traverse(vehicle_id, target)      -> Status
swap_battery(vehicle_id, dock)    -> Status

Status = RUNNING | SUCCESS | FAILURE
```

Non-blocking — called every tick, returns `RUNNING` until done. That is what lets the orchestrator
run a behaviour tree over the whole fleet, and what makes failure visible rather than a hang.

**`FAILURE` is a first-class outcome, not an exception.** The orchestrator recovering from a failed
traverse is the single most compelling thing we can put in the demo video, so make failure
reportable from day one.

---

## Interface 4 — orchestrator ↔ fleet

**Owner: C.** The NVIDIA model on Nebius emits commands; the sim returns telemetry.

- **Commands down** — strict JSON. ⚠️ If we use structured outputs, every object node needs
  `additionalProperties: false` **and** every property listed in `required`. One optional field
  gives a `400 invalid_json_schema` and the run dies in about four seconds. There is no such thing
  as an optional field — use a required key with a permissive type and enforce the real rule in our
  own validator.
- **Telemetry up** — per vehicle: pose, battery state of charge, current task, last status.

The exact schema lands when C writes it. It is listed here so it is understood as a frozen
interface, not an implementation detail.

---

## Folder ownership

```
contracts/        nobody     frozen types + stubs
robot/            Dimitris   vehicle USD, articulation, SPEC.md
envs/traverse/    Dimitris   locomotion env, reward, training config
envs/swap/        B          dock + battery USD, grasp frames, SPEC.md
scene/            C          world USD, panel array, camera rigs, SPEC.md
orchestrator/     C          Nebius client, schema, behaviour tree, telemetry
docs/             shared     rules, research, decisions
```

**Do not edit another person's folder.** If you need something from it, ask for it to be exposed
through the contract instead.

---

## Open decisions — settle one at a time, do not bundle

1. **Which NVIDIA open model, and doing what** — reasoning/planning vs physical VLM vs VLA.
2. **Isaac Sim on Nebius, or MuJoCo-train + Isaac-render.** Settled by a spike, not by opinion.
3. **Battery swap: learned policy or scripted IK.** Scripted first regardless — it is the stub.
4. **Vehicle: wheeled or legged.** Dimitris's call.
5. Keep the existing JS sim as a fast fleet-economics view, or retire it.

Log each one in `docs/decisions.md` when locked: chosen, why, rejected, at what cost.
