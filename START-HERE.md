# START HERE — Solar Sheep → Physical AI

**Nebius x NVIDIA Global AI Hackathon · Physical AI track**
Branch: `physical-ai` · read this before touching anything.

```
DEADLINE   30 Oct 2026, 10:00 PDT = 17:00 GMT
           feature freeze 27 Oct · video 28 Oct · submit 29 Oct (1 day buffer)
```

---

## What we are building

Today `solar-sheep` is a browser Canvas 2D toy: PV-carrying "sheep" wander a pasture, track the
sun, and swap battery packs at a dock. The premise and the energy economy are good. Nothing in it
is learned or physically simulated.

We rebuild it as a **real robot simulation in NVIDIA Omniverse / Isaac Sim**, with trained
policies for movement and battery exchange, and an **NVIDIA open model running on Nebius** doing
the fleet orchestration.

```
 ORCHESTRATION   NVIDIA open model on Nebius Token Factory            <- C
                 which vehicle goes to which panel row, when to swap,
                 what to do when a dock is blocked or a vehicle browns out
                       |  commands down (strict JSON)  /  telemetry up
 BEHAVIOUR       Behaviour tree / task graph                          <- C
                       |
 CONTROL         Trained policies in Isaac Sim on Nebius GPU
                 traverse()      cross the panel array                <- Dimitris
                 swap_battery()  extract + insert at the dock         <- B
```

---

## ⚠️ The rule that can disqualify us

Verbatim from the official rules:

> "create a working software application that runs on either **Nebius Token Factory or Nebius AI
> Cloud** and uses **at least one NVIDIA open source model**."

Both halves are mandatory. Our answer: the sim runs on Nebius AI Cloud GPU, **and** the NVIDIA
model sits inside the control loop — take it out and the fleet stops making decisions. That is the
difference between "we called an LLM once" and an integration that is actually load-bearing.

Good news, also verbatim — **simulation counts, no hardware needed**:

> "…or — if the Project has no physical hardware component — the key application modules in action."

Full rubric and submission checklist: **[`docs/rules.md`](docs/rules.md)**. Argue from that file,
never from memory.

---

## ⚠️ The blocker nobody has cleared yet

**None of us has an NVIDIA GPU.** Every workstream depends on cloud GPU access that does not exist
yet. The $25 + $25 hackathon credits are **Token Factory (inference)** — do not assume they cover
GPU hours. **C owns clearing this in the first 48 hours.**

And it is narrower than "rent a GPU". Straight from the Isaac Sim 6.1 requirements page:

> **"GPUs without RT Cores (A100, H100) are not supported."**

So the default datacentre GPU everyone reaches for **will not run our simulator**. We need
**RTX PRO 6000 Blackwell** (explicitly blessed by NVIDIA, and Nebius lists it) or **L40S** (Ada,
has RT cores — should work, but NVIDIA never names it, so **test it on day 1 rather than assuming**).

**Pre-agreed fallback, so we don't relitigate it at week three:** if Isaac Sim on Nebius is blocked
or too expensive, train in **MuJoCo Playground** and use Isaac Sim only for the final render pass.
The interfaces in [`CONTRACTS.md`](CONTRACTS.md) are written so this swap costs us nothing.

## Versions — pin these, do not improvise

**Isaac Lab 2.3.2 + Isaac Sim 5.1.0.** Isaac Lab 3.0 exists but is **beta**, still landing breaking
changes; a beta install failure costs a week we do not have. Verify what you actually install and
pin that — never pin a version from memory.

---

## Who owns what — do not edit someone else's folder

| Folder | Owner | Contains |
|---|---|---|
| `robot/` | **Dimitris** | vehicle URDF → USD, articulation, joint names |
| `envs/traverse/` | **Dimitris** | locomotion env, reward, training config |
| `envs/swap/` | **B** | dock + battery-module USD, grasp frames |
| `scene/` | **C** | world USD, panel array layout, camera rigs |
| `orchestrator/` | **C** | Nebius client, JSON schema, behaviour tree, telemetry |
| `contracts/` | **nobody** | frozen — changed only by agreement of all three |
| `docs/` | shared | rules, research, decisions |

Everyone codes against a **stub** of the other two, so nobody is ever blocked waiting on someone
else's training run. See [`CONTRACTS.md`](CONTRACTS.md).

---

## First tasks — tonight, in parallel

**Everyone, 15 minutes:** install the NVIDIA agent skills into Claude Code — they do a lot of this
work for us and are not optional:

- `isaac-sim/IsaacSim` → `skills/` (~29 skills: Isaac Sim install, headless deploy, URDF→USD,
  navigation, manipulation IK, rendering, ROS 2 bridge)
- `NVIDIA/skills` → `omniverse-realtime-viewer`, `omniverse-cad-to-simready`,
  `omniverse-usd-performance-tuning`

Then redeem credits: code **`NEBIUS-DEVPOST-GLOBAL26`** → $25 Token Factory, plus the Nebius
Builders Program at `dev.nebius.com/builders` → another $25 + Tavily credits.

### Dimitris — the vehicle
**Do not write a PPO recipe from scratch — fork a working one.** Isaac Lab already ships
`Isaac-Velocity-Rough-Unitree-Go2-v0` and `Isaac-Velocity-Rough-Anymal-D-v0`: quadruped velocity
tracking with terrain curricula. A solar array is geometrically a tilted plane with gaps, which is a
*smaller* change than the rough-terrain generator already handles. Reference: Spot locomotion
trains in **~4 h on an RTX 4090** at 4096 envs — so roughly 2–3 h on a rented RTX PRO 6000, which
means we can afford ~10 training runs across the sprint.

Robot meshes are free: `google-deepmind/mujoco_menagerie` has Unitree Go1/Go2/A1, ANYmal and Spot
under BSD-3.

Your first deliverable is `robot/SPEC.md` — **wheeled or legged**, joint names, action space,
observation space, and the frame where the battery pack attaches. That one file unblocks the other
two. *Legged looks far better on video and fits "sheep"; wheeled trains faster.*
Skills: `urdf-mjcf-to-usd-conversion`, `usd-articulation`, `isaac-sim-robot-navigation`,
`navigation-primitives`, `physics-simulation`.

### B — the battery swap
**Script it. Do not train it.** This is the clearest call in the whole plan. Contact-rich insertion
(peg-in-hole) is a twenty-year open problem — 2025 papers still can't make pure RL insertion
transfer, and it needs force/torque sensing we don't have. Meanwhile real battery-swap docks solve
it *mechanically*: published stations swap in **45 s** using a movable carrier that tolerates
**±9° misalignment**.

So: **learned approach + scripted mechanism.** The nav policy drives the vehicle into a
funnel/V-guide dock, a trigger fires a scripted swap animation, the battery state flips, the
vehicle leaves. A judge cannot tell the difference in a 60-second video, and it works every take.
The intelligence goes into *docking*, not fine manipulation.

Settle the swap geometry with Dimitris in a five-minute conversation — pack **on top, side rail, or
underneath** — and write it into `envs/swap/SPEC.md`. That is the only real coupling between your
two workstreams.
Skills: `manipulation-ik`, `motion-generation`, `spatial-reasoning`.

### C — unblock the GPU, then the world
**Go to the Nebius booth while you are still in the room tonight.** Ask these exact questions —
you will never get the answers faster than in person:

1. Do we get **AI Cloud GPU** credits, or only Token Factory inference credits?
2. Can we get an **RTX PRO 6000 Blackwell** or **L40S** instance? *(Not H100 — Isaac Sim will not
   run on it. Say this to them; it saves a long back-and-forth.)*
3. Is there an Isaac Sim / NGC container image, or do we build from
   `NVIDIA-Omniverse/IsaacSim-dockerfiles`?
4. Which **NVIDIA open models are served on Nebius Token Factory** — specifically Nemotron 3 and
   Cosmos Reason 2? ⚠️ The rule says the model must run on **Nebius**, so NVIDIA's own free
   `build.nvidia.com` endpoint does **not** satisfy the gate. This matters.
Then publish the world addressing scheme — panel array as `(row, slot)`, dock at a fixed named
pose — into `scene/SPEC.md`.
Skills: `isaac-sim-installation`, `isaac-sim-headless-deployment`, `isaac-sim-remote`,
`isaac-sim-orchestrator`, `behavior-tree-generation`.

---

## How we work

- **Verify credentials with a real API call.** Reading the docs is not verification. Record the
  actual HTTP status code in `docs/access.md` (gitignored).
- **Never pin a version from memory.** Resolve latest → verify → pin what you actually got.
- **Push early and often.** Never leave the push to the deadline on a venue network.
- **Log every locked decision** in `docs/decisions.md` — chosen, why, rejected, at what cost. It
  becomes the architecture section of the write-up for free.

## Still to do

- [ ] Repo must be **public with an OSS licence file** before submission (hard requirement, 27 Oct)
- [ ] Written explanation of what changed since 26 Aug (required — the JS sim predates the window)
- [ ] Add B and C as collaborators
