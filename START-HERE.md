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

## How we are scored — and what that means for the video

Four criteria, **scored 1–5, equally weighted** (25% each, max 20). But **ties break on
Technological Implementation first**, then Design, then Impact, then Idea — and with thousands of
entrants scored 1–5 there *will* be ties at the top. **Tech Implementation is worth more than 25%
in practice.**

> "Judges are not required to watch beyond three minutes."
> "Judges are not required to test the Project and may choose to judge based solely on the text
> description, images, and video."

**The video is the deliverable.** Put the Nebius/NVIDIA depth in the **first 30 seconds**.

**Prize strategy:** the rules allow *one Overall Award* **OR** *one Track Award + one Bonus*. We
target **Physical AI Track (Jetson) + Best Use of Tavily ($3,000)** — better odds than chasing the
$20k, and the Physical AI field is small (comparable NVIDIA events drew 21–24 submissions).

**Sim-only wins** — in NVIDIA's own Cosmos Cookoff, 1st *and* 2nd place were both Isaac Sim, and 2nd
place was a 100% simulated drone with no hardware. But the organisers here said *"a scripted demo is
table stakes; hardware reacting live to real-world input is where this track can really shine."*
**Cheap hedge:** 60 s of any real device reacting live — a webcam + ESP32 satisfies "IoT /
on-device intelligence", and an ESP32-based project won Grand Prize at a comparable event.

---

## 🎯 START FROM NEBIUS'S OWN REPO — do not build this from scratch

**https://github.com/nebius/nebius-physical-ai** (Apache-2.0)

Nebius maintains an official control plane for exactly this track, and it already contains the two
things we were going to spend three weeks building:

- **`docs/workbench/guides/quadruped-isaac-lab.md`** — "Train a Quadruped to Run in Isaac Lab".
  ANYmal-C velocity policy, RSL-RL, headless, with eval gates. Reusable almost directly for our
  vehicle:
  ```bash
  npa workbench isaac-lab train --task Isaac-Velocity-Rough-Anymal-C-v0 --num-envs 2048 --steps 500
  npa workbench isaac-lab eval  --task ... --success-metric survival --min-success-rate 0.90
  npa workbench isaac-lab export-onnx
  ```
- **`docs/hackathon-isaac-token-factory.md`** — the Isaac Lab → Token Factory bridge. Sim frames to
  S3, an NVIDIA model reads them and emits a plan. Literally our architecture, already written:
  ```
  [ Isaac Lab GPU stage ] --PNG frames--> [ S3 ] --> [ Token Factory reasoner ] --> plan.json
  ```

**So our differentiation is not the plumbing — it is the solar-panel task design, the battery-swap
dock, and the fleet orchestration.** That is a much better place to spend 44 days, and building on
the sponsor's own repo scores directly against "Technological Implementation".

Also in there: `docs/agent.md`, `docs/cli/groot.md`, `docs/cli/cosmos3.md`, `docs/cli/mjlab.md`,
`docs/workbench/composing-cloud-and-token-factory.md`.

## ✅ The GPU question — mostly answered already

**Isaac Lab needs RT cores.** From Nebius's *own* repo, not just NVIDIA's docs:

> "**RT cores are mandatory.** L40S or RTX PRO 6000 only. H100/H200 lack RT cores and won't
> render/simulate Isaac Lab correctly."

> "No RT cores on datacenter Blackwell, same as H100/H200… this is a **hardware fact rather than a
> software gap that will be fixed**."

So provision **`gpu-l40s-d`** (~$0.74/hr preemptible, $1.55 on-demand, `eu-north1`) or
**`gpu-rtx6000`** (~$0.95 preemptible, $1.80 on-demand). **Never H100/H200/B200/B300** — only
state-based headless training can route there.

### 💸 GPU money — claim the event credits

**Builders & Brews attendees get $100 Token Factory + $100 Nebius AI Cloud** (plus Tavily and
Zapier). ⭐ **The $100 AI Cloud is the GPU budget** — about 130 hours of L40S preemptible, which
covers the whole project. **Claim it at the event; don't leave without it.**

Without it we pay out of pocket: the two promo codes are `utm_promo_code_type=Token_Factory`
(inference only), and *"the free trial program in Nebius AI Cloud has been suspended as of July 13,
2026."* 40 GPU-hours ≈ $30 on L40S preemptible, ≈ $72 on RTX PRO 6000 on-demand.

**Fallback if that's unacceptable:** train in **MuJoCo Playground** (Go1 joystick policy in ~7 min
on one GPU, no RT cores needed) and rent an L40S only for the final render pass. The interfaces in
[`CONTRACTS.md`](CONTRACTS.md) make this swap free.

## Versions and gotchas — pin these, do not improvise

- Nebius's repo pins **Isaac Lab `v3.0.0-beta2.patch1` + Isaac Sim `6.0.1.0`**. It is a beta;
  follow *their* pin rather than inventing one, since their images are built against it.
  Gen-3 training uses `--visualizer none`, **not** `--headless` (that's gen-2 only).
- Container is **GHCR, not NGC**: `ghcr.io/nebius/nebius-physical-ai/npa-isaac-lab`. No NGC
  credentials needed. The Isaac runtime downloads on first use — **pre-warm the cache**, or your
  first run looks broken.
- **Windows users must work in WSL2 Ubuntu.** The `npa` CLI is not Windows-native.
- ⚠️ **`nvidia/Nemotron-3_5-Lightning` burns its entire output budget on reasoning** unless you
  pass `chat_template_kwargs={"enable_thinking": False}`. Nebius measured 506 of 512 tokens spent
  on reasoning, returning truncated JSON. This will look like a bug in your code and is not.

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
