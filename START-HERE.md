# START HERE — Solar Sheep → Physical AI

**Nebius x NVIDIA Global AI Hackathon · Physical AI track**
Branch: `physical-ai` · read this before touching anything.

```
DEADLINE   30 Oct 2026, 10:00 PDT = 17:00 GMT
           feature freeze 27 Oct · video 28 Oct · submit 29 Oct (1 day buffer)
           everything stays LIVE until 15 Dec (judging runs 1–15 Dec)
```

| Person | Owns | First deliverable |
|---|---|---|
| **Dimitris** | the rover + its locomotion policy | `robot/SPEC.md` |
| **B** | the battery-swap dock | `envs/swap/SPEC.md` |
| **C** | cloud GPU, the world, the orchestrator | GPU up + `scene/SPEC.md` |

---

## What we are building

A herd of small, cheap, slow rovers. **Each one carries a PV panel on its back**, drives around a
pasture, and **turns its body to face the sun** — because with skid steer, heading *is* panel
orientation. When its battery fills it drives to a dock, the pack is swapped out from underneath,
and it goes back to work. A fleet brain decides who goes where.

> ⚠️ **They are mobile solar collectors, not panel cleaners.** An earlier draft of this document
> said they traverse and clean a fixed solar array. That was wrong — it is a different robot with a
> different reward function. The sheep *carry* the panel.

```
 ORCHESTRATION  NVIDIA Nemotron on Nebius Token Factory                  <- C
                herd dispatch: which rover to which pasture slot, when to
                swap, what to do when a dock is blocked or a rover browns out
                      | commands down (strict JSON) / telemetry up
 BEHAVIOUR      Behaviour tree / task graph                              <- C
                      |
 CONTROL        Isaac Sim on a Nebius RTX PRO 6000
                traverse()      navigate + sun-orient   (trained)        <- Dimitris
                swap_battery()  drive-over dock swap    (scripted IK)    <- B
```

Remove the Nemotron model and the herd stops making decisions. That is what makes the sponsor
integration **load-bearing** rather than decorative.

---

## The vehicle — DECIDED

**A slow, cheap, 4-wheel skid-steer rover. Panel on top, battery underneath, four low-power motors.**

```
            ☀
             ╲                 panel fixed on top (no gimbal)
          ┌───────┐            body yaw aims it
          │  ▭▭▭  │
          │ ▓▓▓▓▓ │            battery slung UNDERNEATH
          └─╥───╥─┘            4 × cheap low-power motors
            ◎   ◎              slow: ~0.3–0.5 m/s
```

**Why this is right, not a compromise:**

1. **Heading *is* panel orientation.** Skid steer gives body yaw for free, so navigation and
   sun-tracking are **one** control problem, not two. The existing JS sim already models exactly
   this — `s.orient`, `exposure = max(0, cos(angleError)) × skill`. **Our reward function is already
   written, in JavaScript, in this repo.**
2. **Battery underneath is the cleanest swap geometry there is.** Drive over the dock, the dock
   lifts the pack out from below, gravity does the alignment. Much easier than any side or top
   presentation.
3. **Slow and low-power is the product argument.** The sim books **30 W** of motor draw against a
   **300 W** panel. That ratio is *why* a mobile collector can beat a fixed one — it must spend far
   less moving than it gains by aiming. A fast robot destroys its own thesis. **Put that number on
   screen in the video.**
4. **It trains fast.** A 4-wheel skid-steer rover is a far smaller action space than a 12-DoF
   quadruped — so we can afford many training runs instead of one anxious one.

---

## ⚠️ The rule that can disqualify us

> "create a working software application that runs on either **Nebius Token Factory or Nebius AI
> Cloud** and uses **at least one NVIDIA open source model**."

Both halves are mandatory. Ours: the sim runs on Nebius AI Cloud GPU, **and** an NVIDIA Nemotron
model sits inside the control loop.

Good news, also verbatim — **simulation counts, no hardware needed**:
> "…or — if the Project has no physical hardware component — the key application modules in action."

Full rubric and checklist: **[`docs/rules.md`](docs/rules.md)**. Argue from that file, never memory.

## How we are scored

Four criteria, **1–5, equally weighted** (25% each, max 20). But **ties break on Technological
Implementation first**, then Design, Impact, Idea — and with thousands of entrants scored 1–5 there
*will* be ties at the top. **Tech Implementation is worth more than 25% in practice.**

> "Judges are not required to watch beyond three minutes."
> "Judges are not required to test the Project and may judge solely on the description, images and
> video."

**The video is the deliverable.** Nebius/NVIDIA depth in the **first 30 seconds**.

**Prize strategy:** rules allow *one Overall Award* **OR** *one Track Award + one Bonus*. We target
**Physical AI Track (Jetson) + Best Use of Tavily ($3,000)** — better odds than chasing the $20k.

**Sim-only wins:** in NVIDIA's own Cosmos Cookoff, 1st *and* 2nd were both Isaac Sim, 2nd being a
100% simulated drone. But organisers said *"a scripted demo is table stakes; hardware reacting live
to real-world input is where this track can really shine."* **Cheap hedge:** 60 s of any real device
reacting live — a webcam + ESP32 counts as "IoT / on-device intelligence".

---

## 🎯 Build on Nebius's own repo — do not start from scratch

**https://github.com/nebius/nebius-physical-ai** (Apache-2.0) — the sponsor's own control plane,
**97 agent skills**, wrapping the `npa` CLI. It already contains what we were going to spend three
weeks building:

- `docs/workbench/guides/quadruped-isaac-lab.md` — Isaac Lab RL training with eval gates
  ```bash
  npa workbench isaac-lab train --task <task> --num-envs 2048 --steps 500
  npa workbench isaac-lab eval  --success-metric survival --min-success-rate 0.90
  npa workbench isaac-lab export-onnx
  ```
- `docs/hackathon-isaac-token-factory.md` — the Isaac Lab → Token Factory bridge, literally our
  architecture: `[Isaac GPU stage] --frames--> [S3] --> [Token Factory reasoner] --> plan.json`

**Our differentiation is not the plumbing** — it is the mobile-collector task design, the drive-over
dock, and the herd orchestration. Building on the sponsor's repo also scores directly against the
top-weighted criterion.

---

## ⛔ Isaac Sim does NOT run on our laptops

From NVIDIA's own `isaac-sim-installation` skill: the Docker path is *"native-Linux x86_64 only and
**explicitly unsupported on Windows/WSL**"*, and the pip path needs **Python 3.12**.

**So Isaac Sim lives only on the rented Nebius Linux GPU box.** WSL is for the `npa` CLI, nothing
else. Consequences:

- Inner dev loop is `isaac-sim-remote` — a **loopback-only** IPC server on `127.0.0.1:8226`. Reach it
  over an **SSH tunnel** to the box. Never publish that port off-host.
- `isaac-sim-orchestrator` owns the shared env contract (`$ISAAC_SIM_DIR`, `$ISAAC_LAB_DIR`,
  `$WORKSPACE_DIR`) every other Isaac skill assumes. Whoever stands up the box reads it first.
- Run `isaac-sim-validator` on every script before it is filmed or handed over — it catches the
  **black-frame / missing-lights** failure that would silently ruin our ≥1-minute footage.

## ✅ GPU — VERIFIED, not assumed

Checked with real API calls against our actual project on 15 Sep 2026:

```
account    Dimitrios Koutsoumpos · tenant-e00yd9pgnqbpbq8sxh
project    project-u00vpbp7kc00vhag08bn1s  "default-project-us-central1"
region     us-central1                      state ACTIVE

platform   gpu-rtx6000  "NVIDIA® RTX PRO 6000 with Intel Granite Rapids"   ✅ AVAILABLE
preset     1gpu-24vcpu-218gb   (1 GPU · 24 vCPU · 218 GiB)
           8gpu-192vcpu-1744gb (8 GPU)

quota      compute.instance.gpu.rtx6000        32     ← plenty
           compute.instance.preemptible.count   8     ← the cheap tier
           compute.instance.count              12
           compute.gpucluster.count             5
```

RTX PRO 6000 is also available in `eu-south1` and `uk-south2` if we ever need to move.

**Isaac Lab needs RT cores.** From Nebius's own skill: *"Use L40S or RTX Pro 6000… Rendering,
camera-bearing tasks, and deployed workbenches require RT cores and **cannot target B200, H100, or
H200**."* Our project also exposes `gpu-h200-sxm` and `gpu-b200-sxm` — **do not use them.**

⚠️ **`gpu-rtx6000` is not a serverless platform** — it's the managed-Kubernetes / VM path. Provision
via `npa cluster` or a VM, not `--runtime serverless`.

**Credits:** Builders & Brews attendees get **$100 Token Factory + $100 AI Cloud**. The AI Cloud $100
is the GPU budget (~100 h preemptible). Plus `NEBIUS-DEVPOST-GLOBAL26` ($25) and the Builders
Program ($25). Apply AI Cloud codes at **console.nebius.com → Billing → Apply promo code**.

### Getting on the box yourself

```bash
# in WSL2 Ubuntu (the CLI is not Windows-native)
curl -sSL https://storage.eu-north1.nebius.cloud/cli/install.sh | bash
exec -l $SHELL
nebius profile create --profile solar --endpoint api.nebius.cloud \
  --federation-endpoint auth.nebius.com --parent-id project-u00vpbp7kc00vhag08bn1s
# opens a browser tab — log in with the Google account on the tenant
nebius iam whoami          # verify
```

---

## Who owns what — and the three things that stop us colliding

### 1. Folders — do not edit someone else's

```
robot/            Dimitris   rover URDF→USD, articulation, SPEC.md
envs/traverse/    Dimitris   locomotion env, reward, training config
envs/swap/        B          dock + battery USD, grasp frames, SPEC.md
scene/            C          world USD, pasture terrain, camera rigs, SPEC.md
orchestrator/     C          Nebius client, schema, behaviour tree, telemetry
contracts/        nobody     frozen — changed only by agreement of all three
docs/             shared     rules, research, decisions
```

### 2. USD sublayers — folders alone are NOT enough

A `.usd` file is binary as far as git is concerned. **Two people editing one stage means one of them
loses work, and merge cannot help.** USD's answer is layer composition:

```
scene/world.usd            C owns. ONLY sublayer references. Nobody else opens it.
  ├── robot/sheep.usd      Dimitris, exclusively
  ├── envs/swap/dock.usd   B, exclusively
  └── scene/pasture.usd    C, exclusively
```

**Z-up, metres.** Every asset authored **at origin** and placed by `world.usd` — never bake world
position into an asset. Never edit a layer you don't own, even to "just fix" something.
Skill: `usd-composition-architecture` (C sets this up in week 1).

### 3. Stubs — nobody waits for anyone's training run

```
traverse(vehicle_id, target)   -> RUNNING | SUCCESS | FAILURE
swap_battery(vehicle_id, dock) -> RUNNING | SUCCESS | FAILURE
```

Non-blocking, called every tick. `FAILURE` is **first-class, not an exception** — the orchestrator
recovering from a failed traverse is the best thing we can put in the video. C demos the whole loop
against stubs on day one. See [`CONTRACTS.md`](CONTRACTS.md).

### 4. Terrain is a contract too

**Dimitris does not wait for C's pasture.** Train against Isaac Lab's procedural rough-terrain
generator now; `scene/pasture.usd` drops in later. That only works if the *statistics* match — train
on 8° slopes, hand it a 25° pasture, and the rover tips over on day one of integration. So C states
max slope, roughness amplitude, obstacle density, friction range and patch size **up front** in
`scene/SPEC.md`, and Dimitris trains a **curriculum spanning a range**, not a single value.

---

## First tasks

### Everyone — install the skills (15 min)

Three official sets. **Nebius's is the primary one**, not a supplement — it drives the cloud we are
actually billed on.

```bash
npx skills add nvidia/skills          # NVIDIA official marketplace (nvidia-official)
git clone https://github.com/nebius/nebius-physical-ai    # 97 skills in .claude/skills
git clone https://github.com/isaac-sim/IsaacSim           # ~39 skills in .claude/skills
# copy their .claude/skills/* into this project's .claude/skills/
```

⭐ **Then everyone reads `workflows/first-run-setup`** before anything else:
> "get from zero to a first verified result — an ordered, gated path through install, configure,
> credential preflight, cheapest-proof workload, then cluster provisioning, with an **explicit stop
> condition at every step**."
> "The failure mode this skill exists to prevent is spending an hour provisioning a GPU cluster and
> discovering at stage three that a credential did not have exact artifact access."

That *is* our access-check and spike phase, already written by the sponsor, with gates. Use it
instead of inventing our own.

Full per-person skill lists: **[`docs/skills.md`](docs/skills.md)**.

### Dimitris — the rover
Write `robot/SPEC.md`: joint/wheel names, action space, observation space, the **downward-facing
battery swap frame** (B is blocked on exactly this), and whether the panel tilt is fixed or has one
low-power pitch actuator.
Skills: `workflows/train-policy` → `tools/isaac-lab` → `urdf-mjcf-to-usd-conversion` →
`usd-articulation` → `physics-simulation` → `navigation-primitives`.

### B — the battery swap
**Script it, don't train it.** Contact-rich insertion is a twenty-year open problem; real swap docks
solve it *mechanically* — published stations swap in **45 s** with a carrier tolerating **±9°**
misalignment. Learned approach + scripted mechanism. The intelligence goes into *docking*.
Settle the swap geometry with Dimitris in five minutes → `envs/swap/SPEC.md`.
Skills: `usd-articulation` → **`manipulation-ik`** → `motion-generation` → `spatial-reasoning`.

### C — GPU first, then the world
Get an **RTX PRO 6000** up and Isaac Sim running headless. That unblocks everyone. Then publish
`scene/SPEC.md` — `(row, slot)` addressing, dock pose, Z-up/metres, and the **terrain statistics**.
Skills: `workflows/first-run-setup` → `atomic/gpu-selection` → `atomic/teardown-and-cost` →
`tools/gpu-cluster-provisioning` → `tools/token-factory` → `isaac-sim-headless-deployment` →
`usd-composition-architecture`.

---

## How we work

- **Verify credentials with a real API call.** Reading docs is not verification. Record the actual
  HTTP status code in `docs/access.md` (gitignored).
- **Keys go in `.env`** (gitignored). Never in chat, never committed. See `.env.example`.
- **Never pin a version from memory.** Resolve latest → verify → pin what you actually got.
- **Push early and often.** Never leave the push to the deadline.
- **Log every locked decision** in [`docs/decisions.md`](docs/decisions.md) — it becomes the
  write-up's architecture section for free.

## Still to do

- [ ] **Nebius Token Factory key** — the eligibility gate. `tokenfactory.nebius.com`
- [ ] Repo **public with an OSS licence file** before submission (hard requirement, by 27 Oct)
- [ ] Written explanation of what changed since 26 Aug (the JS sim predates the window)
- [ ] Add B and C as collaborators
