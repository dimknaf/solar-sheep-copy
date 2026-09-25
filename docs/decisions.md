# Decisions log

Newest first. Each entry says what was decided, why, and what it supersedes.
Details of every skill mentioned: [docs/skills.md](skills.md).

---

## D9 · 25 Sep 2026 — All new work on branch `real-sim`
- `physical-ai` (39f8594, the 15 Sep hackathon state) and `main` are frozen: no
  commits, pushes, resets or tags.
- The 13 files that existed only uncommitted on one laptop (including the dock fix
  without which `renders/swap_demo.py` fails with `ImportError`) were committed on
  `real-sim` first, and verified in a fresh WSL clone.

## D8 · 25 Sep 2026 — Nebius hosting: plain VM + NVIDIA's container, not npa's managed path
- **Decided:** raw Nebius CLI VM, `gpu-rtx6000-a` in uk-south2 (fallback eu-south1,
  then L40S in eu-north1), image `ubuntu24.04-cuda13.0`, driver ≥ 580.95.05. A
  persistent data disk holds checkpoints and caches; **the VM is deleted after every
  session** (`scripts/teardown.sh` keeps only the data disk).
- **Why:** us-central1 RTX PRO 6000 failed on 15 Sep with 4× `NotEnoughResources`
  (capacity, not our config). At npa bf4788a: npa rejects Nebius CLI 0.12.277, cannot
  target `gpu-rtx6000-a`, `npa cluster` excludes uk-south2, and its Isaac Lab E2E is
  still "pending W9-isaac-lab-e2e-fix".
- **Host preflight:** Nebius CUDA images can lack NVIDIA EGL/GL/Vulkan user-space
  libraries; npa's `cloud_init.yaml.tpl` has the exact-version fix.

## D7 · 25 Sep 2026 — Pin Isaac Lab v3.0.0-EA + Isaac Sim 6.1.0
- **Decided:** tag v3.0.0-EA (ae37b028) via NGC `nvcr.io/nvidia/isaac-lab:3.0.0-rc1`;
  native `uv` install of the same tag as fallback. No upgrade to GA mid-project; never
  Isaac Sim 7.0 alpha.
- **Why:** NVIDIA's own pairing ("built for Isaac Sim 6.1"); the Isaac Lab skills ship
  in this tag and use its CLI. The alternative npa image (Lab 3.0.0b2.post1 + Sim
  6.0.1) bundles mujoco-usd-converter 0.2.0, which authors **no static friction** —
  fatal for skid-steer turning (fixed in converter 0.4.1; Sim 6.1 uses 0.5.0).

## D6 · 25 Sep 2026 — Asset import: the official importer, physics set in Isaac Lab
- **Decided:** import `robot/rover.xml` variants with Isaac Lab `MjcfConverterCfg` →
  Isaac Sim 6.1 MJCF importer (`robot_type="Wheeled"`, `fix_base=False`, multi-physics
  payloads) on the GPU box. Drive and friction come from Isaac Lab's
  `ImplicitActuatorCfg` in SI units (damping 18, effort 2.4, friction 0.60, …).
- **Why:** the importer copies kv 18 into a per-degree USD drive gain (~57× too
  stiff) and writes frictionloss into a unitless legacy coefficient. Isaac Lab's
  actuator config is the official place for these and overrides the USD values — so
  **the importer's gains are ignored on purpose**, and no hand-authored USD post-pass
  is needed.
- **Not done:** the alpha pip `mujoco-usd-converter` on the laptop plus hand edits.

## D5 · 25 Sep 2026 — Physics roles: train on PhysX, check on Newton MuJoCo-Warp
- **Decided:** train with `physics=isaacsim_physx`; evaluate the same checkpoint with
  `physics=newton_mjwarp` (PP vs PN) as the sim-to-sim robustness gate.
- **Why:** PhysX is Omniverse's engine; MuJoCo-Warp runs the contact model the
  prototype was tuned on, so a policy that survives both has not overfitted one
  contact model. If PhysX cannot turn with plausible friction, train on MuJoCo-Warp in
  the same task and use PhysX as the check.
- **Skill overrides:** isaac-sim-validator says "reject PhysX, use Newton" and
  physics-simulation says Kit 110 auto-switches to Newton — both overridden on purpose.
  physics-simulation also labels USD angular gains "Nm/rad"; they are per degree.

## D4 · 25 Sep 2026 — The rover is a PhysX articulation with 4 velocity-driven wheels
- **Why:** PhysX Vehicle / tank drive is CPU-only and cannot be cloned for GPU
  training. Isaac Lab has no wheeled robot; its `NonHolonomicActionCfg` moves the base
  kinematically (naive for sim-to-real). Skid-steer turning must emerge from tyre–ground
  contact, so it is **measured** against `robot/SPEC.md`, not assumed.
- Tyre–ground friction is made explicit: terrain μ 1.0 with `multiply` combine, tyre μ
  0.65 (PhysX has no MJCF `priority`).

## D3 · 25 Sep 2026 — MJX trainer becomes a deadline-only fallback
- `envs/traverse/train.py` (MJX + Brax) is kept but not developed. Its known bugs
  (slope force uses 25.4 kg not 29.4 kg; pack not moved at spawn) are fixed only if we
  fall back to it.

## D2 · 24 Sep 2026 — Goal: a real simulation for sim-to-real, in Omniverse
- Owner: "reality, so this can later go to the real world; simple graphics are fine;
  Omniverse and the skills are what we wanted." And (25 Sep): "no crazy fixes, but also
  not naive isaac… we have the nebius to have a real simulation."
- Photorealism is not a goal; physics fidelity, trained policies, domain randomisation,
  sim-to-sim checks and an exportable policy are.

## D1 · 24 Sep 2026 — Reverses integration-A-B §6 ("one simulator — MuJoCo")
- The 15 Sep decision kept MuJoCo because the tuned turning (elliptic cone, impratio 1,
  soft contacts, velocity servos) had "no UsdPhysics equivalent". Partly wrong: joint
  friction and velocity drives do exist (D6). Cone, impratio and solref/solimp truly
  have no PhysX meaning — the accepted cost is that every headline number in
  `robot/SPEC.md` is **re-measured under PhysX** (gate G2).

---

## Open (owner)
1. **Solar panel: flat (as built), fixed tilt, or a tilt motor?** A flat panel's normal
   is +Z, so yawing toward the sun changes nothing on level ground (flagged 15 Sep;
   `robot/SPEC.md` §11 lists it OPEN). Must be settled before the final rover import —
   it changes mass, CoM, climb and tip-over.
2. **Battery swap:** proposed — train the approach and berthing; lift and latch
   mechanical and physics-driven, never teleported.
3. **Real hardware before 30 Oct:** proposed no; ship the exported policy + I/O
   contract, the PhysX↔MuJoCo-Warp results and a system-identification procedure.
4. **GPU spend:** Token Factory key, credit balance, CLI re-login, preemptible vs
   on-demand.
