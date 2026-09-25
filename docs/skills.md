# Skills map — which skill to load for each step of the real simulation

Built 25 Sep 2026 from a read-only sweep of ~600 upstream SKILL.md files (IsaacLab,
IsaacSim, PhysX/ovphysx, ovrtx/ovstage, usd-*, kit-*, NVIDIA/skills,
nebius-physical-ai, newton, mujoco, simready-foundation), with key claims checked
against source code at the pinned tags.

**✓ = vendored in `.claude/skills/`** (loads automatically in Claude Code; pinned in
`skills-lock.json`). Unmarked = read it upstream at the pinned commit if needed.

## 0. The NVIDIA stack in plain words

- **USD** is the scene/robot file format. **Omniverse** is NVIDIA's family of apps and
  libraries that load, simulate and render USD.
- **Isaac Sim** is Omniverse's robot app: it imports robots (URDF/MJCF/CAD → USD),
  simulates them with **PhysX** (NVIDIA's GPU physics engine) and renders with RTX.
- **Isaac Lab** is the training framework on top: it clones thousands of robot copies
  into one GPU simulation and trains policies (PPO via RSL-RL).
- Since Isaac Lab 3.0 the physics engine is a launch option: `physics=isaacsim_physx`
  (PhysX in Isaac Sim) or `physics=newton_mjwarp` (**Newton**, whose MuJoCo-Warp solver
  is the MuJoCo contact model on the GPU). We **train on PhysX and check on
  MuJoCo-Warp**.
- **ovstage / ovphysx / ovrtx** are kit-less building blocks for custom apps. Not on
  our route.
- **Nebius** supplies the GPU VM (RTX PRO 6000) and **Token Factory**, which hosts the
  NVIDIA Nemotron model the fleet brain calls. **npa** (nebius-physical-ai) is Nebius's
  toolkit; we use its skills, not its managed Isaac path.

## 1. Pins

| Item | Pin |
|---|---|
| Isaac Lab | tag **v3.0.0-EA** (ae37b028), via NGC `nvcr.io/nvidia/isaac-lab:3.0.0-rc1`; fallback: native `uv` install of the same tag |
| Isaac Sim | **6.1.0** (7c206f75) — the pairing NVIDIA ships with Isaac Lab EA |
| MJCF importer | Isaac Sim 6.1 `isaacsim.asset.importer.mjcf` (wraps mujoco-usd-converter 0.5.0 / mujoco 3.11) |
| GPU | Nebius `gpu-rtx6000-a`, uk-south2 → eu-south1 → (L40S eu-north1); image `ubuntu24.04-cuda13.0`, driver **≥ 580.95.05** |
| LLM | Token Factory `nvidia/Nemotron-3_5-Lightning`, `enable_thinking=false` (confirm with the models list) |

Never: Isaac Sim 7.0 alpha, tracking release/3.0.0 HEAD, or upgrading to GA
mid-project.

## 2. Load order by workstream

### W1 · MJCF → USD (rover and dock)
1. ✓ **urdf-mjcf-to-usd-conversion** — official `MJCFImporter`, `robot_type` tokens
   ("Wheeled" is a schema token only), gain/bias pitfall, PhysX/MuJoCo physics
   payloads. Its `./isaaclab.sh -p` / `make_instanceable` section is stale for 3.0.
2. ✓ **isaaclab-preparing-assets-for-newton** — keep `run_asset_transformer=True` and
   `run_multi_physics_conversion=True` (neutral + PhysX + MuJoCo payloads); mechanical
   audit: mass, CoM, inertia, colliders, materials, articulation root, fixed joints.
3. ✓ **usd-articulation** — acceptance gate (`scripts/validate_articulation.py`):
   exactly one ArticulationRootAPI on the chassis, connected joint graph. The pack and
   its latch stay outside the articulation.
- Lower priority: usd-composition-architecture (IS); usd-exchange `usd-authoring`
  (only if something must be authored outside the importer).
- Skip: usd-content-agents converters (wrap the alpha pip converter), usd-optimize (no
  physics).

`scripts/tools/convert_mjcf.py` uses the full Isaac Sim importer when Isaac Sim is
installed and falls back to the kit-less importer otherwise.

### W2 · PhysX behaviour vs the locked spec (`robot/SPEC.md`)
1. ✓ **physics-simulation** — TGS, ≥ 240 Hz for contact-rich scenes, iteration
   counts, contact materials and combine modes, drives; the two flags that force PhysX
   in standalone Isaac Sim scripts.
2. ✓ **isaaclab-using-sensors-actuators** — `ImplicitActuatorCfg` fields (stiffness,
   damping, joint_effort_limit, armature, friction, dynamic_friction,
   viscous_friction); `DCMotorCfg` (torque-speed) and `DelayedPDActuatorCfg` (latency).
3. ✓ **isaaclab-selecting-backends** — one backend working before presets; never copy
   numbers between engines; smoke-test each backend.
4. ✓ **navigation-primitives** — DifferentialController + wheel velocity targets as
   the scripted baseline for speed/spin tests; spawn z from colliders.
5. ✓ **isaaclab-preparing-assets-for-newton** (again) — MJWarp column: more default
   slip, no joint-velocity-limit enforcement, condim/cone/impratio via `Mujoco*Cfg`.
6. ✓ **isaaclab-using-presets** — `PhysicsCfg(PresetCfg)` with isaacsim_physx and
   newton_mjwarp, once both run.
- Terrain friction idiom (Isaac Lab velocity env): terrain μ = 1.0 with
  `friction_combine_mode="multiply"`, so the pair coefficient equals the tyre μ
  (0.65) — the PhysX equivalent of MJCF `priority=1`.
- Tyres stay primitive cylinders (Isaac Lab has smooth cylinder-vs-mesh contact "for
  better wheeled simulation"); add a same-seed determinism check.

### W3 · Isaac Lab environment and RL training
1. ✓ **isaaclab-building-environments** — manager-based; start from the nearest
   maintained task; validation: import → few envs → random actions → shapes → short run.
2. ✓ **isaaclab-using-sensors-actuators** — rangefinders = `MultiMeshRayCasterCfg`
   (plain RayCaster sees one static mesh only); chassis ContactSensor.
3. ✓ **isaaclab-training-rl-agents** — `isaaclab train/play --rl_library rsl_rl`,
   `--viz none` headless, smoke-test first.
4. ✓ **isaaclab-debugging-rl-training** — reward audit, one-variable experiments,
   pick checkpoints by task metrics, not reward.
- Code templates at the EA tag: `contrib/navigation/config/anymal_c/navigation_env_cfg.py`,
  `envs/mdp/commands/pose_2d_command.py`, `terrains/config/rough.py`,
  `sensors/ray_caster/multi_mesh_ray_caster_cfg.py`; ours: `envs/traverse/train.py`
  (obs, REWARD_W, curriculum, degenerate-solution guards).
- Never `NonHolonomicActionCfg` (kinematic base motion). Actions =
  `JointVelocityActionCfg` on the 4 wheels.

### W4 · Battery-swap dock
1. ✓ **urdf-mjcf-to-usd-conversion** — weld → FixedJoint (excludeFromArticulation,
   jointEnabled); second free body → plain rigid body.
2. ✓ **usd-articulation** — dock is its own fixed-base articulation.
3. ✓ **physics-simulation** — position drive on the prismatic lift.
4. ✓ **isaac-sim-workflow** — integrity rule: a scripted mechanism is never presented
   as autonomy.
5. ✓ **isaac-sim-robot-navigation** — driving robots live in standalone demo scripts.
- Lower priority: IsaacLab `isaaclab-planning-manipulation-tasks` (phase gates).
- Repo inputs: `envs/swap/SPEC.md`, `envs/swap/dock_mjcf.py`,
  `envs/swap/scripts/swap_battery.py` (DockHardware Protocol, 6 tests),
  `docs/integration-A-B.md` (geometry).

### W5 · Orchestrator (Nemotron on Token Factory)
1. ✓ **token-factory** — base URL `https://api.tokenfactory.nebius.com/v1/`; key
   starts `v1.`; list models before pinning an id.
2. ✓ **health-preflight** — needs npa + Nebius CLI 0.12.254; optional on our route.
- Lower priority (NVIDIA/skills): cuopt-routing-api-python (deterministic
  herd-to-dock scheduling under the LLM).

### W6 · Nebius GPU box, headless, cost
1. ✓ **gpu-selection** — rendering needs RT cores (no H100/H200/B200).
2. ✓ **third-party-eula-preflight** — `ACCEPT_EULA=Y` for Isaac images.
3. ✓ **isaaclab-installing-isaac-lab** — preflight, route choice, verification.
4. ✓ **isaac-sim-headless-deployment** — no-window / renderer-off, batch loop.
5. ✓ **isaaclab-setup-troubleshooting** — import → random_agent → 1-iteration train.
6. ✓ **isaac-sim-troubleshooting** — `SimulationApp.close()` watchdog, `/dev/shm`
   cleanup.
7. ✓ **teardown-and-cost** + `scripts/teardown.sh` — delete the VM after every
   session; keep only the data disk.
8. ✓ **vm-nebius-auth** — headless CLI login on a VM.
9. ✓ **protect-nebius-infra-details** — no tenant/project IDs in the repo; scan staged
   diffs with npa's confidentiality scanner.
- Nebius host gotcha: CUDA images can lack NVIDIA EGL/GL/Vulkan user-space libraries
  that Isaac needs even headless. npa's `deploy/terraform/cloud_init.yaml.tpl`
  installs the exact-version packages; run the Isaac Sim Compatibility Checker.
- Skip: npa `isaac-lab` (pins beta2 / Sim 6.0.1, E2E still pending), npa cluster /
  mk8s / SkyPilot paths.

### W7 · Simple video
1. ✓ **isaaclab-training-rl-agents** — `isaaclab play --checkpoint …` with
   `VideoRecorderCfg`.
2. ✓ **isaac-sim-rendering** — headless capture, explicit lights, ffmpeg assembly.
3. ✓ **isaac-camera** — fixed overview or chase camera.
4. ✓ **isaac-sim-validator** — start/middle/end frame checks, no black frames.
5. ✓ **isaac-sim-workflow** — acceptance criteria.
- If RTX PRO 6000 shows artefacts: npa isaac-arena settings (legacy RTX on, RT2 and
  path tracing off, ≥ 8 settling renders).

### W8 · Sim-to-real
1. ✓ **isaaclab-randomizing-with-events** — the only `stable` Isaac Lab skill: event
   modes, PhysX vs Newton differences, ADR via DifficultyScheduler + modify_term_cfg.
2. ✓ **isaaclab-transferring-policies-sim-to-sim** — PP/PN/NN/NP matrix with
   joint/body ordering overrides; matched actuator response. We commit to PP + PN
   (a deliberate reduction of the skill's full matrix).
3. ✓ **isaaclab-using-sensors-actuators** — DC-motor and latency actuator models.
4. ✓ **isaaclab-debugging-rl-training** — robustness ablations.
- Export (docs at the EA tag): `policy_deployment/01_io_descriptors`, `05_leapp`,
  `isaaclab_rl/rsl_rl/exporter.py` (ONNX/JIT). Prove ONNX matches PyTorch on recorded
  observations.

## 3. Conflicts between skills, and our resolution

| Conflict | Resolution |
|---|---|
| The importer copies the MJCF velocity gain (kv 18) into a per-degree USD drive gain (~57× too stiff) and writes frictionloss into a unitless legacy coefficient | Set damping 18, friction 0.60, etc. in the Isaac Lab actuator config (SI units). The USD values are overridden on purpose. |
| physics-simulation labels USD angular drive gains "Nm/rad" | Wrong: USD angular gains are per degree. Isaac Lab converts. |
| physics-simulation: Kit 110 auto-switches to Newton; isaac-sim-validator: "reject PhysX, use Newton" | Deliberate override: we train on PhysX (Omniverse's engine). Isaac Lab picks via `physics=`; standalone scripts pass the PhysX-forcing flags. Isaac Sim 6.1 notes say PhysX remains the default. |
| "Newton drops joints whose drive stiffness and damping are both 0" | No longer true at the EA tag when an Isaac Lab actuator config is present. |
| urdf-mjcf-to-usd-conversion says `./isaaclab.sh -p` and `make_instanceable: true` | Stale for 3.0: use `MjcfConverterCfg` / `convert_mjcf.py`; output is instanceable by default. |
| npa isaac-lab pins 3.0.0b2.post1 / Sim 6.0.1 | Not our route: its importer drops static friction; its E2E is pending. Not vendored. |
| npa isaac-arena (RT2 off) vs isaac-sim-rendering (RT2) | Try IS defaults first; npa settings only if artefacts appear. |
| `docs/integration-A-B.md` §6: friction and velocity servos have "no UsdPhysics equivalent" | Outdated: Isaac Lab actuator configs cover them. Only cone, impratio and solref/solimp have no PhysX meaning — which is why the spec numbers are re-measured, not copied. |

## 4. Whole families to skip
- **ovrtx (38), ovstage (15), kit-cae (4), kit-app-template (1):** NVIDIA proprietary —
  never commit (blocked in `.gitignore`); not on our route.
- **ovphysx (7):** standalone PhysX 5.11 (Isaac Sim 6.1 runs 5.9); reference only.
- **usd-optimize (16):** no physics. **usd-content-agents (34):** heavy experimental
  harness.
- **IsaacSim SDG/people/ROS skills**, **NVIDIA/skills** families unrelated to robots,
  **npa** skills for other simulators/robots (Cosmos, VLA, Franka sim2real, etc.).

## 5. How to use this map
1. Before a workstream, load its ✓ skills in order.
2. Isaac Lab skills link to docs with `../../../docs/source/...`; those resolve inside
   `~/IsaacLab` (tag v3.0.0-EA) in WSL and on the VM.
3. Check every number or command a skill gives against the pins in §1 before using it.
4. To add a skill: copy the whole upstream directory unmodified to
   `.claude/skills/<frontmatter-name>/`, add it to `skills-lock.json` (source, commit,
   licence, SHA-256) and `THIRD_PARTY_NOTICES.md`. No symlinks.
