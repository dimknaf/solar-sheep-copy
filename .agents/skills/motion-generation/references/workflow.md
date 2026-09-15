# Motion-generation demo workflow

Use this sequence for new motion-generation demos. It assumes the generic Python
server, rendering, physics, and manipulation validation rules are loaded from
the sibling skills listed in `../SKILL.md`. The controller is pluggable; for the
cuMotion `RmpFlowController` specifics see `cumotion.md`.

## Runtime choice

- **Python server**: default for iteration, stage inspection, marker placement,
  screenshots, and validation probes.
- **Standalone script**: deliverable shape when the demo must own
  `SimulationApp`, app startup, simulation timing, CLI args, and pass/fail
  reporting. Mirror the scene logic through the Python server first unless the
  user explicitly asks to validate by launching `python.sh`.
- **Interactive `BaseSample` extension**: use only after the core sequence is
  stable. Port setup/reset/callback structure, not old IK APIs.

## Demo checklist

1. Define task success as measured state transitions, not just tool motion.
2. Read the current local motion-generation docs/example for the selected robot,
   plus the controller-specific reference (e.g. `cumotion.md`).
3. Build the smallest stage: robot, support surface, one object, lights, and
   only required obstacles.
4. Inspect robot DOFs, link paths, controller tool frames, object AABB,
   rigid-body state, collision APIs, and mass.
5. Add debug markers for the controller target, measured tool pose, physical
   grasp point, object center/grasp point, and relevant axes.
6. Define the axis contract: tool approach axis, object semantic axis, and
   desired final object axis.
7. Compute any tool-local grasp offset in the same orientation used for
   approach; freeze it while closing and lifting.
8. Build the world binding (`SceneQuery`, `ObstacleStrategy`, `WorldBinding`;
   see `world-binding.md`) and the motion-generation controller (cuMotion
   `RmpFlowController`; see `cumotion.md`).
9. Implement an explicit phase machine. Every phase needs:
   - target pose
   - controller reset behavior
   - convergence predicate
   - timeout
   - trace output with measured object/tool state
10. Reset the controller at discontinuous target jumps.
11. Validate gates in order with `manipulation-ik`: pick-up/hold,
    manipulate/flip, and place/release. Do not continue after a failed gate.
12. Capture visual evidence from the latest run with `isaac-sim-remote` helpers.
13. Distill reusable corrections into the owning skill before delivery.

## Phase machine shape

Use named phases rather than hidden timers. Start with `PHASES` from
[`scripts/phase_machine.py`](../scripts/phase_machine.py) and remove phases that do
not apply to the task.

Motion phases should converge on measured state. A timeout in a motion phase is
a failed or incomplete phase, even if a final pose oracle happens to pass.
Fixed-frame completion is only acceptable for intentional dwell, hold-open, or
settle phases.

## Standalone scaffold

Use [`scripts/cumotion/standalone_demo_template.py`](../scripts/cumotion/standalone_demo_template.py)
as a starting point for standalone demos. It owns `SimulationApp` construction and
the required post-app imports.

Do not copy standalone startup calls into Python-server probes. In particular,
`SimulationManager.setup_simulation(...)` belongs in standalone `SimulationApp`
demos. In a running server, stop/play with `app_utils`, reset or recreate the
stage, and step with `await app_utils.update_app_async()`.

## Finish criteria

- The phase machine reaches a terminal phase without unhandled exceptions.
- Motion phases complete by measured convergence, not by motion timeouts.
- Gate validation passes in order and includes object-state evidence.
- The latest run has fresh visual evidence through `isaac-sim-remote`.
- The result uses the requested physical interaction model. If the task is
  contact-only, no pose assist, hidden constraints, or direct object pose writes
  are used as the success path.
