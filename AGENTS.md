# AGENTS.md — guidance for AI agents (and humans) working in this repo

This repo has **two parts** with different rules:

1. **The browser economics sim** — `index.html`, `config.js`, `simulation.js`,
   `render.js`, `main.js` (+ `.tmp-frames.js`, `.tmp-shot.html`). Dependency-free
   vanilla JS. **Everything from here down to "Where to make a given change" applies
   ONLY to these files.**
2. **The real simulation (Python / NVIDIA Omniverse)** — `robot/`, `envs/`,
   `renders/`, `scripts/`, `docs/`. Python with pinned dependencies, heading for Isaac
   Lab on a Nebius GPU. See **[Part 2](#part-2--the-real-simulation-python--omniverse)**
   at the end of this file.

## Part 1 — the browser economics sim

Solar Sheep's browser sim is small and dependency-free. It is intentionally
plain: no framework, no build, no package manager, no tests directory. Read this
before changing those files so you keep it that way.

## Ground rules

- **Vanilla ES2017+ JavaScript, plain `<script>` tags.** No ES modules, no
  bundler, no TypeScript, no transpilation. `index.html` loads scripts in a fixed
  order and relies on global scope (see below). Don't introduce imports/exports.
- **Zero runtime dependencies.** Do not add `npm install`, `package.json`, CDN
  scripts, or any fetch of external resources. Everything must work from a static
  file server or straight from disk.
- **`'use strict';` is the first line** of every JS file. Keep it.
- **Keep it small.** The whole app is ~1200 lines across 4 JS files plus the HTML.
  Favor small, readable changes over abstraction.

## File map & responsibilities

| File | Owns | Touch it when |
| --- | --- | --- |
| `index.html` | Layout, all CSS, UI controls, **script load order** | Changing UI, adding controls, changing styling. |
| `config.js` | The single global `CONFIG` object | Adding/tuning any simulation parameter or price. |
| `simulation.js` | Pure-ish sim core: world init, state machine, energy math, weather, clouds, time stepping, history. | Changing behaviour, physics, energy numbers, or adding sim logic. |
| `render.js` | All Canvas 2D drawing (scene, battery plant, metrics overlay, 24h chart) and the shared `STATE_LABEL` / `STATE_COLOR` maps. | Changing visuals, labels, or colours. |
| `main.js` | The `requestAnimationFrame` loop, speed control, selection, tuning, cost panel, HUD text updates. | Changing interaction, performance cadence, or what the UI shows. |

### Load order matters

`index.html` loads them in this order and each file reads globals from the ones
before it:

```
config.js → simulation.js → render.js → main.js
```

`config.js` defines `CONFIG`. `simulation.js` uses `CONFIG` (via `createSim`)
and defines `createSim`, `stepSim`, `advance`, `sunFactor`, `sunPos`, `shadeAt`,
`solarInW`, `weatherF`, `whNowW`, `plantNowW`, `unitNowW`, `poolStoredWh`,
`sheepNowW` (legacy no-op — the herd no longer discharges directly),
`gridNowW` (plant + roof), `fmtWh`, `hourOfDay`, `dayOf`, `mulberry32`,
`NAMES`, `DAY`, `WEATHER_LABEL`, `WEATHER_COLOR`, `pickWeather`.
`render.js` defines `makeDraw` plus `STATE_LABEL` and `STATE_COLOR`.
`main.js` is an IIFE that consumes all of the above.

If you add a new global that more than one file uses, define it exactly once in
the earliest file that owns the concept, and keep the load order valid.

## Conventions

- **Naming:** camelCase functions/vars, `sim` is the central state object passed
  around, `cfg` is the local alias for `sim.cfg` / `CONFIG`.
- **`main.js` wraps everything in an IIFE** so its locals don't leak; the other
  three files intentionally use top-level globals. Don't "clean up" the globals
  into modules.
- **No classes.** Plain objects and factory functions (`createSim`, `makeDraw`).
- **Determinism:** world init (slots, clouds, per-sheep `wake`/`recall`/`skill`,
  initial weather) is seeded with `mulberry32(1337)` so a fresh sim always starts
  identically. `Math.random()` is used at **runtime** for wander targets,
  particle spawns, cloud respawn, and weather state *transitions* — that's fine,
  but keep init deterministic.

## Units & invariants (important — keep these straight)

- **Time** is in **seconds**. `sim.t` is an absolute seconds-from-start value;
  `DAY = 86400`. Day of day is `t % DAY`. `hourOfDay(t)` and `dayOf(t)` exist —
  use them. Note `createSim` starts at `(startDay - 1) * DAY + startHour * 3600`
  so Day 1 at 06:30 is what `dayOf` reports.
- **Power** is in **watts (W)**; **energy** is in **watt-hours (Wh)**; convert
  with `dtH = dt / 3600`. `fmtWh()` pretty-prints Wh/kWh.
- **SoC** is a **fraction 0–1**, not a percentage, stored on a **pack object**
  `{id, soc, state}`, never on the sheep. Each sheep carries one pack
  (`s.pack`, state `'carry'`); the rest of the pool moves through the battery
  plant (see below). Multiply by `battery.capacityWh` for Wh. Usable range is
  `reserveSoc`…`1`.
- **Fixed substep:** `advance()` steps the sim in fixed `cfg.substep` (0.5 s)
  increments and **carries the fractional remainder in `sim.acc`**, so total
  advance is exactly `speed × realSec` quantised to whole substeps. Do not
  re-introduce per-frame overshoot (`while (t < target)` without a remainder) —
  it makes the sim run up to ~3× too fast.
- **World space** is `1600 × 1000` units; the canvas scales/letterboxes it to
  fit (`S = min(w/worldW, h/worldH)`). Coordinates in sim/render are world
  units, not pixels. Mouse→world conversion lives in `toWorld()` in `main.js`.
- **Chart** is a fixed-size 320×130 canvas; `sim.history` holds per-bucket
  `{b, sAcc, rAcc, dur}` records — sheep and roof grid-energy separately.
- **Orientation:** `s.orient` is a heading in radians (screen coords); the
  sheep body rotates to it but labels/battery bar are drawn unrotated.

## Sheep state machine (refer to `simulation.js:stepSim`)

```
resting → toField → harvesting → toDock → queuing → swapping
  ↑__________________________________________________|
  (fresh pack: daytime → toField, after recall → resting)
```

- `resting` → `toField` when `(t % DAY) >= sunrise + s.wake` and `< s.recall`
  (per-sheep stagger, so the herd does **not** move out as one block).
- `toField` → `harvesting` on arrival (picks a wander target); if the sun is
  down it bails straight to `toDock`.
- `harvesting` → `toDock` when SoC ≥ `fullThreshold` or `(t % DAY) ≥ s.recall`.
- `toDock` → `queuing` at the dock pad.
- `queuing` → `swapping` when first in line **and** an empty pack is waiting on
  the ready rack (otherwise it holds the berth so the empty pack never gets
  stuck).
- `swapping` → after `dock.swapSec`: the sheep's pack is pushed onto the
  conveyor (state `'line'`) and it takes an empty pack from the ready rack.
  Then → `toField` (daylight) or `resting` (night).

While sunning, each sheep **turns toward the sun** at `cfg.sheep.turnRate`
rad/s, with a per-sheep `skill` (0.85–1.0) that both biases its target angle
and scales its **exposure** = `max(0, cos(angleError)) × skill`. Panel output is
`peakW × sun × cloudShade × exposure × weatherF` — facing the sun genuinely
matters. `s.exposure` is also shown in the UI (`face N%`).

When adding a state, update: the `switch` in `stepSim`, `STATE_LABEL`,
`STATE_COLOR`, and any place that special-cases state (e.g. the "on site" count
in `main.js`, `solarInW`'s daylight check, motor-draw checks in the overlay /
details panel).

## Battery plant (the swap line)

A fixed pool of `dock.pool` (50) packs cycles through
`carry → line → discharging → ready → carry`:

- **Single dock** at the warehouse's right wall (`sim.dockPos`); at most one
  sheep swaps at a time (`sim.dock.current`), the rest queue in
  `sim.dock.queue` with positions from `sim.queuePt(i)`.
- **Conveyor** `sim.line`: packs with a `t` timer, `dock.lineSec` (30 s)
  transit, oldest first; on arrival they join `sim.ready` (state `'ready'`).
- **Discharge units** `sim.units` (`dock.units` = 10): `stepPlant()` pulls
  **charged** packs off `sim.ready` (FIFO; empty packs stay put, waiting for a
  sheep to carry them out) and drains them at `grid.dischargeW` each, 95%
  discharge efficiency, down to the reserve floor; drained packs go back to
  `sim.ready`.
- **Ready rack** = `sim.ready` (an array of pack objects). It holds both the
  charged packs awaiting a unit and the empty packs awaiting a sheep. A pack
  is "empty" when `soc <= reserveSoc + 1e-9`.
- **Init:** pools n packs to the sheep (at the reserve floor = the unproduced
  initial reserve) and puts the rest into `sim.ready`.
- **Determinism:** pack ids/order are seeded; only weather/wander/particles
  use runtime `Math.random()`.

## Weather

`sim.weather = { state, left }` with states from `CONFIG.weather`
(`clear` / `cloudy` / `overcast`). Each state has a solar multiplier `f`, a
**target cloud count**, and a pick weight. Every `1–4` sim hours a new state is
rolled (`Math.random()` at runtime). The cloud array grows/shrinks toward the
target count each step. `weatherF(sim)` is the only thing physics code should
call to read it; render uses it for field brightness/tint. Keep the seeded-init
and runtime-random split (see Determinism).

## Energy bookkeeping (the invariant — keep it exact)

`sim.totals = { produced, harvest, delivered, lost }` (Wh) and `sim.daily`
(holds the same four, reset at each midnight) feed the header, the on-canvas
overlay, and the warehouse/plant panels. The accounting must balance **to the
floating-point epsilon**:

```
produced = delivered + lost + (in-pool − initial-reserve)
```

where `in-pool` = `poolStoredWh(sim)` (all 50 packs, wherever they are) and
`initial-reserve` = `sheep.count × reserveSoc × capacityWh`.

- `produced` += gross PV Wh — sheep panel `gainW × dtH` **and** roof `roofW × dtH`.
- `harvest` += Wh actually stored in a pack (after charge efficiency, capped
  by headroom).
- `delivered` += Wh discharged to the grid by the **plant's** units + roof;
  `whDelivered` tracks the roof's share, `plantDelivered` the plant's.
- `lost` += motor draw (booked exactly as the Wh it pulls out of the pack) +
  charge loss + clipped solar (in harvesting, `lossWh = gainWh×dtH −
  chargedWh×dtH` so the row sums exactly) + **discharge loss**
  (`outWh × (1/dischargeEff − 1)` in the plant).

Because energy can sit in any of the 50 packs, the identity only closes over
the **whole pool** — don't "simplify" it back to the herd. `node .tmp-frames.js`
**asserts this** (drift > 1 Wh fails), plus pool bookkeeping (every pack in
exactly one state, `sim.ready` length matches) and that the plant actually
delivers. If you change the charge/discharge math, re-check the identity and
that `plantNowW()` / `whNowW()` / `gridNowW()` still match what's really being
delivered.

## How to run & verify

Serve statically and view in a browser:

```sh
python -m http.server 8765   # → http://localhost:8765
```

**Server lifecycle (for agents):**
- **Probe before starting** — `curl -s -o /dev/null -w "%{http_code}"
  http://localhost:8765/`; a `200` means it's already running, don't start
  another. A `000`/connection-refused means it's down.
- **Start detached** — a server run in your own terminal (even backgrounded
  with `&`) **dies when your session/terminal is closed**. On Windows:
  ```sh
  powershell -NoProfile -Command "Start-Process -WindowStyle Hidden python -WorkingDirectory 'c:\path\to\solar-sheep' -ArgumentList '-m','http.server','8765'"
  ```
  then re-probe. (For humans: just run `python -m http.server 8765` in a
  terminal of their own; if the port is already in use, the server is already
  up — open http://localhost:8765 as-is.)

**Headless smoke test:** `node .tmp-frames.js` runs the real `config.js`,
`simulation.js`, `render.js`, `main.js` inside a `vm` sandbox with a mocked
DOM/canvas, drives 3000 rAF frames, then fast-forwards a full day (~14h,
sunrise → night) via `advance()` so the dock/conveyor/units/ready-rack are all
exercised, then a few more live frames. It **exits non-zero if** the sim
freezes, the rAF loop dies, the energy balance drifts > 1 Wh, the pool
bookkeeping is inconsistent, or the plant delivered nothing. It prints the
clock, weather, per-sheep states/pack-SoC, plant status and the totals
breakdown. Run it after changing `simulation.js` or the frame loop:

```sh
node .tmp-frames.js
# expect: "FRAME LOOP OK"
```

The mock is deliberately crude (proxy-based 2D context, stub elements) — it
checks the sim advances, balances, and doesn't throw, not pixel output.

## Gotchas / things to know before editing

- **`.tmp-frames.js` and `.tmp-shot.html`** are development/verification
  artifacts, not part of the app. They are safe to run (`node .tmp-frames.js`)
  but are not loaded by the app. Don't ship logic that depends on them.
- **`shot-day.png`** is a reference screenshot (also linked from the README).
- **`nul`** is a stray Windows shell-redirect artifact. It is not a source file;
  it can be safely ignored or deleted (never `echo ... > nul`-style output).
- **`server.log`** is just `python -m http.server` access log output — ignore.
- **`render.js` defines `STATE_LABEL`/`STATE_COLOR`** and `main.js`/`simulation.js`
  define `WEATHER_LABEL`/`WEATHER_COLOR` that `render.js` reads. That is
  load-order dependent; if you move these metadata maps, make sure they're
  defined before the files that use them run.
- **The scene canvas uses `ctx.roundRect`** — requires a reasonably modern
  browser. Don't regress to a context that lacks it without a fallback.
- **Don't break the "works from disk" property.** Everything is relative to
  `index.html`; no absolute paths, no server-only features.

## Where to make a given change

- A new tunable number → add to `CONFIG` in `config.js`, read it in
  `simulation.js`/`render.js`, and (if it's a user-facing slider) add the input
  to `index.html` + wire it in `main.js` (including `updateCost()` if it
  affects cost).
- New behaviour/physics → `simulation.js` (`stepSim` and helpers).
- New visual → `render.js` (`makeDraw` helpers).
- New UI control or readout → `index.html` + `main.js`.
- Keep `CONFIG` as the single source of truth; don't hardcode magic numbers
  in the sim when a config key fits.

---

## Part 2 — the real simulation (Python / Omniverse)

Goal: a real simulation for sim-to-real, built with NVIDIA's official tools and run on
a Nebius GPU — Isaac Lab (training) on Isaac Sim (Omniverse's robot app, PhysX
physics). Why each choice was made: [docs/decisions.md](docs/decisions.md).

### Start here
- **Branch:** all work is on `real-sim`. `physical-ai` and `main` are frozen — never
  commit, push, reset or tag them.
- **Skills:** [docs/skills.md](docs/skills.md) says which skill to load for each step,
  in order. The 29 vendored skills live in `.claude/skills/<name>/` as real
  directories and load automatically. Never add symlinks there (Windows checks them
  out as 30-byte stubs); to add a skill, follow docs/skills.md §5.
- **Isaac Lab skills** link to docs with `../../../docs/source/...`; follow those
  inside `~/IsaacLab` (tag v3.0.0-EA) in WSL or on the VM.
- **Pins** (Isaac Lab v3.0.0-EA + Isaac Sim 6.1.0, NGC container, driver ≥ 580.95.05):
  docs/skills.md §1. Check any number or command a skill gives against them.

### Layout
| Path | What |
|---|---|
| `robot/rover.xml`, `robot/SPEC.md` | MuJoCo model of the rover and its locked, measured spec (the targets the Omniverse model must match) |
| `envs/swap/` | Battery-swap dock: engine-agnostic phase machine + `DockHardware` Protocol, MuJoCo dock (`dock_mjcf.py`), 6 tests |
| `envs/traverse/` | MJX + Brax trainer — deadline-only fallback, not developed further |
| `renders/` | MuJoCo renders and the swap demo (`swap_demo.py`) |
| `scripts/teardown.sh` | Deletes every Nebius VM and disk in the tenant except the kept data disk |

### Python environment
- Windows: `.venv` at the repo root. WSL: a separate clone at `~/solar-sheep` on the
  **Linux filesystem** — run Linux-bound work there, never from `/mnt/c`.
- Install: `uv pip install -r requirements.txt --override requirements-overrides.txt`
  (the override is intentional; see the header of requirements.txt).
- Line endings are set by `.gitattributes`: shell scripts and skills are always LF.

### Verify after touching the swap or rover code
```sh
python -m envs.swap.tests.test_swap_phases                          # expect: 6 tests OK
python renders/swap_demo.py --no-video --out /tmp/sd.mp4            # expect: ALL ASSERTIONS PASSED
```

### Nebius and secrets
- Keys only in `.env` (gitignored). Never commit tenant/project IDs, endpoints or
  credentials; read them from the environment. Before committing anything that
  mentions Nebius, scan the staged diff (skill: `protect-nebius-infra-details`):
  `git diff --cached | (cd ~/npa-src/npa/src && python3 -m npa.guardrails.confidentiality --repo-root /tmp --built-in-nebius-infra --stdin-source staged-diff --stdin-is-diff)`
- No GPU spend without the owner's explicit go. **Delete the VM after every session**
  (`bash scripts/teardown.sh --yes`); only the persistent data disk stays.

### Commits
- Explicit pathspecs only (never `git add -A`); no force-push; no AI co-author
  trailers.
