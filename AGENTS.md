# AGENTS.md — guidance for AI agents (and humans) working in this repo

Solar Sheep is a small, dependency-free browser simulation. It is intentionally
plain: no framework, no build, no package manager, no tests directory. Read this
before changing anything so you keep it that way.

## Ground rules

- **Vanilla ES2017+ JavaScript, plain `<script>` tags.** No ES modules, no
  bundler, no TypeScript, no transpilation. `index.html` loads scripts in a fixed
  order and relies on global scope (see below). Don't introduce imports/exports.
- **Zero runtime dependencies.** Do not add `npm install`, `package.json`, CDN
  scripts, or any fetch of external resources. Everything must work from a static
  file server or straight from disk.
- **`'use strict';` is the first line** of every JS file. Keep it.
- **Keep it small.** The whole app is ~900 lines across 4 JS files plus the HTML.
  Favor small, readable changes over abstraction.

## File map & responsibilities

| File | Owns | Touch it when |
| --- | --- | --- |
| `index.html` | Layout, all CSS, UI controls, **script load order** | Changing UI, adding controls, changing styling. |
| `config.js` | The single global `CONFIG` object | Adding/tuning any simulation parameter. |
| `simulation.js` | Pure-ish sim core: world init, state machine, energy math, clouds, time stepping, history. | Changing behaviour, physics, energy numbers, or adding sim logic. |
| `render.js` | All Canvas 2D drawing (scene + 24h chart) and the shared `STATE_LABEL` / `STATE_COLOR` maps. | Changing visuals, labels, or colours. |
| `main.js` | The `requestAnimationFrame` loop, speed control, selection, tuning, HUD/panel text updates. | Changing interaction, performance cadence, or what the UI shows. |

### Load order matters

`index.html` loads them in this order and each file reads globals from the ones
before it:

```
config.js → simulation.js → render.js → main.js
```

`config.js` defines `CONFIG`. `simulation.js` uses `CONFIG` (via `createSim`)
and defines `createSim`, `stepSim`, `advance`, `sunFactor`, `shadeAt`, `solarInW`,
`gridNowW`, `fmtWh`, `hourOfDay`, `dayOf`, `mulberry32`, `NAMES`, `DAY`.
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
- **Determinism:** world init is seeded with `mulberry32(1337)` so the starting
  layout/clouds are reproducible. `Math.random()` is still used at runtime for
  wander targets, particle spawns, and cloud respawn — that's fine, but don't
  silently make init non-deterministic or vice-versa.

## Units & invariants (important — keep these straight)

- **Time** is in **seconds**. `sim.t` is an absolute seconds-from-start value;
  `DAY = 86400`. Day of day is `t % DAY`. `hourOfDay(t)` and `dayOf(t)` exist —
  use them rather than re-deriving.
- **Power** is in **watts (W)**; **energy** is in **watt-hours (Wh)**; convert
  with `dtH = dt / 3600`. `fmtWh()` pretty-prints Wh/kWh.
- **SoC** (`s.soc`) is a **fraction 0–1**, not a percentage. Multiply by
  `battery.capacityWh` for Wh. The usable range is `reserveSoc`…`1`.
- **Fixed substep:** `advance()` steps the sim in fixed `cfg.substep` (0.5 s)
  increments to cover a real-frame delta. Don't pass variable `dt` into
  `stepSim` from the UI; go through `advance`.
- **World space** is `1600 × 1000` units; the canvas scales/letterboxes it to
  fit (`S = min(w/worldW, h/worldH)`). Coordinates in sim/render are world
  units, not pixels. Mouse→world conversion lives in `toWorld()` in `main.js`.
- **Chart** is a fixed-size 320×130 canvas; `sim.history` holds per-bucket
  `{b, accum, dur}` grid-energy records.

## Sheep state machine (refer to `simulation.js:stepSim`)

```
resting → toField → harvesting → toWarehouse → delivering
  ↑_______________________________|  (when empty, and it's daytime → toField)
```

- `resting` → `toField` when `(t % DAY) >= sunrise` and `< recallTime`.
- `toField` → `harvesting` on arrival (picks a wander target); if the sun is
  down it bails straight to `toWarehouse`.
- `harvesting` → `toWarehouse` when SoC ≥ `fullThreshold` or `(t % DAY) ≥ recallTime`.
- `delivering` → back to `resting` (after recall) or `toField` (if daylight)
  once near the reserve floor.

When adding a state, update: the `switch` in `stepSim`, `STATE_LABEL`,
`STATE_COLOR`, and any place that special-cases state (e.g. the "on site" count
in `main.js`, `solarInW`'s daylight check, motor-draw check in the details panel).

## Energy bookkeeping

`sim.totals = { harvest, delivered, lost }` (Wh) feed the header HUD and the
end-to-end efficiency. Keep them consistent:

- `harvest` += actual charged Wh (after charge efficiency, capped by headroom).
- `delivered` += Wh actually discharged to the grid.
- `lost` += motor draw + clamped/clipped solar that couldn't be stored.

If you change the charge/discharge math in `stepSim`, re-check these three
accumulators still add up, and that `gridNowW()` (used by the warehouse panel
and chart) still matches what's really being delivered.

## How to run & verify

Serve statically and view in a browser:

```sh
python -m http.server 8765   # → http://localhost:8765
```

**Headless smoke test:** `node .tmp-frames.js` runs the real `config.js`,
`simulation.js`, `render.js`, `main.js` inside a `vm` sandbox with a mocked
DOM/canvas, drives 3000 frames, and **exits non-zero if the sim freezes or the
rAF loop dies**. It prints the resulting clock, per-sheep states and SoC. Run it
after changing `simulation.js` or the frame loop:

```sh
node .tmp-frames.js
# expect: "FRAME LOOP OK"
```

The mock is deliberately crude (proxy-based 2D context, stub elements) — it
checks the sim advances and doesn't throw, not pixel output.

## Gotchas / things to know before editing

- **`.tmp-frames.js` and `.tmp-shot.html`** are development/verification
  artifacts, not part of the app. They are safe to run (`node .tmp-frames.js`)
  but are not loaded by the app. Don't ship logic that depends on them.
- **`shot-day.png`** is a reference screenshot (also linked from the README).
- **`nul`** is a stray Windows shell-redirect artifact. It is not a source file;
  it can be safely ignored or deleted (never `echo ... > nul`-style output).
- **`server.log`** is just `python -m http.server` access log output — ignore.
- **`render.js` defines `STATE_LABEL`/`STATE_COLOR`** that `main.js` reads. That
  is load-order dependent; if you move state metadata, make sure it's defined
  before `main.js` runs.
- **The scene canvas uses `ctx.roundRect`** — requires a reasonably modern
  browser. Don't regress to a context that lacks it without a fallback.
- **Don't break the "works from disk" property.** Everything is relative to
  `index.html`; no absolute paths, no server-only features.

## Where to make a given change

- A new tunable number → add to `CONFIG` in `config.js`, read it in
  `simulation.js`/`render.js`, and (if it's a user-facing slider) add the input
  to `index.html` + wire it in `main.js`.
- New behaviour/physics → `simulation.js` (`stepSim` and helpers).
- New visual → `render.js` (`makeDraw` helpers).
- New UI control or readout → `index.html` + `main.js`.
- Keep `CONFIG` as the single source of truth; don't hardcode magic numbers
  in the sim when a config key fits.