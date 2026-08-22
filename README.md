# Solar Sheep — autopilot panels

A browser simulation of a herd of solar-panel sheep. Each sheep carries a small PV
panel on its back and a battery. During the day the herd leaves the warehouse,
wanders the pasture sunning its panels, and when a battery fills up (or the sun
drops) it comes home to deliver its stored energy to the grid.

Everything runs client-side in vanilla JavaScript + Canvas 2D. No build step,
no dependencies.

![daytime screenshot](shot-day.png)

## Run it

There is nothing to install. Serve the folder with any static file server and
open it in a browser:

```sh
# from this directory
python -m http.server 8765
# then open http://localhost:8765
```

`index.html` can also be opened directly from disk — all scripts are plain
`<script>` tags with no modules or bundling.

## Controls

| Control | What it does |
| --- | --- |
| `pause` / `1x` / `10x` / `30x` / `100x` | Simulation speed (default `10x`). |
| `+1h` | Jump the clock ahead one simulated hour instantly. |
| `space` | Toggle pause / resume. |
| Click a **sheep** | Show its live details (PV now, battery, grid output, motor draw, today's harvest/delivery). |
| Click the **warehouse** | Show site details (grid intake now, sheep on site, energy in herd, totals, end-to-end efficiency). |
| **Tuning** sliders | Set herd size (1–12), battery capacity (100–2000 Wh), panel rating (50–600 W). |
| **Apply and restart sim** | Rebuild the simulation with the new settings (resets the clock and totals). |

## What the simulation models

- **Day / night** — Sunrise `06:00`, sunset `18:00`, recall to warehouse `17:20`.
  Solar output follows a sine curve peaking at midday. The sim starts on
  Day 1 at `06:30`.
- **Clouds** — A handful of drifting clouds pass over the field; any sheep under
  one gets its panel shaded to 45% of the clear-sky value.
- **The sheep state machine**

  ```
  resting → toField → harvesting → toWarehouse → delivering → (resting or toField)
  ```

  - `resting` — back at the warehouse dock at night.
  - `toField` — driving to its assigned pasture slot.
  - `harvesting` — wandering the pasture, generating power into its battery.
  - `toWarehouse` — driving home (battery full, or recall time reached).
  - `delivering` — at the dock, discharging into the grid until near-empty.

- **Energy model** — Panels produce `panel.peakW` (200 W default) scaled by sun
  and shade. Energy flows into a `battery.capacityWh` (500 Wh default) pack with
  95% charge and 95% discharge efficiency and a 5% reserve SoC floor. Driving
  draws a 30 W motor; delivery pushes up to 60 W into the grid per sheep. A
  battery counts as "full" at 95% SoC.

## Metrics

- **Harvested** — Wh captured from the sun into batteries.
- **Delivered** — Wh actually pushed to the grid.
- **Eff** — end-to-end efficiency, delivered / harvested.
- **Grid power (last 24h)** — a live chart of combined grid output, with the
  night hours shaded and the current instantaneous reading.

## Architecture

| File | Responsibility |
| --- | --- |
| `index.html` | Layout, styling, UI controls, and script load order. |
| `config.js` | `CONFIG` object — the single source of all tunable numbers. |
| `simulation.js` | Core simulation: state machine, energy model, cloud motion, time stepping. |
| `render.js` | All Canvas drawing (field, warehouse, sheep, clouds, sun, particles) and the 24h chart. |
| `main.js` | UI wiring: rAF loop, speed control, selection, tuning, HUD/panel updates. |

### `CONFIG` reference

| Key | Default | Meaning |
| --- | --- | --- |
| `world.w` / `world.h` | `1600` / `1000` | World dimensions the camera scales to fit. |
| `warehouse` | `{x:70,y:380,w:220,h:240}` | Warehouse rectangle (also holds the dock). |
| `startDay` / `startHour` | `1` / `6.5` | Simulation start time. |
| `substep` | `0.5` | Fixed physics substep in seconds. |
| `sunrise` / `sunset` | `21600` / `64800` | Seconds-of-day for sunrise/sunset. |
| `recallTime` | `62400` | Seconds-of-day the herd is called home (17:20). |
| `sun.cloudShade` | `0.45` | Multiplier on panel output when a sheep is in cloud shadow. |
| `clouds` | `{count:8, speed:6, minR:60, maxR:130}` | Cloud population, drift speed, radius range. |
| `sheep.count` | `5` | Herd size. |
| `panel.peakW` | `200` | Panel peak output in watts. |
| `battery.capacityWh` | `500` | Battery capacity in Wh. |
| `battery.chargeEff` / `dischargeEff` | `0.95` / `0.95` | Charge / discharge efficiency. |
| `battery.reserveSoc` | `0.05` | SoC floor (fraction) the battery will not go below. |
| `motor.powerW` | `30` | Continuous motor draw in watts. |
| `motor.travelSpeed` / `harvestSpeed` | `45` / `12` | Drive speed, and slower wander speed while harvesting (world units/s). |
| `grid.dischargeW` | `60` | Max delivery rate to the grid per sheep. |
| `fullThreshold` | `0.95` | SoC at which a battery is considered full. |
| `chart.bucketSec` / `keepSec` | `60` / `86400` | Grid-power chart resolution and 24h window. |