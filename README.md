# Solar Sheep — autopilot panels

A browser simulation of a herd of solar-panel sheep. Each sheep carries a small PV
panel on its back and one battery **pack** taken from a fixed pool of 50. During
the day the herd leaves the warehouse (staggered — each sheep wakes a random time
after sunrise), wanders the pasture turning to track the sun, and when its pack
fills up (or the sun drops) it drives to a **single battery-swap dock** (2 sim
minutes per swap), drops the full pack on a **conveyor** into the **battery
plant**, and takes a fresh empty one back out to keep working. The plant runs up
to **10 discharge units** at a time, draining packs overnight; the warehouse's
roof PV array feeds the grid directly and is metered separately.

Everything runs client-side in vanilla JavaScript + Canvas 2D. No build step,
no dependencies.

## Run it

There is nothing to install. Serve the folder with any static file server and
open it in a browser:

```sh
# from this directory
python -m http.server 8765
# then open http://localhost:8765
```

If the port is already in use, a server is already running — just open
http://localhost:8765. `index.html` can also be opened directly from disk; all
scripts are plain `<script>` tags with no modules or bundling.

## Controls

| Control | What it does |
| --- | --- |
| `pause` / `0.5x` / `1x` / `10x` / `30x` / `100x` | Simulation speed (default `10x`). |
| `+1h` | Jump the clock ahead one simulated hour instantly. |
| `space` | Toggle pause / resume. |
| Click a **sheep** | Live details: PV now, sun exposure, its pack, motor draw, produced/harvested/charged today. |
| Click the **battery plant** | Plant details: discharge now, units busy, packs on the line, ready rack, dock queue, energy in pool, plant vs. roof delivered. |
| Click the **warehouse** | Site details: grid intake now (split plant vs. roof PV), sheep on site, energy carried, totals, end-to-end efficiency. |
| **Tuning** sliders | Herd size (1–12), battery pack capacity (100–2000 Wh), panel rating (50–600 W), roof PV (0–1000 W). |
| **Apply and restart sim** | Rebuild the simulation with the new settings (resets the clock and totals). |

## What the simulation models

- **Day / night** — Sunrise `06:00`, sunset `18:00`, recall home ~`17:20`
  (±15 min per sheep). Solar output follows a sine curve peaking at midday.
  The sim starts on Day 1 at `06:30`.
- **Random weather** — `clear` / `cloudy` / `overcast`, each lasting 1–4
  sim hours, rolled randomly. Weather scales all solar output (×1.0 / ×0.8 /
  ×0.45) and drives the target cloud count (4 / 8 / 14), so the sky visibly
  thickens and thins.
- **Sun tracking** — While sunning, each sheep physically turns toward the sun
  (turn-rate limited, with a per-sheep 85–100% tracking skill). Panel output
  is `peakW × sun × cloudShade × exposure × weather`, where
  `exposure = max(0, cos(angleError)) × skill` — a sheep parked the wrong way
  around genuinely produces less (shown as `face N%` on canvas and in the
  details panel).
- **Clouds** — Drifting cloud blobs shade any panel beneath them to 45% of
  the clear-sky value.
- **The sheep state machine**

  ```
  resting → toField → harvesting → toDock → queuing → swapping
    ↑__________________________________________________|
  ```

  - `resting` — parked at the warehouse at night.
  - `toField` — driving to its assigned pasture slot.
  - `harvesting` — wandering the pasture, generating power into its pack.
  - `toDock` — driving to the dock (pack full, or recall time reached).
  - `queuing` — waiting its turn at the single dock (FIFO; the queue only
    moves when an empty pack is actually waiting on the ready rack).
  - `swapping` — 2 sim minutes on the dock: its full pack goes onto the
    conveyor, it takes an empty one, and heads back out (or to rest at night).

- **The battery plant** — A fixed pool of 50 packs (5 on the sheep, 45 idle)
  cycles `carry → line → discharging → ready → carry`:

  - **Conveyor**: the swapped-out pack rides ~30 s from the dock to the plant's
    intake port.
  - **Discharge units**: up to 10 packs discharge at once, 60 W each at 95%
    efficiency down to the 5% reserve floor — so the plant keeps feeding the
    grid through the evening and night.
  - **Ready rack**: holds both charged packs awaiting a free unit and empty
    packs awaiting a sheep. The whole pool state is drawn on the plant
    building and summarized in the overlay / plant panel.

- **Warehouse roof PV** — A fixed panel on the roof (`roof PV` slider, 300 W
  default) producing `peakW × sun × weather` straight to the grid, no battery.
  Its intake is metered and charted **separately** from the plant's.

- **Energy model** — Packs are `battery.capacityWh` (500 Wh default) with 95%
  charge / 95% discharge efficiency and a 5% reserve SoC floor. Driving draws a
  30 W motor (booked as Wh pulled out of the pack); a pack counts as "full" at
  95% SoC.

## Metrics

The header, the on-screen **LIVE METRICS** overlay (bottom-right of the scene),
and the warehouse/plant panels all agree, and the books balance exactly:
`produced = delivered + lost + energy still in the pool` (all 50 packs).

- **Produced** — gross PV Wh captured (sheep panels + roof).
- **Harvested** — Wh actually stored in packs (after charge efficiency).
- **Delivered** — Wh pushed to the grid (plant units + roof).
- **Lost** — motor burn, charge/discharge losses, and clipped solar that had
  nowhere to go.
- **Eff** — end-to-end efficiency, delivered / produced.
- **PV now / grid now / motor** — live aggregate power (W).
- **In pool** — Wh currently stored anywhere in the 50-pack pool.
- **Grid power (last 24h)** — live two-line chart: green = battery plant,
  amber = roof PV, with night hours shaded (the plant's line extends past
  sunset).
- **Cost estimate** — live breakdown in the Tuning panel as you slide:
  panels (£0.80/W), the 50 battery packs (£120/kWh), roof PV, motors & wheels
  (£150/sheep), inverter (£300) → total £ and £/kWh delivered.

## Architecture

| File | Responsibility |
| --- | --- |
| `index.html` | Layout, styling, UI controls, and script load order. |
| `config.js` | `CONFIG` object — the single source of all tunable numbers and prices. |
| `simulation.js` | Core simulation: state machine, battery plant (dock, conveyor, units, ready rack), weather, orientation/exposure, energy model, roof PV, time stepping, history. |
| `render.js` | All Canvas drawing: field, warehouse + dock, battery plant (conveyor, units, racks), rotating sheep, clouds, sun, particles, metrics overlay, 24h chart. |
| `main.js` | UI wiring: rAF loop, speed control, selection, tuning, cost panel, HUD/panel updates. |

### `CONFIG` reference

| Key | Default | Meaning |
| --- | --- | --- |
| `world.w` / `world.h` | `1600` / `1000` | World dimensions the camera scales to fit. |
| `warehouse` | `{x:70,y:380,w:220,h:240}` | Warehouse rectangle (the dock is on its right wall). |
| `plant` | `{x:70,y:650,w:220,h:190}` | Battery plant building rectangle. |
| `startDay` / `startHour` | `1` / `6.5` | Simulation start time. |
| `substep` | `0.5` | Fixed physics substep in seconds. |
| `sunrise` / `sunset` | `21600` / `64800` | Seconds-of-day for sunrise/sunset. |
| `recallTime` | `62400` | Base seconds-of-day the herd is called home (17:20). |
| `sun.cloudShade` | `0.45` | Multiplier on panel output under a cloud shadow. |
| `clouds` | `{count:8, speed:6, minR:60, maxR:130}` | Initial cloud population, drift speed, radius range. |
| `weather.states` | clear/cloudy/overcast | Per-state solar factor, target cloud count, pick weight. |
| `weather.minDurH` / `maxDurH` | `1` / `4` | Weather spell duration range (sim hours). |
| `sheep.count` | `5` | Herd size. |
| `sheep.wakeSpreadSec` | `2700` | Herd departure stagger window after sunrise. |
| `sheep.recallJitterSec` | `900` | Per-sheep recall time, ± this. |
| `sheep.turnRate` | `0.6` | rad/s — how fast a sheep can turn to face the sun. |
| `sheep.trackSkillMin` / `Max` | `0.85` / `1.0` | Per-sheep tracking quality (sloppy → precise). |
| `panel.peakW` | `200` | Sheep panel peak output in watts. |
| `whPanel.peakW` | `300` | Warehouse roof PV peak output in watts. |
| `battery.capacityWh` | `500` | Pack capacity in Wh. |
| `battery.chargeEff` / `dischargeEff` | `0.95` / `0.95` | Charge / discharge efficiency. |
| `battery.reserveSoc` | `0.05` | SoC floor (fraction) a pack will not go below. |
| `motor.powerW` | `30` | Continuous motor draw in watts. |
| `motor.travelSpeed` / `harvestSpeed` | `45` / `12` | Drive speed, and slower wander speed while harvesting (world units/s). |
| `grid.dischargeW` | `60` | Max grid discharge rate **per plant unit**. |
| `fullThreshold` | `0.95` | SoC at which a pack is considered full. |
| `dock.swapSec` | `120` | Battery-swap time at the single dock (sim seconds). |
| `dock.units` | `10` | Number of plant discharge units. |
| `dock.pool` | `50` | Total battery packs in the plant's pool. |
| `dock.lineSec` | `30` | Conveyor transit time, dock → plant (sim seconds). |
| `chart.bucketSec` / `keepSec` | `60` / `86400` | Grid-power chart resolution and 24h window. |
| `cost.*` | see file | £ prices for panels (per W), batteries (per kWh), drives (per sheep), inverter (fixed). |