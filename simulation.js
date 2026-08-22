'use strict';

function mulberry32(a) {
  return function () {
    a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

const NAMES = ['Lana', 'Woolfy', 'Pip', 'Nimbus', 'Clover', 'Biscuit', 'Moss', 'Pudding', 'Tofu', 'Marigold', 'Echo', 'Roly'];
const DAY = 86400;

const WEATHER_LABEL = { clear: 'clear skies', cloudy: 'cloudy', overcast: 'overcast' };
const WEATHER_COLOR = { clear: '#ffd166', cloudy: '#9fb0c3', overcast: '#7d8aa0' };

function pickWeather(sim, rand) {
  const st = sim.cfg.weather.states;
  const r = (rand || Math.random)();
  let acc = 0;
  for (const name of Object.keys(st)) {
    acc += st[name].w;
    if (r <= acc) return name;
  }
  return 'clear';
}

function sunPos(t, cfg) {
  const d = t % DAY;
  const p = Math.max(0, Math.min(1, (d - cfg.sunrise) / (cfg.sunset - cfg.sunrise)));
  return { x: 1100 + p * 380, y: 92 - Math.sin(Math.PI * p) * 48 };
}

function createSim(cfg) {
  const wh = cfg.warehouse;
  const plant = cfg.plant;
  const dockX = wh.x + wh.w + 52;
  const n = cfg.sheep.count;

  // one dock pad at the warehouse's right wall (queue forms along it, heading
  // north); the conveyor runs from the dock to the plant's intake port
  const dockPos = { x: wh.x + wh.w + 20, y: wh.y + wh.h / 2 };
  const queuePt = i => ({ x: dockPos.x + 8, y: dockPos.y - i * 46 });
  const lineX0 = dockPos.x - 6, lineY0 = dockPos.y;
  const lineX1 = plant.x + plant.w + 14, lineY1 = plant.y + 62;

  const slots = [];
  const colW = (cfg.world.w - dockX - 80) / 3;
  for (let i = 0; i < n; i++) {
    const col = i % 3, row = Math.floor(i / 3);
    slots.push({
      x: dockX + 140 + col * (colW - 70) + (row % 2 ? 80 : 0),
      y: 150 + row * 220 + (col % 2 ? 60 : 0),
    });
  }

  const sim = {
    cfg, t: (cfg.startDay - 1) * DAY + cfg.startHour * 3600,
    sheep: [], clouds: [], parts: [],
    pool: [], units: [], line: [], ready: [],
    dock: { queue: [], current: null, timer: 0 },
    dockPos, queuePt, lineX0, lineY0, lineX1, lineY1,
    history: [],
    totalGridAccum: 0, totalGridDur: 0,
    totals: { harvest: 0, delivered: 0, lost: 0, produced: 0 },
    daily: { produced: 0, harvest: 0, delivered: 0, lost: 0 },
    whDelivered: 0, plantDelivered: 0,
    weather: { state: 'clear', left: 0 },
    selected: null,
  };
  const rng = mulberry32(1337);

  // seed weather (seeded rng keeps init deterministic)
  sim.weather.state = pickWeather(sim, rng);
  sim.weather.left = (cfg.weather.minDurH + rng() * (cfg.weather.maxDurH - cfg.weather.minDurH)) * 3600;

  for (let i = 0; i < n; i++) {
    const homeY = wh.y + 34 + (n === 1 ? (wh.h - 50) / 2 : i * (wh.h - 50) / (n - 1));
    sim.sheep.push({
      id: i, name: NAMES[i % NAMES.length],
      x: dockX, y: homeY,
      home: { x: dockX, y: homeY },
      slot: slots[i], state: 'resting',
      pack: null,   // a member of sim.pool, state 'carry'
      wander: null,
      wake: rng() * cfg.sheep.wakeSpreadSec,
      recall: cfg.recallTime + (rng() * 2 - 1) * cfg.sheep.recallJitterSec,
      skill: cfg.sheep.trackSkillMin + rng() * (cfg.sheep.trackSkillMax - cfg.sheep.trackSkillMin),
      orient: 0,
      exposure: 1,
      produceToday: 0,
      harvestToday: 0, deliverToday: 0,
      lastDay: 1,
    });
  }

  // battery pool: the whole fleet of packs; the first n start on a sheep (as the
  // unproduced initial reserve), the rest start empty and wait on the ready rack
  for (let i = 0; i < cfg.dock.pool; i++) {
    sim.pool.push({ id: i, soc: 0, state: 'ready' });
  }
  for (let u = 0; u < cfg.dock.units; u++) sim.units.push({ p: null });
  for (let i = 0; i < n; i++) {
    const p = sim.pool[i];
    p.soc = cfg.battery.reserveSoc;
    p.state = 'carry';
    sim.sheep[i].pack = p;
  }
  // the remaining packs wait on the ready rack
  for (let i = n; i < cfg.dock.pool; i++) sim.ready.push(sim.pool[i]);

  for (let i = 0; i < cfg.clouds.count; i++) {
    sim.clouds.push({
      x: rng() * (cfg.world.w + 600) - 300,
      y: rng() * cfg.world.h,
      r: cfg.clouds.minR + rng() * (cfg.clouds.maxR - cfg.clouds.minR),
      vx: cfg.clouds.speed * (0.5 + rng()),
    });
  }
  return sim;
}

function hourOfDay(t) { return (t % DAY) / 3600; }
function dayOf(t) { return Math.floor(t / DAY) + 1; }

function sunFactor(t, cfg) {
  const d = t % DAY;
  if (d <= cfg.sunrise || d >= cfg.sunset) return 0;
  return Math.sin(Math.PI * (d - cfg.sunrise) / (cfg.sunset - cfg.sunrise));
}

function weatherF(sim) {
  return sim.cfg.weather.states[sim.weather.state].f;
}

function shadeAt(sim, x, y) {
  for (const cl of sim.clouds) {
    const dx = x - cl.x, dy = y - cl.y;
    if (dx * dx + dy * dy < cl.r * cl.r) return sim.cfg.sun.cloudShade;
  }
  return 1;
}

function solarInW(sim, s) {
  const daylight = s.state === 'harvesting' || s.state === 'toField';
  return daylight ? sim.cfg.panel.peakW * sunFactor(sim.t, sim.cfg) * shadeAt(sim, s.x, s.y) * s.exposure * weatherF(sim) : 0;
}

function whNowW(sim) {
  return sim.cfg.whPanel.peakW * sunFactor(sim.t, sim.cfg) * weatherF(sim);
}

// live discharge rate of one plant unit (0 while empty / at the reserve floor)
function unitNowW(sim, u) {
  if (!u.p) return 0;
  const cap = sim.cfg.battery.capacityWh;
  return u.p.soc * cap > cap * sim.cfg.battery.reserveSoc + 0.01
    ? sim.cfg.grid.dischargeW : 0;
}
function plantNowW(sim) {
  let w = 0;
  for (const u of sim.units) w += unitNowW(sim, u);
  return w;
}

// Wh currently stored in the whole pool (carried, on the line, in units, on racks)
function poolStoredWh(sim) {
  const cap = sim.cfg.battery.capacityWh;
  let w = 0;
  for (const p of sim.pool) w += p.soc * cap;
  return w;
}

function spawnGlow(sim, x, y) {
  sim.parts.push({ x, y, age: 0, max: 0.7 + Math.random() * 0.4 });
}
function spawnSpark(sim, x, y) {
  sim.parts.push({ x: x + (Math.random() - 0.5) * 34, y: y + (Math.random() - 0.5) * 22, age: 0, max: 0.35 + Math.random() * 0.3, spark: true });
}

function pickWander(sim, s) {
  const a = Math.random() * Math.PI * 2, r = 30 + Math.random() * 60;
  const minX = sim.cfg.warehouse.x + sim.cfg.warehouse.w + 110;
  s.wander = {
    x: Math.max(minX, Math.min(sim.cfg.world.w - 80, s.slot.x + Math.cos(a) * r)),
    y: Math.max(90, Math.min(sim.cfg.world.h - 90, s.slot.y + Math.sin(a) * r)),
  };
}
function clampWorld(sim, s) {
  const m = 60;
  s.x = Math.max(m, Math.min(sim.cfg.world.w - m, s.x));
  s.y = Math.max(m, Math.min(sim.cfg.world.h - m, s.y));
}

function driveSim(sim, s, pt, speed, dt) {
  const dx = pt.x - s.x, dy = pt.y - s.y;
  const d = Math.hypot(dx, dy);
  if (d <= 0.5) return true;
  const stepLen = Math.min(d, speed * dt);
  s.x += dx / d * stepLen;
  s.y += dy / d * stepLen;
  const cap = sim.cfg.battery.capacityWh;
  const dtH = dt / 3600;
  const availWh = s.pack.soc * cap - cap * sim.cfg.battery.reserveSoc;
  const useW = Math.min(sim.cfg.motor.powerW, availWh / sim.cfg.battery.dischargeEff / dtH);
  if (useW > 0) {
    // motor burn is booked exactly as the Wh it pulls out of the pack
    const burnWh = useW * dtH / sim.cfg.battery.dischargeEff;
    s.pack.soc = Math.max(sim.cfg.battery.reserveSoc, s.pack.soc - burnWh / cap);
    sim.totals.lost += burnWh;
    sim.daily.lost += burnWh;
  }
  return d <= stepLen + 0.01;
}

function stepPlant(sim, dt) {
  const cfg = sim.cfg;
  const dtH = dt / 3600;
  const cap = cfg.battery.capacityWh;
  const reserveSoc = cfg.battery.reserveSoc;
  let gridNow = 0;

  // conveyor: packs ride in order, oldest first; on arrival join the ready queue
  for (const m of sim.line) m.t += dt;
  const done = sim.line.filter(m => m.t >= cfg.dock.lineSec);
  if (done.length) {
    sim.line = sim.line.filter(m => m.t < cfg.dock.lineSec);
    for (const m of done) {
      m.state = 'ready';
      sim.ready.push(m);
    }
  }

  // assign charged packs from the ready rack to free discharge units (FIFO).
  // Empty packs stay on the rack, waiting for a sheep to carry them out.
  for (const u of sim.units) {
    if (u.p) continue;
    const idx = sim.ready.findIndex(p => p.soc > reserveSoc + 1e-9);
    if (idx === -1) break;
    const p = sim.ready.splice(idx, 1)[0];
    p.state = 'discharging';
    u.p = p;
  }

  // discharge into the grid; a pack that hits the reserve floor goes back to ready
  for (const u of sim.units) {
    const p = u.p;
    if (!p) continue;
    const availWh = p.soc * cap - cap * reserveSoc;
    if (availWh > 0) {
      const outWh = Math.min(cfg.grid.dischargeW * dtH, availWh / cfg.battery.dischargeEff);
      const lossWh = outWh * (1 / cfg.battery.dischargeEff - 1);
      p.soc = (p.soc * cap - outWh / cfg.battery.dischargeEff) / cap;
      sim.totals.delivered += outWh;
      sim.totals.lost += lossWh;
      sim.daily.delivered += outWh;
      sim.daily.lost += lossWh;
      sim.plantDelivered += outWh;
      gridNow += outWh / dtH;
    }
    if (p.soc <= reserveSoc + 1e-9 && p.state === 'discharging') {
      p.state = 'ready';
      u.p = null;
      sim.ready.push(p);
    }
  }

  return gridNow;
}

function stepSim(sim, dt) {
  const cfg = sim.cfg;
  sim.t += dt;
  const t = sim.t;
  const d = t % DAY;
  const sun = sunFactor(t, cfg);
  const wf = weatherF(sim);
  const cap = cfg.battery.capacityWh;
  const reserveSoc = cfg.battery.reserveSoc;
  const dtH = dt / 3600;
  const dtMin = dt / 60;
  let gridNow = 0;

  const sp = sunPos(t, cfg);

  // ---- weather state machine (random weather behaviour)
  sim.weather.left -= dt;
  if (sim.weather.left <= 0) {
    sim.weather.state = pickWeather(sim);
    sim.weather.left = (cfg.weather.minDurH + Math.random() * (cfg.weather.maxDurH - cfg.weather.minDurH)) * 3600;
  }

  // ---- clouds: drift + population tracks the weather's target count
  for (const cl of sim.clouds) {
    cl.x += cl.vx * dt;
    if (cl.x - cl.r > cfg.world.w + 200) {
      cl.x = -cl.r - 100;
      cl.y = Math.random() * cfg.world.h;
    }
  }
  const targetClouds = cfg.weather.states[sim.weather.state].clouds;
  const newCloud = () => ({
    x: -100 - Math.random() * 200,
    y: Math.random() * cfg.world.h,
    r: cfg.clouds.minR + Math.random() * (cfg.clouds.maxR - cfg.clouds.minR),
    vx: cfg.clouds.speed * (0.5 + Math.random()),
  });
  while (sim.clouds.length < targetClouds) sim.clouds.push(newCloud());
  if (sim.clouds.length > targetClouds) sim.clouds.pop();

  for (const s of sim.sheep) {
    const newDay = dayOf(t);
    if (newDay !== s.lastDay) {
      s.lastDay = newDay;
      s.produceToday = 0; s.harvestToday = 0; s.deliverToday = 0;
      sim.daily = { produced: 0, harvest: 0, delivered: 0, lost: 0 };
    }

    // ---- orientation: park facing the sun (turn-rate limited), laggy by skill
    let target;
    if (sun > 0 && (s.state === 'harvesting' || s.state === 'toField')) {
      target = Math.atan2(sp.y - s.y, sp.x - s.x) + (1 - s.skill) * 0.6;
    } else {
      target = Math.atan2(s.home.y - s.y, s.home.x - s.x);
    }
    let da = target - s.orient;
    while (da > Math.PI) da -= 2 * Math.PI;
    while (da < -Math.PI) da += 2 * Math.PI;
    const step = Math.max(-cfg.sheep.turnRate * dt, Math.min(cfg.sheep.turnRate * dt, da));
    s.orient += step;
    const rem = da - step; // residual angle error after this step's turn

    s.exposure = (sun > 0 && (s.state === 'harvesting' || s.state === 'toField'))
      ? Math.max(0, Math.cos(rem)) * s.skill : 0;

    switch (s.state) {
      case 'resting': {
        if (d >= cfg.sunrise + s.wake && d < s.recall) s.state = 'toField';
        break;
      }

      case 'toField': {
        const arrived = driveSim(sim, s, s.slot, cfg.motor.travelSpeed, dt);
        if (sun === 0) s.state = 'toDock';
        else if (arrived) { s.state = 'harvesting'; pickWander(sim, s); }
        break;
      }

      case 'harvesting': {
        const shade = shadeAt(sim, s.x, s.y);
        const gainW = cfg.panel.peakW * sun * shade * s.exposure * wf;
        const availWh = cap - s.pack.soc * cap;
        const chargeInW = (gainW - cfg.motor.powerW) * cfg.battery.chargeEff;
        const chargedW = Math.min(Math.max(0, chargeInW), availWh / dtH);
        // everything produced but not stored (motor + charge loss + clipping)
        const lossWh = (gainW * dtH) - (chargedW * dtH);

        sim.totals.produced += gainW * dtH;
        sim.daily.produced += gainW * dtH;
        s.produceToday += gainW * dtH;

        if (s.wander) {
          const dx = s.wander.x - s.x, dy = s.wander.y - s.y;
          const dlen = Math.hypot(dx, dy);
          if (dlen < 12) { pickWander(sim, s); }
          else {
            const stepLen = Math.min(dlen, cfg.motor.harvestSpeed * dt);
            s.x += dx / dlen * stepLen;
            s.y += dy / dlen * stepLen;
          }
        }
        clampWorld(sim, s);

        if (chargedW > 0) {
          s.pack.soc += (chargedW * dtH) / cap;
          sim.totals.harvest += chargedW * dtH;
          sim.daily.harvest += chargedW * dtH;
          s.harvestToday += chargedW * dtH;
          if (Math.random() < dtMin * 0.12) spawnSpark(sim, s.x, s.y - 8);
        }
        if (lossWh > 0) { sim.totals.lost += lossWh; sim.daily.lost += lossWh; }

        if (s.pack.soc >= cfg.fullThreshold || d >= s.recall) s.state = 'toDock';
        break;
      }

      case 'toDock': {
        const arrived = driveSim(sim, s, sim.dockPos, cfg.motor.travelSpeed, dt);
        if (arrived) { s.state = 'queuing'; s.x = sim.dockPos.x; s.y = sim.dockPos.y; }
        break;
      }

      case 'queuing': {
        const i = sim.dock.queue.indexOf(s);
        if (i === -1) { sim.dock.queue.push(s); }
        else if (i === 0 && !sim.dock.current && sim.ready.some(p => p.soc <= reserveSoc + 1e-9)) {
          // only start a swap when an empty pack is waiting — otherwise hold the berth
          s.x = sim.dockPos.x; s.y = sim.dockPos.y;
          s.state = 'swapping';
          sim.dock.current = s;
          sim.dock.timer = 0;
          if (Math.random() < dtMin * 1.5) spawnSpark(sim, s.x, s.y);
        } else if (i > 0) {
          const q = sim.queuePt(i);
          s.x += (q.x - s.x) * Math.min(1, 6 * dt);
          s.y += (q.y - s.y) * Math.min(1, 6 * dt);
        }
        break;
      }

      case 'swapping': {
        sim.dock.timer += dt;
        if (Math.random() < dtMin * 0.4) spawnSpark(sim, s.x, s.y);
        if (sim.dock.timer >= cfg.dock.swapSec) {
          sim.dock.current = null;
          sim.dock.queue.shift();
          // the old pack rides the conveyor into the plant…
          const oldP = s.pack;
          oldP.state = 'line';
          oldP.t = 0;
          sim.line.push(oldP);
          // …and the sheep takes an empty one back out with it
          const idx = sim.ready.findIndex(p => p.soc <= reserveSoc + 1e-9);
          if (idx !== -1) {
            const fresh = sim.ready.splice(idx, 1)[0];
            fresh.state = 'carry';
            s.pack = fresh;
          } else {
            // no empty left (shouldn't happen - checked when the swap started)
            sim.line.pop();
            oldP.state = 'carry';
          }
          s.state = (sun > 0 && d < s.recall) ? 'toField' : 'resting';
          if (s.state === 'toField') pickWander(sim, s);
        }
        break;
      }
    }
  }

  // ---- battery plant: conveyor + discharge units (runs day and night)
  const plantW = stepPlant(sim, dt);
  gridNow += plantW;

  // ---- warehouse roof PV: fixed panel, straight to the grid
  const roofW = whNowW(sim);
  sim.totals.produced += roofW * dtH;
  sim.daily.produced += roofW * dtH;
  const roofOutWh = roofW * dtH;
  if (roofOutWh > 0) {
    sim.totals.delivered += roofOutWh;
    sim.daily.delivered += roofOutWh;
    sim.whDelivered += roofOutWh;
  }
  gridNow += roofW;
  sim.totalGridAccum += gridNow * dt;
  sim.totalGridDur += dt;

  const bucket = Math.floor(t / cfg.chart.bucketSec);
  if (sim.history.length === 0 || sim.history[sim.history.length - 1].b !== bucket) {
    sim.history.push({ b: bucket, sAcc: 0, rAcc: 0, dur: dt });
  }
  const cur = sim.history[sim.history.length - 1];
  cur.sAcc += (gridNow - roofW) * dt;
  cur.rAcc += roofW * dt;
  cur.dur += dt;
  const cutoffB = Math.floor((t - cfg.chart.keepSec) / cfg.chart.bucketSec);
  while (sim.history.length > 2 && sim.history[1].b < cutoffB) sim.history.shift();

  sim.parts = sim.parts.filter(p => (p.age += dt) < p.max);
}

function advance(sim, realSec, speed) {
  // carry the fractional remainder so total advance is exactly speed * realSec,
  // quantised to whole substeps (no per-frame overshoot)
  const sub = sim.cfg.substep;
  sim.acc = (sim.acc || 0) + realSec * speed;
  let guard = 0;
  while (sim.acc >= sub - 1e-9 && guard < 40000) {
    stepSim(sim, sub);
    sim.acc -= sub;
    guard++;
  }
}

function sheepNowW(sim) {
  // kept as a no-op alias — the herd no longer discharges directly
  return 0;
}

// grid output now = plant discharge units + roof PV
function gridNowW(sim) {
  return plantNowW(sim) + whNowW(sim);
}

function fmtWh(w) {
  if (w < 1000) return Math.round(w) + ' Wh';
  return (w / 1000).toFixed(2) + ' kWh';
}