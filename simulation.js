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
  const dockX = wh.x + wh.w + 52;
  const n = cfg.sheep.count;

  const slots = [];
  const colW = (cfg.world.w - dockX - 80) / 3;
  let maxRow = 0;
  for (let i = 0; i < n; i++) {
    const col = i % 3, row = Math.floor(i / 3);
    maxRow = Math.max(maxRow, row);
    slots.push({
      x: dockX + 140 + col * (colW - 70) + (row % 2 ? 80 : 0),
      y: 150 + row * 220 + (col % 2 ? 60 : 0),
    });
  }

  const sim = {
    cfg, t: (cfg.startDay - 1) * DAY + cfg.startHour * 3600,
    sheep: [], clouds: [], parts: [],
    history: [],
    totalGridAccum: 0, totalGridDur: 0,
    totals: { harvest: 0, delivered: 0, lost: 0, produced: 0 },
    daily: { produced: 0, harvest: 0, delivered: 0, lost: 0 },
    whDelivered: 0,
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
      soc: cfg.battery.reserveSoc,
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
  const availWh = s.soc * cap - cap * sim.cfg.battery.reserveSoc;
  const useW = Math.min(sim.cfg.motor.powerW, availWh / sim.cfg.battery.dischargeEff / dtH);
  if (useW > 0) {
    s.soc = Math.max(sim.cfg.battery.reserveSoc, s.soc - (useW * dtH) / (sim.cfg.battery.dischargeEff * cap));
    sim.totals.lost += useW * dtH;
    sim.daily.lost += useW * dtH;
  }
  return d <= stepLen + 0.01;
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
        if (sun === 0) s.state = 'toWarehouse';
        else if (arrived) { s.state = 'harvesting'; pickWander(sim, s); }
        break;
      }

      case 'harvesting': {
        const shade = shadeAt(sim, s.x, s.y);
        const gainW = cfg.panel.peakW * sun * shade * s.exposure * wf;
        const availWh = cap - s.soc * cap;
        const chargeInW = (gainW - cfg.motor.powerW) * cfg.battery.chargeEff;
        const chargedW = Math.min(Math.max(0, chargeInW), availWh / dtH);
        // chargedWh + losses must exactly equal gainWh * dtH
        const lossWh = (gainW - chargedW / cfg.battery.chargeEff) * dtH;

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
          s.soc += (chargedW * dtH) / cap;
          sim.totals.harvest += chargedW * dtH;
          sim.daily.harvest += chargedW * dtH;
          s.harvestToday += chargedW * dtH;
          if (Math.random() < dtMin * 0.12) spawnSpark(sim, s.x, s.y - 8);
        }
        if (lossWh > 0) { sim.totals.lost += lossWh; sim.daily.lost += lossWh; }

        if (s.soc >= cfg.fullThreshold || d >= s.recall) s.state = 'toWarehouse';
        break;
      }

      case 'toWarehouse': {
        const arrived = driveSim(sim, s, s.home, cfg.motor.travelSpeed, dt);
        if (arrived) { s.state = 'delivering'; s.x = s.home.x; s.y = s.home.y; }
        break;
      }

      case 'delivering': {
        const availWh = s.soc * cap - cap * reserveSoc;
        if (availWh <= 0.01) {
          s.state = (d >= s.recall) ? 'resting' : 'toField';
          break;
        }
        const useW = Math.min(cfg.grid.dischargeW, availWh / dtH);
        const outWh = useW * dtH;
        const lossWh = outWh * (1 / cfg.battery.dischargeEff - 1);
        gridNow += useW;
        s.soc = (s.soc * cap - outWh / cfg.battery.dischargeEff) / cap;
        sim.totals.delivered += outWh;
        sim.totals.lost += lossWh;
        sim.daily.delivered += outWh;
        sim.daily.lost += lossWh;
        s.deliverToday += outWh;
        if (Math.random() < dtMin * 1.2) spawnGlow(sim, s.x, s.y);
        break;
      }
    }
  }

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
  let w = 0;
  const cap = sim.cfg.battery.capacityWh;
  for (const s of sim.sheep) {
    if (s.state === 'delivering') {
      const availWh = s.soc * cap - cap * sim.cfg.battery.reserveSoc;
      if (availWh > 0) w += sim.cfg.grid.dischargeW;
    }
  }
  return w;
}

// legacy alias (excludes roof PV)
function gridNowW(sim) {
  return sheepNowW(sim);
}

function fmtWh(w) {
  if (w < 1000) return Math.round(w) + ' Wh';
  return (w / 1000).toFixed(2) + ' kWh';
}