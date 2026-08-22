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
    cfg, t: cfg.startDay * DAY + cfg.startHour * 3600,
    sheep: [], clouds: [], parts: [],
    history: [],
    totalGridAccum: 0, totalGridDur: 0,
    totals: { harvest: 0, delivered: 0, lost: 0 },
    selected: null,
  };
  const rng = mulberry32(1337);

  for (let i = 0; i < n; i++) {
    const homeY = wh.y + 34 + (n === 1 ? (wh.h - 50) / 2 : i * (wh.h - 50) / (n - 1));
    sim.sheep.push({
      id: i, name: NAMES[i % NAMES.length],
      x: dockX, y: homeY,
      home: { x: dockX, y: homeY },
      slot: slots[i], state: 'resting',
      soc: cfg.battery.reserveSoc,
      wander: null,
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

function shadeAt(sim, x, y) {
  for (const cl of sim.clouds) {
    const dx = x - cl.x, dy = y - cl.y;
    if (dx * dx + dy * dy < cl.r * cl.r) return sim.cfg.sun.cloudShade;
  }
  return 1;
}

function solarInW(sim, s) {
  const daylight = s.state === 'harvesting' || s.state === 'toField';
  return daylight ? sim.cfg.panel.peakW * sunFactor(sim.t, sim.cfg) * shadeAt(sim, s.x, s.y) : 0;
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
  }
  return d <= stepLen + 0.01;
}

function stepSim(sim, dt) {
  const cfg = sim.cfg;
  sim.t += dt;
  const t = sim.t;
  const sun = sunFactor(t, cfg);
  const cap = cfg.battery.capacityWh;
  const reserveSoc = cfg.battery.reserveSoc;
  let gridNow = 0;

  for (const cl of sim.clouds) {
    cl.x += cl.vx * dt;
    if (cl.x - cl.r > cfg.world.w + 200) {
      cl.x = -cl.r - 100;
      cl.y = Math.random() * cfg.world.h;
    }
  }

  for (const s of sim.sheep) {
    const newDay = dayOf(t);
    if (newDay !== s.lastDay) {
      s.lastDay = newDay;
      s.harvestToday = 0; s.deliverToday = 0;
    }
    switch (s.state) {
      case 'resting': {
        if ((t % DAY) >= cfg.sunrise && (t % DAY) < cfg.recallTime) s.state = 'toField';
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
        const gainW = cfg.panel.peakW * sun * shade;
        const dtH = dt / 3600;
        const availWh = cap - s.soc * cap;
        const chargeInW = (gainW - cfg.motor.powerW) * cfg.battery.chargeEff;
        const chargedW = Math.min(Math.max(0, chargeInW), availWh / dtH);
        const wastedW = Math.max(0, gainW - cfg.motor.powerW) - chargedW / cfg.battery.chargeEff;

        if (s.wander) {
          const dx = s.wander.x - s.x, dy = s.wander.y - s.y;
          const d = Math.hypot(dx, dy);
          if (d < 12) { pickWander(sim, s); }
          else {
            const stepLen = Math.min(d, cfg.motor.harvestSpeed * dt);
            s.x += dx / d * stepLen;
            s.y += dy / d * stepLen;
          }
        }
        clampWorld(sim, s);

        if (chargedW > 0) {
          s.soc += (chargedW * dtH) / cap;
          sim.totals.harvest += chargedW * dtH;
          s.harvestToday += chargedW * dtH;
          if (Math.random() < dt * 0.002) spawnSpark(sim, s.x, s.y - 8);
        }
        if (wastedW > 0) sim.totals.lost += wastedW * dtH;

        if (s.soc >= cfg.fullThreshold || (t % DAY) >= cfg.recallTime) s.state = 'toWarehouse';
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
          s.state = ((t % DAY) >= cfg.recallTime) ? 'resting' : 'toField';
          break;
        }
        const dtH = dt / 3600;
        const useW = Math.min(cfg.grid.dischargeW, availWh / dtH);
        gridNow += useW;
        s.soc = (s.soc * cap - useW * dtH / cfg.battery.dischargeEff) / cap;
        sim.totals.delivered += useW * dtH;
        s.deliverToday += useW * dtH;
        if (Math.random() < dt * 0.02) spawnGlow(sim, s.x, s.y);
        break;
      }
    }
  }

  sim.totalGridAccum += gridNow * dt;
  sim.totalGridDur += dt;

  const bucket = Math.floor(t / cfg.chart.bucketSec);
  if (sim.history.length === 0 || sim.history[sim.history.length - 1].b !== bucket) {
    sim.history.push({ b: bucket, accum: gridNow * dt, dur: dt });
  } else {
    const cur = sim.history[sim.history.length - 1];
    cur.accum += gridNow * dt;
    cur.dur += dt;
  }
  const cutoffB = Math.floor((t - cfg.chart.keepSec) / cfg.chart.bucketSec);
  while (sim.history.length > 2 && sim.history[1].b < cutoffB) sim.history.shift();

  sim.parts = sim.parts.filter(p => (p.age += dt) < p.max);
}

function advance(sim, realSec, speed) {
  const target = sim.t + realSec * speed;
  let guard = 0;
  while (sim.t < target - 1e-9 && guard < 40000) {
    stepSim(sim, sim.cfg.substep);
    guard++;
  }
}

function gridNowW(sim) {
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

function fmtWh(w) {
  if (w < 1000) return Math.round(w) + ' Wh';
  return (w / 1000).toFixed(2) + ' kWh';
}
