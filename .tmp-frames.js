'use strict';
const fs = require('fs'), vm = require('vm');

function makeCtx2d() {
  const target = {
    canvas: {},
    createLinearGradient: () => ({ addColorStop() {} }),
    createRadialGradient: () => ({ addColorStop() {} }),
  };
  return new Proxy(target, {
    get(t, p) { if (p in t) return t[p]; return () => {}; },
    set(t, p, v) { t[p] = v; return true; },
  });
}

function makeEl(tag) {
  const e = {
    style: {}, textContent: '', innerHTML: '', className: '', value: '0',
    classList: { add() {}, remove() {}, toggle() {} },
    dataset: {},
    addEventListener() {}, appendChild() {},
    querySelector() { return makeEl('span'); },
    width: 0, height: 0,
  };
  if (tag === 'canvas') {
    e.getContext = () => makeCtx2d();
    e.getBoundingClientRect = () => ({ left: 0, top: 0, width: 800, height: 500 });
    e.parentElement = { clientWidth: 1600, clientHeight: 1000 };
  }
  return e;
}

const els = {};
function getEl(id) {
  if (!els[id]) {
    els[id] = makeEl(id === 'scene' || id === 'chart' ? 'canvas' : 'div');
    if (id === 's-count') els[id].value = '5';
    if (id === 's-batt') els[id].value = '500';
    if (id === 's-panel') els[id].value = '200';
    if (id === 's-roof') els[id].value = '300';
  }
  return els[id];
}

let rafCb = null;
const sandbox = {
  console, performance,
  document: {
    getElementById: getEl,
    createElement: (t) => makeEl(t),
    querySelectorAll: () => ['0', '0.5', '1', '10', '30', '100'].map((s) => ({ dataset: { speed: s }, classList: { toggle() {} }, addEventListener() {} })),
    addEventListener() {},
    body: {},
  },
  window: { addEventListener() {}, devicePixelRatio: 1 },
  requestAnimationFrame: (cb) => { rafCb = cb; return 1; },
};
const context = vm.createContext(sandbox);

for (const f of ['config.js', 'simulation.js', 'render.js']) {
  let src = fs.readFileSync(f, 'utf8');
  if (f === 'config.js') src += '\nglobalThis.CONFIG = CONFIG;';
  vm.runInContext(src, context, { filename: f });
}
const realCreate = context.createSim;
context.createSim = (cfg) => { const s = realCreate(cfg); context.__sim = s; return s; };
vm.runInContext(fs.readFileSync('main.js', 'utf8'), context, { filename: 'main.js' });

const sim = context.__sim;
if (!sim) { console.log('FAIL: sim not created'); process.exit(1); }
// pull sim globals out of the sandbox for the assertions below
const poolStoredWh = context.poolStoredWh;
const unitNowW = context.unitNowW;

const t0 = sim.t;
let t = performance.now();

// phase 1: real rAF frames — checks the loop, rendering path, HUD updates
for (let i = 0; i < 3000; i++) {
  const cb = rafCb; rafCb = null;
  if (!cb) { console.log('FAIL: frame loop died at frame', i); process.exit(1); }
  t += 16.7;
  cb(t);
}

// phase 2: fast-forward ~14h (sunrise → night) in advance() chunks so the dock,
// conveyor, discharge units and ready rack all get exercised
const goal = 14 * 3600;
while (sim.t - t0 < goal) {
  const dt = Math.min(100, goal - (sim.t - t0));
  context.advance(sim, dt / 10, 10); // 10x speed, whole-substep exact
}

// phase 3: a few more live frames at the end (night, plant still draining)
for (let i = 0; i < 200; i++) {
  const cb = rafCb; rafCb = null;
  if (!cb) { console.log('FAIL: frame loop died after fast-forward at frame', i); process.exit(1); }
  t += 16.7;
  cb(t);
}

const h = Math.floor((sim.t % 86400) / 3600), m = Math.floor((sim.t % 3600) / 60);
const dT = ((sim.t - t0) / 3600).toFixed(2);
console.log(`clock after run: Day ${Math.floor(sim.t / 86400) + 1} ${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}  (advanced ${dT} sim-h)`);
console.log('weather:', sim.weather.state, 'clouds:', sim.clouds.length);
console.log('states:', sim.sheep.map(s => `${s.name}:${s.state}`).join(' '));
console.log('soc   :', sim.sheep.map(s => s.pack.soc.toFixed(2)).join(' '));
const plantBusy = sim.units.filter(u => u.p && unitNowW(sim, u) > 0).length;
console.log(`plant : busy=${plantBusy}/${sim.units.length} line=${sim.line.length} ready=${sim.ready.length} queue=${sim.dock.queue.length}`);
const T = sim.totals;
const stored = poolStoredWh(sim);
const cap = sim.cfg.battery.capacityWh;
const reserve0 = sim.sheep.length * sim.cfg.battery.reserveSoc * cap; // unproduced initial reserve
const drift = T.produced - T.delivered - T.lost - (stored - reserve0);
console.log(`totals: produced=${T.produced.toFixed(1)} delivered=${T.delivered.toFixed(1)} lost=${T.lost.toFixed(1)} inPool=${stored.toFixed(1)} plantDel=${sim.plantDelivered.toFixed(1)} roofDel=${sim.whDelivered.toFixed(1)} drift=${drift.toFixed(2)}Wh`);
if (Math.abs(drift) > 1) { console.log('FAIL: energy balance drift > 1 Wh'); process.exit(1); }
if (sim.t - t0 < 300) { console.log('FAIL: sim barely advances - freeze'); process.exit(1); }
if (sim.plantDelivered < 500) { console.log('FAIL: battery plant delivered nothing (expected >500 Wh after 14h)'); process.exit(1); }
// every pack must be accounted for exactly once: 50 total
const nCarry = sim.pool.filter(p => p.state === 'carry').length;
const nLine = sim.pool.filter(p => p.state === 'line').length;
const nDis = sim.pool.filter(p => p.state === 'discharging').length;
const nReady = sim.pool.filter(p => p.state === 'ready').length;
if (sim.pool.length !== sim.cfg.dock.pool || nCarry + nLine + nDis + nReady !== sim.pool.length) {
  console.log(`FAIL: pool bookkeeping: total=${sim.pool.length} carry=${nCarry} line=${nLine} disch=${nDis} ready=${nReady}`);
  process.exit(1);
}
if (sim.ready.length !== nReady) { console.log(`FAIL: ready queue mismatch: array=${sim.ready.length} packs-in-ready-state=${nReady}`); process.exit(1); }
console.log('FRAME LOOP OK');