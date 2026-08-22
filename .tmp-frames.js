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
  }
  return els[id];
}

let rafCb = null;
const sandbox = {
  console, performance,
  document: {
    getElementById: getEl,
    createElement: (t) => makeEl(t),
    querySelectorAll: () => ['0', '1', '10', '30', '100'].map((s) => ({ dataset: { speed: s }, classList: { toggle() {} }, addEventListener() {} })),
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

let t = performance.now();
const t0 = sim.t;
for (let i = 0; i < 3000; i++) {
  const cb = rafCb; rafCb = null;
  if (!cb) { console.log('FAIL: frame loop died at frame', i); process.exit(1); }
  t += 16.7;
  cb(t);
}
const h = Math.floor((sim.t % 86400) / 3600), m = Math.floor((sim.t % 3600) / 60);
const dT = ((sim.t - t0) / 60).toFixed(0);
console.log(`clock after 3000 frames: Day ${Math.floor(sim.t / 86400) + 1} ${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}  (advanced ${dT} sim-min)`);
console.log('states:', sim.sheep.map(s => `${s.name}:${s.state}`).join(' '));
console.log('soc   :', sim.sheep.map(s => s.soc.toFixed(2)).join(' '));
if (sim.t - t0 < 300) { console.log('FAIL: sim barely advances - freeze'); process.exit(1); }
console.log('FRAME LOOP OK');
