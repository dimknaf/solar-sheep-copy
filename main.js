'use strict';

(function () {
  const cfg = CONFIG;
  let sim = createSim(cfg);
  let speed = 10;
  let last = performance.now();
  let slowT = 0;

  const draw = makeDraw();
  const scene = document.getElementById('scene');
  draw.setup(scene);
  draw.resize();
  window.addEventListener('resize', () => { draw.resize(); });

  const clockEl = document.getElementById('clock');
  const tHar = document.getElementById('t-har');
  const tDel = document.getElementById('t-del');
  const tLost = document.getElementById('t-lost');
  const tEff = document.getElementById('t-eff');
  const gnow = document.getElementById('gnow');
  const detailsEl = document.getElementById('details');
  const listEl = document.getElementById('list');
  const costEl = document.getElementById('cost');
  let rowCache = [];

  function buildList() {
    listEl.innerHTML = '';
    rowCache = sim.sheep.map(s => {
      const row = document.createElement('div');
      row.className = 'row';
      row.innerHTML = `<span class="dot"></span><span class="name"></span><span class="st"></span><span class="bar"><i></i></span><span class="wh"></span>`;
      row.addEventListener('click', () => {
        sim.selected = sim.selected === s.id ? null : s.id;
        sim.selectedWh = null;
      });
      listEl.appendChild(row);
      return row;
    });
  }
  buildList();

  function updateList() {
    sim.sheep.forEach((s, i) => {
      const row = rowCache[i];
      if (!row) return;
      row.classList.toggle('sel', sim.selected === s.id);
      row.querySelector('.dot').style.background = STATE_COLOR[s.state];
      row.querySelector('.name').textContent = s.name;
      row.querySelector('.st').textContent = STATE_LABEL[s.state];
      const bar = row.querySelector('.bar i');
      const frac = (s.soc - sim.cfg.battery.reserveSoc) / (1 - sim.cfg.battery.reserveSoc);
      bar.style.width = (Math.max(0, Math.min(1, frac)) * 100) + '%';
      bar.style.background = s.soc > 0.35 ? '#80ed99' : (s.soc > 0.15 ? '#ffd166' : '#ff6b6b');
      row.querySelector('.wh').textContent = fmtWh(s.deliverToday);
    });
  }

  function statRows(rows) {
    return rows.map(([k, v, cls]) =>
      `<div class="kv"><span>${k}</span><b class="${cls || ''}">${v}</b></div>`).join('');
  }

  function updateDetails() {
    if (!detailsEl) return;
    if (sim.selectedWh) {
      const cap = sim.cfg.battery.capacityWh;
      const sheepW = sheepNowW(sim);
      const roofW = whNowW(sim);
      const onSite = sim.sheep.filter(s =>
        (s.state === 'resting' || s.state === 'delivering') && Math.hypot(s.x - s.home.x, s.y - s.home.y) < 60).length;
      const stored = sim.sheep.reduce((a, s) => a + s.soc * cap, 0);
      const avgW = sim.totalGridDur > 0 ? sim.totalGridAccum / sim.totalGridDur : 0;
      detailsEl.innerHTML = `
        <h3>Warehouse & grid</h3>
        ${statRows([
          ['grid intake (now)', Math.round(sheepW + roofW) + ' W', (sheepW + roofW) > 0 ? 'good' : ''],
          ['  from herd', Math.round(sheepW) + ' W', sheepW > 0 ? 'good' : ''],
          ['  from roof PV', Math.round(roofW) + ' W', roofW > 0 ? 'good' : ''],
          ['on site', onSite + ' / ' + sim.sheep.length],
          ['energy in herd', fmtWh(stored)],
          ['total delivered', fmtWh(sim.totals.delivered), 'good'],
          ['  of which roof', fmtWh(sim.whDelivered)],
          ['total produced', fmtWh(sim.totals.produced)],
          ['total lost', fmtWh(sim.totals.lost)],
          ['end-to-end efficiency', sim.totals.produced > 1 ? Math.round(100 * sim.totals.delivered / sim.totals.produced) + '%' : '-'],
          ['avg intake since start', Math.round(avgW) + ' W'],
        ])}
        <div class="hint">click a sheep for its live numbers</div>`;
      return;
    }
    const s = sim.sheep.find(x => x.id === sim.selected);
    if (!s) { detailsEl.innerHTML = ''; return; }
    const cap = sim.cfg.battery.capacityWh;
    const pvW = solarInW(sim, s);
    const outW = (s.state === 'delivering' && (s.soc - sim.cfg.battery.reserveSoc) * cap > 0.01) ? sim.cfg.grid.dischargeW : 0;
    const motorW = (s.state === 'toField' || s.state === 'toWarehouse' || s.state === 'harvesting') ? sim.cfg.motor.powerW : 0;
    detailsEl.innerHTML = `
      <h3>${s.name} - ${STATE_LABEL[s.state]}</h3>
      ${statRows([
        ['PV production (now)', Math.round(pvW) + ' W', pvW > 1 ? 'good' : ''],
        ['sun exposure', Math.round(s.exposure * 100) + '% (skill ' + Math.round(s.skill * 100) + '%)'],
        ['battery', Math.round(s.soc * 100) + '% (' + Math.round(s.soc * cap) + ' / ' + cap + ' Wh)'],
        ['to grid (now)', outW ? Math.round(outW) + ' W' : '0 W', outW ? 'good' : ''],
        ['motor draw', motorW + ' W'],
        ['produced today', fmtWh(s.produceToday)],
        ['harvested today', fmtWh(s.harvestToday)],
        ['delivered today', fmtWh(s.deliverToday), 'good'],
      ])}`;
  }

  function updateHUD() {
    const h = Math.floor((sim.t % 86400) / 3600);
    const m = Math.floor((sim.t % 3600) / 60);
    clockEl.textContent = `Day ${dayOf(sim.t)} - ${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}`;
    tHar.textContent = fmtWh(sim.totals.harvest);
    tDel.textContent = fmtWh(sim.totals.delivered);
    tLost.textContent = fmtWh(sim.totals.lost);
    tEff.textContent = sim.totals.produced > 1 ? Math.round(100 * sim.totals.delivered / sim.totals.produced) + '%' : '-';
    gnow.textContent = Math.round(sheepNowW(sim) + whNowW(sim));
  }

  function toWorld(e) {
    const r = scene.getBoundingClientRect();
    const cfgW = sim.cfg.world;
    const S = Math.min(r.width / cfgW.w, r.height / cfgW.h);
    const ox = (r.width - cfgW.w * S) / 2, oy = (r.height - cfgW.h * S) / 2;
    return { x: (e.clientX - r.left - ox) / S, y: (e.clientY - r.top - oy) / S };
  }

  scene.addEventListener('click', e => {
    const w = toWorld(e);
    const wh = sim.cfg.warehouse;
    if (w.x >= wh.x - 30 && w.x <= wh.x + wh.w + 100 && w.y >= wh.y - 30 && w.y <= wh.y + wh.h + 30) {
      sim.selectedWh = true;
      sim.selected = null;
      return;
    }
    sim.selectedWh = null;
    let best = null, bd = 1e9;
    for (const s of sim.sheep) {
      const d = (s.x - w.x) ** 2 + (s.y - w.y) ** 2;
      if (d < bd) { bd = d; best = s; }
    }
    sim.selected = best && bd < 45 * 45 ? best.id : null;
  });

  const spdBtns = [...document.querySelectorAll('.spd[data-speed]')];
  function setSpeed(v) {
    speed = v;
    spdBtns.forEach(b => b.classList.toggle('active', +b.dataset.speed === v));
  }
  spdBtns.forEach(b => b.addEventListener('click', () => setSpeed(+b.dataset.speed)));
  document.getElementById('skip').addEventListener('click', () => {
    advance(sim, 3600 / Math.max(1, speed), Math.max(1, speed));
  });
  window.addEventListener('keydown', e => {
    if (e.code === 'Space' && e.target === document.body) {
      e.preventDefault();
      setSpeed(speed === 0 ? 10 : 0);
    }
  });

  const sCount = document.getElementById('s-count');
  const sBatt = document.getElementById('s-batt');
  const sPanel = document.getElementById('s-panel');
  const sRoof = document.getElementById('s-roof');
  const vCount = document.getElementById('v-count');
  const vBatt = document.getElementById('v-batt');
  const vPanel = document.getElementById('v-panel');
  const vRoof = document.getElementById('v-roof');

  function updateCost() {
    if (!costEl) return;
    const c = cfg.cost;
    const n = +sCount.value, batt = +sBatt.value, panelW = +sPanel.value, roof = +sRoof.value;
    const panels = c.panelGBPperW * panelW * n;
    const roofC = c.panelGBPperW * roof;
    const batts = c.batteryGBPperKWh * batt / 1000 * n;
    const drives = c.driveGBPperSheep * n;
    const inv = c.inverterGBP;
    const total = panels + roofC + batts + drives + inv;
    const kwh = sim.totals.delivered / 1000;
    costEl.innerHTML = statRows([
      [`panels ${n} × ${panelW} W`, '£' + Math.round(panels)],
      [`batteries ${n} × ${batt} Wh`, '£' + Math.round(batts)],
      [`roof PV ${roof} W`, '£' + Math.round(roofC)],
      ['motors & wheels', '£' + Math.round(drives)],
      ['inverter', '£' + Math.round(inv)],
      ['total cost', '£' + Math.round(total), 'good'],
      ['per kWh delivered', kwh > 0.01 ? '£' + (total / kwh).toFixed(2) : '-'],
    ]);
  }

  sCount.addEventListener('input', () => { vCount.textContent = sCount.value; updateCost(); });
  sBatt.addEventListener('input', () => { vBatt.textContent = sBatt.value + ' Wh'; updateCost(); });
  sPanel.addEventListener('input', () => { vPanel.textContent = sPanel.value + ' W'; updateCost(); });
  sRoof.addEventListener('input', () => { vRoof.textContent = sRoof.value + ' W'; updateCost(); });

  document.getElementById('apply').addEventListener('click', () => {
    cfg.sheep.count = +sCount.value;
    cfg.battery.capacityWh = +sBatt.value;
    cfg.panel.peakW = +sPanel.value;
    cfg.whPanel.peakW = +sRoof.value;
    sim = createSim(cfg);
    sim.selectedWh = false;
    buildList();
    updateList();
    updateDetails();
    updateCost();
  });
  updateCost();

  setSpeed(10);

  function frame(now) {
    const dtReal = Math.min(0.1, (now - last) / 1000);
    last = now;
    if (speed > 0) advance(sim, dtReal, speed);
    const roofW = whNowW(sim);
    const pvW = sim.sheep.reduce((a, s) => a + solarInW(sim, s), 0) + roofW;
    const gridW = sheepNowW(sim) + roofW;
    const motorW = sim.sheep.filter(s =>
      s.state === 'toField' || s.state === 'toWarehouse' || s.state === 'harvesting').length * sim.cfg.motor.powerW;
    draw.draw(sim, pvW, gridW, motorW);
    draw.drawChart(sim);
    updateHUD();
    slowT += dtReal;
    if (slowT > 0.25) { slowT = 0; updateList(); updateDetails(); }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
})();