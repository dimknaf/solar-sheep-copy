'use strict';

const STATE_LABEL = {
  resting: 'resting',
  toField: 'to field',
  harvesting: 'sunning',
  toWarehouse: 'coming home',
  delivering: 'to grid',
};
const STATE_COLOR = {
  resting: '#9aa7b8',
  toField: '#7aa2ff',
  harvesting: '#ffd166',
  toWarehouse: '#ff9f68',
  delivering: '#80ed99',
};

function makeDraw() {
  let canvas, ctx, dpr = 1;

  function setup(c) {
    canvas = c;
    ctx = canvas.getContext('2d');
  }

  function resize() {
    dpr = window.devicePixelRatio || 1;
    const stage = canvas.parentElement;
    const w = stage.clientWidth - 28, h = stage.clientHeight - 28;
    canvas.style.width = w + 'px';
    canvas.style.height = h + 'px';
    canvas.width = Math.max(1, Math.round(w * dpr));
    canvas.height = Math.max(1, Math.round(h * dpr));
  }

  function draw(sim, pvW, gridW, motorW) {
    const cfg = sim.cfg;
    const S = Math.min(canvas.width / cfg.world.w, canvas.height / cfg.world.h);
    const ox = (canvas.width - cfg.world.w * S) / 2;
    const oy = (canvas.height - cfg.world.h * S) / 2;

    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = '#0c1016';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.setTransform(S, 0, 0, S, ox, oy);

    const t = sim.t;
    const sun = sunFactor(t, cfg);

    drawField(sim, sun);
    drawWarehouse(sim);
    drawClouds(sim);
    for (const s of sim.sheep) drawSheep(sim, s);
    drawParticles(sim);
    drawWeatherTint(sim);
    drawTint(sim, sun, t);
    drawSunBadge(sim, sun);
    drawOverlay(sim, pvW, gridW, motorW);
    ctx.restore();
  }

  function drawField(sim, sun) {
    const cfg = sim.cfg;
    const g = ctx.createLinearGradient(0, 0, 0, cfg.world.h);
    const lit = (0.55 + sun * 0.45) * weatherF(sim);
    g.addColorStop(0, shadeColor('#5d8a45', lit));
    g.addColorStop(1, shadeColor('#4a7a3a', lit));
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, cfg.world.w, cfg.world.h);

    ctx.strokeStyle = 'rgba(255,255,255,0.05)';
    ctx.lineWidth = 1;
    for (let x = 100; x < cfg.world.w; x += 140) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, cfg.world.h); ctx.stroke();
    }
    for (let y = 100; y < cfg.world.h; y += 140) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(cfg.world.w, y); ctx.stroke();
    }

    ctx.fillStyle = 'rgba(0,0,0,0.15)';
    ctx.font = '600 26px system-ui';
    ctx.textAlign = 'center';
    ctx.fillText('P A S T U R E', (cfg.warehouse.x + cfg.warehouse.w + cfg.world.w) / 2 + 30, cfg.world.h - 40);
  }

  function drawWarehouse(sim) {
    const wh = sim.cfg.warehouse;
    ctx.save();
    ctx.fillStyle = 'rgba(0,0,0,0.28)';
    ctx.beginPath();
    ctx.roundRect(wh.x + 6, wh.y + 8, wh.w, wh.h, 10);
    ctx.fill();
    ctx.fillStyle = '#2b3242';
    ctx.beginPath(); ctx.roundRect(wh.x, wh.y, wh.w, wh.h, 10); ctx.fill();
    ctx.fillStyle = '#39415a';
    ctx.beginPath(); ctx.roundRect(wh.x, wh.y, wh.w, 46, [10, 10, 0, 0]); ctx.fill();
    ctx.fillStyle = '#e8ecf2';
    ctx.font = '700 22px system-ui';
    ctx.textAlign = 'center';
    ctx.fillText('WAREHOUSE', wh.x + wh.w / 2, wh.y + 31);

    // roof PV strip (separate from the sheep herd)
    const roofW = whNowW(sim);
    const roofGlow = Math.min(1, roofW / sim.cfg.whPanel.peakW);
    ctx.fillStyle = `rgba(43,74,128,${0.5 + roofGlow * 0.5})`;
    ctx.strokeStyle = `rgba(120,170,255,${0.25 + roofGlow * 0.7})`;
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.roundRect(wh.x + 12, wh.y + 8, wh.w - 24, 7, 3); ctx.fill();
    ctx.stroke();

    const gridActive = sim.sheep.some(s => s.state === 'delivering') || roofW > 1;
    const gx = wh.x + wh.w / 2, gy = wh.y + 95;
    ctx.strokeStyle = gridActive ? '#80ed99' : '#5c6b80';
    ctx.lineWidth = 3;
    bolt(ctx, gx - 14, gy - 26, 52);
    ctx.font = '600 15px system-ui';
    ctx.fillStyle = gridActive ? '#80ed99' : '#8fa0b5';
    ctx.fillText('GRID INTAKE', gx, gy + 58);
    const wNow = sheepNowW(sim) + roofW;
    ctx.font = '700 24px system-ui';
    ctx.fillText(wNow > 0.5 ? Math.round(wNow) + ' W' : 'idle', gx, gy + 88);
    ctx.font = '600 12px system-ui';
    ctx.fillStyle = roofW > 0.5 ? '#7aa2ff' : '#5c6b80';
    ctx.fillText('roof ' + Math.round(roofW) + ' W', gx, gy + 108);

    const n = sim.cfg.sheep.count;
    ctx.strokeStyle = 'rgba(255,255,255,0.08)';
    ctx.lineWidth = 1;
    for (let i = 0; i < n; i++) {
      const dy = wh.y + 34 + (n === 1 ? (wh.h - 50) / 2 : i * (wh.h - 50) / (n - 1));
      ctx.beginPath();
      ctx.roundRect(wh.x + wh.w + 26, dy - 20, 62, 40, 8);
      ctx.stroke();
    }
    ctx.restore();
  }

  function bolt(ctx, x, y, hgt) {
    ctx.beginPath();
    ctx.moveTo(x + 16, y);
    ctx.lineTo(x, y + hgt * 0.55);
    ctx.lineTo(x + 12, y + hgt * 0.55);
    ctx.lineTo(x - 4, y + hgt);
    ctx.lineTo(x + 22, y + hgt * 0.4);
    ctx.lineTo(x + 10, y + hgt * 0.4);
    ctx.closePath();
    ctx.stroke();
  }

  function drawClouds(sim) {
    ctx.save();
    for (const cl of sim.clouds) {
      const g = ctx.createRadialGradient(cl.x, cl.y, cl.r * 0.2, cl.x, cl.y, cl.r);
      g.addColorStop(0, 'rgba(0,0,0,0.20)');
      g.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.fillStyle = g;
      ctx.beginPath(); ctx.arc(cl.x, cl.y, cl.r, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = 'rgba(255,255,255,0.05)';
      ctx.beginPath(); ctx.arc(cl.x, cl.y, cl.r * 0.55, 0, Math.PI * 2); ctx.fill();
    }
    ctx.restore();
  }

  function drawSheep(sim, s) {
    const sel = sim.selected === s.id;
    const sun = sunFactor(sim.t, sim.cfg);
    const shade = shadeAt(sim, s.x, s.y);
    const glowAmt = Math.min(1, sun * shade * s.exposure * weatherF(sim)) * (s.state === 'harvesting' || s.state === 'toField' ? 1 : 0.25);

    ctx.save();
    ctx.translate(s.x, s.y);

    // ground shadow stays unrotated
    ctx.fillStyle = 'rgba(0,0,0,0.25)';
    ctx.beginPath(); ctx.ellipse(2, 6, 24, 18, 0, 0, Math.PI * 2); ctx.fill();

    ctx.save();
    ctx.rotate(s.orient);

    // nose wedge marks which way the panels face
    ctx.fillStyle = '#c9c0ae';
    ctx.beginPath();
    ctx.moveTo(-30, -5); ctx.lineTo(-40, 0); ctx.lineTo(-30, 5);
    ctx.closePath(); ctx.fill();

    const wheelPulse = 1 + Math.sin(sim.t * 14 + s.id) * 0.06;
    ctx.fillStyle = '#14181f';
    for (const [wx, wy] of [[-19, -22], [19, -22], [-19, 22], [19, 22]]) {
      ctx.beginPath();
      ctx.ellipse(wx, wy, 7 * wheelPulse, 5, 0, 0, Math.PI * 2);
      ctx.fill();
    }

    const puffs = [[-16, -6], [-14, 8], [-18, 2], [-6, -14], [2, -10], [-2, 0], [2, 10], [-8, 14], [10, -6], [10, 8], [0, -4]];
    for (let i = puffs.length - 1; i >= 0; i--) {
      const [px, py] = puffs[i];
      ctx.fillStyle = i % 2 ? '#f2ede2' : '#e9e2d3';
      ctx.beginPath(); ctx.arc(px, py, 12, 0, Math.PI * 2); ctx.fill();
    }

    if (sel) {
      ctx.strokeStyle = '#ffd166';
      ctx.lineWidth = 2.5;
      ctx.setLineDash([6, 5]);
      ctx.lineDashOffset = -sim.t * 20;
      ctx.beginPath(); ctx.arc(0, 0, 34, 0, Math.PI * 2); ctx.stroke();
      ctx.setLineDash([]);
    }

    ctx.fillStyle = `rgba(43,74,128,${0.55 + glowAmt * 0.35})`;
    ctx.strokeStyle = `rgba(120,170,255,${0.25 + glowAmt * 0.6})`;
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.roundRect(-17, -8, 34, 16, 3); ctx.fill();
    ctx.stroke();
    ctx.strokeStyle = `rgba(140,190,255,${0.3 + glowAmt * 0.4})`;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(-6, -8); ctx.lineTo(-6, 8);
    ctx.moveTo(6, -8); ctx.lineTo(6, 8);
    ctx.stroke();

    ctx.restore(); // unrotate for upright labels

    const bw = 56, bh = 7;
    ctx.fillStyle = 'rgba(10,14,20,0.75)';
    ctx.beginPath(); ctx.roundRect(-bw / 2, -46, bw, bh, 3); ctx.fill();
    const frac = (s.soc - sim.cfg.battery.reserveSoc) / (1 - sim.cfg.battery.reserveSoc);
    const barCol = s.soc > 0.35 ? '#80ed99' : (s.soc > 0.15 ? '#ffd166' : '#ff6b6b');
    ctx.fillStyle = barCol;
    if (frac > 0.01) {
      ctx.beginPath(); ctx.roundRect(-bw / 2 + 1, -45, (bw - 2) * Math.max(0, Math.min(1, frac)), bh - 2, 2); ctx.fill();
    }
    ctx.font = '600 12px system-ui';
    ctx.textAlign = 'center';
    ctx.fillStyle = 'rgba(255,255,255,0.92)';
    ctx.fillText(`${s.name.toUpperCase()} - ${STATE_LABEL[s.state]}`, 0, -70);

    const pvW = solarInW(sim, s);
    const outW = (s.state === 'delivering' && (s.soc - sim.cfg.battery.reserveSoc) * sim.cfg.battery.capacityWh > 0.01) ? sim.cfg.grid.dischargeW : 0;
    ctx.font = '600 11px system-ui';
    ctx.fillStyle = `rgba(150,210,255,${0.75 + glowAmt * 0.25})`;
    ctx.fillText(`PV ${Math.round(pvW)} W   ${Math.round(s.soc * 100)}%   face ${Math.round(s.exposure * 100)}%${outW ? `   grid ${Math.round(outW)} W` : ''}`, 0, -56);
    ctx.restore();
  }

  function drawParticles(sim) {
    for (const p of sim.parts) {
      const k = p.age / p.max;
      if (p.spark) {
        ctx.fillStyle = `rgba(140,200,255,${(1 - k) * 0.9})`;
        ctx.beginPath(); ctx.arc(p.x, p.y, 2.2 * (1 - k * 0.5), 0, Math.PI * 2); ctx.fill();
      } else {
        const dx = sim.cfg.warehouse.x + sim.cfg.warehouse.w / 2 - p.x;
        const dy = sim.cfg.warehouse.y + sim.cfg.warehouse.h / 2 - p.y;
        const d = Math.hypot(dx, dy) || 1;
        const px = p.x + dx / d * k * 26;
        const py = p.y + dy / d * k * 26;
        ctx.fillStyle = `rgba(128,237,153,${(1 - k) * 0.8})`;
        ctx.beginPath(); ctx.arc(px, py, 3.2 * (1 - k * 0.6), 0, Math.PI * 2); ctx.fill();
      }
    }
  }

  function drawWeatherTint(sim) {
    const f = weatherF(sim);
    if (f >= 0.95) return;
    ctx.fillStyle = `rgba(150,160,175,${(1 - f) * 0.16})`;
    ctx.fillRect(0, 0, sim.cfg.world.w, sim.cfg.world.h);
  }

  function drawTint(sim, sun, t) {
    const d = t % 86400;
    const cfg = sim.cfg;
    let a = 0;
    if (d < cfg.sunrise) { a = 0.55; }
    else if (d < cfg.sunrise + 2 * 3600) { a = 0.4 * (1 - (d - cfg.sunrise) / (2 * 3600)); }
    else if (d > cfg.sunset - 2 * 3600 && d < cfg.sunset) { a = 0.4 * (1 - (cfg.sunset - d) / (2 * 3600)); }
    else if (d >= cfg.sunset) { a = 0.62; }
    if (a <= 0) return;
    // night / pre-dawn dark blue, dawn & dusk orange
    const col = d < cfg.sunrise ? '30,40,80' : (d >= cfg.sunset ? '8,12,30' : '255,140,60');
    ctx.fillStyle = `rgba(${col},${a})`;
    ctx.fillRect(0, 0, sim.cfg.world.w, sim.cfg.world.h);
  }

  function drawSunBadge(sim, sun) {
    const cfg = sim.cfg;
    const d = sim.t % 86400;
    const p = Math.max(0, Math.min(1, (d - cfg.sunrise) / (cfg.sunset - cfg.sunrise)));
    const cx = 1100 + p * 380;
    const cy = 92 - Math.sin(Math.PI * p) * 48;
    ctx.save();
    ctx.strokeStyle = 'rgba(255,255,255,0.15)';
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(1100 + 190, 92, 190, Math.PI, Math.PI * 2); ctx.stroke();
    const r = 16 + sun * 6;
    const g = ctx.createRadialGradient(cx, cy, r * 0.3, cx, cy, r * 3);
    g.addColorStop(0, `rgba(255,220,120,${0.5 + sun * 0.5})`);
    g.addColorStop(1, 'rgba(255,220,120,0)');
    ctx.fillStyle = g;
    ctx.beginPath(); ctx.arc(cx, cy, r * 3, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = `rgba(255,214,90,${0.35 + sun * 0.65})`;
    ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.fill();
    if (sun === 0) {
      ctx.fillStyle = 'rgba(220,230,255,0.7)';
      ctx.beginPath(); ctx.arc(1180, 60, 12, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = 'rgba(20,28,52,0.9)';
      ctx.beginPath(); ctx.arc(1174, 56, 10, 0, Math.PI * 2); ctx.fill();
    }
    ctx.restore();
  }

  function costOf(sim) {
    const c = sim.cfg.cost;
    const n = sim.cfg.sheep.count;
    const gbp = c.panelGBPperW * sim.cfg.panel.peakW * n
      + c.batteryGBPperKWh * sim.cfg.battery.capacityWh / 1000 * n
      + sim.cfg.whPanel.peakW * c.panelGBPperW
      + c.driveGBPperSheep * n
      + c.inverterGBP;
    const kwh = sim.totals.delivered / 1000;
    return { gbp, perKWh: kwh > 0.01 ? gbp / kwh : null };
  }

  function drawOverlay(sim, pvW, gridW, motorW) {
    const cfg = sim.cfg;
    const rows = [
      ['PV now', Math.round(pvW) + ' W', '#7aa2ff'],
      ['grid now', Math.round(gridW) + ' W', '#80ed99'],
      ['motor', Math.round(motorW) + ' W', '#ff9f68'],
      ['in herd', fmtWh(sim.sheep.reduce((a, s) => a + s.soc * cfg.battery.capacityWh, 0)), '#e8ecf2'],
      ['produced', fmtWh(sim.totals.produced), '#e8ecf2'],
      ['lost', fmtWh(sim.totals.lost), '#ff6b6b'],
      ['today', fmtWh(sim.daily.delivered) + ' to grid', '#80ed99'],
      ['cost', '£' + Math.round(costOf(sim).gbp), '#ffd166'],
    ];
    const w = 236, rh = 20, pad = 12;
    const h = pad * 2 + 22 + rows.length * rh + 8;
    const x0 = cfg.world.w - w - 16, y0 = cfg.world.h - h - 16;

    ctx.save();
    ctx.fillStyle = 'rgba(13,19,32,0.84)';
    ctx.strokeStyle = 'rgba(255,255,255,0.10)';
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.roundRect(x0, y0, w, h, 12); ctx.fill(); ctx.stroke();

    ctx.font = '700 13px system-ui';
    ctx.textAlign = 'left';
    ctx.fillStyle = '#ffd166';
    ctx.fillText('LIVE METRICS', x0 + pad, y0 + pad + 10);
    ctx.fillStyle = WEATHER_COLOR[sim.weather.state];
    ctx.textAlign = 'right';
    ctx.fillText(WEATHER_LABEL[sim.weather.state], x0 + w - pad, y0 + pad + 10);

    rows.forEach((r, i) => {
      const yy = y0 + pad + 30 + i * rh;
      ctx.textAlign = 'left';
      ctx.font = '500 12.5px system-ui';
      ctx.fillStyle = '#8fa0b5';
      ctx.fillText(r[0], x0 + pad, yy);
      ctx.textAlign = 'right';
      ctx.font = '600 12.5px system-ui';
      ctx.fillStyle = r[2];
      ctx.fillText(r[1], x0 + w - pad, yy);
    });
    ctx.restore();
  }

  function shadeColor(hex, lit) {
    const n = parseInt(hex.slice(1), 16);
    const r = Math.round(((n >> 16) & 255) * lit);
    const g = Math.round(((n >> 8) & 255) * lit);
    const b = Math.round((n & 255) * lit);
    return `rgb(${r},${g},${b})`;
  }

  function drawChart(sim) {
    const c = document.getElementById('chart');
    if (!c) return;
    const cc = c.getContext('2d');
    const W = c.width, H = c.height;
    cc.clearRect(0, 0, W, H);
    cc.fillStyle = '#0d1320';
    cc.fillRect(0, 0, W, H);

    const t = sim.t;
    const nowB = Math.floor(t / sim.cfg.chart.bucketSec);
    const span = Math.floor(sim.cfg.chart.keepSec / sim.cfg.chart.bucketSec);
    const startB = nowB - span;

    cc.fillStyle = 'rgba(120,140,180,0.10)';
    const spanB = Math.floor(sim.cfg.chart.keepSec / 60);
    const startB2 = Math.floor(t / 60) - spanB;
    for (let b = startB2; b < startB2 + spanB; b++) {
      const hod = ((b * 60) % 86400) / 3600;
      if (hod < sim.cfg.sunrise / 3600 || hod >= sim.cfg.sunset / 3600) {
        const x = ((b - startB2) / spanB) * W;
        cc.fillRect(x, 0, W / spanB + 0.5, H);
      }
    }

    let maxW = 1;
    for (const h of sim.history) {
      maxW = Math.max(maxW, h.sAcc / Math.max(0.01, h.dur), h.rAcc / Math.max(0.01, h.dur));
    }
    maxW *= 1.1;

    cc.strokeStyle = 'rgba(255,255,255,0.06)';
    cc.lineWidth = 1;
    for (const yy of [0.25, 0.5, 0.75]) {
      cc.beginPath(); cc.moveTo(0, H * yy); cc.lineTo(W, H * yy); cc.stroke();
    }

    const xOf = b => ((b - startB) / span) * W;
    const yOf = p => H - 20 - (p / maxW) * (H - 32);

    // sheep series (green)
    cc.beginPath();
    let started = false;
    for (const h of sim.history) {
      const x = xOf(h.b);
      const y = yOf(h.sAcc / Math.max(0.01, h.dur));
      if (x < 0) continue;
      if (!started) { cc.moveTo(x, y); started = true; }
      else cc.lineTo(x, y);
    }
    cc.strokeStyle = '#80ed99';
    cc.lineWidth = 2;
    cc.stroke();
    cc.lineTo(W, H - 16); cc.lineTo(0, H - 16);
    cc.closePath();
    cc.fillStyle = 'rgba(128,237,153,0.10)';
    cc.fill();

    // roof series (amber)
    cc.beginPath();
    started = false;
    for (const h of sim.history) {
      const x = xOf(h.b);
      const y = yOf(h.rAcc / Math.max(0.01, h.dur));
      if (x < 0) continue;
      if (!started) { cc.moveTo(x, y); started = true; }
      else cc.lineTo(x, y);
    }
    cc.strokeStyle = '#ffd166';
    cc.lineWidth = 1.5;
    cc.stroke();

    cc.font = '10px system-ui';
    cc.fillStyle = 'rgba(255,255,255,0.35)';
    cc.textAlign = 'left';
    cc.fillText('-24h', 4, H - 4);
    cc.fillStyle = '#80ed99';
    cc.fillText('sheep', 34, H - 4);
    cc.fillStyle = '#ffd166';
    cc.fillText('roof', 70, H - 4);
    cc.textAlign = 'right';
    cc.fillStyle = 'rgba(255,255,255,0.35)';
    cc.fillText('now', W - 4, H - 4);
    cc.fillText(Math.round(maxW / 1.1) + 'W', W - 4, 10);
  }

  return { setup, resize, draw, drawChart };
}