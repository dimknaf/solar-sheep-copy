'use strict';

const CONFIG = {
  world: { w: 1600, h: 1000 },
  warehouse: { x: 70, y: 380, w: 220, h: 240 },

  startDay: 1,
  startHour: 6.5,
  substep: 0.5,

  sunrise: 6 * 3600,
  sunset: 18 * 3600,
  recallTime: 17 * 3600 + 20 * 60,

  sun: { cloudShade: 0.45 },
  clouds: { count: 8, speed: 6, minR: 60, maxR: 130 },

  weather: {
    // name → { solar multiplier, target cloud count, pick weight }
    states: {
      clear:    { f: 1.0,  clouds: 4,  w: 0.45 },
      cloudy:   { f: 0.8,  clouds: 8,  w: 0.35 },
      overcast: { f: 0.45, clouds: 14, w: 0.20 },
    },
    minDurH: 1,
    maxDurH: 4,
  },

  sheep: {
    count: 5,
    wakeSpreadSec: 45 * 60,    // per-sheep stagger after sunrise
    recallJitterSec: 15 * 60,  // per-sheep recall time, ± this
    turnRate: 0.6,             // rad/s, how fast a sheep can turn to face the sun
    trackSkillMin: 0.85,       // per-sheep tracking quality (sloppy → precise)
    trackSkillMax: 1.0,
  },

  panel: { peakW: 200 },
  whPanel: { peakW: 300 },     // fixed warehouse roof PV, feeds the grid directly
  battery: {
    capacityWh: 500,
    chargeEff: 0.95,
    dischargeEff: 0.95,
    reserveSoc: 0.05,
  },
  motor: { powerW: 30, travelSpeed: 45, harvestSpeed: 12 },
  grid: { dischargeW: 60 },
  fullThreshold: 0.95,
  chart: { bucketSec: 60, keepSec: 24 * 3600 },

  cost: {
    panelGBPperW: 0.8,
    batteryGBPperKWh: 120,
    driveGBPperSheep: 150,
    inverterGBP: 300,
  },
};