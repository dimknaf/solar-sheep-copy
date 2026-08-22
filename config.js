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

  sheep: { count: 5 },
  panel: { peakW: 200 },
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
};
