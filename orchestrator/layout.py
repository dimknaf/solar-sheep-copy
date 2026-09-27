"""orchestrator/layout.py -- where things are on the site (world frame, metres).

World frame: Z up, X east, Y north (CONTRACTS). One Isaac env grid, everything within
about +/-7 m of the origin. ``Layout.default(n_rovers=5)``:

                        y
       row 1   (-3.5,4.4) (-1.5,4.4) ( 0.5,4.4) ( 2.5,4.4)      pasture: 2 rows x 4 slots,
       row 0   (-3.5,2.4) (-1.5,2.4) ( 0.5,2.4) ( 2.5,2.4)      2.0 m pitch, row 0 nearest
                                                                 the dock (slot = column)
     queue:  3      2      1      0    approach   BERTH   exit
          (-6.7) (-5.4) (-4.1) (-2.8)  (-1.5,0)  (0,0)  (2.0,0)  --> dock +X  --- x
                                               [dock pad 1.10 x 1.00, cabinet at +Y]
       park:  (-5.0,-2.4) (-3.5,-2.4) (-2.0,-2.4) (-0.5,-2.4) (1.0,-2.4)   facing north

* Dock origin = berth at (0, 0), dock yaw 0: the rover drives ONTO the pad along +X
  (envs/swap/SPEC.md: approach at (-1.50, 0) in the dock frame, vehicle +X forward) and,
  the dock being drive-over (fore/aft magazine, dock_mjcf.py), drives OFF it forward to
  ``exit_xy``. Everything dock-relative (approach, queue, exit) follows ``dock_yaw``.
* Queue points run straight back from the approach along dock -X, 1.3 m apart
  (>= 1.2 m; rover footprint 0.91 x 1.01 m), bending south after the fourth.
* Harvest cells: approach -> farthest cell is 5.95 m; cells 2.0 m apart, so two rovers
  spinning in place (0.68 m swept radius) keep 0.64 m of air between them.
* Park row south of the queue, 1.5 m pitch, facing north (yaw +pi/2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

XY = Tuple[float, float]

BERTH_RADIUS_M = 0.35     # "on the berth" for the validator (swap tolerance itself is 0.035 m)
NEAR_DOCK_M = 1.0         # within this of the berth/approach/queue = "at the dock"


@dataclass
class Layout:
    dock_xy: XY = (0.0, 0.0)          # dock origin in the world (= berth); Isaac spawns the dock here
    dock_yaw: float = 0.0             # rad; the berth direction (+X of the dock) in the world
    rows: int = 2
    slots: int = 4
    grid_origin: XY = (-3.5, 2.4)     # cell (0, 0)
    slot_pitch: float = 2.0           # along +X
    row_pitch: float = 2.0            # along +Y
    approach_back_m: float = 1.5      # approach standoff behind the berth (SPEC.md, LOCKED)
    exit_fwd_m: float = 2.0           # drive-off point ahead of the berth
    queue_pitch: float = 1.3
    queue_straight: int = 4           # queue points on the straight line before it bends
    park_origin: XY = (-5.0, -2.4)
    park_pitch: float = 1.5
    park_yaw: float = math.pi / 2

    # ---- construction ---------------------------------------------------------
    @classmethod
    def default(cls, n_rovers: int = 5) -> "Layout":
        slots = max(4, -(-n_rovers // 2))       # 2 rows, at least one cell per rover
        return cls(slots=slots)

    # ---- dock frame -> world -------------------------------------------------
    def _dock_to_world(self, dx: float, dy: float) -> XY:
        c, s = math.cos(self.dock_yaw), math.sin(self.dock_yaw)
        return (self.dock_xy[0] + c * dx - s * dy, self.dock_xy[1] + s * dx + c * dy)

    @property
    def berth_xy(self) -> XY:
        return self._dock_to_world(0.0, 0.0)

    @property
    def berth_yaw(self) -> float:
        return self.dock_yaw

    @property
    def approach_xy(self) -> XY:
        return self._dock_to_world(-self.approach_back_m, 0.0)

    @property
    def exit_xy(self) -> XY:
        return self._dock_to_world(self.exit_fwd_m, 0.0)

    def queue_xy(self, i: int) -> XY:
        """i = 0 is next in line (1.3 m behind the approach point)."""
        i = max(0, int(i))
        k = min(i, self.queue_straight - 1)
        dx = -self.approach_back_m - self.queue_pitch * (k + 1)
        dy = -self.queue_pitch * (i - k)            # bend toward dock -Y after the straight
        return self._dock_to_world(dx, dy)

    def park_xy(self, i: int) -> XY:
        return (self.park_origin[0] + self.park_pitch * int(i), self.park_origin[1])

    # ---- pasture grid --------------------------------------------------------
    def slot_xy(self, row: int, slot: int) -> XY:
        return (self.grid_origin[0] + self.slot_pitch * slot,
                self.grid_origin[1] + self.row_pitch * row)

    def in_grid(self, row: int, slot: int) -> bool:
        return 0 <= row < self.rows and 0 <= slot < self.slots

    def cells(self) -> List[Tuple[int, int]]:
        return [(r, s) for r in range(self.rows) for s in range(self.slots)]

    def spread_cells(self) -> List[Tuple[int, int]]:
        """Checkerboard first, so the first few rovers fan out rather than bunch up."""
        cs = self.cells()
        return [c for c in cs if (c[0] + c[1]) % 2 == 0] + [c for c in cs if (c[0] + c[1]) % 2]

    def nearest_free_cell(self, x: float, y: float, taken: Iterable[Tuple[int, int]],
                          exclude: Sequence[Tuple[int, int]] = ()) -> Optional[Tuple[int, int]]:
        bad = set(taken) | set(exclude)
        best, best_d = None, float("inf")
        for c in self.cells():
            if c in bad:
                continue
            cx, cy = self.slot_xy(*c)
            d = math.hypot(cx - x, cy - y)
            if d < best_d:
                best, best_d = c, d
        return best

    # ---- predicates ----------------------------------------------------------
    def at_berth(self, x: float, y: float) -> bool:
        bx, by = self.berth_xy
        return math.hypot(x - bx, y - by) <= BERTH_RADIUS_M

    def near_dock(self, x: float, y: float) -> bool:
        pts = [self.berth_xy, self.approach_xy] + [self.queue_xy(i) for i in range(6)]
        return any(math.hypot(x - px, y - py) <= NEAR_DOCK_M for px, py in pts)

    def detour_xy(self, x: float, y: float, tx: float, ty: float, side: int = 1,
                  off_m: float = 1.0) -> XY:
        """A detour point beside the straight line (x,y)->(tx,ty) (traverse-failure retry)."""
        dx, dy = tx - x, ty - y
        d = math.hypot(dx, dy) or 1.0
        mx, my = (x + tx) / 2, (y + ty) / 2
        return (round(mx - side * dy / d * off_m, 3), round(my + side * dx / d * off_m, 3))

    def describe(self) -> dict:
        """Compact geometry for the LLM prompt and the log."""
        r2 = lambda p: [round(p[0], 2), round(p[1], 2)]
        return {"grid": {"rows": self.rows, "slots": self.slots,
                         "cell00": r2(self.slot_xy(0, 0)),
                         "pitch_m": [self.slot_pitch, self.row_pitch]},
                "berth": r2(self.berth_xy), "approach": r2(self.approach_xy),
                "queue0": r2(self.queue_xy(0))}
