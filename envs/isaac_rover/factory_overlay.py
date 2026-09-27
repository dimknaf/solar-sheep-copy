"""envs/isaac_rover/factory_overlay.py -- the text and graphics drawn on every frame of the factory clip.

Imported lazily by envs/isaac_rover/factory.py. Pillow only (``from PIL import Image, ImageDraw``);
if Pillow is missing, ``available()`` is False and the factory writes the clip without overlays.
Pure drawing: it takes a finished RTX frame (H x W x 3 uint8) and a plain ``info`` dict that the
factory fills every captured step, and returns a new frame. Nothing here touches Isaac Sim.

    info = {
      "clock": "10:40", "timelapse": 120.0, "t_phys": 63.2,
      "sun_az_deg": 135.0, "sun_el_deg": 31.0, "weather": "clear", "weather_f": 1.0,
      "rovers": [{"id", "state", "stage", "soc", "exposure", "pv_w", "has_pack", "status",
                  "reason", "tag_px": (x, y) or None}],
      "dock": {"status", "current", "phase", "queue": [...], "empty_packs", "swaps", "mode"},
      "brain": {"label", "calls", "cost", "in_flight", "cmd_json", "cmd_reason", "summary"},
      "banner": str or None,          # a fault / recovery line in red
      "title": str,
    }
"""

from __future__ import annotations

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # the factory checks available() and runs without overlays
    Image = ImageDraw = ImageFont = None

FONT_PATHS = ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "DejaVuSans.ttf",
              "/usr/share/fonts/dejavu/DejaVuSans.ttf")
FONT_BOLD_PATHS = ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "DejaVuSans-Bold.ttf",
                   "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf")
MONO_PATHS = ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", "DejaVuSansMono.ttf",
              "/usr/share/fonts/dejavu/DejaVuSansMono.ttf")

PANEL = (12, 16, 22, 178)          # translucent dark panel
TEXT = (236, 240, 244, 255)
MUTED = (168, 178, 190, 255)
ACCENT = (255, 196, 64, 255)       # sun / energy
GOOD = (96, 208, 128, 255)
BAD = (255, 92, 84, 255)
LLM = (118, 185, 0, 255)           # NVIDIA green for the Nemotron label
RULES = (120, 170, 255, 255)
STATE_COLOR = {
    "resting": (140, 150, 165, 255), "to_field": (120, 170, 255, 255),
    "harvesting": (255, 196, 64, 255), "to_dock": (200, 140, 255, 255),
    "queuing": (230, 120, 220, 255), "swapping": (96, 208, 128, 255), "fault": (255, 92, 84, 255),
}

_fonts: dict = {}


def available() -> bool:
    return Image is not None


def _font(size: int, kind: str = "regular"):
    key = (size, kind)
    if key in _fonts:
        return _fonts[key]
    paths = {"bold": FONT_BOLD_PATHS, "mono": MONO_PATHS}.get(kind, FONT_PATHS)
    font = None
    for p in paths:
        try:
            font = ImageFont.truetype(p, size)
            break
        except OSError:
            continue
    if font is None:
        try:
            font = ImageFont.load_default(size=size)       # Pillow >= 10.1
        except TypeError:
            font = ImageFont.load_default()
    _fonts[key] = font
    return font


def _soc_color(soc: float):
    if soc >= 0.9:
        return GOOD
    if soc >= 0.4:
        return ACCENT
    return BAD


def _clip(text: str, n: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "..."


def draw(frame: np.ndarray, info: dict) -> np.ndarray:
    """Return ``frame`` with the factory overlays drawn on it (same size, uint8 RGB)."""
    if Image is None:
        return frame
    h, w = frame.shape[:2]
    s = h / 1080.0                                        # layout is designed at 1080p
    px = lambda v: int(round(v * s))                      # noqa: E731
    base = Image.fromarray(np.ascontiguousarray(frame[..., :3])).convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    f_title, f_big = _font(px(26), "bold"), _font(px(22), "bold")
    f_txt, f_small, f_mono = _font(px(19)), _font(px(16)), _font(px(15), "mono")

    # -- top-left: clock, sun, weather ---------------------------------------------------------
    x0, y0 = px(24), px(22)
    d.rounded_rectangle((x0, y0, x0 + px(560), y0 + px(132)), px(12), fill=PANEL)
    d.text((x0 + px(16), y0 + px(10)), info.get("title", "Solar Sheep factory"), font=f_title, fill=TEXT)
    tl = info.get("timelapse", 0.0)
    d.text((x0 + px(16), y0 + px(48)),
           f"Sim time {info.get('clock', '--:--')}  ·  time-lapse {tl:.0f}x  ·  "
           f"physics t {info.get('t_phys', 0.0):.1f} s", font=f_txt, fill=TEXT)
    d.text((x0 + px(16), y0 + px(78)),
           f"Sun az {info.get('sun_az_deg', 0.0):.0f}°  el {info.get('sun_el_deg', 0.0):.0f}°  "
           f"·  weather {info.get('weather', '?')} (x{info.get('weather_f', 1.0):.2f})",
           font=f_txt, fill=ACCENT)
    d.text((x0 + px(16), y0 + px(104)), "Isaac Lab 3.0 · PhysX · trained RL traverse policy",
           font=f_small, fill=MUTED)

    # -- left column: one card per rover -------------------------------------------------------
    rovers = info.get("rovers", [])
    cy = y0 + px(150)
    card_h = px(78)
    for r in rovers:
        d.rounded_rectangle((x0, cy, x0 + px(480), cy + card_h - px(8)), px(10), fill=PANEL)
        col = STATE_COLOR.get(r.get("state", ""), TEXT)
        d.text((x0 + px(14), cy + px(8)), r.get("id", "?"), font=f_big, fill=TEXT)
        label = r.get("state", "?") + (f" · {r['stage']}" if r.get("stage") else "")
        d.text((x0 + px(130), cy + px(11)), _clip(label, 30), font=f_txt, fill=col)
        # SoC bar
        bx0, by0, bw, bh = x0 + px(14), cy + px(42), px(230), px(18)
        d.rectangle((bx0, by0, bx0 + bw, by0 + bh), outline=MUTED, width=max(1, px(1)))
        if r.get("has_pack", True):
            soc = max(0.0, min(1.0, float(r.get("soc", 0.0))))
            d.rectangle((bx0 + px(2), by0 + px(2), bx0 + px(2) + int((bw - px(4)) * soc), by0 + bh - px(2)),
                        fill=_soc_color(soc))
            d.text((bx0 + bw + px(10), by0 - px(2)), f"{soc * 100:4.0f}%", font=f_txt, fill=TEXT)
        else:
            d.text((bx0 + px(8), by0 - px(1)), "NO PACK", font=f_small, fill=BAD)
        d.text((bx0 + bw + px(80), by0 - px(2)),
               f"face {r.get('exposure', 0.0) * 100:3.0f}%  {r.get('pv_w', 0.0):3.0f} W",
               font=f_small, fill=ACCENT)
        if r.get("status") == "FAILURE":
            d.text((x0 + px(300), cy + px(11)), _clip("FAIL " + r.get("reason", ""), 14), font=f_small, fill=BAD)
        cy += card_h
        if r.get("tag_px") is not None:                    # name tag above the rover in the 3D view
            tx, ty = r["tag_px"]
            if 0 <= tx < w and 0 <= ty < h:
                tw = px(12) * len(r.get("id", "")) + px(46)
                d.rounded_rectangle((tx - tw // 2, ty - px(26), tx + tw // 2, ty), px(6), fill=PANEL)
                d.text((tx - tw // 2 + px(8), ty - px(24)), r.get("id", ""), font=f_small, fill=col)
                if r.get("has_pack", True):
                    d.text((tx - tw // 2 + px(8) + px(10) * len(r.get("id", "")), ty - px(24)),
                           f" {float(r.get('soc', 0.0)) * 100:.0f}%", font=f_small, fill=TEXT)

    # -- top-right: dock ------------------------------------------------------------------------
    dk = info.get("dock", {})
    dw = px(480)
    dx0 = w - dw - px(24)
    d.rounded_rectangle((dx0, y0, dx0 + dw, y0 + px(156)), px(12), fill=PANEL)
    d.text((dx0 + px(16), y0 + px(10)), "Battery-swap dock", font=f_big, fill=TEXT)
    busy = bool(dk.get("current"))
    d.text((dx0 + px(16), y0 + px(44)),
           _clip(f"{dk.get('status', 'idle')}" + (f": {dk['current']}" if busy else "")
                 + (f" · {dk['phase']}" if dk.get("phase") else ""), 44),
           font=f_txt, fill=GOOD if busy else MUTED)
    q = dk.get("queue", [])
    d.text((dx0 + px(16), y0 + px(72)), _clip("queue: " + (", ".join(q) if q else "empty"), 46),
           font=f_txt, fill=TEXT)
    d.text((dx0 + px(16), y0 + px(100)),
           f"empty packs on rack: {dk.get('empty_packs', 0)}   swaps done: {dk.get('swaps', 0)}",
           font=f_txt, fill=TEXT)
    d.text((dx0 + px(16), y0 + px(128)), _clip(dk.get("mode", ""), 56), font=f_small, fill=MUTED)

    # -- bottom: the fleet brain ----------------------------------------------------------------
    br = info.get("brain", {})
    bh_ = px(150)
    bx, byy = px(24), h - bh_ - px(24)
    d.rounded_rectangle((bx, byy, w - px(24), byy + bh_), px(12), fill=PANEL)
    label = br.get("label", "rule fallback")
    is_llm = not label.startswith("rule")
    d.text((bx + px(16), byy + px(10)), "Fleet brain:", font=f_big, fill=TEXT)
    d.text((bx + px(170), byy + px(10)), _clip(label, 70), font=f_big, fill=LLM if is_llm else RULES)
    d.text((w - px(560), byy + px(14)),
           f"LLM calls {br.get('calls', 0)}  ·  est. ${br.get('cost', 0.0):.4f}"
           + ("  ·  thinking..." if br.get("in_flight") else ""), font=f_small, fill=MUTED)
    d.text((bx + px(16), byy + px(46)), _clip(br.get("cmd_json", ""), 150), font=f_mono, fill=TEXT)
    d.text((bx + px(16), byy + px(74)), _clip("reason: " + br.get("cmd_reason", ""), 120), font=f_txt,
           fill=ACCENT)
    d.text((bx + px(16), byy + px(106)), _clip("plan: " + br.get("summary", ""), 140), font=f_small,
           fill=MUTED)

    # -- fault / recovery banner ---------------------------------------------------------------
    banner = info.get("banner")
    if banner:                                            # just above the brain panel, centred
        bw_ = px(1000)
        bx1, by1 = (w - bw_) // 2, byy - px(58)
        d.rounded_rectangle((bx1, by1, bx1 + bw_, by1 + px(44)), px(10), fill=(90, 14, 14, 215))
        d.text((bx1 + px(16), by1 + px(10)), _clip(banner, 90), font=f_txt, fill=(255, 225, 220, 255))

    out = Image.alpha_composite(base, layer).convert("RGB")
    return np.asarray(out)
