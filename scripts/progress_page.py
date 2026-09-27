#!/usr/bin/env python3
"""scripts/progress_page.py -- the owner's local progress page for the rover work.

    python scripts/progress_page.py                  # reads runs/gpu/runs, writes runs/gpu/progress.html

One self-contained HTML file on the laptop: no CDN, no fonts, no scripts from anywhere, all
text escaped. It shows
  - the milestone timeline (runs/gpu/milestones.json, written by the agent as things happen),
  - the physics-check results (any measure_*.json the rover measurement wrote),
  - the newest video clips of every run (Omniverse RTX renders pulled by pull_runs.sh),
  - links to the live views, which work only while scripts/gpu/watch.sh is running.
Standard library only, so it is safe to run on the laptop.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def esc(x) -> str:
    return html.escape(str(x), quote=True)


def clips(runs: Path) -> list[tuple[str, list[Path]]]:
    """(run name, newest clips first) for every run that has videos."""
    out = []
    for vdir in sorted(runs.rglob("videos")):
        if not vdir.is_dir():
            continue
        mp4s = sorted(vdir.rglob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
        if mp4s:
            out.append((str(vdir.parent.relative_to(runs)), mp4s))
    out.sort(key=lambda r: r[1][0].stat().st_mtime, reverse=True)
    for mp4 in sorted(runs.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True):
        out.append((mp4.stem, [mp4]))
    return out


def measurements(runs: Path) -> list[tuple[str, dict]]:
    res = []
    for p in sorted(runs.rglob("measure_*.json")):
        try:
            res.append((p.stem, json.loads(p.read_text(encoding="utf-8"))))
        except (OSError, ValueError):
            continue
    return res


def render(runs: Path, out: Path, milestones_file: Path) -> None:
    rel = lambda p: os.path.relpath(p, out.parent).replace(os.sep, "/")  # noqa: E731
    try:
        milestones = json.loads(milestones_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        milestones = []

    parts = []
    for m in reversed(milestones):
        vid = ""
        if m.get("clip"):
            clip = (out.parent / m["clip"])
            if clip.exists():
                vid = f'<video src="{esc(rel(clip))}" controls loop muted preload="metadata"></video>'
        nums = "".join(f"<li><b>{esc(k)}</b> {esc(v)}</li>" for k, v in (m.get("numbers") or {}).items())
        parts.append(
            f'<article class="ms"><div class="when">{esc(m.get("time", ""))}</div>'
            f'<h3>{esc(m.get("title", ""))}</h3><p>{esc(m.get("detail", ""))}</p>'
            f'{"<ul>" + nums + "</ul>" if nums else ""}{vid}</article>')
    timeline = "\n".join(parts) or "<p class='muted'>No milestones yet.</p>"

    tables = []
    for name, res in measurements(runs):
        rows = "".join(
            f"<tr><td>{esc(k)}</td><td>{esc(v.get('distance_m'))}</td><td>{esc(v.get('speed_mps'))}</td>"
            f"<td>{esc(v.get('yaw_rate'))}</td><td>{'yes' if v.get('tipped') else 'no'}</td></tr>"
            for k, v in res.get("lanes", {}).items())
        checks = " · ".join(f"{'✅' if ok else '❌'} {esc(c)}" for c, ok in res.get("checks", {}).items())
        tables.append(
            f"<h3>{esc(name)} <span class='muted'>({esc(res.get('physics'))}, {esc(res.get('hz'))} Hz)</span></h3>"
            f"<p>{checks}</p><table><tr><th>lane</th><th>distance m</th><th>speed m/s</th>"
            f"<th>yaw rad/s</th><th>tipped</th></tr>{rows}</table>")
    measured = "\n".join(tables) or "<p class='muted'>No physics checks yet.</p>"

    vids = []
    for run, mp4s in clips(runs):
        cells = "".join(
            f'<figure><video src="{esc(rel(p))}" controls loop muted preload="metadata"></video>'
            f'<figcaption>{esc(p.name)} · {esc(time.strftime("%d %b %H:%M", time.localtime(p.stat().st_mtime)))}'
            f"</figcaption></figure>" for p in mp4s[:3])
        vids.append(f"<h3>{esc(run)}</h3><div class='grid'>{cells}</div>")
    videos = "\n".join(vids) or "<p class='muted'>No clips yet.</p>"

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="120">
<title>Rover Progress</title>
<style>
:root {{ --bg:#f7f7f4; --fg:#1d1f21; --muted:#6b6f75; --card:#ffffff; --line:#e3e3de; --accent:#b8500f; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg:#141517; --fg:#e8e8e6; --muted:#9a9ea5; --card:#1d1f22; --line:#2c2f33; --accent:#f08a3e; }} }}
body {{ margin:0; background:var(--bg); color:var(--fg); font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif; }}
main {{ max-width:1100px; margin:0 auto; padding:24px 16px 64px; }}
h1 {{ font-size:24px; margin:0 0 4px; }} h2 {{ font-size:18px; margin:32px 0 12px; border-bottom:1px solid var(--line); padding-bottom:6px; }}
h3 {{ font-size:15px; margin:14px 0 6px; }} .muted {{ color:var(--muted); }}
.live a {{ display:inline-block; margin:6px 10px 0 0; padding:8px 12px; border:1px solid var(--line); border-radius:8px;
  background:var(--card); color:var(--accent); text-decoration:none; font-weight:600; }}
.ms {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px 16px; margin:10px 0; }}
.ms .when {{ color:var(--muted); font-size:13px; }} .ms p {{ margin:4px 0; }} .ms ul {{ margin:6px 0; padding-left:18px; }}
video {{ width:100%; max-width:640px; border-radius:8px; background:#000; display:block; margin-top:8px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(300px,1fr)); gap:12px; }}
figure {{ margin:0; }} figcaption {{ color:var(--muted); font-size:13px; margin-top:4px; }}
table {{ border-collapse:collapse; width:100%; max-width:700px; }} td,th {{ border-bottom:1px solid var(--line); padding:4px 8px; text-align:left; }}
</style></head><body><main>
<h1>Solar Sheep — the rover in Omniverse</h1>
<p class="muted">Local page, refreshed from the GPU box every few minutes. Updated {esc(time.strftime("%d %b %Y %H:%M"))}.</p>
<div class="live"><a href="http://localhost:8080">Live 3D world (Viser)</a><a href="http://localhost:6006">Training charts (TensorBoard)</a></div>
<p class="muted">The live links work only while the secure tunnel (scripts/gpu/watch.sh) is running.</p>
<h2>Milestones</h2>
{timeline}
<h2>Physics check against robot/SPEC.md</h2>
{measured}
<h2>Newest clips from every run</h2>
{videos}
</main></body></html>
"""
    out.write_text(page, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", default=str(REPO / "runs" / "gpu" / "runs"))
    ap.add_argument("--out", default=str(REPO / "runs" / "gpu" / "progress.html"))
    ap.add_argument("--milestones", default=str(REPO / "runs" / "gpu" / "milestones.json"))
    a = ap.parse_args()
    runs, out = Path(a.runs), Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    render(runs, out, Path(a.milestones))
    print(out)


if __name__ == "__main__":
    main()
