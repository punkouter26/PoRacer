r"""Live training dashboard. Serves a page that polls the running job's
progress.json, so you can watch a run without tailing a log.

    ..\.venv\Scripts\python.exe mjx_training\dashboard.py
    ..\.venv\Scripts\python.exe mjx_training\dashboard.py --run walk03 --port 8765

Then open http://localhost:8765. With no --run it follows the most recently
modified directory under runs/, so it picks up whatever is training now.
"""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

RUNS = Path(__file__).resolve().parent / "runs"

PAGE = r"""<!doctype html>
<meta charset="utf-8">
<title>Creature training</title>
<style>
  :root {
    color-scheme: dark;
    --bg: #0e1116; --panel: #171b22; --line: #262c36;
    --fg: #e6edf3; --dim: #8b949e;
    --a: #4ea1ff; --b: #3fb950; --c: #d29922;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--fg);
         font: 14px/1.5 ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif; }
  header { padding: 20px 24px 8px; }
  h1 { margin: 0; font-size: 18px; font-weight: 600; letter-spacing: -.01em; }
  .sub { color: var(--dim); font-size: 13px; margin-top: 2px; }
  .wrap { padding: 12px 24px 32px; max-width: 1100px; }
  .tiles { display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); margin-bottom: 18px; }
  .tile { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 12px 14px; }
  .tile .k { color: var(--dim); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }
  .tile .v { font-size: 26px; font-weight: 600; font-variant-numeric: tabular-nums; margin-top: 4px; }
  .tile .u { font-size: 13px; color: var(--dim); font-weight: 400; }
  .card { background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
          padding: 14px 16px 8px; margin-bottom: 14px; }
  .card h2 { margin: 0 0 6px; font-size: 13px; font-weight: 600; color: var(--dim);
             text-transform: uppercase; letter-spacing: .04em; }
  canvas { width: 100%; height: 170px; display: block; }
  .bar { height: 6px; background: #21262d; border-radius: 3px; overflow: hidden; margin-top: 10px; }
  .bar > i { display: block; height: 100%; background: var(--a); border-radius: 3px; transition: width .4s; }
  .foot { color: var(--dim); font-size: 12px; margin-top: 14px; }
  .dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; background: var(--b);
         margin-right: 6px; vertical-align: 1px; }
  .dot.off { background: #6e7681; }
</style>
<header>
  <h1>Creature training <span id="run" style="color:var(--dim);font-weight:400"></span></h1>
  <div class="sub"><span class="dot" id="dot"></span><span id="status">connecting…</span></div>
</header>
<div class="wrap">
  <div class="tiles">
    <div class="tile"><div class="k">Speed</div><div class="v"><span id="t-speed">—</span><span class="u"> m/s</span></div></div>
    <div class="tile"><div class="k">Reward</div><div class="v" id="t-reward">—</div></div>
    <div class="tile"><div class="k">Episode length</div><div class="v"><span id="t-len">—</span><span class="u"> /1000</span></div></div>
    <div class="tile"><div class="k">Steps</div><div class="v" id="t-steps">—</div></div>
    <div class="tile"><div class="k">Elapsed</div><div class="v" id="t-time">—</div></div>
  </div>
  <div class="bar"><i id="prog" style="width:0%"></i></div>
  <div class="foot" id="eta"></div>

  <div class="card"><h2>Forward speed (m/s)</h2><canvas id="c-speed"></canvas></div>
  <div class="card"><h2>Episode reward</h2><canvas id="c-reward"></canvas></div>
  <div class="card"><h2>Episode length (steps before falling)</h2><canvas id="c-len"></canvas></div>
</div>
<script>
const TOTAL = 60000000;

function draw(id, pts, color) {
  const cv = document.getElementById(id);
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth, h = cv.clientHeight;
  cv.width = w * dpr; cv.height = h * dpr;
  const x = cv.getContext('2d'); x.scale(dpr, dpr);
  x.clearRect(0, 0, w, h);
  if (!pts.length) return;

  const pad = {l: 44, r: 8, t: 10, b: 18};
  const ys = pts.map(p => p[1]);
  let lo = Math.min(...ys, 0), hi = Math.max(...ys);
  if (hi === lo) hi = lo + 1;
  const xs = pts.map(p => p[0]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs) || 1;
  const px = v => pad.l + (w - pad.l - pad.r) * ((v - x0) / ((x1 - x0) || 1));
  const py = v => pad.t + (h - pad.t - pad.b) * (1 - (v - lo) / (hi - lo));

  // gridlines + y labels
  x.strokeStyle = '#262c36'; x.fillStyle = '#8b949e';
  x.font = '11px ui-monospace, monospace'; x.lineWidth = 1;
  for (let i = 0; i <= 3; i++) {
    const v = lo + (hi - lo) * i / 3, y = Math.round(py(v)) + .5;
    x.beginPath(); x.moveTo(pad.l, y); x.lineTo(w - pad.r, y); x.stroke();
    x.fillText(v >= 100 ? v.toFixed(0) : v.toFixed(2), 4, y + 3);
  }
  // area + line
  const g = x.createLinearGradient(0, pad.t, 0, h - pad.b);
  g.addColorStop(0, color + '55'); g.addColorStop(1, color + '00');
  x.beginPath(); x.moveTo(px(pts[0][0]), py(pts[0][1]));
  pts.forEach(p => x.lineTo(px(p[0]), py(p[1])));
  x.lineTo(px(pts[pts.length - 1][0]), h - pad.b); x.lineTo(px(pts[0][0]), h - pad.b);
  x.closePath(); x.fillStyle = g; x.fill();

  x.beginPath(); x.moveTo(px(pts[0][0]), py(pts[0][1]));
  pts.forEach(p => x.lineTo(px(p[0]), py(p[1])));
  x.strokeStyle = color; x.lineWidth = 2; x.lineJoin = 'round'; x.stroke();

  const last = pts[pts.length - 1];
  x.beginPath(); x.arc(px(last[0]), py(last[1]), 3.5, 0, 7); x.fillStyle = color; x.fill();
}

const fmt = n => n.toLocaleString();
function hms(s) {
  s = Math.round(s);
  const m = Math.floor(s / 60);
  return m >= 60 ? `${Math.floor(m/60)}h ${m%60}m` : `${m}m ${String(s%60).padStart(2,'0')}s`;
}

async function tick() {
  let d;
  try {
    const r = await fetch('data?_=' + Date.now());
    d = await r.json();
  } catch (e) {
    document.getElementById('status').textContent = 'server unreachable';
    document.getElementById('dot').className = 'dot off';
    return;
  }
  document.getElementById('run').textContent = d.run ? '· ' + d.run : '';
  const ev = d.evals || [];
  if (!ev.length) {
    document.getElementById('status').textContent = 'waiting for first eval (JIT compiling)…';
    document.getElementById('dot').className = 'dot off';
    return;
  }
  const e = ev[ev.length - 1];
  document.getElementById('status').textContent =
      `${ev.length} evals · live` + (d.running ? '' : ' · process not running');
  document.getElementById('dot').className = 'dot' + (d.running ? '' : ' off');

  document.getElementById('t-speed').textContent = e.speed.toFixed(3);
  document.getElementById('t-reward').textContent = e.reward.toFixed(0);
  document.getElementById('t-len').textContent = e.ep_len.toFixed(0);
  document.getElementById('t-steps').textContent = fmt(e.steps);
  document.getElementById('t-time').textContent = hms(e.elapsed);

  const pct = Math.min(100, 100 * e.steps / TOTAL);
  document.getElementById('prog').style.width = pct + '%';
  const rate = e.steps / Math.max(e.elapsed, 1);
  document.getElementById('eta').textContent = e.steps > 0
      ? `${pct.toFixed(0)}% of ${fmt(TOTAL)} steps · ${fmt(Math.round(rate))} steps/s · ~${hms((TOTAL - e.steps) / rate)} remaining`
      : '';

  draw('c-speed',  ev.map(p => [p.steps, p.speed]),  '#4ea1ff');
  draw('c-reward', ev.map(p => [p.steps, p.reward]), '#3fb950');
  draw('c-len',    ev.map(p => [p.steps, p.ep_len]), '#d29922');
}
tick(); setInterval(tick, 4000);
addEventListener('resize', tick);
</script>
"""


def latest_run(explicit: str | None) -> Path | None:
  if explicit:
    p = RUNS / explicit
    return p if p.exists() else None
  candidates = [d for d in RUNS.glob("*") if (d / "progress.json").exists()]
  if not candidates:
    # a run that has started but not produced its first eval yet
    candidates = [d for d in RUNS.glob("*") if d.is_dir()]
  return max(candidates, key=lambda d: d.stat().st_mtime) if candidates else None


def build_handler(run_name: str | None):
  class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # keep the console quiet
      pass

    def _send(self, body: bytes, ctype: str):
      self.send_response(200)
      self.send_header("Content-Type", ctype)
      self.send_header("Cache-Control", "no-store")
      self.send_header("Content-Length", str(len(body)))
      self.end_headers()
      self.wfile.write(body)

    def do_GET(self):
      if self.path.startswith("/data"):
        run = latest_run(run_name)
        payload = {"run": run.name if run else None, "evals": [], "running": False}
        if run:
          pj = run / "progress.json"
          if pj.exists():
            try:
              raw = json.loads(pj.read_text())
            except json.JSONDecodeError:
              raw = []  # mid-write; the next poll will catch it
            for e in raw:
              L = max(e.get("episode_length", 1), 1)
              payload["evals"].append({
                  "steps": e.get("steps", 0),
                  "reward": e.get("reward", 0.0),
                  "ep_len": e.get("episode_length", 0.0),
                  "speed": e.get("eval/episode_x_velocity", 0.0) / L,
                  "elapsed": e.get("elapsed_s", 0.0),
              })
            # "running" == progress.json touched in the last 10 minutes
            import time
            payload["running"] = (time.time() - pj.stat().st_mtime) < 600
        self._send(json.dumps(payload).encode(), "application/json")
      else:
        self._send(PAGE.encode("utf-8"), "text/html; charset=utf-8")

  return Handler


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--run", default=None, help="run name (default: most recent)")
  ap.add_argument("--port", type=int, default=8765)
  args = ap.parse_args()

  run = latest_run(args.run)
  print(f"following: {run if run else '(no run yet)'}")
  print(f"dashboard: http://localhost:{args.port}   (Ctrl+C to stop)")
  HTTPServer(("127.0.0.1", args.port), build_handler(args.run)).serve_forever()


if __name__ == "__main__":
  main()
