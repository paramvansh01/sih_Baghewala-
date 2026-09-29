"""Local web server for the live digital twin (standard library only).

  GET  /                 live dashboard
  GET  /api/stream       server-sent events: one JSON snapshot per simulated day
  GET  /api/snapshot     latest snapshot (JSON)
  POST /api/select       {"well": "BGW-S01", "fleet": "twin"|"practice"}
  POST /api/control      {"running": bool, "seconds_per_day": float}
  POST /api/fault        {"well": "...", "fault": "none"|preset}   (card simulation only)
  GET  /report/          the batch results report (artifacts/)
"""
from __future__ import annotations

import json
import socket
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np

from ..config import ROOT
from .engine import FLEETS, LiveField

STATIC = Path(__file__).parent / "static"
ARTIFACTS = ROOT / "artifacts"


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o).__name__)


def _dumps(obj):
    # NaN/inf are not valid JSON for the browser's JSON.parse -> send null
    return json.dumps(_clean(obj), default=_default, allow_nan=False)


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return [_clean(v) for v in o.tolist()]
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    return o


def make_handler(field: LiveField):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(ARTIFACTS), **kw)

        def log_message(self, *_):
            pass

        def _json(self, obj, code=200):
            body = _dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _body(self):
            n = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(n) or b"{}")

        def do_GET(self):
            route = urlsplit(self.path).path  # ignore query strings when routing
            if route in ("/", "/index.html"):
                body = (STATIC / "dashboard.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif route == "/api/snapshot":
                with field.lock:
                    snap = field.snapshot
                self._json(snap or {})
            elif route == "/api/stream":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                seen = -1
                try:
                    while True:
                        with field.new_snapshot:
                            field.new_snapshot.wait_for(lambda: field.version != seen, timeout=15)
                            snap, v = field.snapshot, field.version
                        if v == seen or snap is None:
                            self.wfile.write(b": keep-alive\n\n")
                        else:
                            seen = v
                            self.wfile.write(b"data: " + _dumps(snap).encode() + b"\n\n")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return
            elif route == "/report" or route.startswith("/report/"):
                self.path = "/index.html" if route in ("/report", "/report/") else route[len("/report"):]
                super().do_GET()
            else:
                super().do_GET()  # figures/... for the report

        def do_POST(self):
            try:
                data = self._body()
            except json.JSONDecodeError:
                return self._json(dict(error="bad json"), 400)
            route = urlsplit(self.path).path
            if route == "/api/select":
                ids = {w.id for w in field.fleets["twin"]}
                if data.get("well") not in ids or data.get("fleet") not in FLEETS:
                    return self._json(dict(error="unknown well or fleet"), 400)
                field.selected = dict(well=data["well"], fleet=data["fleet"])
            elif route == "/api/control":
                if "running" in data:
                    field.running = bool(data["running"])
                if "seconds_per_day" in data:
                    lo, hi = field.lcfg["seconds_per_day_range"]
                    field.seconds_per_day = float(min(max(float(data["seconds_per_day"]), lo), hi))
            elif route == "/api/fault":
                if data.get("fault") not in ["none"] + list(field.lcfg["fault_presets"]):
                    return self._json(dict(error="unknown fault"), 400)
                field.faults[data.get("well")] = data["fault"]
            else:
                return self._json(dict(error="not found"), 404)
            field.refresh()
            self._json(dict(ok=True))

    return Handler


def port_in_use(port):
    for fam, host in ((socket.AF_INET, "127.0.0.1"), (socket.AF_INET6, "::1")):
        with socket.socket(fam, socket.SOCK_STREAM) as s:
            s.settimeout(0.3)
            if s.connect_ex((host, port)) == 0:
                return True
    return False


def run(port):
    if port_in_use(port):
        raise SystemExit(f"port {port} is already used by another program; pass --port <free port>")
    from ..serve import build_html
    if (ARTIFACTS / "report.md").exists():
        build_html()
    print("loading models and initialising 2 x 34 wells from end-of-history state ...", flush=True)
    field = LiveField()
    field.tick()
    threading.Thread(target=field.run_forever, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(field))
    srv.daemon_threads = True
    print(f"live twin: http://127.0.0.1:{port}/   report: http://127.0.0.1:{port}/report/", flush=True)
    srv.serve_forever()
