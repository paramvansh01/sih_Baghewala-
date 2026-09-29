"""Render artifacts/report.md to artifacts/index.html and serve artifacts/ on localhost."""
from __future__ import annotations

import functools
import http.server
import socketserver

import markdown

from .config import ROOT

A = ROOT / "artifacts"

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Baghewala Digital Twin</title>
<style>
:root {{ --bg:#ffffff; --fg:#1d1d1f; --muted:#5f6368; --line:#e3e3e6; --head:#f4f5f7; --accent:#1a5fb4; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg:#16181c; --fg:#e8e8ea; --muted:#a0a4ab; --line:#2f333a; --head:#1f2228; --accent:#8ab4f8; }}
  img {{ background:#fff; }}
}}
body {{ margin:0; background:var(--bg); color:var(--fg);
       font:15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
main {{ max-width:1100px; margin:0 auto; padding:24px 16px 64px; }}
h1 {{ font-size:1.7rem; margin:.2em 0 .6em; }}
h2 {{ font-size:1.25rem; margin:2em 0 .6em; padding-top:.6em; border-top:1px solid var(--line); }}
a {{ color:var(--accent); }}
code {{ font:13px/1.4 ui-monospace, SFMono-Regular, Menlo, monospace; background:var(--head);
        padding:1px 4px; border-radius:4px; }}
.table-wrap {{ overflow-x:auto; margin:1em 0; }}
table {{ border-collapse:collapse; font-size:13px; }}
th, td {{ border:1px solid var(--line); padding:5px 8px; text-align:left; vertical-align:top; }}
th {{ background:var(--head); }}
td {{ font-variant-numeric:tabular-nums; }}
img {{ max-width:100%; height:auto; border:1px solid var(--line); border-radius:6px; margin:.6em 0; }}
</style></head>
<body><main>
{body}
</main></body></html>
"""


def build_html():
    md = (A / "report.md").read_text()
    body = markdown.markdown(md, extensions=["tables"])
    body = body.replace("<table>", '<div class="table-wrap"><table>').replace("</table>", "</table></div>")
    out = A / "index.html"
    out.write_text(PAGE.format(body=body))
    return out


def _port_in_use(port):
    import socket
    for host in ("127.0.0.1", "::1"):
        fam = socket.AF_INET6 if ":" in host else socket.AF_INET
        with socket.socket(fam, socket.SOCK_STREAM) as s:
            s.settimeout(0.3)
            if s.connect_ex((host, port)) == 0:
                return True
    return False


def serve(port=8790):
    if _port_in_use(port):
        raise SystemExit(f"port {port} is already used by another program; pass --port <free port>")
    build_html()
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(A))
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", port), handler) as httpd:
        print(f"serving http://localhost:{port}/  (Ctrl+C to stop)", flush=True)
        httpd.serve_forever()
