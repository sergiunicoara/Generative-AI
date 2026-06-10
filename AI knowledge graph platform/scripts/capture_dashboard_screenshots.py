"""Capture admin-dashboard screenshots for the performance scorecard.

Reproducible asset generation: starts the admin dashboard in demo mode (zero
backend required), drives a headless Chromium via Playwright, visits each tab,
and writes tight PNGs into ``docs/assets/``.

Usage:
    python scripts/capture_dashboard_screenshots.py

If a dashboard is already serving on the target port it is reused; otherwise a
demo-mode uvicorn server is started and torn down automatically.

Requires: ``pip install playwright`` then ``playwright install chromium``.
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"
PORT = int(os.environ.get("DASHBOARD_PORT", "8001"))
BASE = f"http://127.0.0.1:{PORT}/admin/"

# (tab label as shown in the UI, output filename slug)
TABS = [
    ("Graph Health",  "dashboard-graph-health"),
    ("Conflicts",     "dashboard-conflicts"),
    ("Communities",   "dashboard-communities"),
    ("GDPR & PII",    "dashboard-gdpr"),
    ("Calibration",   "dashboard-calibration"),
]

# Tight content selector (the dashboard root inside Dash's react entry point).
CONTENT_SELECTOR = "#react-entry-point > div"


def _port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _start_server() -> subprocess.Popen | None:
    if _port_open(PORT):
        print(f"[capture] reusing dashboard already serving on :{PORT}")
        return None
    print(f"[capture] starting demo-mode dashboard on :{PORT} ...")
    env = {**os.environ, "GRAPHRAG_DASHBOARD_DEMO": "1"}
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:app",
         "--host", "127.0.0.1", "--port", str(PORT)],
        cwd=str(ROOT), env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    for _ in range(60):
        if _port_open(PORT):
            time.sleep(1.5)  # let the ASGI app finish wiring routes
            print("[capture] dashboard is up")
            return proc
        time.sleep(0.5)
    proc.terminate()
    raise RuntimeError(f"dashboard did not come up on :{PORT}")


def main() -> int:
    ASSETS.mkdir(parents=True, exist_ok=True)
    proc = _start_server()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1440, "height": 900},
                                    device_scale_factor=2)
            # Warm up — load the page, set tenant to 'aerospace', then click a
            # non-default tab so Dash initialises all callbacks before we capture.
            page.goto(BASE, wait_until="domcontentloaded")
            page.wait_for_timeout(1500)
            # Switch tenant to 'aerospace' via the debounced input
            try:
                tenant_input = page.locator("#tenant-input")
                tenant_input.click(timeout=3000)
                page.keyboard.press("Control+A")
                page.keyboard.type("aerospace")
                page.keyboard.press("Tab")   # blur triggers debounce callback
                page.wait_for_timeout(3000)  # let all tab callbacks re-fire
            except Exception:
                pass
            try:
                page.get_by_text("Conflicts", exact=True).first.click(timeout=5000)
                page.wait_for_timeout(2000)
            except Exception:
                pass

            for label, slug in TABS:
                page.goto(BASE, wait_until="domcontentloaded")
                page.wait_for_timeout(1500)   # let Dash JS init
                # Click the target tab.
                try:
                    page.get_by_text(label, exact=True).first.click(timeout=5000)
                except Exception:
                    pass
                page.wait_for_timeout(2500)  # let the tab content swap in
                # Plotly charts only exist on some tabs; wait if present.
                try:
                    page.wait_for_selector(".js-plotly-plot", timeout=8000)
                    # wait for all Plotly charts to finish rendering their SVG
                    page.wait_for_selector(".js-plotly-plot .main-svg", timeout=5000)
                except Exception:
                    pass
                page.wait_for_timeout(3500)  # let charts/tables finish sizing
                out = ASSETS / f"{slug}.png"
                el = page.query_selector(CONTENT_SELECTOR)
                (el or page).screenshot(path=str(out))
                print(f"[capture] wrote {out.relative_to(ROOT)}")
            browser.close()
    finally:
        if proc is not None:
            proc.terminate()
            print("[capture] stopped demo dashboard")
    print("[capture] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
