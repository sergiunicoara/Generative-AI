"""Capture the client-facing Energy demo UI from the running local API."""

from __future__ import annotations

import os
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "presentation" / "energy_demo_ui_capture"
BASE = os.environ.get("ENERGY_DEMO_URL", "http://127.0.0.1:8000")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1)
        page.goto(f"{BASE}/auth/dev-login?next=/energy-demo", wait_until="networkidle")
        page.screenshot(path=str(OUT / "dashboard_current.png"), full_page=True)

        page.locator("button").nth(1).click()
        page.wait_for_timeout(250)
        page.screenshot(path=str(OUT / "dashboard_insufficient_evidence.png"), full_page=True)

        page.locator("summary").click()
        page.wait_for_timeout(150)
        page.screenshot(path=str(OUT / "dashboard_technical_trace.png"), full_page=True)

        # Text-based selectors here (not `.nth()`): the remediation panel's
        # own buttons come after everything the two captures above rely on,
        # but stay off positional indices for its own markup too, so future
        # panel changes don't silently shift what gets captured.
        page.get_by_role("button", name="Request telemetry from Snowflake").first.click()
        page.wait_for_timeout(150)
        page.get_by_label("Owner").fill("ops-team")
        page.get_by_label("Reason").fill("Confirm gearbox temperature via Snowflake export")
        page.get_by_role("button", name="Submit request").click()
        page.wait_for_timeout(400)
        page.screenshot(path=str(OUT / "dashboard_evidence_remediation.png"), full_page=True)
        browser.close()

    print(f"Captured Energy UI screenshots in {OUT}")


if __name__ == "__main__":
    main()
