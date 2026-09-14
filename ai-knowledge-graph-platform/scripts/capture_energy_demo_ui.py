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
        # The demo database may already contain a prior request from an
        # earlier capture. Create one only when the panel is empty, then show
        # a single concrete remediation case at readable scale.
        if page.locator(".remediation-request").count() == 0:
            page.get_by_role("button", name="Request telemetry from Snowflake").first.click()
            page.wait_for_timeout(150)
            form = page.locator(".remediation-form:not([hidden])")
            form.get_by_label("Owner").fill("ops-team")
            form.get_by_label("Reason").fill("Confirm gearbox temperature via Snowflake export")
            form.get_by_role("button", name="Submit request").click()
            page.wait_for_timeout(400)

        # Keep the capture honest but legible: it shows the actual no-inference
        # boundary, one target asset, its request action, and one persisted
        # governed request. The unmodified UI still lists every affected asset.
        page.locator(".remediation-asset").evaluate_all(
            "items => items.slice(1).forEach(item => item.hidden = true)"
        )
        page.locator(".remediation-request").evaluate_all(
            "items => items.slice(1).forEach(item => item.hidden = true)"
        )
        page.locator("#evidence-remediation").screenshot(
            path=str(OUT / "dashboard_evidence_remediation.png")
        )

        # Publication and quarantine audit panel: loads client-side via
        # fetch() on page load, so wait for it to leave its initial
        # "Loading…" state before capturing. By default the demo fixture is
        # fully conformant (the empty-quarantine state), which is itself the
        # required capture. Start the server with
        # ENERGY_DEMO_INCLUDE_INVALID_FIXTURE=1 to additionally capture a
        # real quarantined-record example -- this script only detects and
        # captures whatever the running server actually publishes, it can't
        # flip that flag itself. Confirmed live: the durable governance store
        # dedupes a publish by the *published* graph's own content hash, and
        # the invalid record never reaches that graph (it's quarantined
        # first) -- so against an artifacts/energy/ store that already has a
        # prior publish for this tenant, the flag alone won't show a new
        # quarantined record. Also set ENERGY_GOVERNANCE_DB_URL to a fresh
        # sqlite path (or start from a clean artifacts/energy/) when you need
        # the populated example, not just the flag.
        page.wait_for_function(
            "document.getElementById('publication-summary')?.textContent"
            ".indexOf('Loading') === -1"
        )
        # Publication history (the rollback timeline) loads independently
        # via its own fetch -- wait for it too, so the same panel capture
        # below never catches it mid-"Loading…".
        page.wait_for_function(
            "document.getElementById('publication-history')?.textContent"
            ".indexOf('Loading') === -1"
        )
        page.locator("#publication-audit").screenshot(
            path=str(OUT / "dashboard_publication_audit.png")
        )
        # The demo UI has no rollback control (rollback stays an
        # authorised API/admin operation) -- so by default the timeline
        # shows a single, non-rollback entry. Note that rather than
        # silently capturing a one-line timeline as if it were the full
        # feature.
        if page.locator("#publication-history li").count() < 2:
            print(
                "Publication history has no rollback entry in this run -- "
                "captured the single-publication timeline only. Seed a "
                "durable rollback (GovernanceStore.rollback(), an "
                "authorised admin operation) against the running server's "
                "store for a populated timeline example."
            )
        if page.locator(".quarantine-record").count() > 0:
            page.locator(".quarantine-record").first.locator("summary").click()
            page.wait_for_timeout(150)
            page.locator("#publication-audit").screenshot(
                path=str(OUT / "dashboard_publication_audit_quarantined.png")
            )
        else:
            print(
                "No quarantined records in this run -- captured the empty "
                "state only. Set ENERGY_DEMO_INCLUDE_INVALID_FIXTURE=1 on the "
                "server for a populated example."
            )
        browser.close()

    print(f"Captured Energy UI screenshots in {OUT}")


if __name__ == "__main__":
    main()
