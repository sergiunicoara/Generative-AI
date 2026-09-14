"""Capture the live GraphDB SPARQL result used by the client teaser."""

from __future__ import annotations

from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "presentation" / "energy_demo_ui_capture" / "graphdb_sparql_wt01.png"
BASE = "http://127.0.0.1:7200"
QUERY = """PREFIX energy: <https://example.energy.demo/ontology#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?asset ?label ?component ?workOrder WHERE {
  ?asset rdfs:label ?label ; energy:hasComponent ?component .
  ?workOrder energy:concernsAsset ?asset .
  FILTER(?asset = <https://example.energy.demo/asset/WT-01>)
}"""


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1)
        page.goto(f"{BASE}/sparql#user", wait_until="domcontentloaded", timeout=15_000)
        page.wait_for_timeout(10_000)
        repository = page.get_by_text("energy-demo", exact=True).first
        if repository.is_visible():
            repository.click()
            page.wait_for_timeout(4_000)
        editor = page.locator(".CodeMirror").first
        editor.wait_for(timeout=15_000)
        editor.evaluate("(node, query) => node.CodeMirror.setValue(query)", QUERY)
        page.get_by_role("button", name="Run", exact=True).click()
        page.get_by_text("Showing results from", exact=False).wait_for(timeout=15_000)
        page.screenshot(path=str(OUT), full_page=False)
        browser.close()
    print(OUT)


if __name__ == "__main__":
    main()
