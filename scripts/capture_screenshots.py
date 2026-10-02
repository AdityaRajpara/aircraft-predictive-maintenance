"""Capture dashboard screenshots for the README (dev tool; needs `pip install playwright`
and `playwright install chromium`). Run `make predict` first."""
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

PORT = 8765
OUT = Path("docs/images")
SHOTS = [
    # (tab, file, element to wait for, viewport height). Streamlit scrolls inside its own
    # container, so full_page does not help; size the viewport to the content instead.
    ("Fleet overview", "dashboard_fleet.png", ".js-plotly-plot:visible", 800),
    ("Maintenance queue", "dashboard_maintenance.png", "[data-testid='stDataFrame']", 900),
    ("Engine drill-down", "dashboard_engine.png", ".js-plotly-plot:visible", 1400),
]


def wait_up(url: str, timeout: float = 30) -> None:
    end = time.time() + timeout
    while time.time() < end:
        try:
            urllib.request.urlopen(url, timeout=2)
            return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("streamlit did not start")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    srv = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "dashboard/app.py", "--server.headless", "true",
         "--server.port", str(PORT), "--browser.gatherUsageStats", "false"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        wait_up(f"http://localhost:{PORT}")
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(
                viewport={"width": 1200, "height": 800}, color_scheme="light", device_scale_factor=2
            )
            page.goto(f"http://localhost:{PORT}")
            page.get_by_role("tab", name="Fleet overview").wait_for()
            for tab, filename, ready, height in SHOTS:
                page.set_viewport_size({"width": 1200, "height": height})
                page.get_by_role("tab", name=tab).click()
                page.locator(ready).first.wait_for(state="visible", timeout=20000)
                page.wait_for_timeout(1500)  # let charts finish animating
                page.screenshot(path=str(OUT / filename))
                print("saved", OUT / filename)
            browser.close()
    finally:
        srv.terminate()


if __name__ == "__main__":
    main()
