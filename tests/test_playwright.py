"""Playwright browser tests for the Salmon IBM Shiny app.

Requires a running app on http://localhost:8123.
Run with:  micromamba run -n shiny python -m pytest tests/test_playwright.py -v --headed
"""
import re
import subprocess
import time

import pytest
from playwright.sync_api import Page, expect

APP_URL = "http://localhost:8123"
# Columbia init takes ~12s in a background thread
COLUMBIA_INIT_TIMEOUT = 30_000
STEP_TIMEOUT = 15_000


@pytest.fixture(scope="module", autouse=True)
def _ensure_app_running():
    """Skip the whole module if the Shiny app is not reachable."""
    import urllib.request
    try:
        urllib.request.urlopen(APP_URL, timeout=5)
    except Exception:
        pytest.skip(f"Shiny app not running at {APP_URL}")


# ---------- App load & basic UI ----------

def test_app_loads(page: Page):
    """App loads, title bar and sidebar are present."""
    page.goto(APP_URL)
    # Title bar has both <span> and <strong> with "Salmon IBM" — use the strong tag
    expect(page.get_by_role("strong")).to_be_visible(timeout=10_000)
    expect(page.locator("text=Individual-Based Migration Model")).to_be_visible()


def test_sidebar_controls_present(page: Page):
    """Sidebar contains landscape selector and key parameter inputs."""
    page.goto(APP_URL)
    expect(page.locator("#landscape")).to_be_visible(timeout=10_000)
    expect(page.locator("#n_agents")).to_be_visible()
    expect(page.locator("#n_steps")).to_be_visible()
    expect(page.locator("#map_field")).to_be_visible()


def test_run_controls_present(page: Page):
    """Run/Step/Pause/Reset buttons are present."""
    page.goto(APP_URL)
    for btn_id in ["btn_run", "btn_step", "btn_pause", "btn_reset"]:
        expect(page.locator(f"#{btn_id}")).to_be_visible(timeout=10_000)


# ---------- Curonian Lagoon ----------

def test_curonian_map_renders(page: Page):
    """Curonian Lagoon: deck.gl canvas appears after Step."""
    page.goto(APP_URL)
    page.wait_for_selector("#landscape", timeout=10_000)
    # Default landscape is curonian — click Step to init + render
    page.click("#btn_step")
    # deck.gl renders into a canvas inside the map widget
    # Map widget container is present once shiny-deckgl initializes
    map_div = page.locator("#map")
    expect(map_div).to_be_visible(timeout=STEP_TIMEOUT)


def test_curonian_status_updates(page: Page):
    """After stepping, status text shows alive count."""
    page.goto(APP_URL)
    page.wait_for_selector("#btn_step", timeout=10_000)
    page.click("#btn_step")
    status = page.locator(".status-badge")
    expect(status).to_contain_text(re.compile(r"\d+/\d+ alive"), timeout=STEP_TIMEOUT)


def test_curonian_progress_updates(page: Page):
    """After stepping, progress text shows t = 1 h."""
    page.goto(APP_URL)
    page.wait_for_selector("#btn_step", timeout=10_000)
    page.click("#btn_step")
    progress = page.locator("text=t = 1 h")
    expect(progress).to_be_visible(timeout=STEP_TIMEOUT)


def test_curonian_legend_visible(page: Page):
    """Legend overlay appears after stepping."""
    page.goto(APP_URL)
    page.wait_for_selector("#btn_step", timeout=10_000)
    page.click("#btn_step")
    legend = page.locator(".map-legend")
    expect(legend).to_be_visible(timeout=STEP_TIMEOUT)


def test_curonian_charts_render(page: Page):
    """Charts tab: survival, energy, behavior iframes load after stepping."""
    page.goto(APP_URL)
    page.wait_for_selector("#btn_step", timeout=10_000)
    page.click("#btn_step")
    # Wait for status to confirm step completed
    expect(page.locator(".status-badge")).to_contain_text(
        re.compile(r"\d+/\d+ alive"), timeout=STEP_TIMEOUT,
    )
    # Switch to Charts tab
    page.click("text=Charts")
    # Plotly iframes should appear
    for name in ["survival", "energy", "behavior"]:
        iframe = page.locator(f'iframe[src*="{name}.html"]')
        expect(iframe).to_be_visible(timeout=10_000)


# ---------- Columbia River ----------

def test_columbia_landscape_switch(page: Page):
    """Switching to Columbia River initializes HexMesh simulation."""
    page.goto(APP_URL)
    page.wait_for_selector("#landscape", timeout=10_000)
    page.select_option("#landscape", "columbia")
    # Columbia init takes ~12s — wait for status to update
    status = page.locator(".status-badge")
    expect(status).to_contain_text(
        re.compile(r"\d+/\d+ alive"), timeout=COLUMBIA_INIT_TIMEOUT,
    )


def test_columbia_map_renders(page: Page):
    """Columbia River: deck.gl canvas appears after landscape switch + Step."""
    page.goto(APP_URL)
    page.wait_for_selector("#landscape", timeout=10_000)
    page.select_option("#landscape", "columbia")
    # Wait for init to complete
    expect(page.locator(".status-badge")).to_contain_text(
        re.compile(r"\d+/\d+ alive"), timeout=COLUMBIA_INIT_TIMEOUT,
    )
    page.click("#btn_step")
    # Map widget container is present once shiny-deckgl initializes
    map_div = page.locator("#map")
    expect(map_div).to_be_visible(timeout=STEP_TIMEOUT)


def test_columbia_run_advances_progress(page: Page):
    """Columbia River: Run button advances simulation time."""
    page.goto(APP_URL)
    page.wait_for_selector("#landscape", timeout=10_000)
    page.select_option("#landscape", "columbia")
    expect(page.locator(".status-badge")).to_contain_text(
        re.compile(r"\d+/\d+ alive"), timeout=COLUMBIA_INIT_TIMEOUT,
    )
    # Use Run (not Step) — single Step can get swallowed during
    # the ~11s binary widget.update() for 200K water cells.
    page.click("#btn_run")
    progress = page.locator('[id="progress_text"]')
    expect(progress).to_contain_text(
        re.compile(r"t = [1-9]"), timeout=COLUMBIA_INIT_TIMEOUT,
    )
    page.click("#btn_pause")


# ---------- Map field switching ----------

def test_map_field_switch(page: Page):
    """Changing the map color field updates the legend title."""
    page.goto(APP_URL)
    page.wait_for_selector("#btn_step", timeout=10_000)
    page.click("#btn_step")
    expect(page.locator(".map-legend")).to_be_visible(timeout=STEP_TIMEOUT)
    # Default is temperature — switch to depth
    page.select_option("#map_field", "depth")
    legend_title = page.locator(".map-legend-title")
    expect(legend_title).to_contain_text("Depth", timeout=10_000)


# ---------- Run / Pause ----------

def test_run_and_pause(page: Page):
    """Run button advances multiple steps; Pause stops it."""
    page.goto(APP_URL)
    page.wait_for_selector("#btn_step", timeout=10_000)
    # First step to initialize the simulation
    page.click("#btn_step")
    expect(page.locator(".status-badge")).to_contain_text(
        re.compile(r"\d+/\d+ alive"), timeout=STEP_TIMEOUT,
    )
    page.click("#btn_run")
    # Wait for progress to advance beyond t = 1 h
    progress = page.locator('[id="progress_text"]')
    expect(progress).to_contain_text(re.compile(r"t = [2-9]"), timeout=STEP_TIMEOUT)
    page.click("#btn_pause")
    # After pause, verify simulation stopped: progress text should settle on a
    # value and then not change. expect().to_have_text with a 2s timeout waits
    # for the text to remain stable.
    progress.wait_for(state="visible", timeout=5000)
    text1 = progress.inner_text()
    # If simulation is truly paused, text1 stays equal for 2s.
    expect(progress).to_have_text(text1, timeout=2000)
    text2 = progress.inner_text()
    assert text1 == text2, f"Simulation didn't pause: {text1} → {text2}"


# ---------- Reset ----------

def test_reset_reinitializes(page: Page):
    """Reset button reinitializes the simulation to t=0."""
    page.goto(APP_URL)
    page.wait_for_selector("#btn_step", timeout=10_000)
    page.click("#btn_step")
    expect(page.locator("text=t = 1 h")).to_be_visible(timeout=STEP_TIMEOUT)
    page.click("#btn_reset")
    expect(page.locator("text=t = 0 h")).to_be_visible(timeout=STEP_TIMEOUT)
