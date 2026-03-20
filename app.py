"""Baltic Salmon IBM — Shiny for Python Application (Lagoon Field Station theme)."""
from __future__ import annotations

import asyncio
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from shiny import App, reactive, render, ui

from shiny_deckgl import (
    CARTO_DARK,
    CARTO_POSITRON,
    CoordinateSystem,
    MapWidget,
    encode_binary_attribute,
    head_includes,
    layer,
    map_view,
    orthographic_view,
    path_layer,
    scatterplot_layer,
)

from salmon_ibm.config import load_config
from salmon_ibm.bioenergetics import BioParams
from salmon_ibm.simulation import Simulation
from heximpy.hxnparser import HexMap as HxnHexMap, GridMeta
from ui.sidebar import sidebar_panel
from ui.run_controls import run_controls_panel
from ui.science_tab import science_panel
from ui.viewer_tab import viewer_panel

# --- Plotly theme constants (Lagoon palette) ---
PLOT_BG = "rgba(11, 31, 44, 0.0)"
PAPER_BG = "rgba(19, 47, 62, 0.0)"
GRID_COLOR = "rgba(42, 122, 122, 0.15)"
AXIS_COLOR = "#6a8a8a"
TEXT_COLOR = "#e4e8e6"
ACCENT_COLOR = "#e8d5b7"
TEAL = "#3d9b8f"
SALMON_COLOR = "#d4826a"

# --- Plotly light theme constants ---
PLOT_BG_LIGHT = "rgba(248, 250, 249, 0.0)"
PAPER_BG_LIGHT = "rgba(255, 255, 255, 0.0)"
GRID_COLOR_LIGHT = "rgba(42, 122, 122, 0.1)"
AXIS_COLOR_LIGHT = "#4a7a7a"
TEXT_COLOR_LIGHT = "#1a3d50"
ACCENT_COLOR_LIGHT = "#8a7040"

# Behavior colors — nature-inspired
BEH_NAMES = ["Hold", "Random", "CWR", "Upstream", "Downstream"]
BEH_COLORS = ["#7a8b7a", "#4a8fa8", "#3d9b8f", "#d4826a", "#b8963e"]

# Bathymetric colorscale
BATHY_COLORSCALE = [
    [0.0, "#0b1f2c"],
    [0.2, "#1a3d50"],
    [0.4, "#2a7a7a"],
    [0.6, "#3d9b8f"],
    [0.8, "#7ac4a5"],
    [1.0, "#c8e6c9"],
]

TEMP_COLORSCALE = [
    [0.0, "#0b1f2c"],
    [0.15, "#1a3d50"],
    [0.3, "#2a7a7a"],
    [0.5, "#4a8fa8"],
    [0.7, "#e8d5b7"],
    [0.85, "#d4826a"],
    [1.0, "#b05a3f"],
]


def _base_layout(theme="dark", **overrides):
    """Shared Plotly layout for all charts."""
    is_light = theme == "light"
    _plot_bg = PLOT_BG_LIGHT if is_light else PLOT_BG
    _paper_bg = PAPER_BG_LIGHT if is_light else PAPER_BG
    _grid = GRID_COLOR_LIGHT if is_light else GRID_COLOR
    _axis = AXIS_COLOR_LIGHT if is_light else AXIS_COLOR
    _text = TEXT_COLOR_LIGHT if is_light else TEXT_COLOR
    layout = dict(
        plot_bgcolor=_plot_bg,
        paper_bgcolor=_paper_bg,
        font=dict(family="Work Sans, sans-serif", color=_text, size=12),
        margin=dict(l=48, r=16, t=36, b=44),
        xaxis=dict(
            gridcolor=_grid, zerolinecolor=_grid,
            tickfont=dict(size=10, color=_axis),
            title_font=dict(size=11, color=_axis),
        ),
        yaxis=dict(
            gridcolor=_grid, zerolinecolor=_grid,
            tickfont=dict(size=10, color=_axis),
            title_font=dict(size=11, color=_axis),
        ),
        legend=dict(
            font=dict(size=11, color=_text),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
        ),
    )
    layout.update(overrides)
    return layout


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    """Convert hex color '#rrggbb' to (r, g, b) integers."""
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _colorscale_rgb(z: np.ndarray, colorscale: list) -> np.ndarray:
    """Map array values → (N, 3) uint8 RGB using a Plotly-style colorscale."""
    z_min, z_max = float(np.nanmin(z)), float(np.nanmax(z))
    if z_max <= z_min:
        z_norm = np.zeros(len(z), dtype=np.float32)
    else:
        z_norm = ((z - z_min) / (z_max - z_min)).astype(np.float32)
    z_norm = np.clip(z_norm, 0, 1)
    stops = np.array([s[0] for s in colorscale], dtype=np.float32)
    colors = np.array([_hex_to_rgb(s[1]) for s in colorscale], dtype=np.float32)
    r = np.interp(z_norm, stops, colors[:, 0]).astype(np.uint8)
    g = np.interp(z_norm, stops, colors[:, 1]).astype(np.uint8)
    b = np.interp(z_norm, stops, colors[:, 2]).astype(np.uint8)
    return np.column_stack([r, g, b])


BEH_COLORS_RGB = [_hex_to_rgb(c) for c in BEH_COLORS]
BEH_COLORS_ARRAY = np.array(
    [list(_hex_to_rgb(c)) + [240] for c in BEH_COLORS], dtype=np.uint8
)

# Hex vertex offsets — pointy-top (vertex at top/bottom, matching HexSim display)
# Vertices at 30°,90°,...,330°
_HEX_ANGLES = np.arange(6) * (np.pi / 3) + (np.pi / 6)
_HEX_DX = np.cos(_HEX_ANGLES).astype(np.float64)
_HEX_DY = np.sin(_HEX_ANGLES).astype(np.float64)


def _build_hex_polygons(cx, cy, edge_scaled):
    """Compute hex vertex positions from centers.

    Returns flat (n*6, 2) float32 array and startIndices list.
    cx, cy: 1D arrays of center coordinates (already scaled/flipped).
    edge_scaled: hex edge length in the same coordinate units.
    """
    n = len(cx)
    # Vectorized: (n, 6) for each vertex
    vx = cx[:, None] + edge_scaled * _HEX_DX[None, :]
    vy = cy[:, None] + edge_scaled * _HEX_DY[None, :]
    verts = np.column_stack([vx.ravel(), vy.ravel()]).astype(np.float32)  # (n*6, 2)
    start_indices = (np.arange(n, dtype=np.int32) * 6).tolist()
    return verts, start_indices


def _subsample_indices(n, max_pts):
    """Return sorted random indices for subsampling large meshes."""
    if n > max_pts:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, max_pts, replace=False)
        idx.sort()
        return idx
    return np.arange(n)


# Geo-anchor profiles for meter→lat/lon conversion.
# Each profile defines two endpoints along the long axis (data_cols)
# and a meters-per-degree-latitude for cross-axis offset.
_GEO_ANCHORS = {
    "columbia": {
        # Mouth (Astoria, low data_cols) → Bonneville Dam (upstream, high data_cols)
        "start_lon": -123.83, "start_lat": 46.19,
        "end_lon": -121.94, "end_lat": 45.64,
        "m_per_deg_lat": 111_000.0,
    },
    "curonian": {
        # South end (Nemunas delta, low data_cols) → North end (Klaipėda strait, high data_cols)
        "start_lon": 21.20, "start_lat": 55.26,
        "end_lon": 21.08, "end_lat": 55.72,
        "m_per_deg_lat": 111_000.0,
    },
}
_DEFAULT_ANCHOR = "columbia"


def _detect_landscape(workspace_path: str | None = None) -> str:
    """Detect landscape from workspace path name."""
    if workspace_path:
        lc = str(workspace_path).lower()
        if "curonian" in lc or "lagoon" in lc:
            return "curonian"
    return "columbia"


def _hexsim_to_lonlat(cx, cy, landscape=None):
    """Convert HexSim meter coordinates to approximate lon/lat.

    cx is the long axis (data_cols direction).
    cy is the short axis (data_rows direction, cross-axis).
    Maps linearly between two anchor points defined per landscape,
    with cy adding a cross-axis latitude offset.
    """
    key = landscape or _DEFAULT_ANCHOR
    a = _GEO_ANCHORS.get(key, _GEO_ANCHORS[_DEFAULT_ANCHOR])
    cx_max = cx.max() if len(cx) > 0 else 1.0
    t = cx / cx_max  # 0 = start, 1 = end
    lon = a["start_lon"] + t * (a["end_lon"] - a["start_lon"])
    lat = (a["start_lat"] + t * (a["end_lat"] - a["start_lat"])
           + cy / a["m_per_deg_lat"])
    return lon, lat


def _build_water_binary(mesh, z, cscale, idx, scale=1.0, landscape=None):
    """Build binary-encoded position and color arrays for the water layer.

    Returns (positions_binary, colors_binary, n_points).
    HexSim meshes use grid coordinates scaled to pixel-like range for orthographic view.
    """
    rgb = _colorscale_rgb(z, cscale)
    is_hexsim = hasattr(mesh, '_edge')
    if is_hexsim:
        # Scale grid coordinates; negate Y to flip (row 0 = top of grid → bottom of screen)
        positions = np.column_stack([
            mesh.centroids[idx, 1] * scale,
            -mesh.centroids[idx, 0] * scale,
        ]).astype(np.float32)
    else:
        positions = np.column_stack([
            mesh.centroids[idx, 1] * scale,
            mesh.centroids[idx, 0] * scale,
        ]).astype(np.float32)
    colors = np.column_stack([
        rgb[idx, 0], rgb[idx, 1], rgb[idx, 2],
        np.full(len(idx), 220, dtype=np.uint8),
    ]).astype(np.uint8)
    return (
        encode_binary_attribute(positions),
        encode_binary_attribute(colors),
        len(idx),
    )


def _build_color_binary(mesh, z, cscale, idx):
    """Build only the color binary for a partial update."""
    rgb = _colorscale_rgb(z, cscale)
    colors = np.column_stack([
        rgb[idx, 0], rgb[idx, 1], rgb[idx, 2],
        np.full(len(idx), 220, dtype=np.uint8),
    ]).astype(np.uint8)
    return encode_binary_attribute(colors)


def _build_agent_data(sim, mesh, scale=1.0, landscape=None):
    """Build agent layer data: list of dicts with position, color, info."""
    alive = sim.pool.alive
    if not alive.any():
        return []

    tris = sim.pool.tri_idx[alive]
    behaviors = sim.pool.behavior[alive]
    energies = sim.pool.ed_kJ_g[alive]
    agent_ids = np.where(alive)[0]
    is_hexsim = hasattr(mesh, '_edge')

    pts = []
    for tri, beh, ed, aid in zip(tris, behaviors, energies, agent_ids):
        beh_int = int(beh)
        rc, gc, bc = BEH_COLORS_RGB[beh_int]
        if is_hexsim:
            # Scaled grid coordinates; negate Y to flip
            x = float(mesh.centroids[tri, 1] * scale)
            y = float(-mesh.centroids[tri, 0] * scale)
        else:
            x = float(mesh.centroids[tri, 1] * scale)
            y = float(mesh.centroids[tri, 0] * scale)
        pts.append({
            "position": [round(x, 2), round(y, 2)],
            "color": [rc, gc, bc, 240],
            "info": f"#{aid} {BEH_NAMES[beh_int]} | {ed:.1f} kJ/g",
        })
    return pts


def _build_agent_binary(sim, mesh, scale=1.0):
    """Build binary-encoded agent position + color arrays for deck.gl."""
    alive = sim.pool.alive
    if not alive.any():
        return None, None, 0
    tris = sim.pool.tri_idx[alive]
    behaviors = sim.pool.behavior[alive]
    is_hexsim = hasattr(mesh, '_edge')
    if is_hexsim:
        positions = np.column_stack([
            mesh.centroids[tris, 1] * scale,
            -mesh.centroids[tris, 0] * scale,
        ]).astype(np.float32)
    else:
        positions = np.column_stack([
            mesh.centroids[tris, 1] * scale,
            mesh.centroids[tris, 0] * scale,
        ]).astype(np.float32)
    colors = BEH_COLORS_ARRAY[behaviors]
    n = int(alive.sum())
    return encode_binary_attribute(positions), encode_binary_attribute(colors), n


class TrailBuffer:
    """NumPy rolling buffer for agent movement trails."""
    MAX_AGENTS = 2000
    TRAIL_LEN = 10

    def __init__(self):
        self._buf = np.zeros((self.MAX_AGENTS, self.TRAIL_LEN, 2), dtype=np.float32)
        self._ptr = 0
        self._fill = 0
        self._tracked = None

    def update(self, alive_mask, all_positions_xy):
        """Update trail buffer with positions indexed by global agent ID.

        all_positions_xy: (n_total, 2) array — positions for ALL agents,
        indexed by agent index (not just alive ones).
        """
        alive_idx = np.where(alive_mask)[0]
        n = len(alive_idx)
        if n == 0:
            return
        if self._tracked is None:
            if n <= self.MAX_AGENTS:
                self._tracked = alive_idx.copy()
            else:
                step = max(1, n // self.MAX_AGENTS)
                self._tracked = alive_idx[::step][:self.MAX_AGENTS]
            self._fill = 0
            self._ptr = 0
        nt = min(len(self._tracked), self.MAX_AGENTS)
        valid = self._tracked[:nt]
        # Filter to tracked agents that are still alive
        still_alive = alive_mask[valid]
        alive_tracked = valid[still_alive]
        if len(alive_tracked) > 0 and len(all_positions_xy) > 0:
            # Index into full position array by global agent ID
            m = min(len(alive_tracked), nt)
            self._buf[:m, self._ptr, :] = all_positions_xy[alive_tracked[:m]]
        self._ptr = (self._ptr + 1) % self.TRAIL_LEN
        self._fill = min(self._fill + 1, self.TRAIL_LEN)

    def clear(self):
        self._buf[:] = 0
        self._fill = 0
        self._ptr = 0
        self._tracked = None

    def build_paths(self):
        if self._fill < 2 or self._tracked is None:
            return []
        nt = min(len(self._tracked), self.MAX_AGENTS)
        order = [(self._ptr - self._fill + i) % self.TRAIL_LEN for i in range(self._fill)]
        paths = []
        for i in range(nt):
            trail = self._buf[i, order, :].tolist()
            if trail[0] == trail[-1]:
                continue
            paths.append({"path": trail, "color": [100, 180, 160, 100]})
        return paths


# Blank map style for non-geographic (orthographic) rendering
import json as _json
BLANK_STYLE = _json.dumps({"version": 8, "sources": {}, "layers": [
    {"id": "background", "type": "background",
     "paint": {"background-color": "#0b1f2c"}}
]})

# Static assets directory
WWW_DIR = Path(__file__).parent / "www"

# Max hex background points for deck.gl (200K → ~5.5 MB HTML, less banding)
MAX_DECK_POINTS = 200_000
MAX_HEX_POINTS = 500_000  # SolidPolygonLayer is more GPU-efficient

# --- shiny-deckgl map widget (shared by both landscapes) ---
TOOLTIP_STYLE = {
    "backgroundColor": "rgba(19, 47, 62, 0.92)",
    "color": "#e4e8e6",
    "fontSize": "11px",
    "fontFamily": "Work Sans, sans-serif",
    "borderRadius": "4px",
    "padding": "6px 10px",
    "border": "1px solid rgba(42, 122, 122, 0.3)",
}

map_widget = MapWidget(
    "map",
    view_state={"longitude": 0, "latitude": 0, "zoom": 1},
    style=BLANK_STYLE,
    tooltip={"html": "{info}", "style": TOOLTIP_STYLE},
    controller=True,
    parameters={"clearColor": [11/255, 31/255, 44/255, 1]},
)

viewer_map_widget = MapWidget(
    "viewer_map",
    view_state={"longitude": 0, "latitude": 0, "zoom": 1},
    style=BLANK_STYLE,
    tooltip={"html": "{info}", "style": TOOLTIP_STYLE},
    controller=True,
    parameters={"clearColor": [11/255, 31/255, 44/255, 1]},
)

# ── HexSim workspace discovery ───────────────────────────────────────────────

def _find_workspaces() -> dict[str, str]:
    """Find HexSim workspace directories (contain .grid files)."""
    base = Path(".")
    workspaces = {}
    for grid_file in base.rglob("*.grid"):
        ws_dir = grid_file.parent
        name = ws_dir.name
        workspaces[str(ws_dir)] = name
    return workspaces


def _list_hxn_layers(ws_path: str) -> dict[str, str]:
    """List .hxn layers in a workspace's Spatial Data/Hexagons/ folder."""
    ws = Path(ws_path)
    hex_dir = ws / "Spatial Data" / "Hexagons"
    layers = {}
    if hex_dir.exists():
        for subdir in sorted(hex_dir.iterdir()):
            if not subdir.is_dir():
                continue
            hxn_files = list(subdir.glob("*.hxn"))
            if hxn_files:
                layers[str(hxn_files[0])] = subdir.name
    # Also check for .hxn files directly in the workspace
    for hxn in sorted(ws.glob("*.hxn")):
        layers[str(hxn)] = hxn.stem
    return layers

# --- Theme toggle JS (flash prevention + toggle logic) ---
THEME_JS = """
(function() {
    var stored = localStorage.getItem('salmon-ibm-theme') || 'dark';
    document.documentElement.setAttribute('data-theme', stored);
})();

function toggleTheme() {
    var html = document.documentElement;
    var current = html.getAttribute('data-theme') || 'dark';
    var next = current === 'dark' ? 'light' : 'dark';
    html.setAttribute('data-theme', next);
    localStorage.setItem('salmon-ibm-theme', next);
    var btn = document.getElementById('theme-toggle-btn');
    if (btn) btn.textContent = next === 'dark' ? '\\u263E' : '\\u2600';
    if (window.Shiny) Shiny.setInputValue('theme_mode', next);
}

document.addEventListener('DOMContentLoaded', function() {
    var btn = document.getElementById('theme-toggle-btn');
    var theme = document.documentElement.getAttribute('data-theme') || 'dark';
    if (btn) btn.textContent = theme === 'dark' ? '\\u263E' : '\\u2600';
});

document.addEventListener('shiny:connected', function() {
    var theme = document.documentElement.getAttribute('data-theme') || 'dark';
    Shiny.setInputValue('theme_mode', theme);
});
"""

# --- App UI ---
app_ui = ui.page_sidebar(
    sidebar_panel(),
    ui.head_content(
        ui.tags.script(THEME_JS),
        ui.tags.link(rel="stylesheet", href="style.css"),
        ui.tags.meta(name="viewport", content="width=device-width, initial-scale=1"),
    ),
    head_includes(),
    run_controls_panel(),
    ui.navset_tab(
        ui.nav_panel(
            "Map",
            ui.div(
                ui.div(
                    map_widget.ui(height="520px"),
                    ui.output_ui("map_legend"),
                    style="position: relative;",
                ),
                class_="chart-card",
            ),
            value="map",
        ),
        ui.nav_panel(
            "Charts",
            ui.row(
                ui.column(
                    6,
                    ui.div(
                        ui.div("Population Survival", class_="chart-card-title"),
                        ui.output_ui("survival_plot"),
                        class_="chart-card",
                    ),
                ),
                ui.column(
                    6,
                    ui.div(
                        ui.div("Energy Reserve", class_="chart-card-title"),
                        ui.output_ui("energy_plot"),
                        class_="chart-card",
                    ),
                ),
            ),
            ui.row(
                ui.column(
                    12,
                    ui.div(
                        ui.div("Behavioral State Distribution", class_="chart-card-title"),
                        ui.output_ui("behavior_plot"),
                        class_="chart-card",
                    ),
                ),
            ),
            value="charts",
        ),
        ui.nav_panel(
            "Science",
            science_panel(),
            value="science",
        ),
        ui.nav_panel(
            "HexSim Viewer",
            viewer_panel(),
            value="viewer",
        ),
        id="main_tabs",
    ),
    title=ui.div(
        ui.tags.span(
            "Salmon IBM",
            class_="navbar-title",
        ),
        ui.tags.span(
            " \u2014 Individual-Based Migration Model",
            class_="navbar-subtitle",
        ),
        ui.tags.button(
            "\u263E",
            id="theme-toggle-btn",
            onclick="toggleTheme()",
            class_="theme-toggle",
        ),
        style="display: flex; align-items: center; width: 100%;",
    ),
)


# --- Server ---
def server(input, output, session):
    sim_state = reactive.Value(None)
    running = reactive.Value(False)
    history = reactive.Value([])
    trail_buffer = TrailBuffer()

    @reactive.effect
    @reactive.event(input.btn_reset, input.landscape, ignore_none=False)
    async def _init_sim():
        landscape = input.landscape()

        if landscape == "columbia":
            cfg = load_config("config_columbia.yaml")
        else:
            cfg = load_config("config_curonian_hexsim.yaml")

        # Apply estuary overrides from UI controls
        est = cfg.setdefault("estuary", {})
        est.setdefault("salinity_cost", {}).update({
            "S_opt": input.s_opt(), "S_tol": input.s_tol(), "k": input.sal_k(),
        })
        est.setdefault("do_avoidance", {}).update({
            "lethal": input.do_lethal(), "high": input.do_high(),
        })
        est.setdefault("seiche_pause", {}).update({
            "dSSHdt_thresh_m_per_15min": input.seiche_thresh(),
        })

        # Capture reactive values before entering thread
        n_agents = input.n_agents()
        rng_seed = input.rng_seed()
        ra, rb, rq = input.ra(), input.rb(), input.rq()
        ed_mortal = input.ed_mortal()
        t_opt, t_max = input.t_opt(), input.t_max()
        ed_init = input.ed_init()

        # Validate parameters
        errors = []
        if ra is not None and ra <= 0:
            errors.append("RA must be > 0")
        if rq is not None and rq <= 0:
            errors.append("RQ must be > 0")
        if t_max is not None and t_opt is not None and t_max <= t_opt:
            errors.append(f"T lethal ({t_max}) must be > T optimal ({t_opt})")
        if ed_init is not None and ed_mortal is not None and ed_init <= ed_mortal:
            errors.append(f"Init ED ({ed_init}) must be > Lethal ED ({ed_mortal})")
        if errors:
            ui.notification_show(" | ".join(errors), type="warning", duration=8)
            return

        # Run blocking init in a thread to avoid freezing the event loop
        sim = await asyncio.to_thread(
            Simulation, cfg, n_agents=n_agents, data_dir="data", rng_seed=rng_seed,
        )
        sim.bio_params = BioParams(
            RA=ra, RB=rb, RQ=rq, ED_MORTAL=ed_mortal,
            T_OPT=t_opt, T_MAX=t_max,
        )
        sim._activity_lut = sim._build_activity_lut()
        sim.pool.ed_kJ_g[:] = ed_init
        sim_state.set(sim)
        history.set([])
        running.set(False)
        trail_buffer.clear()

        # Update field selector: HexSim meshes have spatial data layers
        is_hexsim = hasattr(sim.mesh, "n_cells")
        if is_hexsim and hasattr(sim, 'landscape') and 'spatial_data' in sim.landscape:
            sd = sim.landscape["spatial_data"]
            choices = {name: name for name in sorted(sd.keys())}
            if not choices:
                choices = {"depth": "Grid Values"}
            ui.update_select("map_field", choices=choices)
        elif is_hexsim:
            # HexSim without scenario loader — use environment fields
            choices = {}
            if hasattr(sim.env, 'fields'):
                for k in sim.env.fields:
                    choices[k] = k.replace("_", " ").title()
            choices["depth"] = "Bathymetry"
            ui.update_select("map_field", choices=choices)
        else:
            ui.update_select("map_field", choices={
                "temperature": "Temperature",
                "salinity": "Salinity",
                "ssh": "Sea Surface Height",
                "depth": "Bathymetry",
            })

    @reactive.effect
    @reactive.event(input.btn_step)
    async def _step():
        try:
            sim = sim_state.get()
            if sim is None:
                await _init_sim()
                sim = sim_state.get()
            await asyncio.to_thread(sim.step)
            # Update trail buffer with full-population position array
            if sim.pool.alive.any():
                mesh = sim.mesh
                all_tris = sim.pool.tri_idx
                is_hex = hasattr(mesh, '_edge')
                scale = _cached_scale if _cached_scale not in (0, 1.0) and is_hex else (
                        80.0 / max(abs(mesh.centroids).max(), 1) if is_hex else 1.0)
                if is_hex:
                    all_xy = np.column_stack([mesh.centroids[all_tris, 1] * scale,
                                              -mesh.centroids[all_tris, 0] * scale]).astype(np.float32)
                else:
                    all_xy = np.column_stack([mesh.centroids[all_tris, 1],
                                              mesh.centroids[all_tris, 0]]).astype(np.float32)
                trail_buffer.update(sim.pool.alive, all_xy)
            history.set(sim.history.copy())
        except Exception as e:
            ui.notification_show(f"Step error: {e}", type="error", duration=10)

    @reactive.effect
    @reactive.event(input.btn_run)
    async def _run():
        running.set(True)
        try:
            sim = sim_state.get()
            if sim is None:
                await _init_sim()
                sim = sim_state.get()
            steps = input.n_steps()
            while running.get() and sim.current_t < steps:
                speed = input.speed()
                def _batch():
                    for _ in range(speed):
                        if sim.current_t >= steps:
                            break
                        sim.step()
                t_batch = time.perf_counter()
                await asyncio.to_thread(_batch)
                elapsed = time.perf_counter() - t_batch
                if sim.pool.alive.any():
                    mesh = sim.mesh
                    all_tris = sim.pool.tri_idx
                    is_hex = hasattr(mesh, '_edge')
                    scale = _cached_scale if _cached_scale not in (0, 1.0) and is_hex else (
                        80.0 / max(abs(mesh.centroids).max(), 1) if is_hex else 1.0)
                    if is_hex:
                        all_xy = np.column_stack([mesh.centroids[all_tris, 1] * scale,
                                                  -mesh.centroids[all_tris, 0] * scale]).astype(np.float32)
                    else:
                        all_xy = np.column_stack([mesh.centroids[all_tris, 1],
                                                  mesh.centroids[all_tris, 0]]).astype(np.float32)
                    trail_buffer.update(sim.pool.alive, all_xy)
                history.set(sim.history.copy())
                await asyncio.sleep(max(0.05, 0.25 - elapsed))
        except Exception as e:
            ui.notification_show(f"Simulation error: {e}", type="error", duration=10)
        finally:
            running.set(False)

    @reactive.effect
    @reactive.event(input.btn_pause)
    def _pause():
        running.set(False)

    _cached_theme = ""

    @reactive.effect
    @reactive.event(input.theme_mode)
    async def _update_theme():
        nonlocal _cached_theme
        theme = input.theme_mode()
        if not theme or theme == _cached_theme:
            return
        _cached_theme = theme

        is_light = theme == "light"

        # HexSim grids use blank style; only non-HexSim (TriMesh) uses basemap
        sim = sim_state.get()
        is_hexsim = sim is not None and hasattr(sim.mesh, '_edge')
        if is_hexsim:
            # Blank style bg color adapts via CSS, no basemap change needed
            pass
        else:
            style = CARTO_POSITRON if is_light else CARTO_DARK
            await map_widget.set_style(session, style)
        await viewer_map_widget.set_style(session, BLANK_STYLE)

        # Force full map rebuild if simulation is loaded
        if sim is not None:
            nonlocal _cached_subsample_idx, _cached_landscape
            _cached_subsample_idx = None
            _cached_landscape = ""  # force full update on next render

    @render.text
    def status_text():
        _ = history.get()  # re-render on each step
        sim = sim_state.get()
        if sim is None:
            return "Awaiting initialization"
        alive = int(sim.pool.alive.sum())
        total = sim.pool.n
        arrived = int(sim.pool.arrived.sum())
        return f"{alive}/{total} alive \u00b7 {arrived} arrived"

    @output
    @render.ui
    def live_stats():
        h = history.get()
        if not h:
            return ui.div(class_="live-stats-bar")
        last = h[-1]
        t = last.get("time", 0)
        n_alive = last.get("n_alive", 0)
        n_arrived = last.get("n_arrived", 0)
        beh = last.get("behavior_counts", {})
        return ui.div(
            ui.span(f"t = {t} h", class_="stat-val"),
            ui.span("|", class_="stat-sep"),
            ui.span(f"{n_alive:,} alive", class_="stat-val stat-alive"),
            ui.span("|", class_="stat-sep"),
            ui.span(f"{n_arrived:,} arrived", class_="stat-val"),
            ui.span("|", class_="stat-sep"),
            ui.span(
                f"\u2191{beh.get(3,0)} \u2193{beh.get(1,0)} \u2192{beh.get(0,0)} \u25cb{beh.get(4,0)}",
                class_="stat-val stat-behaviors",
            ),
            class_="live-stats-bar",
        )

    @render.text
    def progress_text():
        _ = history.get()  # re-render on each step
        sim = sim_state.get()
        if sim is None:
            return "t = 0 h"
        return f"t = {sim.current_t} h"

    @render.ui
    def map_legend():
        sim = sim_state.get()
        _ = history.get()
        if sim is None:
            return ui.HTML("")

        field_name = input.map_field()
        if field_name == "depth":
            cscale = BATHY_COLORSCALE
            cbar_title = "Depth (m)"
            z = sim.mesh.depth
        elif field_name in sim.env.fields:
            z = sim.env.fields[field_name]
            cscale = TEMP_COLORSCALE
            field_labels = {
                "temperature": "Temp (\u00b0C)",
                "salinity": "Sal (PSU)",
                "ssh": "SSH (m)",
            }
            cbar_title = field_labels.get(field_name, field_name)
        else:
            cscale = BATHY_COLORSCALE
            cbar_title = "Depth (m)"
            z = sim.mesh.depth

        z_min = f"{float(np.nanmin(z)):.1f}"
        z_max = f"{float(np.nanmax(z)):.1f}"
        gradient = ", ".join(f"{s[1]} {int(s[0]*100)}%" for s in cscale)

        # Grid type label
        is_hexsim = hasattr(sim.mesh, "n_cells")
        grid_label = f"Hexagonal Grid ({sim.mesh.n_cells:,} cells)" if is_hexsim else "Triangular Mesh"

        # Behavior chips + agent count (only if agents alive)
        beh_html = ""
        n_alive = int(sim.pool.alive.sum())
        if n_alive > 0:
            chips = []
            beh_counts = {}
            for b in range(5):
                beh_counts[b] = int((sim.pool.behavior[sim.pool.alive] == b).sum())
            for i, (name, color) in enumerate(zip(BEH_NAMES, BEH_COLORS)):
                count = beh_counts.get(i, 0)
                chips.append(
                    f'<span class="map-legend-beh-item">'
                    f'<span class="map-legend-beh-dot" style="background:{color}"></span>'
                    f'{name} ({count})</span>'
                )
            beh_html = (
                f'<div class="map-legend-section">Agents ({n_alive:,})</div>'
                f'<div class="map-legend-behaviors">{"".join(chips)}</div>'
            )

        return ui.HTML(
            f'<div class="map-legend">'
            f'<div class="map-legend-section">{grid_label}</div>'
            f'<div class="map-legend-title">{cbar_title}</div>'
            f'<div class="map-legend-bar" style="background:linear-gradient(to right,{gradient})"></div>'
            f'<div class="map-legend-range"><span>{z_min}</span><span>{z_max}</span></div>'
            f'{beh_html}'
            f'</div>'
        )

    # --- Map update: three-tier strategy ---
    # Cache state for change detection
    _cached_landscape = ""
    _cached_field = ""
    _cached_subsample_idx = None  # Columbia subsample / Curonian water indices (computed once)
    _cached_scale = 1.0
    _cached_water_n = 0

    # Dynamic fields whose colors change on env.advance()
    DYNAMIC_FIELDS = {"temperature", "salinity", "ssh"}

    def _resolve_field(sim):
        """Resolve current field → (z, colorscale)."""
        field_name = input.map_field()
        is_hexsim = hasattr(sim.mesh, "n_cells")

        # HexSim: check spatial_data dict first
        if is_hexsim and hasattr(sim, 'landscape') and 'spatial_data' in sim.landscape:
            sd = sim.landscape.get("spatial_data", {})
            if field_name in sd:
                return sd[field_name], TEMP_COLORSCALE

        # Standard fields
        if field_name == "depth":
            if hasattr(sim.mesh, 'depth'):
                return sim.mesh.depth, BATHY_COLORSCALE
            # HexSim meshes don't have depth — return zeros
            return np.zeros(sim.mesh.n_cells if is_hexsim else sim.mesh.n_triangles), BATHY_COLORSCALE
        elif field_name in sim.env.fields:
            return sim.env.fields[field_name], TEMP_COLORSCALE

        # Fallback: first available spatial data layer or temperature
        if is_hexsim and hasattr(sim, 'landscape'):
            sd = sim.landscape.get("spatial_data", {})
            if sd:
                first_key = next(iter(sd))
                return sd[first_key], TEMP_COLORSCALE
        if hasattr(sim.mesh, 'depth'):
            return sim.mesh.depth, BATHY_COLORSCALE
        return np.zeros(sim.mesh.n_cells if is_hexsim else 100), BATHY_COLORSCALE

    def _water_idx(sim):
        """Return subsample indices for the current mesh."""
        nonlocal _cached_subsample_idx
        mesh = sim.mesh
        is_hexsim = hasattr(mesh, "n_cells")
        if is_hexsim:
            if _cached_subsample_idx is None:
                n = mesh.n_cells
                if n > MAX_HEX_POINTS:
                    # Stride-based subsampling: preserves spatial structure
                    # (random subsampling scatters hexagons across the grid)
                    step = max(1, n // MAX_HEX_POINTS)
                    _cached_subsample_idx = np.arange(0, n, step)
                else:
                    _cached_subsample_idx = np.arange(n)
            return _cached_subsample_idx
        if _cached_subsample_idx is None:
            _cached_subsample_idx = np.where(mesh.water_mask)[0]
        return _cached_subsample_idx

    def _view_state(sim, landscape=None):
        """Return the appropriate view_state for the current landscape."""
        mesh = sim.mesh
        is_hexsim = hasattr(mesh, '_edge')
        if is_hexsim:
            cx = mesh.centroids[:, 1] * _cached_scale
            cy = -mesh.centroids[:, 0] * _cached_scale
            lon = float(cx.mean())
            lat = float(cy.mean())
            # Zoom so hexagons are visible (~4px per edge)
            edge_s = mesh._edge * _cached_scale
            min_hex_px = 4
            zoom_for_hex = float(np.log2(min_hex_px * 360 / (256 * max(edge_s, 1e-6))))
            extent = max(float(cx.max() - cx.min()), float(cy.max() - cy.min()), 0.01)
            zoom_for_extent = float(np.log2(360 / extent))
            zoom = max(zoom_for_hex, zoom_for_extent)
            return {"longitude": lon, "latitude": lat, "zoom": zoom}
        return {"longitude": 21.07, "latitude": 55.31, "zoom": 10}

    def _agent_layer(sim, landscape=None):
        """Build the complete agent ScatterplotLayer dict."""
        agent_data = _build_agent_data(sim, sim.mesh, scale=_cached_scale,
                                        landscape=landscape)
        return scatterplot_layer(
            "agents", agent_data,
            getPosition="@@d.position",
            getFillColor="@@d.color",
            getRadius=150,
            radiusMinPixels=5,
            radiusMaxPixels=12,
            stroked=True,
            getLineColor=[0, 0, 0, 140],
            lineWidthMinPixels=1,
            pickable=True,
        )

    _prev_agent_count = 0

    def _should_transition(sim):
        nonlocal _prev_agent_count
        n = int(sim.pool.alive.sum())
        changed = n != _prev_agent_count
        _prev_agent_count = n
        return not changed

    def _agent_layer_binary(sim, scale=1.0, use_transitions=False):
        pos_bin, col_bin, n = _build_agent_binary(sim, sim.mesh, scale=scale)
        is_hex = hasattr(sim.mesh, '_edge')
        if n == 0:
            return scatterplot_layer("agents", {"length": 0},
                getPosition=encode_binary_attribute(np.zeros((0, 2), dtype=np.float32)),
                getFillColor=encode_binary_attribute(np.zeros((0, 4), dtype=np.uint8)),
            )
        props = dict(
            getPosition=pos_bin,
            getFillColor=col_bin,
            getRadius=150 if not is_hex else 0.0001,
            radiusMinPixels=5,
            radiusMaxPixels=12,
            stroked=True,
            getLineColor=[0, 0, 0, 140],
            lineWidthMinPixels=1,
            pickable=True,
        )
        if use_transitions:
            props["transitions"] = {"getPosition": {"duration": 200, "type": "spring"}}
        return scatterplot_layer("agents", {"length": n}, **props)

    async def _full_update(sim, landscape=None):
        """Send everything: positions + colors + agents (~3 MB)."""
        nonlocal _cached_water_n
        z, cscale = _resolve_field(sim)
        idx = _water_idx(sim)
        pos_bin, col_bin, n = _build_water_binary(
            sim.mesh, z, cscale, idx, scale=_cached_scale,
            landscape=landscape,
        )
        _cached_water_n = n

        is_hexsim = hasattr(sim.mesh, "n_cells")
        if is_hexsim:
            # SolidPolygonLayer with pre-computed hex vertices
            mesh = sim.mesh
            cx = mesh.centroids[idx, 1] * _cached_scale
            cy = -mesh.centroids[idx, 0] * _cached_scale
            edge_s = mesh._edge * _cached_scale
            verts, start_idx = _build_hex_polygons(cx, cy, edge_s)
            rgb = _colorscale_rgb(z, cscale)
            colors = np.column_stack([
                rgb[idx, 0], rgb[idx, 1], rgb[idx, 2],
                np.full(n, 220, dtype=np.uint8),
            ]).astype(np.uint8)
            water = layer(
                "SolidPolygonLayer", "water",
                data={"length": n, "startIndices": start_idx},
                getPolygon=encode_binary_attribute(verts),
                getFillColor=encode_binary_attribute(colors),
                filled=True,
                extruded=False,
                pickable=False,
            )
        else:
            water = layer(
                "ScatterplotLayer", "water",
                data={"length": n},
                getPosition=pos_bin,
                getFillColor=col_bin,
                getRadius=30,
                radiusMinPixels=2,
                radiusMaxPixels=6,
                pickable=False,
            )
        await map_widget.update(
            session,
            layers=[water, _agent_layer_binary(sim, scale=_cached_scale)],
            view_state=_view_state(sim, landscape=landscape),
            views=[map_view(controller=True)],
        )

    async def _color_and_agent_update(sim, landscape=None):
        """Send new colors + agents."""
        is_hexsim = hasattr(sim.mesh, '_edge')
        if is_hexsim:
            # SolidPolygonLayer can't do partial color update; do full rebuild
            await _full_update(sim, landscape=landscape)
            return
        z, cscale = _resolve_field(sim)
        idx = _water_idx(sim)
        col_bin = _build_color_binary(sim.mesh, z, cscale, idx)
        layers = [
            {"id": "water", "getFillColor": col_bin,
             "data": {"length": _cached_water_n}},
            _agent_layer_binary(sim, scale=_cached_scale, use_transitions=_should_transition(sim)),
        ]
        if input.show_trails():
            trail_data = trail_buffer.build_paths()
            if trail_data:
                layers.insert(0, layer("PathLayer", "trails",
                    data=trail_data,
                    getPath="@@d.path",
                    getColor="@@d.color",
                    widthMinPixels=1,
                    widthMaxPixels=3,
                    jointRounded=True,
                    capRounded=True,
                ))
        else:
            # Explicitly hide trail layer (shallow merge keeps stale layers visible)
            layers.insert(0, {"id": "trails", "visible": False})
        await map_widget.partial_update(session, layers)

    async def _agent_only_update(sim, landscape=None):
        """Send only agents, water layer untouched in JS cache (~5 KB)."""
        layers = [_agent_layer_binary(sim, scale=_cached_scale, use_transitions=_should_transition(sim))]
        if input.show_trails():
            trail_data = trail_buffer.build_paths()
            if trail_data:
                layers.insert(0, layer("PathLayer", "trails",
                    data=trail_data,
                    getPath="@@d.path",
                    getColor="@@d.color",
                    widthMinPixels=1,
                    widthMaxPixels=3,
                    jointRounded=True,
                    capRounded=True,
                ))
        else:
            layers.insert(0, {"id": "trails", "visible": False})
        await map_widget.partial_update(session, layers)

    @reactive.effect
    async def _update_map():
        nonlocal _cached_landscape, _cached_field
        nonlocal _cached_subsample_idx, _cached_scale

        sim = sim_state.get()
        _ = history.get()
        if sim is None:
            return

        landscape = input.landscape()
        field_name = input.map_field()
        is_hexsim = hasattr(sim.mesh, "n_cells")
        if is_hexsim:
            # Compute scale to keep pseudo-lat/lon within ±80 degrees
            max_coord = max(abs(sim.mesh.centroids[:, 0]).max(),
                            abs(sim.mesh.centroids[:, 1]).max())
            scale = 80.0 / max(max_coord, 1)
        else:
            scale = 1.0

        # Detect what changed
        landscape_changed = landscape != _cached_landscape
        field_changed = field_name != _cached_field

        if landscape_changed:
            # Reset cache for new landscape
            _cached_landscape = landscape
            _cached_field = field_name
            _cached_subsample_idx = None
            _cached_scale = scale
            if is_hexsim:
                await map_widget.set_style(session, BLANK_STYLE)
            else:
                style = CARTO_POSITRON if _cached_theme == "light" else CARTO_DARK
                await map_widget.set_style(session, style)
            await _full_update(sim, landscape=landscape)
        elif field_changed:
            _cached_field = field_name
            _cached_scale = scale
            await _color_and_agent_update(sim, landscape=landscape)
        else:
            # Step only — choose path based on field dynamism
            _cached_scale = scale
            if field_name in DYNAMIC_FIELDS:
                await _color_and_agent_update(sim, landscape=landscape)
            else:
                await _agent_only_update(sim, landscape=landscape)

    # --- Chart helper: Plotly → HTML file + iframe (no widget/comm layer) ---
    def _plotly_iframe(fig, name, height="280px"):
        """Write a Plotly Figure to www/<name>.html and return an iframe.

        Script tags in innerHTML don't execute (browser security), so we
        can't use to_html() inline. Instead, write a standalone HTML file
        and serve it via iframe — same pattern as the deck.gl map.
        """
        html_str = pio.to_html(
            fig, include_plotlyjs="cdn", full_html=True,
            config={"displayModeBar": False, "responsive": True},
        )
        chart_path = WWW_DIR / f"{name}.html"
        chart_path.write_text(html_str, encoding="utf-8")
        ts = int(time.time() * 1000)
        return ui.tags.iframe(
            src=f"{name}.html?t={ts}",
            width="100%",
            height=height,
            style="border: none; border-radius: 8px; background: transparent;",
        )

    # --- Survival Plot ---
    @render.ui
    def survival_plot():
        h = history.get()
        theme = input.theme_mode() or "dark"
        # Throttle during Run: skip chart regeneration every 5th step
        if running.get() and h and len(h) > 5 and len(h) % 5 != 0:
            return ui.tags.iframe(src="survival.html", width="100%",
                height="280px", style="border:none;border-radius:8px;background:transparent;")
        fig = go.Figure()
        if not h:
            fig.update_layout(**_base_layout(theme=theme, height=280))
            return _plotly_iframe(fig, "survival")
        times = [r["time"] for r in h]
        alive = [r["n_alive"] for r in h]
        fig.add_trace(go.Scatter(
            x=times, y=alive, mode="lines",
            line=dict(color=TEAL, width=2.5),
            fill="tozeroy",
            fillcolor="rgba(61, 155, 143, 0.15)",
            name="Alive",
        ))
        fig.update_layout(**_base_layout(
            theme=theme,
            height=280,
            xaxis_title="Hour",
            yaxis_title="N alive",
            showlegend=False,
        ))
        return _plotly_iframe(fig, "survival")

    # --- Energy Plot ---
    @render.ui
    def energy_plot():
        h = history.get()
        theme = input.theme_mode() or "dark"
        if running.get() and h and len(h) > 5 and len(h) % 5 != 0:
            return ui.tags.iframe(src="energy.html", width="100%",
                height="280px", style="border:none;border-radius:8px;background:transparent;")
        fig = go.Figure()
        if not h:
            fig.update_layout(**_base_layout(theme=theme, height=280))
            return _plotly_iframe(fig, "energy")
        times = [r["time"] for r in h]
        ed = [r.get("mean_ed", 0) for r in h]
        accent = ACCENT_COLOR_LIGHT if theme == "light" else ACCENT_COLOR
        fig.add_trace(go.Scatter(
            x=times, y=ed, mode="lines",
            line=dict(color=accent, width=2.5),
            fill="tozeroy",
            fillcolor="rgba(232, 213, 183, 0.1)",
            name="Mean ED",
        ))
        fig.add_hline(
            y=4.0, line_dash="dot", line_color=SALMON_COLOR, line_width=1.5,
            annotation_text="Mortality",
            annotation_font=dict(size=10, color=SALMON_COLOR),
            annotation_position="top left",
        )
        fig.update_layout(**_base_layout(
            theme=theme,
            height=280,
            xaxis_title="Hour",
            yaxis_title="kJ/g",
            showlegend=False,
        ))
        return _plotly_iframe(fig, "energy")

    # --- Behavior Plot ---
    @render.ui
    def behavior_plot():
        h = history.get()
        theme = input.theme_mode() or "dark"
        if running.get() and h and len(h) > 5 and len(h) % 5 != 0:
            return ui.tags.iframe(src="behavior.html", width="100%",
                height="280px", style="border:none;border-radius:8px;background:transparent;")
        fig = go.Figure()
        if not h:
            fig.update_layout(**_base_layout(theme=theme, height=280))
            return _plotly_iframe(fig, "behavior")
        times = [r["time"] for r in h]
        for b in range(5):
            counts = [r.get("behavior_counts", {}).get(b, 0) for r in h]
            fig.add_trace(go.Scatter(
                x=times, y=counts, mode="lines", name=BEH_NAMES[b],
                stackgroup="one",
                line=dict(color=BEH_COLORS[b], width=0.5),
                fillcolor=BEH_COLORS[b],
            ))
        fig.update_layout(**_base_layout(
            theme=theme,
            height=280,
            xaxis_title="Hour",
            yaxis_title="Count",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
                font=dict(size=10),
            ),
        ))
        return _plotly_iframe(fig, "behavior")


    # ── HexSim Viewer tab ─────────────────────────────────────────────────
    _viewer_initialized = False

    @reactive.effect
    async def _init_viewer_workspaces():
        """Populate workspace dropdown on startup."""
        workspaces = await asyncio.to_thread(_find_workspaces)
        if workspaces:
            ui.update_select("viewer_workspace", choices=workspaces)

    @reactive.effect
    @reactive.event(input.viewer_workspace)
    async def _update_viewer_layers():
        """Update layer dropdown when workspace changes."""
        ws = input.viewer_workspace()
        if not ws:
            return
        layers = await asyncio.to_thread(_list_hxn_layers, ws)
        ui.update_select("viewer_layer", choices=layers)

    @render.ui
    def viewer_map_container():
        return ui.div(
            viewer_map_widget.ui(height="520px"),
            ui.output_ui("viewer_legend"),
            style="position: relative;",
        )

    @reactive.effect
    @reactive.event(input.viewer_load)
    async def _load_viewer_layer():
        nonlocal _viewer_initialized
        hxn_path = input.viewer_layer()
        ws_path = input.viewer_workspace()
        if not hxn_path:
            ui.notification_show("No layer selected", type="warning")
            return

        try:
            hm = await asyncio.to_thread(HxnHexMap.from_file, hxn_path)
        except Exception as e:
            ui.notification_show(f"Read error: {e}", type="error", duration=10)
            return

        values = hm.values.astype(np.float32)

        # Determine water cells (non-zero)
        water_mask = values != 0.0
        water_flat = np.where(water_mask)[0]
        n_water = len(water_flat)

        if n_water == 0:
            ui.notification_show("No non-zero cells found", type="warning")
            return

        # Get georef from .grid file
        ws = Path(ws_path)
        grid_files = list(ws.glob("*.grid"))
        if grid_files:
            gm = GridMeta.from_file(grid_files[0])
            edge = gm.edge
            grid_x_extent = gm.x_extent
            grid_y_extent = gm.y_extent
        elif hm.format == "plain":
            edge = hm.cell_size if hm.cell_size > 0 else 13.876
            grid_x_extent = hm.width * 1.5 * edge
            grid_y_extent = hm.height * np.sqrt(3.0) * edge
        else:
            edge = 13.876
            grid_x_extent = hm.height * 1.5 * edge
            grid_y_extent = hm.width * np.sqrt(3.0) * edge

        # Build row/col arrays properly (handles narrow grids where odd
        # rows have width-1 cells — simple flat//width is WRONG for those).
        # This matches hxn_viewer.py _hex_centers() exactly.
        h, w = hm.height, hm.width
        if hm.flag == 0:
            # Wide grid: all rows have w cells
            all_rows = np.repeat(np.arange(h), w)
            all_cols = np.tile(np.arange(w), h)
        else:
            # Narrow grid: even rows have w cells, odd rows have w-1
            row_list, col_list = [], []
            for r in range(h):
                rw = w if r % 2 == 0 else w - 1
                row_list.append(np.full(rw, r, dtype=np.int32))
                col_list.append(np.arange(rw, dtype=np.int32))
            all_rows = np.concatenate(row_list)
            all_cols = np.concatenate(col_list)

        # Filter to non-zero (water) cells
        data_rows = all_rows[water_flat]
        data_cols = all_cols[water_flat]
        water_values = values[water_flat]

        # Hex center coordinates (matches hxnparser.HexMap.hex_to_xy)
        # Pointy-top spacing (verified against HexSim 4.0.20)
        cx = np.sqrt(3.0) * edge * (data_cols.astype(np.float64) + 0.5 * (data_rows % 2))
        cy = 1.5 * edge * data_rows.astype(np.float64)

        # Subsample for deck.gl — stride-based to preserve spatial tiling
        max_pts = 500_000
        if n_water > max_pts:
            step = max(1, n_water // max_pts)
            idx = np.arange(0, n_water, step)
        else:
            idx = np.arange(n_water)

        # Scale to keep pseudo-lat/lon within ±80 degrees
        max_coord = max(abs(cy).max(), abs(cx).max(), 1)
        viewer_scale = 80.0 / max_coord
        sx = cx[idx] * viewer_scale
        sy = -cy[idx] * viewer_scale
        n_pts = len(idx)

        # Hex polygon vertices
        edge_s = edge * viewer_scale
        verts, start_idx = _build_hex_polygons(sx, sy, edge_s)

        # Color by value
        z = water_values[idx]
        rgb = _colorscale_rgb(z, TEMP_COLORSCALE)
        colors = np.column_stack([
            rgb[:, 0], rgb[:, 1], rgb[:, 2],
            np.full(n_pts, 220, dtype=np.uint8),
        ]).astype(np.uint8)

        center_x = float(np.mean(sx))
        center_y = float(np.mean(sy))
        # Compute zoom so hexagons are visible (~4 pixels per edge)
        # At zoom Z, 1 degree ≈ 256 * 2^Z / 360 pixels
        # For edge_s degrees to be 4px: zoom = log2(4 * 360 / (256 * edge_s))
        min_hex_pixels = 4
        zoom_for_hex = float(np.log2(min_hex_pixels * 360 / (256 * max(edge_s, 1e-6))))
        # Also compute zoom to fit extent
        extent = max(float(sx.max() - sx.min()), float(sy.max() - sy.min()), 0.01)
        zoom_for_extent = float(np.log2(360 / extent))
        # Use the LARGER zoom (closer in) — show hex detail, user can zoom out
        zoom = max(zoom_for_hex, zoom_for_extent)

        water_layer = layer(
            "SolidPolygonLayer", "viewer_water",
            data={"length": n_pts, "startIndices": start_idx},
            getPolygon=encode_binary_attribute(verts),
            getFillColor=encode_binary_attribute(colors),
            filled=True,
            extruded=False,
            pickable=False,
        )

        try:
            await viewer_map_widget.set_style(session, BLANK_STYLE)
            await viewer_map_widget.update(
                session,
                layers=[water_layer],
                view_state={"longitude": center_x, "latitude": center_y, "zoom": float(zoom)},
                views=[map_view(controller=True)],
            )
            _viewer_initialized = True
            ui.notification_show(
                f"Loaded {n_pts:,} hexagons (zoom={zoom:.1f})",
                type="message", duration=5,
            )
        except Exception as e:
            ui.notification_show(f"Map update failed: {e}", type="error", duration=10)
            import traceback
            traceback.print_exc()

        # Store metadata for display
        _viewer_meta.set({
            "format": hm.format,
            "version": hm.version,
            "ncols": hm.height,   # data rows = grid ncols (short axis)
            "nrows": hm.width,    # data stride = grid nrows (long axis)
            "total_cells": hm.n_hexagons,
            "water_cells": n_water,
            "cell_size": hm.cell_size if hm.cell_size > 0 else None,
            "edge": edge,
            "origin": hm.origin,
            "nodata": hm.nodata,
            "vmin": float(np.nanmin(water_values)),
            "vmax": float(np.nanmax(water_values)),
            "vmean": float(np.nanmean(water_values)),
            "n_unique": int(len(np.unique(water_values))),
            "layer_name": Path(hxn_path).parent.name,
        })

    _viewer_meta = reactive.Value({})

    @render.ui
    def viewer_metadata():
        meta = _viewer_meta.get()
        if not meta:
            return ui.HTML(
                '<div class="param-hint">Select a workspace and layer, '
                'then click Load Layer.</div>'
            )

        origin = meta.get("origin", (0, 0))
        cell_size_str = f'{meta["cell_size"]:.3f} m' if meta["cell_size"] else "from .grid"
        hex_area = (3.0 * np.sqrt(3.0) / 2.0) * meta["edge"] ** 2

        return ui.HTML(f'''
            <div class="viewer-meta">
                <strong class="viewer-meta-title">{meta["layer_name"]}</strong><br>
                <span class="viewer-meta-label">Format:</span> {meta["format"]}<br>
                <span class="viewer-meta-label">Version:</span> {meta["version"]}<br>
                <span class="viewer-meta-label">Grid:</span> {meta["ncols"]} &times; {meta["nrows"]}<br>
                <span class="viewer-meta-label">Total hexagons:</span> {meta["total_cells"]:,}<br>
                <span class="viewer-meta-label">Water cells:</span> {meta["water_cells"]:,}
                    ({100*meta["water_cells"]/meta["total_cells"]:.1f}%)<br>
                <span class="viewer-meta-label">Hex edge:</span> {meta["edge"]:.3f} m<br>
                <span class="viewer-meta-label">Hex area:</span> {hex_area:.1f} m&sup2;<br>
                <span class="viewer-meta-label">Cell size:</span> {cell_size_str}<br>
                <span class="viewer-meta-label">Origin:</span> ({origin[0]:.1f}, {origin[1]:.1f})<br>
                <hr class="viewer-meta-hr">
                <span class="viewer-meta-label">Values:</span><br>
                &nbsp; min={meta["vmin"]:.2f} &nbsp; max={meta["vmax"]:.2f}<br>
                &nbsp; mean={meta["vmean"]:.2f} &nbsp; unique={meta["n_unique"]}<br>
            </div>
        ''')

    @render.ui
    def viewer_legend():
        meta = _viewer_meta.get()
        if not meta:
            return ui.HTML("")
        gradient = ", ".join(f"{s[1]} {int(s[0]*100)}%" for s in TEMP_COLORSCALE)
        return ui.HTML(
            f'<div class="map-legend">'
            f'<div class="map-legend-title">{meta["layer_name"]}</div>'
            f'<div class="map-legend-bar" style="background:linear-gradient(to right,{gradient})"></div>'
            f'<div class="map-legend-range"><span>{meta["vmin"]:.1f}</span><span>{meta["vmax"]:.1f}</span></div>'
            f'</div>'
        )


app = App(app_ui, server, static_assets=str(Path(__file__).parent / "www"))
