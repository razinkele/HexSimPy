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
    MapWidget,
    encode_binary_attribute,
    h3_hexagon_layer,
    head_includes,
    layer,
    loading_widget,
    map_view,
    reset_view_widget,
    trips_layer,
)
import h3

from salmon_ibm.config import load_config
from salmon_ibm.bioenergetics import BioParams
from salmon_ibm.simulation import Simulation
from heximpy.hxnparser import HexMap as HxnHexMap, GridMeta
from ui.sidebar import sidebar_panel
from ui.science_tab import science_panel
from ui.viewer_tab import viewer_panel
from ui.charts_panel import charts_panel

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
    [0.0, "#c8e6c9"],
    [0.2, "#7ac4a5"],
    [0.4, "#50b8a8"],
    [0.6, "#3a9090"],
    [0.8, "#2a6070"],
    [1.0, "#1a3d50"],
]

TEMP_COLORSCALE = [
    [0.0, "#1a3d50"],
    [0.15, "#2a6070"],
    [0.3, "#3a9090"],
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
            gridcolor=_grid,
            zerolinecolor=_grid,
            tickfont=dict(size=10, color=_axis),
            title_font=dict(size=11, color=_axis),
        ),
        yaxis=dict(
            gridcolor=_grid,
            zerolinecolor=_grid,
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
def _build_water_binary(mesh, z, cscale, idx, scale=1.0, landscape=None):
    """Build binary-encoded position and color arrays for the water layer.

    Returns (positions_binary, colors_binary, n_points).
    HexSim meshes use grid coordinates scaled to pixel-like range for orthographic view.
    """
    rgb = _colorscale_rgb(z, cscale)
    is_hexsim = hasattr(mesh, "_edge")
    if is_hexsim:
        # Scale grid coordinates; negate Y to flip (row 0 = top of grid → bottom of screen)
        positions = np.column_stack(
            [
                mesh.centroids[idx, 1] * scale,
                -mesh.centroids[idx, 0] * scale,
            ]
        ).astype(np.float32)
    else:
        positions = np.column_stack(
            [
                mesh.centroids[idx, 1] * scale,
                mesh.centroids[idx, 0] * scale,
            ]
        ).astype(np.float32)
    colors = np.column_stack(
        [
            rgb[idx, 0],
            rgb[idx, 1],
            rgb[idx, 2],
            np.full(len(idx), 220, dtype=np.uint8),
        ]
    ).astype(np.uint8)
    return (
        encode_binary_attribute(positions),
        encode_binary_attribute(colors),
        len(idx),
    )


def _build_color_binary(mesh, z, cscale, idx):
    """Build only the color binary for a partial update."""
    rgb = _colorscale_rgb(z, cscale)
    colors = np.column_stack(
        [
            rgb[idx, 0],
            rgb[idx, 1],
            rgb[idx, 2],
            np.full(len(idx), 220, dtype=np.uint8),
        ]
    ).astype(np.uint8)
    return encode_binary_attribute(colors)


class TripBuffer:
    """Accumulates agent positions as [lon, lat, timestamp] paths for TripsLayer."""

    MAX_AGENTS = 2000
    TRAIL_LEN = 40  # max waypoints kept per agent

    def __init__(self):
        # (MAX_AGENTS, TRAIL_LEN, 3) — lon, lat, timestamp
        self._buf = np.zeros((self.MAX_AGENTS, self.TRAIL_LEN, 3), dtype=np.float32)
        self._beh = np.zeros(self.MAX_AGENTS, dtype=np.int8)  # latest behavior per slot
        self._ptr = 0  # next write slot (ring buffer)
        self._fill = 0  # how many slots have been written
        self._tracked = None  # global agent indices we track
        self._time = 0  # simulation timestamp counter

    def update(self, alive_mask, all_positions_xy, behaviors=None):
        """Append current positions as a new timestep.

        all_positions_xy: (n_total, 2) array — [lon, lat] for ALL agents.
        behaviors: (n_total,) int array — behavior index per agent (optional).
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
                self._tracked = alive_idx[::step][: self.MAX_AGENTS]
            self._fill = 0
            self._ptr = 0
        nt = min(len(self._tracked), self.MAX_AGENTS)
        valid = self._tracked[:nt]
        still_alive = alive_mask[valid]
        # Write into correct buffer slots using the original slot indices
        # (not a compact 0..m range, which would misalign after deaths)
        alive_slots = np.where(still_alive)[0]
        dead_slots = np.where(~still_alive)[0]
        alive_agents = valid[alive_slots]
        # Mark dead agent slots with NaN so build_trips can filter them
        if len(dead_slots) > 0:
            self._buf[dead_slots, self._ptr, :] = np.nan
        if len(alive_agents) > 0 and len(all_positions_xy) > 0:
            self._buf[alive_slots, self._ptr, 0] = all_positions_xy[alive_agents, 0]
            self._buf[alive_slots, self._ptr, 1] = all_positions_xy[alive_agents, 1]
            self._buf[alive_slots, self._ptr, 2] = self._time
            if behaviors is not None:
                self._beh[alive_slots] = behaviors[alive_agents]
        self._ptr = (self._ptr + 1) % self.TRAIL_LEN
        self._fill = min(self._fill + 1, self.TRAIL_LEN)
        self._time += 1

    def clear(self):
        self._buf[:] = 0
        self._beh[:] = 0
        self._fill = 0
        self._ptr = 0
        self._tracked = None
        self._time = 0

    @property
    def current_time(self):
        return self._time

    def build_trips(self):
        """Build TripsLayer-compatible data: list of {path, timestamps, color}.

        Path contains ``[lon, lat, time]`` triplets (matching the format that
        ``shiny_deckgl.format_trips()`` produces).  Timestamps are normalized
        to start from 0 so they align with the JS ``currentTime`` range.
        """
        if self._fill < 2 or self._tracked is None:
            return [], 0, 0
        nt = min(len(self._tracked), self.MAX_AGENTS)
        order = np.array(
            [(self._ptr - self._fill + i) % self.TRAIL_LEN for i in range(self._fill)]
        )
        # Compute t_min from the step counter (not from a buffer slot that
        # may be stale if agent 0 died).
        t_min = float(self._time - self._fill)
        t_max = float(self._time - 1)

        # Bulk extract: (nt, fill, 3) — vectorized NaN/stationarity check
        all_wp = self._buf[:nt, order, :]
        has_nan = np.any(np.isnan(all_wp), axis=2)  # (nt, fill)
        valid_counts = (~has_nan).sum(axis=1)  # (nt,)

        # Stationarity: compare all positions to first valid position
        first_valid = np.argmax(~has_nan, axis=1)  # (nt,)
        ref = all_wp[np.arange(nt), first_valid, :2]  # (nt, 2)
        same_pos = np.where(
            has_nan,
            True,
            (all_wp[:, :, 0] == ref[:, 0:1]) & (all_wp[:, :, 1] == ref[:, 1:2]),
        )
        stationary = np.all(same_pos, axis=1)

        # Filter: >= 2 valid waypoints AND not stationary
        keep_idx = np.where((valid_counts >= 2) & ~stationary)[0]

        # Pre-compute colors (clamp behavior index)
        beh_indices = np.clip(self._beh[:nt], 0, len(BEH_COLORS) - 1)
        # Cache RGB tuples to avoid re-parsing hex strings
        _color_cache = [_hex_to_rgb(c) for c in BEH_COLORS]

        trips = []
        for i in keep_idx:
            valid_wp = all_wp[i, ~has_nan[i]]
            norm_ts = valid_wp[:, 2] - t_min
            path_3d = np.column_stack([valid_wp[:, :2], norm_ts])
            r, g, b = _color_cache[int(beh_indices[i])]
            trips.append(
                {
                    "path": path_3d.tolist(),
                    "timestamps": norm_ts.tolist(),
                    "color": [r, g, b, 240],
                }
            )
        return trips, t_min, t_max


# Blank map style for non-geographic (orthographic) rendering
import base64 as _b64

_blank_json = '{"version":8,"sources":{},"layers":[{"id":"background","type":"background","paint":{"background-color":"#f0f0f0"}}]}'
BLANK_STYLE = (
    "data:application/json;base64," + _b64.b64encode(_blank_json.encode()).decode()
)

# Static assets directory
WWW_DIR = Path(__file__).parent / "www"

# Max hex background points for deck.gl (200K → ~5.5 MB HTML, less banding)
MAX_DECK_POINTS = 200_000
MAX_HEX_POINTS = 200_000  # ScatterplotLayer: 200K points ≈ 2MB binary, handles easily

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
    parameters={"clearColor": [240 / 255, 240 / 255, 240 / 255, 1]},
)

viewer_map_widget = MapWidget(
    "viewer_map",
    view_state={"longitude": 0, "latitude": 0, "zoom": 1},
    style=BLANK_STYLE,
    tooltip={"html": "{info}", "style": TOOLTIP_STYLE},
    controller=True,
    parameters={"clearColor": [11 / 255, 31 / 255, 44 / 255, 1]},
)

# ── HexSim workspace discovery ───────────────────────────────────────────────


def _pick_h3_resolution(hexsim_edge_m: float) -> int:
    """Choose the highest H3 resolution whose cells are ≤ hexsim_edge_m.

    This oversamples: every HexSim cell maps to 1–2 H3 cells, preserving
    fine detail (narrow rivers, patch boundaries). For typical inputs:
      - HexSim 13.87 m → H3 res 12 (edge 10.83 m)
      - HexSim 100 m   → H3 res 10 (edge 75.86 m)
      - HexSim 500 m   → H3 res 9  (edge 200.79 m)
    """
    for res in range(16):
        if h3.average_hexagon_edge_length(res, unit="m") <= hexsim_edge_m:
            return res
    return 15


_H3_LANDSCAPE_PREFIX = "h3:"


def _find_workspaces() -> dict[str, str]:
    """Find viewable workspaces.

    Two kinds:
      * HexSim — directories containing a ``.grid`` file.  Path = dir.
      * H3 landscape — ``data/*_h3_landscape.nc`` produced by
        ``scripts/build_*_h3_landscape.py``.  Path is prefixed with
        ``"h3:"`` so the layer-loader can branch on it.
    """
    base = Path(".")
    workspaces: dict[str, str] = {}
    for grid_file in base.rglob("*.grid"):
        ws_dir = grid_file.parent
        workspaces[str(ws_dir)] = ws_dir.name
    for nc in sorted((base / "data").glob("*_h3_landscape.nc")):
        # Display name: "Nemunas H3 (res 9)" — try to read resolution
        # attribute, fall back to filename if attrs aren't readable.
        try:
            import xarray as xr
            ds = xr.open_dataset(nc, engine="h5netcdf")
            res = int(ds.attrs.get("h3_resolution", -1))
            ds.close()
            stem = nc.stem.replace("_h3_landscape", "")
            display = f"{stem.title()} H3 (res {res})" if res >= 0 else stem
        except Exception:
            display = nc.stem
        workspaces[f"{_H3_LANDSCAPE_PREFIX}{nc}"] = display
    return workspaces


def _list_hxn_layers(ws_path: str) -> dict[str, str]:
    """List displayable layers in a workspace.

    For HexSim workspaces: ``.hxn`` files under ``Spatial Data/Hexagons``.
    For H3 landscapes (``ws_path`` starts with ``h3:``): the variables
    in the NetCDF — ``depth`` plus the first-day snapshot of each
    forcing var (``tos``/``sos``/``uo``/``vo``).  Layer keys are
    ``h3:<nc_path>:<varname>[:<time_idx>]``.
    """
    layers: dict[str, str] = {}
    if ws_path.startswith(_H3_LANDSCAPE_PREFIX):
        nc_path = ws_path[len(_H3_LANDSCAPE_PREFIX):]
        layers[f"{_H3_LANDSCAPE_PREFIX}{nc_path}:depth"] = (
            "Depth (EMODnet bathymetry)"
        )
        try:
            import xarray as xr
            ds = xr.open_dataset(nc_path, engine="h5netcdf")
            data_vars = set(ds.data_vars) if hasattr(ds, "data_vars") else set()
            ds.close()
            for var, label in [
                ("tos", "Surface temperature (day 0)"),
                ("sos", "Surface salinity (day 0)"),
                ("uo",  "Eastward current (day 0)"),
                ("vo",  "Northward current (day 0)"),
            ]:
                if var in data_vars:
                    key = f"{_H3_LANDSCAPE_PREFIX}{nc_path}:{var}:0"
                    layers[key] = label
        except Exception:
            pass
        return layers

    ws = Path(ws_path)
    hex_dir = ws / "Spatial Data" / "Hexagons"
    if hex_dir.exists():
        for subdir in sorted(hex_dir.iterdir()):
            if not subdir.is_dir():
                continue
            hxn_files = list(subdir.glob("*.hxn"))
            if hxn_files:
                layers[str(hxn_files[0])] = subdir.name
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
    ui.busy_indicators.use(spinners=False),
    ui.head_content(
        ui.tags.script(THEME_JS),
        ui.tags.link(rel="stylesheet", href="style.css"),
        ui.tags.meta(name="viewport", content="width=device-width, initial-scale=1"),
        # Disable MapLibre world tiling for the orthographic hex grid
        ui.tags.script("""
            (function() {
                var _check = setInterval(function() {
                    var inst = window.__deckgl_instances && window.__deckgl_instances['map'];
                    if (inst && inst.map) {
                        inst.map.setRenderWorldCopies(false);
                        clearInterval(_check);
                    }
                }, 500);
                // Map loading overlay
                var _loaderTimeout = null;
                function _waitShiny(cb) {
                    if (typeof Shiny !== 'undefined' && Shiny.addCustomMessageHandler) cb();
                    else setTimeout(function(){ _waitShiny(cb); }, 200);
                }
                _waitShiny(function() {
                    Shiny.addCustomMessageHandler('map_loader_show', function(msg) {
                        var el = document.getElementById('map-loader-overlay');
                        var txt = document.getElementById('map-loader-text');
                        if (el) { el.style.display = 'flex'; }
                        if (txt) { txt.textContent = msg.text || 'Loading grid...'; }
                        // Auto-hide when deck.gl gets layers (map rendered)
                        if (_loaderTimeout) clearTimeout(_loaderTimeout);
                        var _poll = setInterval(function() {
                            var inst = window.__deckgl_instances && window.__deckgl_instances['map'];
                            if (inst && inst.lastLayers && inst.lastLayers.length > 0) {
                                var water = inst.lastLayers.find(function(l) { return l.id === 'water'; });
                                if (water && water.data && (water.data.length > 0 || water.data.attributes)) {
                                    if (el) el.style.display = 'none';
                                    clearInterval(_poll);
                                }
                            }
                        }, 500);
                        // Fallback: auto-hide after 30s
                        _loaderTimeout = setTimeout(function() {
                            if (el) el.style.display = 'none';
                            clearInterval(_poll);
                        }, 30000);
                    });
                    Shiny.addCustomMessageHandler('map_loader_hide', function() {
                        var el = document.getElementById('map-loader-overlay');
                        if (el) { el.style.display = 'none'; }
                        if (_loaderTimeout) { clearTimeout(_loaderTimeout); _loaderTimeout = null; }
                    });
                });
            })();
        """),
    ),
    head_includes(),
    ui.navset_tab(
        ui.nav_panel(
            "Map",
            ui.div(
                ui.div(
                    map_widget.ui(height="520px"),
                    ui.div(
                        ui.div(
                            ui.tags.div(class_="map-loader-spinner"),
                            ui.tags.div("Loading grid...", id="map-loader-text"),
                            class_="map-loader-content",
                        ),
                        id="map-loader-overlay",
                        class_="map-loader-overlay",
                        style="display: none;",
                    ),
                    ui.output_ui("map_legend"),
                    style="position: relative;",
                ),
                charts_panel(),
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
                        ui.div(
                            "Behavioral State Distribution", class_="chart-card-title"
                        ),
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
            "\u263e",
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
    step_stats = reactive.Value(
        {"t": 0, "alive": 0, "dead": 0, "arrived": 0, "behaviors": {}}
    )
    trip_buffer = TripBuffer()

    @reactive.effect
    @reactive.event(input.btn_reset, input.landscape, ignore_none=False)
    async def _init_sim():
        await _do_init_sim()

    async def _do_init_sim():
        landscape = input.landscape()

        # Show loading overlay on the map
        try:
            grid_name = (
                "Columbia River" if landscape == "columbia" else "Curonian Lagoon"
            )
            await session.send_custom_message(
                "map_loader_show", {"text": f"Loading {grid_name} grid..."}
            )
        except Exception:
            import logging

            logging.getLogger(__name__).debug(
                "Message send failed (session may not be ready)"
            )

        if landscape == "columbia":
            cfg = load_config("config_columbia.yaml")
        else:
            cfg = load_config("config_curonian_hexsim.yaml")

        # Apply estuary overrides from UI controls
        est = cfg.setdefault("estuary", {})
        est.setdefault("salinity_cost", {}).update(
            {
                "S_opt": input.s_opt(),
                "S_tol": input.s_tol(),
                "k": input.sal_k(),
            }
        )
        est.setdefault("do_avoidance", {}).update(
            {
                "lethal": input.do_lethal(),
                "high": input.do_high(),
            }
        )
        est.setdefault("seiche_pause", {}).update(
            {
                "dSSHdt_thresh_m_per_15min": input.seiche_thresh(),
            }
        )

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
            Simulation,
            cfg,
            n_agents=n_agents,
            data_dir="data",
            rng_seed=rng_seed,
        )
        sim.bio_params = BioParams(
            RA=ra,
            RB=rb,
            RQ=rq,
            ED_MORTAL=ed_mortal,
            T_OPT=t_opt,
            T_MAX=t_max,
        )
        sim._activity_lut = sim._build_activity_lut()
        sim.pool.ed_kJ_g[:] = ed_init
        sim_state.set(sim)

        # Update loader with cell count, then hide after a short delay
        try:
            is_hex = hasattr(sim.mesh, "n_cells")
            n_cells = (
                sim.mesh.n_cells if is_hex else getattr(sim.mesh, "n_triangles", 0)
            )
            await session.send_custom_message(
                "map_loader_show",
                {"text": f"Rendering {n_cells:,} cells..."},
            )
        except Exception:
            import logging

            logging.getLogger(__name__).debug(
                "Message send failed (session may not be ready)"
            )

        # ── Initialize streaming chart bins ──
        ws = getattr(sim.mesh, "_workspace", None)
        if ws and "Gradient [ upstream ]" in ws.hexmaps:
            up_hm = ws.hexmaps["Gradient [ upstream ]"]
            up_vals = up_hm.values[up_hm.values > 0]
            max_dist = float(up_vals.max()) if len(up_vals) > 0 else 1.0
            sim._upstream_distances = up_hm.values[sim.mesh._water_full_idx].astype(
                np.float64
            )
        else:
            max_dist = float(abs(sim.mesh.centroids[:, 0]).max())
            sim._upstream_distances = sim.mesh.centroids[:, 0].copy()
        sim._migration_bins = np.linspace(0, max_dist, 51)

        history.set([])

        # Push chart reset to JS
        try:
            n_agents = int(cfg["grid"].get("n_agents", input.n_agents()))
            await session.send_custom_message(
                "chart_reset",
                {
                    "max_time": int(input.n_steps()),
                    "n_agents": n_agents,
                    "river_length_km": float(max_dist),
                    "n_bins": 50,
                    "bin_edges": sim._migration_bins.tolist(),
                },
            )
        except Exception:
            import logging

            logging.getLogger(__name__).debug(
                "Message send failed (session may not be ready)"
            )  # Session not ready yet on first load

        running.set(False)
        trip_buffer.clear()

        # Update field selector: HexSim meshes have spatial data layers
        is_hexsim = hasattr(sim.mesh, "n_cells")
        if is_hexsim and hasattr(sim, "landscape") and "spatial_data" in sim.landscape:
            sd = sim.landscape["spatial_data"]
            choices = {name: name for name in sorted(sd.keys())}
            if not choices:
                choices = {"depth": "Grid Values"}
            ui.update_select("map_field", choices=choices, selected="depth")
        elif is_hexsim:
            # HexSim without scenario loader — use environment fields
            choices = {}
            choices["depth"] = "Bathymetry"
            if hasattr(sim.env, "fields"):
                for k in sim.env.fields:
                    choices[k] = k.replace("_", " ").title()
            ui.update_select("map_field", choices=choices, selected="depth")
        else:
            ui.update_select(
                "map_field",
                choices={
                    "depth": "Bathymetry",
                    "temperature": "Temperature",
                    "salinity": "Salinity",
                    "ssh": "Sea Surface Height",
                },
                selected="depth",
            )

    async def _push_chart_data(sim):
        """Push chart update to JS — lightweight JSON, no reactive cascade."""
        try:
            pool = sim.pool
            alive_mask = pool.alive.astype(bool)
            n_alive = int(alive_mask.sum())
            n_total = len(pool.alive)

            # Behavior counts: 0=Hold, 1=Random, 2=CWR, 3=Upstream, 4=Downstream
            beh = pool.behavior[alive_mask] if n_alive > 0 else np.array([], dtype=int)
            beh_counts = {
                "upstream": int((beh == 3).sum()),
                "downstream": int((beh == 4).sum()),
                "hold": int((beh == 0).sum()),
                "random": int((beh == 1).sum()),
                "cwr": int((beh == 2).sum()),
            }

            # Migration histogram
            if n_alive > 0 and hasattr(sim, "_upstream_distances"):
                dists = sim._upstream_distances[pool.tri_idx[alive_mask]]
                bin_counts, _ = np.histogram(dists, bins=sim._migration_bins)
            else:
                bin_counts = np.zeros(50, dtype=int)

            await session.send_custom_message(
                "chart_update",
                {
                    "t": sim.current_t,
                    "alive": n_alive,
                    "dead": n_total - n_alive,
                    "arrived": int(getattr(pool, "n_arrived", 0)),
                    "behaviors": beh_counts,
                    "migration_bins": bin_counts.tolist(),
                },
            )
        except Exception as exc:
            import logging

            logging.getLogger(__name__).debug("chart push failed: %s", exc)

    @reactive.effect
    @reactive.event(input.btn_step)
    async def _step():
        nonlocal _cached_scale
        # Simulation step — fatal on error
        try:
            sim = sim_state.get()
            if sim is None:
                await _do_init_sim()
                sim = sim_state.get()
            await asyncio.to_thread(sim.step)
        except Exception as e:
            import logging

            logging.getLogger(__name__).exception("Simulation step failed")
            ui.notification_show(f"Simulation error: {e}", type="error", duration=10)
            return

        # UI updates — non-fatal
        try:
            if sim.pool.alive.any():
                mesh = sim.mesh
                all_tris = sim.pool.tri_idx
                is_hex = hasattr(mesh, "_edge")
                scale = (
                    _cached_scale
                    if _cached_scale is not None and is_hex
                    else (80.0 / max(abs(mesh.centroids).max(), 1) if is_hex else 1.0)
                )
                _cached_scale = scale
                if is_hex:
                    all_xy = np.column_stack(
                        [
                            mesh.centroids[all_tris, 1] * scale,
                            -mesh.centroids[all_tris, 0] * scale,
                        ]
                    ).astype(np.float32)
                else:
                    all_xy = np.column_stack(
                        [mesh.centroids[all_tris, 1], mesh.centroids[all_tris, 0]]
                    ).astype(np.float32)
                trip_buffer.update(sim.pool.alive, all_xy, sim.pool.behavior)
            pool = sim.pool
            alive_mask = pool.alive.astype(bool)
            n_alive = int(alive_mask.sum())
            beh = pool.behavior[alive_mask] if n_alive > 0 else np.array([], dtype=int)
            step_stats.set(
                {
                    "t": sim.current_t,
                    "alive": n_alive,
                    "dead": len(pool.alive) - n_alive,
                    "arrived": int(getattr(pool, "n_arrived", 0)),
                    "behaviors": {i: int((beh == i).sum()) for i in range(5)},
                }
            )
            history.set(sim.history.copy())
            await _push_chart_data(sim)
        except Exception:
            import logging

            logging.getLogger(__name__).exception(
                "UI update failed at t=%d", sim.current_t
            )

    @reactive.effect
    @reactive.event(input.btn_run)
    async def _start_run():
        sim = sim_state.get()
        if sim is None:
            try:
                await _do_init_sim()
            except Exception as e:
                import logging

                logging.getLogger(__name__).exception("Init failed")
                ui.notification_show(f"Init failed: {e}", type="error", duration=10)
                return
        running.set(True)

    @reactive.effect
    async def _run_tick():
        """Self-scheduling run loop using invalidate_later.

        Each invocation does one batch, sets reactive values, then returns.
        Shiny flushes outputs between invocations so the UI stays live.
        """
        nonlocal _cached_scale
        if not running.get():
            return
        sim = sim_state.get()
        if sim is None:
            running.set(False)
            return
        steps = input.n_steps()
        if sim.current_t >= steps or not sim.pool.alive.any():
            running.set(False)
            return

        speed = input.speed()

        def _batch():
            for _ in range(speed):
                if sim.current_t >= steps or not sim.pool.alive.any():
                    break
                sim.step()

        # Simulation batch — fatal on error
        try:
            t_batch = time.perf_counter()
            await asyncio.to_thread(_batch)
            elapsed = time.perf_counter() - t_batch
        except Exception as e:
            import logging

            logging.getLogger(__name__).exception("Simulation batch failed")
            ui.notification_show(f"Simulation error: {e}", type="error", duration=10)
            running.set(False)
            return

        # Visualization updates — non-fatal
        try:
            if sim.pool.alive.any():
                mesh = sim.mesh
                all_tris = sim.pool.tri_idx
                is_hex = hasattr(mesh, "_edge")
                scale = (
                    _cached_scale
                    if _cached_scale is not None and is_hex
                    else (80.0 / max(abs(mesh.centroids).max(), 1) if is_hex else 1.0)
                )
                _cached_scale = scale
                if is_hex:
                    all_xy = np.column_stack(
                        [
                            mesh.centroids[all_tris, 1] * scale,
                            -mesh.centroids[all_tris, 0] * scale,
                        ]
                    ).astype(np.float32)
                else:
                    all_xy = np.column_stack(
                        [mesh.centroids[all_tris, 1], mesh.centroids[all_tris, 0]]
                    ).astype(np.float32)
                trip_buffer.update(sim.pool.alive, all_xy, sim.pool.behavior)
            pool = sim.pool
            alive_mask = pool.alive.astype(bool)
            n_alive = int(alive_mask.sum())
            beh = pool.behavior[alive_mask] if n_alive > 0 else np.array([], dtype=int)
            history.set(sim.history.copy())
            await _trips_update(is_live=True)
            speed_val = speed
            is_final = sim.current_t >= steps or not sim.pool.alive.any()
            should_update_ui = (
                is_final or speed_val <= 2 or sim.current_t % max(1, speed_val) == 0
            )
            if should_update_ui:
                step_stats.set(
                    {
                        "t": sim.current_t,
                        "alive": n_alive,
                        "dead": len(pool.alive) - n_alive,
                        "arrived": int(getattr(pool, "n_arrived", 0)),
                        "behaviors": {i: int((beh == i).sum()) for i in range(5)},
                    }
                )
                await _push_chart_data(sim)
        except Exception:
            import logging

            logging.getLogger(__name__).exception(
                "Visualization failed at t=%d", sim.current_t
            )

        # Check if run completed during this batch
        if sim.current_t >= steps or not sim.pool.alive.any():
            running.set(False)
            return

        # Schedule next tick after a short delay — Shiny flushes outputs in between
        delay_ms = int(max(50, 250 - elapsed * 1000))
        reactive.invalidate_later(delay_ms / 1000)

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
        is_hexsim = sim is not None and hasattr(sim.mesh, "_edge")
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
        s = step_stats.get()
        sim = sim_state.get()
        if sim is None:
            return ""
        total = len(sim.pool.alive)
        return f"{s['alive']}/{total} alive \u00b7 {s['arrived']} arrived"

    @output
    @render.ui
    def live_stats():
        s = step_stats.get()
        if s["t"] == 0 and s["alive"] == 0:
            return ui.div(class_="live-stats-bar")
        t = s["t"]
        n_alive = s["alive"]
        n_arrived = s["arrived"]
        beh = s["behaviors"]
        return ui.div(
            ui.span(f"t = {t} h", class_="stat-val"),
            ui.span("|", class_="stat-sep"),
            ui.span(f"{n_alive:,} alive", class_="stat-val stat-alive"),
            ui.span("|", class_="stat-sep"),
            ui.span(f"{n_arrived:,} arrived", class_="stat-val"),
            ui.span("|", class_="stat-sep"),
            ui.span(
                f"\u2191{beh.get(3, 0)} \u2193{beh.get(4, 0)} \u25a0{beh.get(0, 0)} \u25cb{beh.get(1, 0)} \u2736{beh.get(2, 0)}",
                class_="stat-val stat-behaviors",
            ),
            class_="live-stats-bar",
        )

    @render.text
    def progress_text():
        s = step_stats.get()
        return f"t = {s['t']} h"

    @render.ui
    def map_legend():
        sim = sim_state.get()
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
        elif (
            hasattr(sim, "landscape")
            and "spatial_data" in sim.landscape
            and field_name in sim.landscape["spatial_data"]
        ):
            z = sim.landscape["spatial_data"][field_name]
            cscale = TEMP_COLORSCALE
            cbar_title = field_name
        else:
            cscale = BATHY_COLORSCALE
            cbar_title = "Depth (m)"
            z = sim.mesh.depth if hasattr(sim.mesh, "depth") else np.zeros(1)

        z_min = f"{float(np.nanmin(z)):.1f}"
        z_max = f"{float(np.nanmax(z)):.1f}"
        gradient = ", ".join(f"{s[1]} {int(s[0] * 100)}%" for s in cscale)

        # Grid type label
        is_hexsim = hasattr(sim.mesh, "n_cells")
        grid_label = (
            f"Hexagonal Grid ({sim.mesh.n_cells:,} cells)"
            if is_hexsim
            else "Triangular Mesh"
        )

        # Behavior chips (no counts — those update in the status bar)
        chips = []
        for name, color in zip(BEH_NAMES, BEH_COLORS):
            chips.append(
                f'<span class="map-legend-beh-item">'
                f'<span class="map-legend-beh-dot" style="background:{color}"></span>'
                f"{name}</span>"
            )
        beh_html = (
            f'<div class="map-legend-section">Behaviors</div>'
            f'<div class="map-legend-behaviors">{"".join(chips)}</div>'
        )

        return ui.HTML(
            f'<div class="map-legend">'
            f'<div class="map-legend-section">{grid_label}</div>'
            f'<div class="map-legend-title">{cbar_title}</div>'
            f'<div class="map-legend-bar" style="background:linear-gradient(to right,{gradient})"></div>'
            f'<div class="map-legend-range"><span>{z_min}</span><span>{z_max}</span></div>'
            f"{beh_html}"
            f"</div>"
        )

    # --- Map update: three-tier strategy ---
    # Cache state for change detection
    _cached_landscape = ""
    _cached_field = ""
    _cached_subsample_idx = (
        None  # Columbia subsample / Curonian water indices (computed once)
    )
    _cached_scale = None
    _cached_water_n = 0
    _cached_water_layer = None  # Preserved across steps to prevent grid disappearance

    # Dynamic fields whose colors change on env.advance()
    DYNAMIC_FIELDS = {"temperature", "salinity", "ssh"}

    def _resolve_field(sim):
        """Resolve current field → (z, colorscale)."""
        field_name = input.map_field()
        is_hexsim = hasattr(sim.mesh, "n_cells")

        # HexSim: check spatial_data dict first
        if is_hexsim and hasattr(sim, "landscape") and "spatial_data" in sim.landscape:
            sd = sim.landscape.get("spatial_data", {})
            if field_name in sd:
                return sd[field_name], TEMP_COLORSCALE

        # Standard fields
        if field_name == "depth":
            if hasattr(sim.mesh, "depth"):
                return sim.mesh.depth, BATHY_COLORSCALE
            # HexSim meshes don't have depth — return zeros
            return np.zeros(
                sim.mesh.n_cells if is_hexsim else sim.mesh.n_triangles
            ), BATHY_COLORSCALE
        elif field_name in sim.env.fields:
            return sim.env.fields[field_name], TEMP_COLORSCALE

        # Fallback: first available spatial data layer or temperature
        if is_hexsim and hasattr(sim, "landscape"):
            sd = sim.landscape.get("spatial_data", {})
            if sd:
                first_key = next(iter(sd))
                return sd[first_key], TEMP_COLORSCALE
        if hasattr(sim.mesh, "depth"):
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
        is_hexsim = hasattr(mesh, "_edge")
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

    def _build_trips_layer(is_live: bool = False):
        """Build a shiny_deckgl ``trips_layer`` dict for partial_update.

        Uses ``_tripsAnimation`` so the built-in RAF loop in deckgl-init.js
        handles animation — no custom JS needed, no competing setProps.

        In *live* mode (during Run), ``currentTime`` is pinned to the end of
        the trail and speed=0 so the head is always visible.
        In *replay* mode (paused), speed>0 so the animation loops.
        """
        # Empty / hidden layer when trails toggle is off or no data
        if not input.show_trails():
            return trips_layer(
                "anim-trails",
                [],
                currentTime=0,
                trailLength=0,
                visible=False,
            )

        trips_data, t_min, t_max = trip_buffer.build_trips()
        if not trips_data:
            return trips_layer(
                "anim-trails",
                [],
                currentTime=0,
                trailLength=0,
                visible=False,
            )

        loop_len = max(t_max - t_min, 1)
        trail_len = max(1.0, min(loop_len * 0.5, 15))

        return trips_layer(
            "anim-trails",
            trips_data,
            visible=True,  # must be explicit — partial_update merges, so
            # a prior visible=False would stick otherwise
            getPath="@@d.path",
            getTimestamps="@@d.timestamps",
            getColor="@@d.color",
            widthMinPixels=5,
            widthMaxPixels=10,
            jointRounded=True,
            capRounded=True,
            trailLength=trail_len,
            currentTime=loop_len if is_live else 0,
            _tripsAnimation={
                "loopLength": loop_len,
                "speed": 0 if is_live else 3,
            },
        )

    async def _full_update(sim, landscape=None):
        """Send everything: positions + colors + agents (~3 MB)."""
        nonlocal _cached_water_n, _cached_water_layer
        z, cscale = _resolve_field(sim)
        idx = _water_idx(sim)
        pos_bin, col_bin, n = _build_water_binary(
            sim.mesh,
            z,
            cscale,
            idx,
            scale=_cached_scale,
            landscape=landscape,
        )
        _cached_water_n = n

        is_hexsim = hasattr(sim.mesh, "n_cells")
        if is_hexsim:
            # ScatterplotLayer for hex grid — avoids SolidPolygonLayer WebGL
            # buffer issues with large polygon counts + startIndices
            mesh = sim.mesh
            positions = np.column_stack(
                [
                    mesh.centroids[idx, 1] * _cached_scale,
                    -mesh.centroids[idx, 0] * _cached_scale,
                ]
            ).astype(np.float32)
            rgb = _colorscale_rgb(z, cscale)
            colors = np.column_stack(
                [
                    rgb[idx, 0],
                    rgb[idx, 1],
                    rgb[idx, 2],
                    np.full(n, 220, dtype=np.uint8),
                ]
            ).astype(np.uint8)
            water = layer(
                "ScatterplotLayer",
                "water",
                data={"length": n},
                getPosition=encode_binary_attribute(positions),
                getFillColor=encode_binary_attribute(colors),
                getRadius=0.0001,
                radiusMinPixels=3,
                radiusMaxPixels=8,
                pickable=False,
            )
        else:
            water = layer(
                "ScatterplotLayer",
                "water",
                data={"length": n},
                getPosition=pos_bin,
                getFillColor=col_bin,
                getRadius=30,
                radiusMinPixels=2,
                radiusMaxPixels=6,
                pickable=False,
            )
        _cached_water_layer = water
        update_layers = [water, _build_trips_layer()]
        await map_widget.update(
            session,
            layers=update_layers,
            view_state=_view_state(sim, landscape=landscape),
            views=[map_view(controller=True)],
        )

    async def _hex_color_update(sim, landscape=None):
        """Rebuild hex grid with new field colors — no set_style, no view state change."""
        nonlocal _cached_water_layer
        is_hexsim = hasattr(sim.mesh, "_edge")
        z, cscale = _resolve_field(sim)
        idx = _water_idx(sim)
        n = len(idx)

        if is_hexsim:
            mesh = sim.mesh
            scale = _cached_scale
            positions = np.column_stack(
                [
                    mesh.centroids[idx, 1] * scale,
                    -mesh.centroids[idx, 0] * scale,
                ]
            ).astype(np.float32)
            rgb = _colorscale_rgb(z, cscale)
            colors = np.column_stack(
                [
                    rgb[idx, 0],
                    rgb[idx, 1],
                    rgb[idx, 2],
                    np.full(n, 220, dtype=np.uint8),
                ]
            ).astype(np.uint8)
            water = layer(
                "ScatterplotLayer",
                "water",
                data={"length": n},
                getPosition=encode_binary_attribute(positions),
                getFillColor=encode_binary_attribute(colors),
                getRadius=0.0001,
                radiusMinPixels=3,
                radiusMaxPixels=8,
                pickable=False,
            )
            _cached_water_layer = water
            layers = [water]
        else:
            col_bin = _build_color_binary(sim.mesh, z, cscale, idx)
            water = {
                "id": "water",
                "getFillColor": col_bin,
                "data": {"length": _cached_water_n},
            }
            _cached_water_layer = water
            layers = [water]

        layers.append(_build_trips_layer())
        await map_widget.partial_update(session, layers)

    async def _trips_update(is_live: bool = False):
        """Push only the trips layer — water grid stays untouched."""
        await map_widget.partial_update(session, [_build_trips_layer(is_live)])

    @reactive.effect
    async def _update_map():
        nonlocal _cached_landscape, _cached_field
        nonlocal _cached_subsample_idx, _cached_scale

        sim = sim_state.get()
        _ = history.get()
        if sim is None:
            return
        # During Run, _run_tick handles trips updates directly.
        # Skip all map work here to avoid competing partial_updates.
        if running.get():
            return

        try:
            landscape = input.landscape()
            field_name = input.map_field()

            landscape_changed = landscape != _cached_landscape
            field_changed = field_name != _cached_field

            if landscape_changed:
                is_hexsim = hasattr(sim.mesh, "n_cells")
                if is_hexsim:
                    max_coord = max(
                        abs(sim.mesh.centroids[:, 0]).max(),
                        abs(sim.mesh.centroids[:, 1]).max(),
                    )
                    scale = 80.0 / max(max_coord, 1)
                else:
                    scale = 1.0
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
                try:
                    await session.send_custom_message("map_loader_hide", {})
                except Exception:
                    import logging

                    logging.getLogger(__name__).debug("map_loader_hide send failed")
            elif field_changed:
                _cached_field = field_name
                await _hex_color_update(sim, landscape=landscape)
            else:
                await _trips_update()
        except Exception:
            import logging

            logging.getLogger(__name__).exception("Map update failed")
            ui.notification_show("Map update failed", type="warning", duration=5)

    @reactive.effect
    @reactive.event(input.show_trails)
    async def _toggle_trails():
        """Refresh trips layer when the Trails toggle changes."""
        if sim_state.get() is None:
            return
        try:
            await _trips_update()
        except Exception:
            import logging

            logging.getLogger(__name__).exception("Failed to toggle trails")

    # --- Chart helper: Plotly → HTML file + iframe (no widget/comm layer) ---
    def _plotly_iframe(fig, name, height="280px"):
        """Write a Plotly Figure to www/<name>.html and return an iframe.

        Script tags in innerHTML don't execute (browser security), so we
        can't use to_html() inline. Instead, write a standalone HTML file
        and serve it via iframe — same pattern as the deck.gl map.
        """
        html_str = pio.to_html(
            fig,
            include_plotlyjs="cdn",
            full_html=True,
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
            return ui.tags.iframe(
                src="survival.html",
                width="100%",
                height="280px",
                style="border:none;border-radius:8px;background:transparent;",
            )
        fig = go.Figure()
        if not h:
            fig.update_layout(**_base_layout(theme=theme, height=280))
            return _plotly_iframe(fig, "survival")
        times = [r["time"] for r in h]
        alive = [r["n_alive"] for r in h]
        fig.add_trace(
            go.Scatter(
                x=times,
                y=alive,
                mode="lines",
                line=dict(color=TEAL, width=2.5),
                fill="tozeroy",
                fillcolor="rgba(61, 155, 143, 0.15)",
                name="Alive",
            )
        )
        fig.update_layout(
            **_base_layout(
                theme=theme,
                height=280,
                xaxis_title="Hour",
                yaxis_title="N alive",
                showlegend=False,
            )
        )
        return _plotly_iframe(fig, "survival")

    # --- Energy Plot ---
    @render.ui
    def energy_plot():
        h = history.get()
        theme = input.theme_mode() or "dark"
        if running.get() and h and len(h) > 5 and len(h) % 5 != 0:
            return ui.tags.iframe(
                src="energy.html",
                width="100%",
                height="280px",
                style="border:none;border-radius:8px;background:transparent;",
            )
        fig = go.Figure()
        if not h:
            fig.update_layout(**_base_layout(theme=theme, height=280))
            return _plotly_iframe(fig, "energy")
        times = [r["time"] for r in h]
        ed = [r.get("mean_ed", 0) for r in h]
        accent = ACCENT_COLOR_LIGHT if theme == "light" else ACCENT_COLOR
        fig.add_trace(
            go.Scatter(
                x=times,
                y=ed,
                mode="lines",
                line=dict(color=accent, width=2.5),
                fill="tozeroy",
                fillcolor="rgba(232, 213, 183, 0.1)",
                name="Mean ED",
            )
        )
        fig.add_hline(
            y=4.0,
            line_dash="dot",
            line_color=SALMON_COLOR,
            line_width=1.5,
            annotation_text="Mortality",
            annotation_font=dict(size=10, color=SALMON_COLOR),
            annotation_position="top left",
        )
        fig.update_layout(
            **_base_layout(
                theme=theme,
                height=280,
                xaxis_title="Hour",
                yaxis_title="kJ/g",
                showlegend=False,
            )
        )
        return _plotly_iframe(fig, "energy")

    # --- Behavior Plot ---
    @render.ui
    def behavior_plot():
        h = history.get()
        theme = input.theme_mode() or "dark"
        if running.get() and h and len(h) > 5 and len(h) % 5 != 0:
            return ui.tags.iframe(
                src="behavior.html",
                width="100%",
                height="280px",
                style="border:none;border-radius:8px;background:transparent;",
            )
        fig = go.Figure()
        if not h:
            fig.update_layout(**_base_layout(theme=theme, height=280))
            return _plotly_iframe(fig, "behavior")
        times = [r["time"] for r in h]
        for b in range(5):
            counts = [r.get("behavior_counts", {}).get(b, 0) for r in h]
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=counts,
                    mode="lines",
                    name=BEH_NAMES[b],
                    stackgroup="one",
                    line=dict(color=BEH_COLORS[b], width=0.5),
                    fillcolor=BEH_COLORS[b],
                )
            )
        fig.update_layout(
            **_base_layout(
                theme=theme,
                height=280,
                xaxis_title="Hour",
                yaxis_title="Count",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=10),
                ),
            )
        )
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

    _viewer_progress = reactive.Value(
        {"pct": 0, "stage": "", "active": False, "error": None}
    )

    async def _set_progress(pct: int, stage: str) -> None:
        _viewer_progress.set(
            {"pct": pct, "stage": stage, "active": True, "error": None}
        )
        # Yield to the event loop so the reactive update flushes to the
        # browser before the next CPU-heavy stage runs.
        await asyncio.sleep(0)

    @render.ui
    def viewer_map_container():
        return ui.div(
            viewer_map_widget.ui(height="520px"),
            ui.output_ui("viewer_legend"),
            ui.output_ui("viewer_progress"),
            style="position: relative;",
        )

    @render.ui
    def viewer_progress():
        p = _viewer_progress.get()
        if p.get("error"):
            return ui.HTML(
                f'<div class="viewer-progress-box viewer-progress-error">'
                f'<span class="viewer-progress-stage">⚠ {p["error"]}</span>'
                f"</div>"
            )
        if not p.get("active"):
            return ui.HTML("")
        pct = max(0, min(100, int(p["pct"])))
        stage = p["stage"] or "Loading…"
        return ui.HTML(
            f'<div class="viewer-progress-box">'
            f'  <div class="viewer-progress-head">'
            f'    <span class="viewer-progress-stage">{stage}</span>'
            f'    <span class="viewer-progress-pct">{pct}%</span>'
            f"  </div>"
            f'  <div class="viewer-progress-bar">'
            f'    <div class="viewer-progress-fill" style="width:{pct}%"></div>'
            f"  </div>"
            f"</div>"
        )

    async def _load_viewer_h3_landscape(layer_key: str):
        """Render an H3 landscape NetCDF directly via h3_hexagon_layer.

        ``layer_key`` is ``"h3:<nc_path>:<var>[:<time_idx>]"``.  The
        NetCDF already carries ``h3_id`` (uint64) + per-cell field
        arrays — no HexSim conversion needed.
        """
        nonlocal _viewer_initialized
        # Parse the key: skip prefix, split on ":" — first piece is path.
        body = layer_key[len(_H3_LANDSCAPE_PREFIX):]
        parts = body.split(":")
        if len(parts) < 2:
            ui.notification_show(
                f"Bad H3 layer key: {layer_key}", type="error", duration=10,
            )
            return
        nc_path = parts[0]
        var = parts[1]
        time_idx = int(parts[2]) if len(parts) > 2 else None

        await _set_progress(5, f"Reading {Path(nc_path).name}:{var}…")

        def _read_h3_landscape() -> dict:
            import xarray as xr
            with xr.open_dataset(nc_path, engine="h5netcdf") as ds:
                if var not in ds:
                    raise KeyError(f"variable {var!r} not in {nc_path}")
                arr = ds[var].values
                if arr.ndim == 2:  # (time, cell)
                    if time_idx is None:
                        raise ValueError(
                            f"{var} is time-varying; layer key must include "
                            f"a time index"
                        )
                    arr = arr[time_idx]
                h3_ids = ds["h3_id"].values
                water_mask = ds["water_mask"].values.astype(bool)
                res = int(ds.attrs.get("h3_resolution", -1))
            return {
                "h3_ids": h3_ids,
                "values": arr.astype(np.float32),
                "water_mask": water_mask,
                "resolution": res,
            }

        try:
            data = await asyncio.to_thread(_read_h3_landscape)
        except Exception as e:
            _viewer_progress.set(
                {"pct": 0, "stage": "", "active": False,
                 "error": f"H3 landscape read error: {e}"}
            )
            ui.notification_show(
                f"H3 landscape read error: {e}", type="error", duration=10,
            )
            return

        await _set_progress(40, "Filtering water cells…")
        h3_ids = data["h3_ids"]
        values = data["values"]
        water_mask = data["water_mask"]
        valid = water_mask & np.isfinite(values)
        h3_ids = h3_ids[valid]
        values = values[valid]
        n_cells = len(h3_ids)
        if n_cells == 0:
            _viewer_progress.set(
                {"pct": 0, "stage": "", "active": False,
                 "error": "no water cells in this layer"}
            )
            return

        await _set_progress(70, f"Colorizing {n_cells:,} hexagons…")
        rgb = _colorscale_rgb(values, TEMP_COLORSCALE)
        # h3_hexagon_layer expects per-feature dicts with `hex` (str) +
        # `color`.  Convert h3_ids (uint64) → strings via h3.int_to_str.
        import h3
        data_rows = [
            {"hex": h3.int_to_str(int(h)),
             "color": [int(r), int(g), int(b), 220]}
            for h, (r, g, b) in zip(h3_ids.tolist(), rgb.tolist())
        ]

        await _set_progress(85, "Computing camera target…")
        # Centre on the median water cell (same anchor strategy as the
        # existing H3 viewer path) — guarantees the initial view lands
        # on real hexes, not on an empty bbox-midpoint bend interior.
        mid = n_cells // 2
        mid_lat, mid_lon = h3.cell_to_latlng(h3.int_to_str(int(h3_ids[mid])))
        # Zoom: the existing viewer code pins ≥ 2 px per H3 cell using
        # h3.average_hexagon_edge_length(res).  Reproduce that here.
        h3_edge_m = h3.average_hexagon_edge_length(data["resolution"], unit="m")
        min_hex_px = 3
        zoom = float(np.log2(min_hex_px * 156543.03 / max(h3_edge_m, 1.0)))
        zoom = max(0.0, min(22.0, zoom))

        water_layer = h3_hexagon_layer(
            "viewer_water",
            data=data_rows,
            getHexagon="@@=d.hex",
            getFillColor="@@=d.color",
            stroked=False,
            filled=True,
            extruded=False,
            pickable=False,
            highPrecision=True,
        )

        try:
            await _set_progress(92, "Uploading to deck.gl…")
            await viewer_map_widget.update(
                session,
                layers=[water_layer],
                view_state={
                    "longitude": float(mid_lon),
                    "latitude": float(mid_lat),
                    "zoom": zoom,
                },
            )
            _viewer_initialized = True
            _viewer_meta.set({
                "format": "h3_landscape",
                "version": "—",
                "ncols": 1,
                "nrows": n_cells,
                "total_cells": n_cells,
                "water_cells": n_cells,
                "cell_size": None,
                "edge": h3_edge_m,
                "origin": (mid_lat, mid_lon),
                "nodata": None,
                "vmin": float(values.min()),
                "vmax": float(values.max()),
                "vmean": float(values.mean()),
                "n_unique": int(len(np.unique(values))),
                "layer_name": f"{Path(nc_path).stem}:{var}",
            })
            ui.notification_show(
                f"Loaded {n_cells:,} H3 cells (res {data['resolution']}, "
                f"zoom={zoom:.1f})",
                type="message", duration=5,
            )
            await _set_progress(100, f"Loaded {n_cells:,} H3 cells")
            await asyncio.sleep(0.6)
            _viewer_progress.set(
                {"pct": 0, "stage": "", "active": False, "error": None}
            )
        except Exception as e:
            _viewer_progress.set(
                {"pct": 0, "stage": "", "active": False,
                 "error": f"Map update failed: {e}"}
            )
            ui.notification_show(
                f"Map update failed: {e}", type="error", duration=10,
            )
            import logging
            logging.getLogger(__name__).exception(
                "H3 landscape viewer update failed"
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

        # H3 landscape branch: layer key is "h3:<nc>:<var>[:<time_idx>]".
        # Skip the HexSim hexmap-parsing and georef-meters dance —
        # the NetCDF already has h3_id + lat/lon and the rendering
        # path below uses h3_hexagon_layer which consumes H3 IDs
        # directly.
        if hxn_path.startswith(_H3_LANDSCAPE_PREFIX):
            await _load_viewer_h3_landscape(hxn_path)
            return

        await _set_progress(5, f"Reading {Path(hxn_path).name} …")
        try:
            hm = await asyncio.to_thread(HxnHexMap.from_file, hxn_path)
        except Exception as e:
            _viewer_progress.set(
                {"pct": 0, "stage": "", "active": False, "error": f"Read error: {e}"}
            )
            ui.notification_show(f"Read error: {e}", type="error", duration=10)
            return

        values = hm.values.astype(np.float32)

        # Determine water cells (non-zero)
        water_mask = values != 0.0
        water_flat = np.where(water_mask)[0]
        n_water = len(water_flat)

        if n_water == 0:
            _viewer_progress.set(
                {"pct": 0, "stage": "", "active": False,
                 "error": "No non-zero cells found"}
            )
            ui.notification_show("No non-zero cells found", type="warning")
            return

        await _set_progress(
            25, f"Reading georef · {n_water:,} water cells of {len(values):,}"
        )
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

        await _set_progress(40, "Building hex row/col indices…")
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

        await _set_progress(55, "Computing hex centers…")
        # Hex center coordinates (matches hxnparser.HexMap.hex_to_xy)
        # Pointy-top spacing (verified against HexSim 4.0.20)
        cx = (
            np.sqrt(3.0) * edge * (data_cols.astype(np.float64) + 0.5 * (data_rows % 2))
        )
        # Flip y so row 0 is at the top (screen convention matches HexSim viewer).
        cy = -1.5 * edge * data_rows.astype(np.float64)

        # Subsample HexSim cells BEFORE H3 conversion — stride-based so the
        # river shape is preserved. 300k is enough after H3 dedup to fill
        # most of the visible area without oversized JSON.
        max_pts = 300_000
        if n_water > max_pts:
            step = max(1, n_water // max_pts)
            idx = np.arange(0, n_water, step)
        else:
            idx = np.arange(n_water)

        # Center on a real water cell — the middle one in flat-index order.
        cx_sub = cx[idx]
        cy_sub = cy[idx]
        mid = len(cx_sub) // 2
        cx_center = float(cx_sub[mid])
        cy_center = float(cy_sub[mid])

        # Pseudo-geography: place the anchor hex at (lat=0, lng=0) on the
        # equator so meters→degrees is locally isotropic (1° ≈ 111,320 m in
        # both axes). HexSim's native CRS isn't carried in .grid/.hxn, so
        # anchoring at equator-prime-meridian is the neutral choice.
        m_to_deg = 1.0 / 111320.0
        lons = ((cx_sub - cx_center) * m_to_deg).astype(np.float64)
        lats = ((cy_sub - cy_center) * m_to_deg).astype(np.float64)

        await _set_progress(72, "Converting cells to H3…")
        # Pick the highest H3 resolution whose cells fit inside a HexSim cell
        # — oversamples ~1.5× so every HexSim cell spawns 1–2 H3 cells and
        # fine features (narrow rivers) survive.
        h3_res = _pick_h3_resolution(float(edge))
        h3_edge_m = h3.average_hexagon_edge_length(h3_res, unit="m")

        # Scalar h3 call × N — ~5 μs each; 300k cells ≈ 1.5 s. No vectorized
        # C entry point in h3-py 4.x yet, so we loop.
        h3_ids = [
            h3.latlng_to_cell(float(la), float(lo), h3_res)
            for la, lo in zip(lats, lons)
        ]
        h3_ids_arr = np.array(h3_ids)

        await _set_progress(82, "Aggregating into H3 cells…")
        # Dedup: multiple HexSim cells can land in one H3 cell. Average their
        # values so the legend/colorbar still matches the data distribution.
        z = water_values[idx].astype(np.float64)
        unique_h3, inverse = np.unique(h3_ids_arr, return_inverse=True)
        sums = np.bincount(inverse, weights=z)
        counts = np.bincount(inverse)
        h3_values = sums / counts  # mean per H3 cell
        n_h3 = len(unique_h3)

        # Colors per unique H3 cell
        rgb = _colorscale_rgb(h3_values.astype(np.float32), TEMP_COLORSCALE)

        await _set_progress(88, f"Building {n_h3:,} H3 hexagons…")
        # deck.gl's H3HexagonLayer takes a per-feature dict with `hex` + `color`.
        # Binary attributes aren't practical here because hex IDs are strings.
        data_rows = [
            {"hex": str(h), "color": [int(r), int(g), int(b), 220]}
            for h, (r, g, b) in zip(unique_h3.tolist(), rgb.tolist())
        ]

        water_layer = h3_hexagon_layer(
            "viewer_water",
            data=data_rows,
            getHexagon="@@=d.hex",
            getFillColor="@@=d.color",
            stroked=False,
            filled=True,
            extruded=False,
            pickable=False,
            highPrecision=True,
        )

        # Zoom fit. At zoom Z on the equator: meters/pixel = 156543.03 / 2^Z.
        extent_m = max(
            float(cx_sub.max() - cx_sub.min()),
            float(cy_sub.max() - cy_sub.min()),
            1.0,
        )
        pixels_target = 500
        zoom_fit = np.log2(pixels_target * 156543.03 / extent_m)
        # Clamp so each H3 cell is ≥ 2 px, using the actual H3 edge (not the
        # HexSim edge, since H3 cell size is what the user actually sees).
        min_hex_px = 2
        zoom_hex = np.log2(min_hex_px * 156543.03 / max(h3_edge_m, 1.0))
        zoom = float(max(zoom_fit, zoom_hex))
        zoom = max(0.0, min(22.0, zoom))

        try:
            await _set_progress(92, "Uploading to deck.gl…")
            await viewer_map_widget.update(
                session,
                layers=[water_layer],
                view_state={
                    "longitude": 0.0,
                    "latitude": 0.0,
                    "zoom": float(zoom),
                },
            )
            _viewer_initialized = True
            # Store metadata only on success (map and legend stay in sync)
            _viewer_meta.set(
                {
                    "format": hm.format,
                    "version": hm.version,
                    "ncols": hm.height,
                    "nrows": hm.width,
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
                }
            )
            ui.notification_show(
                f"Loaded {n_h3:,} hexagons (zoom={zoom:.1f})",
                type="message",
                duration=5,
            )
            await _set_progress(100, f"Loaded {n_h3:,} hexagons")
            # Brief pause at 100% so the user registers completion, then hide.
            await asyncio.sleep(0.6)
            _viewer_progress.set(
                {"pct": 0, "stage": "", "active": False, "error": None}
            )
        except Exception as e:
            _viewer_progress.set(
                {"pct": 0, "stage": "", "active": False,
                 "error": f"Map update failed: {e}"}
            )
            ui.notification_show(f"Map update failed: {e}", type="error", duration=10)
            import logging

            logging.getLogger(__name__).exception("Viewer map update failed")

    _viewer_meta = reactive.Value({})

    @render.ui
    def viewer_metadata():
        meta = _viewer_meta.get()
        if not meta:
            return ui.HTML(
                '<div class="param-hint">Select a workspace and layer, '
                "then click Load Layer.</div>"
            )

        origin = meta.get("origin", (0, 0))
        cell_size_str = (
            f"{meta['cell_size']:.3f} m" if meta["cell_size"] else "from .grid"
        )
        hex_area = (3.0 * np.sqrt(3.0) / 2.0) * meta["edge"] ** 2

        return ui.HTML(f"""
            <div class="viewer-meta">
                <strong class="viewer-meta-title">{meta["layer_name"]}</strong><br>
                <span class="viewer-meta-label">Format:</span> {meta["format"]}<br>
                <span class="viewer-meta-label">Version:</span> {meta["version"]}<br>
                <span class="viewer-meta-label">Grid:</span> {meta["ncols"]} &times; {meta["nrows"]}<br>
                <span class="viewer-meta-label">Total hexagons:</span> {meta["total_cells"]:,}<br>
                <span class="viewer-meta-label">Water cells:</span> {meta["water_cells"]:,}
                    ({100 * meta["water_cells"] / meta["total_cells"]:.1f}%)<br>
                <span class="viewer-meta-label">Hex edge:</span> {meta["edge"]:.3f} m<br>
                <span class="viewer-meta-label">Hex area:</span> {hex_area:.1f} m&sup2;<br>
                <span class="viewer-meta-label">Cell size:</span> {cell_size_str}<br>
                <span class="viewer-meta-label">Origin:</span> ({origin[0]:.1f}, {origin[1]:.1f})<br>
                <hr class="viewer-meta-hr">
                <span class="viewer-meta-label">Values:</span><br>
                &nbsp; min={meta["vmin"]:.2f} &nbsp; max={meta["vmax"]:.2f}<br>
                &nbsp; mean={meta["vmean"]:.2f} &nbsp; unique={meta["n_unique"]}<br>
            </div>
        """)

    @render.ui
    def viewer_legend():
        meta = _viewer_meta.get()
        if not meta:
            return ui.HTML("")
        gradient = ", ".join(f"{s[1]} {int(s[0] * 100)}%" for s in TEMP_COLORSCALE)
        return ui.HTML(
            f'<div class="map-legend">'
            f'<div class="map-legend-title">{meta["layer_name"]}</div>'
            f'<div class="map-legend-bar" style="background:linear-gradient(to right,{gradient})"></div>'
            f'<div class="map-legend-range"><span>{meta["vmin"]:.1f}</span><span>{meta["vmax"]:.1f}</span></div>'
            f"</div>"
        )


app = App(app_ui, server, static_assets=str(Path(__file__).parent / "www"))
