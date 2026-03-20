"""Test hex grid rendering: create synthetic grids, render with hxn_viewer, verify with playwright.

Creates several small test .hxn files with known patterns, renders them
to PNG using matplotlib (hxn_viewer reference renderer), and optionally
opens an interactive deck.gl HTML page for visual comparison.
"""
import sys
sys.path.insert(0, ".")

import numpy as np
import math
from pathlib import Path
from heximpy.hxnparser import HexMap


# ============================================================================
# 1. Create synthetic test grids
# ============================================================================

def make_test_grid(name, width, height, flag, values, edge=100.0):
    """Create a HexMap with known values for testing."""
    hm = HexMap(
        format="patch_hexmap",
        version=8,
        width=width,
        height=height,
        flag=flag,
        max_val=float(values.max()),
        min_val=float(values.min()),
        hexzero=0.0,
        values=values.astype(np.float32),
        _edge=edge,
    )
    return hm


def create_test_grids():
    """Create a suite of test grids with known patterns."""
    grids = {}

    # --- Grid 1: 5x5 wide, all ones ---
    # Simple grid where every cell = 1.0
    n = 5 * 5
    grids["5x5_wide_ones"] = make_test_grid(
        "5x5_wide_ones", width=5, height=5, flag=0,
        values=np.ones(n), edge=100.0,
    )

    # --- Grid 2: 5x5 wide, checkerboard ---
    vals = np.zeros(25)
    for i in range(25):
        r, c = i // 5, i % 5
        vals[i] = 1.0 if (r + c) % 2 == 0 else 2.0
    grids["5x5_wide_checker"] = make_test_grid(
        "5x5_wide_checker", width=5, height=5, flag=0,
        values=vals, edge=100.0,
    )

    # --- Grid 3: 5x5 narrow (flag=1) ---
    # Even rows: 5 cells, odd rows: 4 cells
    # Total: 3*5 + 2*4 = 23
    vals = np.arange(1, 24, dtype=np.float32)
    grids["5x5_narrow_seq"] = make_test_grid(
        "5x5_narrow_seq", width=5, height=5, flag=1,
        values=vals, edge=100.0,
    )

    # --- Grid 4: 3x7 narrow, diagonal stripe ---
    h, w = 7, 3
    row_list, col_list = [], []
    for r in range(h):
        rw = w if r % 2 == 0 else w - 1
        row_list.extend([r] * rw)
        col_list.extend(range(rw))
    rows = np.array(row_list)
    cols = np.array(col_list)
    n = len(rows)
    vals = np.zeros(n, dtype=np.float32)
    for i in range(n):
        if cols[i] == rows[i] % w:
            vals[i] = 1.0
    grids["3x7_narrow_diag"] = make_test_grid(
        "3x7_narrow_diag", width=3, height=7, flag=1,
        values=vals, edge=100.0,
    )

    # --- Grid 5: 10x10 wide, ring pattern ---
    n = 10 * 10
    vals = np.zeros(n, dtype=np.float32)
    for i in range(n):
        r, c = i // 10, i % 10
        dist = max(abs(r - 4.5), abs(c - 4.5))
        vals[i] = dist
    grids["10x10_wide_ring"] = make_test_grid(
        "10x10_wide_ring", width=10, height=10, flag=0,
        values=vals, edge=100.0,
    )

    # --- Grid 6: Owl-scale (380x1430 wide, flag=0) but tiny (10x20) ---
    n = 10 * 20
    vals = np.zeros(n, dtype=np.float32)
    for i in range(n):
        r, c = i // 10, i % 10
        vals[i] = float(r + 1)  # horizontal stripes
    grids["10x20_wide_stripes"] = make_test_grid(
        "10x20_wide_stripes", width=10, height=20, flag=0,
        values=vals, edge=100.0,
    )

    return grids


# ============================================================================
# 2. Render with reference renderer (hxn_viewer)
# ============================================================================

def render_reference(hm, name, output_dir):
    """Render a HexMap using the hxn_viewer reference renderer."""
    from heximpy.hxn_viewer import _hex_centers
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    edge = hm._effective_edge()
    cx, cy, vals = _hex_centers(hm, include_zero=True)

    # Build hex polygons
    angles = np.arange(6) * (np.pi / 3)  # flat-top
    dx = edge * np.cos(angles)
    dy = edge * np.sin(angles)

    polygons = []
    colors = []
    v_min, v_max = vals.min(), vals.max()
    for i in range(len(cx)):
        verts = [(cx[i] + dx[j], cy[i] + dy[j]) for j in range(6)]
        polygons.append(verts)
        if v_max > v_min:
            t = (vals[i] - v_min) / (v_max - v_min)
        else:
            t = 0.5
        colors.append(plt.cm.viridis(t))

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    coll = PolyCollection(polygons, facecolors=colors, edgecolors="gray", linewidths=0.5)
    ax.add_collection(coll)
    ax.set_xlim(cx.min() - 2 * edge, cx.max() + 2 * edge)
    ax.set_ylim(cy.min() - 2 * edge, cy.max() + 2 * edge)
    ax.set_aspect("equal")
    ax.set_title(f"{name} (ref renderer)")

    # Add row/col labels for small grids
    if len(cx) <= 50:
        h, w = hm.height, hm.width
        if hm.flag == 0:
            all_rows = np.repeat(np.arange(h), w)
            all_cols = np.tile(np.arange(w), h)
        else:
            rl, cl = [], []
            for r in range(h):
                rw = w if r % 2 == 0 else w - 1
                rl.extend([r] * rw)
                cl.extend(range(rw))
            all_rows = np.array(rl)
            all_cols = np.array(cl)

        for i in range(len(cx)):
            ax.text(cx[i], cy[i], f"{all_rows[i]},{all_cols[i]}",
                    ha="center", va="center", fontsize=6, color="white")

    out = output_dir / f"{name}_reference.png"
    fig.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


# ============================================================================
# 3. Render with deck.gl (same pipeline as the app)
# ============================================================================

def render_deckgl_html(hm, name, output_dir, edge_override=None):
    """Generate a standalone HTML file with deck.gl rendering of the hex grid."""
    from heximpy.hxn_viewer import _hex_centers

    edge = edge_override or hm._effective_edge()
    cx_all, cy_all, vals_all = _hex_centers(hm, include_zero=True)

    # Scale to pseudo-geographic coords (same as app.py)
    max_coord = max(abs(cy_all).max(), abs(cx_all).max(), 1)
    scale = 80.0 / max_coord
    sx = cx_all * scale
    sy = -cy_all * scale  # negate Y
    edge_s = edge * scale

    # Build hex vertices (flat-top, same as app.py)
    angles = np.arange(6) * (np.pi / 3)
    hex_dx = np.cos(angles)
    hex_dy = np.sin(angles)

    # Normalize values for color
    v_min, v_max = vals_all.min(), vals_all.max()
    if v_max > v_min:
        t = (vals_all - v_min) / (v_max - v_min)
    else:
        t = np.full(len(vals_all), 0.5)

    # Build GeoJSON-like polygon data for deck.gl
    features = []
    for i in range(len(sx)):
        verts = []
        for j in range(6):
            verts.append([float(sx[i] + edge_s * hex_dx[j]),
                          float(sy[i] + edge_s * hex_dy[j])])
        verts.append(verts[0])  # close polygon
        r = int(30 + 200 * t[i])
        g = int(100 + 100 * (1 - t[i]))
        b = int(50 + 150 * t[i])
        features.append({
            "polygon": verts,
            "color": [r, g, b, 220],
            "value": float(vals_all[i]),
        })

    import json
    center_lon = float(np.mean(sx))
    center_lat = float(np.mean(sy))

    # Compute zoom for visible hexagons
    min_hex_px = 6
    zoom_hex = math.log2(min_hex_px * 360 / (256 * max(edge_s, 1e-6)))
    extent = max(float(sx.max() - sx.min()), float(sy.max() - sy.min()), 0.01)
    zoom_extent = math.log2(360 / extent)
    zoom = max(zoom_hex, zoom_extent)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{name} — deck.gl hex test</title>
<script src="https://unpkg.com/deck.gl@latest/dist.min.js"></script>
<script src="https://unpkg.com/maplibre-gl@3.6.2/dist/maplibre-gl.js"></script>
<link href="https://unpkg.com/maplibre-gl@3.6.2/dist/maplibre-gl.css" rel="stylesheet"/>
<style>body{{margin:0}}#map{{width:100vw;height:100vh}}
#info{{position:absolute;top:10px;left:10px;background:rgba(0,0,0,0.7);color:white;
padding:8px 12px;font:13px monospace;border-radius:4px;z-index:1}}</style>
</head>
<body>
<div id="info">{name} — {len(features)} hexagons, edge={edge:.1f}, zoom={zoom:.1f}</div>
<div id="map"></div>
<script>
const data = {json.dumps(features)};

const map = new maplibregl.Map({{
  container: 'map',
  style: {{version:8, sources:{{}}, layers:[
    {{id:'bg',type:'background',paint:{{'background-color':'#1a1a2e'}}}}
  ]}},
  center: [{center_lon}, {center_lat}],
  zoom: {zoom},
}});

const overlay = new deck.MapboxOverlay({{
  layers: [
    new deck.SolidPolygonLayer({{
      id: 'hexagons',
      data: data,
      getPolygon: d => d.polygon,
      getFillColor: d => d.color,
      filled: true,
      extruded: false,
      pickable: true,
    }}),
  ],
}});

map.addControl(new maplibregl.NavigationControl());
map.on('load', () => map.addControl(overlay));
</script>
</body>
</html>"""

    out = output_dir / f"{name}_deckgl.html"
    out.write_text(html, encoding="utf-8")
    print(f"  Saved: {out}")
    return out


# ============================================================================
# 4. Save .hxn files for testing in original HexSim viewer
# ============================================================================

def save_hxn(hm, name, output_dir):
    """Save HexMap to .hxn file."""
    out = output_dir / f"{name}.hxn"
    hm.to_file(str(out))
    print(f"  Saved: {out}")
    return out


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    output_dir = Path("tests/hex_test_output")
    output_dir.mkdir(exist_ok=True)

    grids = create_test_grids()

    print(f"\nCreated {len(grids)} test grids:")
    for name, hm in grids.items():
        print(f"\n--- {name} ---")
        print(f"  width={hm.width} height={hm.height} flag={hm.flag}")
        print(f"  n_hexagons={hm.n_hexagons} values.len={len(hm.values)}")
        print(f"  edge={hm._effective_edge()}")

        # Save .hxn file
        save_hxn(hm, name, output_dir)

        # Render reference (matplotlib)
        render_reference(hm, name, output_dir)

        # Render deck.gl HTML
        render_deckgl_html(hm, name, output_dir)

    # Also render real workspaces for comparison
    print("\n\n=== Real Workspaces ===")
    real_tests = [
        ("HexSimPLE_habitat", "HexSimPLE",
         "HexSimPLE/Spatial Data/Hexagons/Habitat Map/Habitat Map.1.hxn"),
        ("Owl_habitat", "NorthernSpottedOwl/Northern Spotted Owls",
         None),
    ]
    for name, ws_path, hxn_path in real_tests:
        ws = Path(ws_path)
        if not ws.exists():
            print(f"\n--- {name}: SKIP (not found) ---")
            continue
        if hxn_path is None:
            hex_dir = ws / "Spatial Data" / "Hexagons"
            for subdir in hex_dir.iterdir():
                if subdir.is_dir():
                    hxns = list(subdir.glob("*.hxn"))
                    if hxns:
                        hxn_path = str(hxns[0])
                        break
        if hxn_path is None:
            continue

        from heximpy.hxnparser import GridMeta
        hm = HexMap.from_file(hxn_path)
        gm = GridMeta.from_file(list(ws.glob("*.grid"))[0])
        hm._edge = gm.edge

        print(f"\n--- {name} ---")
        print(f"  width={hm.width} height={hm.height} flag={hm.flag} edge={gm.edge:.2f}")
        render_deckgl_html(hm, name, output_dir, edge_override=gm.edge)

    print(f"\n\nAll outputs in: {output_dir.resolve()}")
    print("Open the *_deckgl.html files in a browser to compare with *_reference.png")
