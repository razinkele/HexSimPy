# hxnparser

Python parser for EPA HexSim workspace files (`.hxn`, `.grid`, `.hbf`).

Reads and writes HexSim hexagonal raster maps, provides hex grid geometry utilities, and exports to GeoDataFrame, shapefile, GeoTIFF, and CSV.

## Installation

Requires Python 3.10+ and the following packages:

```
numpy
geopandas
shapely
rasterio
```

Install with conda (recommended for GDAL/rasterio):

```bash
conda install numpy geopandas shapely rasterio
```

## Quick Start

### Read a single .hxn file

```python
from heximpy.hxnparser import HexMap

hm = HexMap.from_file("HabitatMap.hxn")
print(f"{hm.width}x{hm.height}, {hm.n_hexagons} hexagons")
print(f"Format: {hm.format}, values range: {hm.min_val} - {hm.max_val}")
```

### Load a full workspace

```python
from heximpy.hxnparser import Workspace

ws = Workspace.from_dir("Columbia River Migration Model/")
print(f"Grid: {ws.grid.ncols}x{ws.grid.nrows}, edge={ws.grid.edge:.3f}m")
print(f"Layers: {ws.layer_names}")
print(f"Barriers: {len(ws.barriers)}")

# Access individual layers
river = ws.hexmaps["River [ extent ]"]
```

### Export to GeoDataFrame / shapefile

```python
# From workspace (edge auto-populated from .grid)
gdf = ws.hexmaps["Habitat Map"].to_geodataframe()

# From standalone file (must provide edge)
hm = HexMap.from_file("HabitatMap.hxn")
gdf = hm.to_geodataframe(edge=13.876, crs="EPSG:32610")

# Save as shapefile
hm.to_shapefile("output.shp", edge=13.876)
```

### Export to CSV

```python
hm.to_csv("output.csv")                   # skip zeros (default)
hm.to_csv("output.csv", skip_zeros=False)  # include all hexagons
```

Output format matches `hxn2csv.c`:

```
Hex ID,Score
42,1.000000
43,0.750000
```

### Export to GeoTIFF

```python
hm.to_geotiff("output.tif", crs="EPSG:32610")
```

### Write .hxn files

```python
hm = HexMap.from_file("HabitatMap.hxn")
hm.to_file("copy.hxn")                          # same format
hm.to_file("converted.hxn", format="plain")     # force format
```

### Hex geometry

```python
hm = HexMap.from_file("River.hxn")

# Neighbors (even-q column-offset, flat-top)
nbrs = hm.neighbors(row=10, col=5)

# Grid coordinates to spatial coordinates
x, y = hm.hex_to_xy(row=10, col=5)

# Spatial coordinates to nearest grid cell
row, col = hm.xy_to_hex(x=150.0, y=200.0)

# Hex distance (minimum steps)
dist = hm.hex_distance((0, 0), (3, 5))

# Polygon vertices for rendering
vertices = hm.hex_polygon(row=10, col=5)
```

### Read barrier files

```python
from heximpy.hxnparser import read_barriers

barriers = read_barriers("Fish Ladder Available.1.hbf")
for b in barriers:
    print(f"Hex {b.hex_id}, edge {b.edge}: {b.class_name}")
```

## Supported Formats

### .hxn (Hexmap) - Two variants

| Format | Magic | Header | Notes |
|--------|-------|--------|-------|
| PATCH_HEXMAP | `"PATCH_HEXMAP"` | 37 bytes | HexSim v8+, used by Columbia River workspace |
| Plain | (none) | 44 bytes | Includes cell_size, origin, nodata |

Both are auto-detected by `HexMap.from_file()`.

### .grid (Workspace geometry)

Binary format with `"PATCH_GRID"` magic (67-byte header). Contains grid dimensions and georeferencing. Parsed by `GridMeta.from_file()`.

### .hbf (Barriers)

Text format with classification (`C`) and edge (`E`) lines. Parsed by `read_barriers()`.

## Hex Grid Convention

This module uses **even-q column-offset** with **flat-top hexagons** (HexSim standard).

```
  col0  col1  col2  col3
   __    __    __    __
  /  \__/  \__/  \__/  \   row 0
  \__/  \__/  \__/  \__/
  /  \__/  \__/  \__/  \   row 1
  \__/  \__/  \__/  \__/
```

Odd columns are shifted down by half a row. Neighbor offsets:

- **Even columns:** `(-1,0) (1,0) (0,-1) (0,1) (-1,-1) (-1,1)`
- **Odd columns:** `(-1,0) (1,0) (0,-1) (0,1) (1,-1) (1,1)`

Reference: [Red Blob Games - Hexagonal Grids](https://www.redblobgames.com/grids/hexagons/#coordinates-offset)

## Running Tests

```bash
cd heximpy
python -m pytest tests/ -v
```

Tests use real `.hxn`, `.grid`, and `.hbf` files from the Columbia River workspace when available, with `skipif` guards for portability.

## Binary Format Details

See `docs/superpowers/specs/2026-03-12-hxn-parser-design.md` for the full binary layout tables, derived from the C source code (`hxn2csv.c`, `get_hexmap_means.c`).

Key corrections from prior implementations:
- PATCH_HEXMAP header is **37 bytes** (not 25)
- Rows at offset 16, cols at offset 20 (not swapped)
- `max_val`, `min_val`, `hexzero` are header metadata, not data values
