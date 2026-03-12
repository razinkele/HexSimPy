# HXN Parser Design Spec

## Overview

A standalone `hxnparser.py` module in `heximpy/` that provides a single authoritative parser for HexSim workspace files (`.hxn`, `.grid`, `.hbf`). Replaces scattered partial implementations across `salmon_ibm/hexsim.py`, `HexSim.py.txt`, and `HexSimFormat.txt` with one reusable library.

**Scope:** Read + Write + Export + Hex geometry + Workspace loading.

**Dependencies:** numpy, scipy, geopandas, shapely, rasterio.

## Data Model

Three dataclasses form the core:

### HexMap

Represents a parsed `.hxn` file. Supports both PATCH_HEXMAP and plain formats.

```python
@dataclass
class HexMap:
    format: str              # "patch_hexmap" or "plain"
    version: int
    width: int               # ncols
    height: int              # nrows
    values: np.ndarray       # flat float32/int32 array
    flag: int = 0            # 0=wide, 1=narrow (PATCH_HEXMAP only)
    max_val: float = 0.0     # PATCH_HEXMAP only
    min_val: float = 0.0     # PATCH_HEXMAP only
    hexzero: float = 0.0     # PATCH_HEXMAP only
    cell_size: float = 0.0   # plain format only
    origin: tuple = (0.0, 0.0)  # plain format only
    nodata: int = 0          # plain format only
    dtype_code: int = 1      # 1=float32, 2=int32 (plain only)
    _edge: float = 0.0       # set by Workspace loader
```

Hexagon count formula (from C code):
- Wide (flag=0): `ncols * nrows`
- Narrow (flag=1): `ncols * n_wide + (ncols - 1) * n_narrow`
  - Even nrows: `n_wide = nrows / 2`, `n_narrow = n_wide`
  - Odd nrows: `n_wide = (nrows + 1) / 2`, `n_narrow = n_wide - 1`

### GridMeta

Parsed `.grid` file with workspace georeferencing.

```python
@dataclass
class GridMeta:
    ncols: int
    nrows: int
    x_extent: float
    y_extent: float
    row_spacing: float
    edge: float              # hex edge length in meters (row_spacing / sqrt(3))
```

### Barrier

Single barrier edge from `.hbf` file.

```python
@dataclass
class Barrier:
    hex_id: int
    edge: int
    classification: int
    class_name: str = ""
```

## File I/O

### Reading

```python
hexmap = HexMap.from_file("HabitatMap.hxn")       # auto-detects format
grid = GridMeta.from_file("workspace.grid")
barriers = read_barriers("barriers.hbf")           # -> list[Barrier]
```

**Format detection:** First 12 bytes. If `b"PATCH_HEXMAP"` -> PATCH_HEXMAP parser. If `b"PATCH_GRID"` -> reject with clear error. Otherwise -> plain 44-byte header.

**PATCH_HEXMAP binary layout (matching C code):**

| Offset | Type | Size | Content |
|--------|------|------|---------|
| 0-11 | char[12] | 12B | Magic: `"PATCH_HEXMAP"` |
| 12-15 | uint32 | 4B | Version |
| 16-19 | uint32 | 4B | Number of columns (ncols) |
| 20-23 | uint32 | 4B | Number of rows (nrows) |
| 24 | uint8 | 1B | Flag (0=wide, 1=narrow) |
| 25-28 | float32 | 4B | Max hexmap score |
| 29-32 | float32 | 4B | Min hexmap score |
| 33-36 | float32 | 4B | Hexagon zero score |

Data: float32 LE values until `b"HISTORY"` marker or EOF.
Footer (optional): variable-length ASCII metadata + 256 NaN padding.

**Plain .hxn binary layout:**

| Offset | Type | Size | Content |
|--------|------|------|---------|
| 0-3 | int32 | 4B | Version |
| 4-7 | int32 | 4B | Width (ncols) |
| 8-11 | int32 | 4B | Height (nrows) |
| 12-19 | float64 | 8B | Cell size |
| 20-27 | float64 | 8B | Origin X |
| 28-35 | float64 | 8B | Origin Y |
| 36-39 | int32 | 4B | Data type code (1=float32, 2=int32) |
| 40-43 | int32 | 4B | No-data value |

Data: `width * height` values in row-major order.

### Writing

```python
hexmap.to_file("output.hxn")                       # writes in original format
hexmap.to_file("output.hxn", format="patch_hexmap") # force specific format
```

## Hex Grid Geometry

All geometry as methods on `HexMap`, using even-q column-offset convention (flat-top, HexSim standard).

```python
hexmap.neighbors(row, col)                   # -> list[(row, col)], up to 6
hexmap.hex_to_xy(row, col)                   # -> (x, y) spatial units
hexmap.xy_to_hex(x, y)                       # -> (row, col) nearest hex
hexmap.hex_distance((r1, c1), (r2, c2))      # -> int step count (cube coords)
hexmap.hex_polygon(row, col)                 # -> list[(x, y)] 6 vertices
hexmap.n_hexagons                            # property, respects wide/narrow
```

**Neighbor offsets (even-q column-offset):**
- Even columns: `[(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1)]`
- Odd columns: `[(-1,0), (1,0), (0,-1), (0,1), (1,-1), (1,1)]`

**Coordinate transforms:**
- `hex_to_xy`: uses `_edge` (from Workspace/GridMeta) or `cell_size` (plain) or unit spacing
- `xy_to_hex`: inverse, rounds to nearest hex
- `hex_distance`: converts offset -> cube coordinates, then Manhattan/2

## Export & Conversion

```python
gdf = hexmap.to_geodataframe(edge=13.876)
gdf = hexmap.to_geodataframe(edge=13.876, include_empty=True)
hexmap.to_shapefile("output.shp", edge=13.876)
hexmap.to_geotiff("output.tif")
hexmap.to_csv("output.csv", skip_zeros=True)
```

- `to_geodataframe`: generates Shapely Polygon per hex cell. Columns: `hex_id`, `row`, `col`, `value`, `geometry`. Skips nodata/zero by default. `edge` required for PATCH_HEXMAP; auto-populated via Workspace.
- `to_shapefile`: delegates to `to_geodataframe().to_file()`.
- `to_geotiff`: rectangular raster approximation via rasterio.
- `to_csv`: matches `hxn2csv.c` output format (`Hex ID,Score`), `skip_zeros=True` by default.
- CRS: not assumed. Pass `crs="EPSG:32610"` to export methods if needed.

## Workspace Loader

```python
@dataclass
class Workspace:
    grid: GridMeta
    hexmaps: dict[str, HexMap]   # layer name -> HexMap
    barriers: list[Barrier]
    path: Path

    @classmethod
    def from_dir(cls, path) -> Workspace: ...

    @property
    def layer_names(self) -> list[str]: ...
```

**Expected directory structure:**
```
workspace/
  *.grid
  *.hbf  (optional)
  Spatial Data/
    Hexagons/
      Layer Name/
        Layer Name.1.hxn
      Another Layer/
        Another Layer.1.hxn
```

**Discovery logic:**
1. Find first `.grid` file in root -> parse as `GridMeta`
2. Glob `Spatial Data/Hexagons/*/*.hxn` -> parse each, key by parent folder name
3. Find any `.hbf` files in root -> parse as barriers
4. Auto-set `_edge` on each `HexMap` from `GridMeta.edge`

**Error handling:**
- Missing `.grid` -> `FileNotFoundError`
- Missing `Spatial Data/Hexagons/` -> `FileNotFoundError`
- No `.hxn` files found -> warning (not error)

## Testing Strategy

- Round-trip tests: read .hxn -> write .hxn -> read again -> compare
- Validate against C code output: parse example files, compare hex counts with C formula
- Geometry: known hex coordinates, neighbor lists, distances
- Export: check GeoDataFrame row counts and column names
- Workspace: load Columbia River workspace, verify layer discovery

## Source Files Analyzed

**C code (authoritative for binary format):**
- `heximpy/hxn2csv.c` — PATCH_HEXMAP reader + CSV export
- `heximpy/get_hexmap_means.c` — PATCH_HEXMAP reader + averaging
- `heximpy/build_hexmap_hexagons.c` — hex grid construction + neighbor logic

**Existing Python (to be replaced):**
- `salmon_ibm/hexsim.py` — partial dual-format reader + HexMesh
- `HexSim.py.txt` — plain-format reader + geometry + shapefile export
- `HexSimFormat.txt` — format documentation (partially AI-generated)

**Example .hxn files available:** 65+ across Columbia River, HexSimPLE, and migration corridor workspaces.
