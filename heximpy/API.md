# hxnparser API Reference

## Module: `heximpy.hxnparser`

```python
from heximpy.hxnparser import HexMap, GridMeta, Barrier, Workspace, read_barriers
```

---

## Constants

| Name | Value | Description |
|------|-------|-------------|
| `_HXN_MAGIC` | `b"PATCH_HEXMAP"` | 12-byte magic for PATCH_HEXMAP .hxn files |
| `_HXN_HEADER_SIZE` | `37` | Bytes before float32 data in PATCH_HEXMAP |
| `_GRID_MAGIC` | `b"PATCH_GRID"` | 10-byte magic for .grid files |
| `_HISTORY_MARKER` | `b"HISTORY"` | Terminates PATCH_HEXMAP data section |

---

## `HexMap`

Parsed hex-map data from a `.hxn` file.

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `format` | `str` | — | `"patch_hexmap"` or `"plain"` |
| `version` | `int` | — | File format version |
| `height` | `int` | — | Number of rows (nrows) |
| `width` | `int` | — | Number of columns (ncols) |
| `flag` | `int` | — | `0` = wide, `1` = narrow |
| `max_val` | `float` | — | Maximum hexmap score (PATCH_HEXMAP header metadata) |
| `min_val` | `float` | — | Minimum hexmap score (PATCH_HEXMAP header metadata) |
| `hexzero` | `float` | — | Hexagon zero score (PATCH_HEXMAP header metadata) |
| `values` | `np.ndarray` | — | Flat float32 array of hex values |
| `cell_size` | `float` | `0.0` | Cell size (plain format only) |
| `origin` | `tuple` | `(0.0, 0.0)` | Origin (x, y) in spatial units (plain format only) |
| `nodata` | `int` | `0` | No-data sentinel (plain format only) |
| `dtype_code` | `int` | `1` | `1` = float32, `2` = int32 (plain format only) |
| `_edge` | `float` | `0.0` | Hex edge length; set by `Workspace` loader |

### Properties

#### `n_hexagons -> int`

Total hexagon count, computed from grid dimensions and flag.

- **Wide** (`flag=0`): `width * height`
- **Narrow** (`flag=1`): `width * n_wide + (width - 1) * n_narrow`
  - Even height: `n_wide = height // 2`, `n_narrow = n_wide`
  - Odd height: `n_wide = (height + 1) // 2`, `n_narrow = n_wide - 1`

### Class Methods

#### `HexMap.from_file(path) -> HexMap`

Read a `.hxn` file, auto-detecting PATCH_HEXMAP vs plain format.

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `str \| Path` | Path to `.hxn` file |

**Returns:** `HexMap`

**Raises:**
- `ValueError` if the file is a PATCH_GRID (`.grid`) file

### Instance Methods

#### `to_file(path, *, format=None) -> None`

Write a `.hxn` file.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str \| Path` | — | Output file path |
| `format` | `str \| None` | `None` | `"patch_hexmap"` or `"plain"`. Defaults to `self.format` |

**Raises:** `ValueError` for unknown format strings.

---

#### `neighbors(row, col) -> list[tuple[int, int]]`

Return up to 6 neighbor positions using even-q column-offset convention (flat-top hexagons). Excludes out-of-bounds neighbors.

| Parameter | Type | Description |
|-----------|------|-------------|
| `row` | `int` | Row index |
| `col` | `int` | Column index |

**Returns:** List of `(row, col)` tuples.

**Neighbor offsets:**
- Even columns: `(-1,0) (1,0) (0,-1) (0,1) (-1,-1) (-1,1)`
- Odd columns: `(-1,0) (1,0) (0,-1) (0,1) (1,-1) (1,1)`

---

#### `hex_to_xy(row, col) -> tuple[float, float]`

Convert hex grid coordinates to Cartesian spatial coordinates.

| Parameter | Type | Description |
|-----------|------|-------------|
| `row` | `int` | Row index |
| `col` | `int` | Column index |

**Returns:** `(x, y)` in spatial units.

**Formula:**
- `x = 1.5 * edge * col`
- `y = sqrt(3) * edge * (row + 0.5 * (col % 2))`

Uses `_effective_edge()`: `_edge` > `cell_size` > `1.0` fallback.

---

#### `xy_to_hex(x, y) -> tuple[int, int]`

Convert Cartesian coordinates to nearest hex grid position.

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `float` | X coordinate |
| `y` | `float` | Y coordinate |

**Returns:** `(row, col)` of nearest hexagon.

---

#### `hex_distance(a, b) -> int`

Minimum hex step count between two positions.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | `tuple[int, int]` | First `(row, col)` |
| `b` | `tuple[int, int]` | Second `(row, col)` |

**Returns:** Integer distance (cube-coordinate Manhattan / 2).

---

#### `hex_polygon(row, col) -> list[tuple[float, float]]`

Return 6 vertices of the flat-top hexagon at the given position.

| Parameter | Type | Description |
|-----------|------|-------------|
| `row` | `int` | Row index |
| `col` | `int` | Column index |

**Returns:** List of 6 `(x, y)` vertex tuples, counterclockwise from the right.

---

#### `to_csv(path, *, skip_zeros=True) -> None`

Export to CSV in `hxn2csv.c` format.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str \| Path` | — | Output file path |
| `skip_zeros` | `bool` | `True` | Skip hexagons with value `0.0` |

**Output format:**
```
Hex ID,Score
42,1.000000
43,0.750000
```

Hex IDs are 1-based.

---

#### `to_geodataframe(*, edge=None, include_empty=False, crs=None) -> GeoDataFrame`

Convert to GeoDataFrame with hexagonal polygon geometry.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `edge` | `float \| None` | `None` | Override hex edge length for geometry |
| `include_empty` | `bool` | `False` | Include hexagons with value `0.0` |
| `crs` | `str \| None` | `None` | Coordinate reference system (e.g. `"EPSG:32610"`) |

**Returns:** `GeoDataFrame` with columns: `hex_id`, `row`, `col`, `value`, `geometry`.

**Warns** if no edge is set for PATCH_HEXMAP files (geometry will use unit spacing).

**Requires:** `geopandas`, `shapely`

---

#### `to_shapefile(path, *, edge=None, include_empty=False, crs=None) -> None`

Export to shapefile. Delegates to `to_geodataframe().to_file()`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str \| Path` | — | Output `.shp` path |
| `edge` | `float \| None` | `None` | Override hex edge length |
| `include_empty` | `bool` | `False` | Include zero-value hexagons |
| `crs` | `str \| None` | `None` | Coordinate reference system |

**Requires:** `geopandas`, `shapely`

---

#### `to_geotiff(path, *, crs=None) -> None`

Export as GeoTIFF (rectangular raster approximation).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str \| Path` | — | Output `.tif` path |
| `crs` | `str \| None` | `None` | Coordinate reference system |

**Requires:** `rasterio`

---

## `GridMeta`

Metadata from a PATCH_GRID (`.grid`) file.

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `ncols` | `int` | — | Number of columns |
| `nrows` | `int` | — | Number of rows |
| `x_extent` | `float` | — | X extent in spatial units |
| `y_extent` | `float` | — | Y extent in spatial units |
| `row_spacing` | `float` | — | Row spacing in spatial units |
| `edge` | `float` | — | Hex edge length (`row_spacing / sqrt(3)`) |
| `version` | `int` | `0` | File format version |
| `n_hexes` | `int` | `0` | Total hexagon count from header |
| `flag` | `int` | `0` | Grid flag |

### Class Methods

#### `GridMeta.from_file(path) -> GridMeta`

Read a PATCH_GRID `.grid` file (67-byte header).

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `str \| Path` | Path to `.grid` file |

**Returns:** `GridMeta`

**Raises:** `ValueError` if file does not start with `b"PATCH_GRID"`.

---

## `Barrier`

A single barrier edge entry from an `.hbf` file.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `hex_id` | `int` | Hexagon ID |
| `edge` | `int` | Edge index (0-5) |
| `classification` | `int` | Classification ID |
| `class_name` | `str` | Human-readable classification name |

---

## `read_barriers(path) -> list[Barrier]`

Parse an `.hbf` barrier file.

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `str \| Path` | Path to `.hbf` file |

**Returns:** List of `Barrier` objects with `class_name` populated from classification lines.

**File format:**
```
C <id> <p1> <p2> "<name>"     # classification definition
E <hex_id> <edge> <class_id>  # barrier edge
```

Falls back to empty string for undefined classification IDs.

---

## `Workspace`

A HexSim workspace directory containing grid, hexmaps, and barriers.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `grid` | `GridMeta` | Parsed `.grid` metadata |
| `hexmaps` | `dict[str, HexMap]` | Layer name -> HexMap |
| `barriers` | `list[Barrier]` | All barriers from `.hbf` files |
| `path` | `Path` | Workspace directory path |

### Properties

#### `layer_names -> list[str]`

Sorted list of hexmap layer names.

### Class Methods

#### `Workspace.from_dir(path) -> Workspace`

Load a HexSim workspace from a directory.

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `str \| Path` | Workspace directory path |

**Returns:** `Workspace`

**Expected directory structure:**
```
workspace/
  *.grid
  Spatial Data/
    Hexagons/
      Layer Name/
        Layer Name.1.hxn
    barriers/  (optional)
      Barrier Name/
        Barrier Name.1.hbf
```

**Behavior:**
- Finds first `.grid` file in root, parses as `GridMeta`
- Globs `Spatial Data/Hexagons/*/*.hxn`, keys each by parent folder name
- Auto-sets `_edge` on each `HexMap` from `GridMeta.edge`
- Looks for `.hbf` files in `Spatial Data/barriers/`, falls back to workspace root

**Raises:**
- `FileNotFoundError` if no `.grid` file found
- `FileNotFoundError` if `Spatial Data/Hexagons/` missing

**Warns** if no `.hxn` files found in hexagon directory.
