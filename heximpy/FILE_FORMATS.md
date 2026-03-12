# HexSim File Format Reference

This document describes the binary and text file formats used by [EPA HexSim](https://www.epa.gov/hexsim), derived from analysis of the original C source code (`hxn2csv.c`, `get_hexmap_means.c`, `build_hexmap_hexagons.c`).

---

## Workspace Directory Structure

A HexSim workspace is organized as follows:

```
workspace/
  ModelName.grid                          # Grid geometry (one per workspace)
  Spatial Data/
    Hexagons/
      Layer Name/
        Layer Name.1.hxn                  # Hex-map data (one or more per layer)
        Layer Name.2.hxn                  # Multiple time steps possible
      Another Layer/
        Another Layer.1.hxn
    barriers/                             # Optional
      Barrier Name/
        Barrier Name.1.hbf               # Barrier edge definitions
```

The `.grid` file defines the spatial framework (dimensions, georeferencing). Each `.hxn` file holds per-hexagon values for one layer at one time step. Barrier `.hbf` files define movement restrictions between hexagons.

---

## .hxn (Hex-Map) Files

Two binary formats exist. Both store per-hexagon floating-point scores. Format is auto-detected by the first 12 bytes.

### PATCH_HEXMAP Format

Used by HexSim v8+ (all Columbia River workspace files use this format).

**Magic:** `"PATCH_HEXMAP"` (12 ASCII bytes, no null terminator in header)

#### Header (37 bytes)

| Offset | Size | C Type | Field | Description |
|--------|------|--------|-------|-------------|
| 0 | 12 | `char[12]` | magic | `"PATCH_HEXMAP"` |
| 12 | 4 | `uint32` | version | File format version number |
| 16 | 4 | `uint32` | rows | Number of rows (height) |
| 20 | 4 | `uint32` | cols | Number of columns (width) |
| 24 | 1 | `bool` | type | Grid type: `0` = wide, `1` = narrow |
| 25 | 4 | `float32` | max_val | Maximum hex-map score (metadata, not a data value) |
| 29 | 4 | `float32` | min_val | Minimum hex-map score (metadata, not a data value) |
| 33 | 4 | `float32` | hexzero | Hexagon zero score (metadata, not a data value) |

**Total header: 37 bytes. Data begins at byte offset 37.**

The `type` field is stored as C `bool` (1 byte). Only values `0` and `1` are valid.

> **Common bug:** Several existing parsers use a header size of 25 bytes, treating `max_val`, `min_val`, and `hexzero` as data values. This corrupts the first three hexagon scores. The correct header size is 37 bytes.

> **Common bug:** Some parsers swap `rows` and `cols` (reading `cols` at offset 16 instead of `rows`). The C source (`hxn2csv.c` line 45-46) reads `rows` first, then `cols`.

#### Data Section

Immediately after the 37-byte header:

- **Type:** `float32` (IEEE 754, little-endian)
- **Count:** Determined by the hexagon count formula (see below)
- **Order:** Row-major, hexagon IDs are 1-based in CSV output

The data section ends when either:
1. All expected hexagons have been read, or
2. The `"HISTORY"` marker byte sequence is encountered

#### History Footer (optional)

After the data values, some files contain:
- The ASCII string `"HISTORY"` (7 bytes)
- Variable-length ASCII metadata text
- 256 bytes of NaN padding

The footer is informational and not required for parsing.

#### Hexagon Count Formula

The number of data values depends on the grid type:

**Wide grid** (`type = 0`):
```
hexagons = cols * rows
```

**Narrow grid** (`type = 1`):
```
if rows is even:
    n_wide   = rows / 2
    n_narrow = n_wide
else:
    n_wide   = (rows + 1) / 2
    n_narrow = n_wide - 1

hexagons = cols * n_wide + (cols - 1) * n_narrow
```

In a narrow grid, odd-numbered rows have one fewer column than even-numbered rows. Wide rows have `cols` hexagons, narrow rows have `cols - 1`.

### Plain Format

Used by some standalone hex-map files (no magic string prefix).

#### Header (44 bytes)

| Offset | Size | C Type | Field | Description |
|--------|------|--------|-------|-------------|
| 0 | 4 | `int32` | version | File format version |
| 4 | 4 | `int32` | width | Number of columns (ncols) |
| 8 | 4 | `int32` | height | Number of rows (nrows) |
| 12 | 8 | `float64` | cell_size | Cell size in spatial units |
| 20 | 8 | `float64` | origin_x | X coordinate of grid origin |
| 28 | 8 | `float64` | origin_y | Y coordinate of grid origin |
| 36 | 4 | `int32` | dtype_code | Data type: `1` = float32, `2` = int32 |
| 40 | 4 | `int32` | nodata | No-data sentinel value |

**Total header: 44 bytes. Data begins at byte offset 44.**

Note: In plain format, `width` (ncols) is at offset 4 and `height` (nrows) at offset 8 -- the opposite order from PATCH_HEXMAP.

#### Data Section

- **Count:** `width * height` values, always in row-major order
- **Type:** `float32` if `dtype_code = 1`, `int32` if `dtype_code = 2` (both little-endian)
- **Grid type:** Always wide (`cols * rows` hexagons)
- No HISTORY footer

### Format Detection

To distinguish the two formats, read the first 12 bytes:
- If they equal `"PATCH_HEXMAP"` (bytes `50 41 54 43 48 5f 48 45 58 4d 41 50`) -> PATCH_HEXMAP format
- If the first 10 bytes equal `"PATCH_GRID"` -> this is a `.grid` file, not a hex-map (reject)
- Otherwise -> plain format (rewind and read 44-byte header)

---

## .grid (Workspace Geometry) Files

One `.grid` file per workspace defines the spatial grid framework that all hex-map layers share.

**Magic:** `"PATCH_GRID"` (10 ASCII bytes)

### Header (67 bytes)

| Offset | Size | C Type | Field | Description |
|--------|------|--------|-------|-------------|
| 0 | 10 | `char[10]` | magic | `"PATCH_GRID"` |
| 10 | 4 | `uint32` | version | File format version |
| 14 | 4 | `uint32` | n_hexagons | Total hexagon count |
| 18 | 4 | `uint32` | ncols | Number of columns |
| 22 | 4 | `uint32` | nrows | Number of rows |
| 26 | 1 | `uint8` | flag | Grid flag |
| 27 | 8 | `float64` | x_extent | X extent in spatial units |
| 35 | 8 | `float64` | zero_1 | Always `0.0` |
| 43 | 8 | `float64` | zero_2 | Always `0.0` |
| 51 | 8 | `float64` | y_extent | Y extent in spatial units |
| 59 | 8 | `float64` | row_spacing | Row spacing in spatial units |

**Total header: 67 bytes.**

The five `float64` values at offset 27-66 are a georeferencing tuple. The second and third values are always zero in observed files.

### Derived Values

**Hex edge length** (the side length of each flat-top hexagon):

```
edge = row_spacing / sqrt(3)
```

This edge length is needed for spatial operations on PATCH_HEXMAP files, which do not store their own georeferencing. When loading a workspace, the edge from the `.grid` file is propagated to all hex-map layers.

---

## .hbf (Barrier) Files

Plain text files defining barriers (movement restrictions) between hexagon edges.

### Line Types

**Classification lines** define barrier types:
```
C <class_id> <param1> <param2> "<name>"
```

| Token | Type | Description |
|-------|------|-------------|
| `C` | literal | Line type marker |
| `class_id` | int | Unique classification ID |
| `param1` | float | Classification parameter 1 (e.g., permeability) |
| `param2` | float | Classification parameter 2 |
| `name` | string | Human-readable name, in double quotes |

**Edge lines** assign barriers to specific hexagon edges:
```
E <hex_id> <edge> <class_id>
```

| Token | Type | Description |
|-------|------|-------------|
| `E` | literal | Line type marker |
| `hex_id` | int | Hexagon ID (1-based) |
| `edge` | int | Edge index (0-5, corresponding to the 6 sides of a hexagon) |
| `class_id` | int | References a classification defined by a `C` line |

### Example

```
C 1 0.5 0.0 "Dam"
C 2 0.3 0.0 "Waterfall"
E 1234 3 1
E 1235 0 2
E 1236 3 1
```

This defines two barrier classifications ("Dam" and "Waterfall"), then places a Dam barrier on edge 3 of hexagons 1234 and 1236, and a Waterfall barrier on edge 0 of hexagon 1235.

Classification lines must appear before edge lines that reference them.

---

## Hex Grid Convention

HexSim uses **even-q column-offset** coordinates with **flat-top hexagons**.

### Layout

```
  col0  col1  col2  col3
   __    __    __    __
  /  \__/  \__/  \__/  \   row 0
  \__/  \__/  \__/  \__/
  /  \__/  \__/  \__/  \   row 1
  \__/  \__/  \__/  \__/
  /  \__/  \__/  \__/  \   row 2
  \__/  \__/  \__/  \__/
```

Odd-numbered columns are shifted down by half a row.

### Neighbor Offsets

Each hexagon has up to 6 neighbors. The offset from `(row, col)` to each neighbor depends on column parity:

**Even columns** (`col % 2 == 0`):
| Direction | (drow, dcol) |
|-----------|-------------|
| North | (-1, 0) |
| South | (+1, 0) |
| Northwest | (0, -1) |
| Northeast | (0, +1) |
| Southwest | (-1, -1) |
| Southeast | (-1, +1) |

**Odd columns** (`col % 2 == 1`):
| Direction | (drow, dcol) |
|-----------|-------------|
| North | (-1, 0) |
| South | (+1, 0) |
| Northwest | (0, -1) |
| Northeast | (0, +1) |
| Southwest | (+1, -1) |
| Southeast | (+1, +1) |

### Hexagon Geometry

For flat-top hexagons with edge length `e`:

- **Column spacing:** `1.5 * e`
- **Row spacing:** `sqrt(3) * e`
- **Hex center at (row, col):**
  - `x = 1.5 * e * col`
  - `y = sqrt(3) * e * (row + 0.5 * (col % 2))`

### Data Storage Order

Hexagons are stored in **row-major order** (all columns in row 0, then row 1, etc.). Hexagon IDs in CSV output are **1-based** (the first hexagon is ID 1, not 0).

For narrow grids, odd rows have one fewer hexagon than even rows. The storage still follows row-major order, but the total count is computed using the narrow formula.

### Wide vs Narrow Grids

In a **wide grid** (`type = 0`), every row has the same number of columns. Total hexagons = `cols * rows`.

In a **narrow grid** (`type = 1`), odd rows have one fewer column. The `build_hexmap_hexagons.c` code demonstrates this: when iterating rows, if `row % 2 != 0`, the last column is skipped (`col == Cols_Min` is skipped, where `Cols_Min = Cols_Max - 1`).

---

## Byte Order

All binary values are **little-endian** (x86 native). This applies to integers (`uint32`, `int32`), floats (`float32`), and doubles (`float64`) across all file formats.

---

## Source Code Reference

These formats were derived from the following C programs included in the `heximpy/` directory:

| File | Purpose | Key Lines |
|------|---------|-----------|
| `hxn2csv.c` | PATCH_HEXMAP reader + CSV export | Lines 43-50: header reads |
| `get_hexmap_means.c` | PATCH_HEXMAP reader + averaging | Lines 40-47: header reads (confirms layout) |
| `build_hexmap_hexagons.c` | Hex grid construction + neighbor logic | Lines 105-155: wide/narrow row iteration |
