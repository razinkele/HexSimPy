"""HexSim workspace loader and HexMesh class.

Reads EPA HexSim workspaces via ``heximpy.hxnparser`` and constructs a
water-only hexagonal mesh that duck-types TriMesh for the simulation loop.

HexSim uses **flat-top hexagons** in an odd-q column-offset grid.
Hex edge length ≈ 13.876 m (hex area ≈ 500 m²).

Note: PATCH_HEXMAP files store dimensions transposed relative to
GridMeta — ``hm.width`` is the data stride (= grid nrows), and
``hm.height`` is the number of data-rows (= grid ncols).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from heximpy.hxnparser import GridMeta, HexMap, Workspace, read_barriers


# ── Hex grid geometry ────────────────────────────────────────────────────────

def _hex_neighbors_offset(row: int, col: int, ncols: int, nrows: int,
                          n_data: int) -> list[int]:
    """Compute flat-index neighbors for (row, col) in an offset hex grid.

    Uses flat-top odd-q column-offset convention (HexSim standard):
        Even cols: neighbors at (r-1,c),(r+1,c), (r,c-1),(r,c+1), (r-1,c-1),(r-1,c+1)
        Odd cols:  neighbors at (r-1,c),(r+1,c), (r,c-1),(r,c+1), (r+1,c-1),(r+1,c+1)

    Reference: https://www.redblobgames.com/grids/hexagons/#coordinates-offset
    """
    if col % 2 == 0:  # even column
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1)]
    else:             # odd column
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1), (1, -1), (1, 1)]

    result = []
    for dr, dc in offsets:
        nr, nc = row + dr, col + dc
        if 0 <= nr < nrows and 0 <= nc < ncols:
            flat = nr * ncols + nc
            if flat < n_data:
                result.append(flat)
    return result


# ── HexMesh class ────────────────────────────────────────────────────────────

class HexMesh:
    """Water-only hexagonal mesh that duck-types TriMesh.

    Only water cells are stored (compacted). All array indices refer to the
    compact water-cell numbering unless stated otherwise.
    """

    def __init__(self, centroids, depth, neighbors, areas, water_full_idx,
                 full_to_compact, ncols, nrows, n_data, *, edge=1.0,
                 workspace=None):
        self.centroids = centroids        # (N_water, 2)  [y, x] in meters
        self.depth = depth                # (N_water,)
        self.neighbors = neighbors        # (N_water, 6)  compact indices, -1 = none
        self.areas = areas                # (N_water,)    hex area in m²
        self.n_cells = len(centroids)
        self.water_mask = np.ones(self.n_cells, dtype=bool)  # all True

        # Internal mappings
        self._water_full_idx = water_full_idx    # (N_water,) full flat indices
        self._full_to_compact = full_to_compact  # dict full_idx → compact_idx
        self._ncols = ncols
        self._nrows = nrows
        self._n_data = n_data
        self._edge = edge                        # hex edge length in meters
        self._tree = cKDTree(centroids)
        self._workspace = workspace              # hxnparser Workspace (optional)

        # Precompute padded water-neighbor arrays for vectorized movement
        self._water_nbrs = neighbors.copy()
        self._water_nbr_count = np.sum(neighbors >= 0, axis=1).astype(np.intp)

    @property
    def n_triangles(self) -> int:
        """Alias for TriMesh compatibility."""
        return self.n_cells

    def water_neighbors(self, idx: int) -> list[int]:
        """Water-only neighbors of compact cell idx."""
        nbrs = self.neighbors[idx]
        return [int(n) for n in nbrs if n >= 0]

    def find_triangle(self, y: float, x: float) -> int:
        """Find nearest water cell to (y, x) coordinates."""
        _, idx = self._tree.query([y, x])
        return int(idx)

    def gradient(self, field: np.ndarray, idx: int) -> tuple[float, float]:
        """Normalized gradient of field at cell idx.

        Same algorithm as TriMesh.gradient but with up to 6 neighbors.
        """
        nbrs = [n for n in self.neighbors[idx] if n >= 0]
        if not nbrs:
            return (0.0, 0.0)

        c0 = self.centroids[idx]
        f0 = field[idx]
        dy, dx = 0.0, 0.0

        for n in nbrs:
            cn = self.centroids[n]
            df = field[n] - f0
            dc = cn - c0
            norm = np.sqrt(dc[0] ** 2 + dc[1] ** 2)
            if norm > 0:
                dy += df * dc[0] / norm
                dx += df * dc[1] / norm

        mag = np.sqrt(dy ** 2 + dx ** 2)
        if mag > 0:
            dy /= mag
            dx /= mag
        return (dy, dx)

    @classmethod
    def from_hexsim(cls, workspace_dir: str | Path,
                    species: str = "chinook",
                    extent_layer: str | None = None,
                    depth_layer: str | None = None) -> HexMesh:
        """Load a HexSim workspace directory → HexMesh.

        Parameters
        ----------
        workspace_dir : path to the HexSim workspace (contains .grid file
            and Spatial Data/Hexagons/ subdirectory).
        species : "chinook" or "steelhead" — selects depth layer.
        extent_layer : name of the water extent layer (auto-detected if None).
        depth_layer : name of the depth layer (auto-detected if None).
        """
        ws = Workspace.from_dir(workspace_dir)
        edge = ws.grid.edge

        # 1. Read water extent (auto-detect: "River [ extent ]" or "Lagoon [ extent ]")
        if extent_layer:
            extent_hm = ws.hexmaps.get(extent_layer)
        else:
            for name in ["River [ extent ]", "Lagoon [ extent ]"]:
                extent_hm = ws.hexmaps.get(name)
                if extent_hm is not None:
                    break
        if extent_hm is None:
            raise FileNotFoundError(
                f"No extent layer found in {workspace_dir}"
            )
        extent = extent_hm.values
        n_data = len(extent)

        # PATCH_HEXMAP stores dimensions transposed relative to GridMeta:
        #   hm.width  = data stride (long axis)  = grid nrows
        #   hm.height = data rows   (short axis)  = grid ncols
        # The flat array is row-major with hm.width as stride.
        # After decomposing: data_row → spatial col (x), data_col → spatial row (y).
        data_stride = extent_hm.width   # columns per data-row
        data_nrows = extent_hm.height   # number of data-rows

        # Water mask: any nonzero value = water
        water_flat = np.where(extent != 0.0)[0]
        n_water = len(water_flat)

        # 2. Build full→compact mapping
        full_to_compact = {}
        for compact_idx, full_idx in enumerate(water_flat):
            full_to_compact[int(full_idx)] = compact_idx

        # 3. Compute centroids — flat-top odd-q convention
        #    Decompose flat indices using data stride (hm.width).
        #    Same layout as hxn_viewer: data_cols → x (long axis),
        #    data_rows → y (short axis).  No axis swap needed.
        data_rows = water_flat // data_stride   # 0..data_nrows-1  (short axis)
        data_cols = water_flat % data_stride     # 0..data_stride-1 (long axis)
        col_spacing = 1.5 * edge
        row_spacing = np.sqrt(3.0) * edge
        cx = data_cols.astype(np.float64) * col_spacing
        cy = data_rows.astype(np.float64) * row_spacing + (data_cols % 2) * (row_spacing / 2.0)
        centroids = np.column_stack([cy, cx])  # (N_water, 2) as [y, x] in meters

        # 4. Read depth (try specific, then generic names)
        depth_hm = None
        if depth_layer:
            depth_hm = ws.hexmaps.get(depth_layer)
        if depth_hm is None:
            for name in [f"River Depth [ {species} ]", "River [ depth ]",
                         "Lagoon [ depth ]"]:
                depth_hm = ws.hexmaps.get(name)
                if depth_hm is not None:
                    break
        if depth_hm is not None:
            depth = depth_hm.values[water_flat].astype(np.float64)
        else:
            depth = np.ones(n_water)

        # 5. Build neighbor graph (compact indices)
        #    Use data_stride for flat-index neighbor computation
        neighbors = np.full((n_water, 6), -1, dtype=np.intp)
        for ci in range(n_water):
            fi = int(water_flat[ci])
            r, c = fi // data_stride, fi % data_stride
            full_nbrs = _hex_neighbors_offset(r, c, data_stride, data_nrows, n_data)
            ni = 0
            for fn in full_nbrs:
                if fn in full_to_compact:
                    neighbors[ci, ni] = full_to_compact[fn]
                    ni += 1

        # 6. Constant hex areas (flat-top: A = (3√3/2) × edge²)
        hex_area = (3.0 * np.sqrt(3.0) / 2.0) * edge ** 2  # in m²
        areas = np.full(n_water, hex_area)

        return cls(centroids, depth, neighbors, areas,
                   water_flat, full_to_compact, data_stride, data_nrows,
                   n_data, edge=edge, workspace=ws)
