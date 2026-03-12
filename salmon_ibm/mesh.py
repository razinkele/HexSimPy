"""Triangular mesh constructed from a regular lat/lon grid."""
from __future__ import annotations

import numpy as np
from scipy.spatial import Delaunay
import xarray as xr


class TriMesh:
    """Unstructured triangular mesh over the Curonian Lagoon domain."""

    def __init__(self, nodes, triangles, mask_per_node, depth_per_node,
                 delaunay=None):
        self.nodes = nodes
        self.triangles = triangles
        self.n_triangles = len(triangles)
        self.centroids = nodes[triangles].mean(axis=1)
        v = nodes[triangles]
        ab = v[:, 1] - v[:, 0]
        ac = v[:, 2] - v[:, 0]
        self.areas = 0.5 * np.abs(ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0])
        tri_mask_sum = mask_per_node[triangles].sum(axis=1)
        self.water_mask = tri_mask_sum == 3
        self.depth = depth_per_node[triangles].mean(axis=1)

        # Use Delaunay object directly instead of slow _build_neighbors()
        self._delaunay = delaunay if delaunay is not None else Delaunay(nodes)
        self.neighbors = self._neighbors_from_delaunay(self._delaunay)

        # Precompute water neighbor padded array for O(1) lookup
        self._precompute_water_neighbors()

    @classmethod
    def from_netcdf(cls, path):
        ds = xr.open_dataset(path, engine="scipy")
        lat = ds["lat"].values
        lon = ds["lon"].values
        mask = ds["mask"].values
        depth = ds["depth"].values
        nodes = np.column_stack([lat.ravel(), lon.ravel()])
        mask_flat = mask.ravel()
        depth_flat = depth.ravel()
        ds.close()
        tri = Delaunay(nodes)
        return cls(nodes, tri.simplices, mask_flat, depth_flat, delaunay=tri)

    @staticmethod
    def _build_neighbors(triangles):
        """Build neighbor array via edge-sharing (slow Python loop).

        Kept for testing / validation; production code uses
        ``_neighbors_from_delaunay`` instead.
        """
        n_tri = len(triangles)
        neighbors = np.full((n_tri, 3), -1, dtype=int)
        edge_to_tri = {}
        for i, tri in enumerate(triangles):
            for e in range(3):
                edge = tuple(sorted((tri[e], tri[(e + 1) % 3])))
                edge_to_tri.setdefault(edge, []).append(i)
        for i, tri in enumerate(triangles):
            ni = 0
            for e in range(3):
                edge = tuple(sorted((tri[e], tri[(e + 1) % 3])))
                for other in edge_to_tri[edge]:
                    if other != i:
                        neighbors[i, ni] = other
                        ni += 1
                        break
        return neighbors

    @staticmethod
    def _neighbors_from_delaunay(delaunay):
        """Extract per-triangle neighbor array from a Delaunay object.

        ``Delaunay.neighbors`` is an (n_triangles, 3) array where entry
        ``[i, j]`` is the index of the triangle that is the neighbor
        opposite vertex ``j`` of triangle ``i``, or -1 at the boundary.
        We just need to ensure the dtype is int.
        """
        return delaunay.neighbors.astype(int)

    def _precompute_water_neighbors(self):
        """Build padded water-neighbor lookup arrays.

        ``self._water_nbrs[i, :count]`` gives the water-neighbor indices
        for triangle *i*, where *count* = ``self._water_nbr_count[i]``.
        """
        n = self.n_triangles
        self._water_nbrs = np.full((n, 3), -1, dtype=np.intp)
        self._water_nbr_count = np.zeros(n, dtype=np.intp)
        for i in range(n):
            k = 0
            for j in range(3):
                nb = self.neighbors[i, j]
                if nb >= 0 and self.water_mask[nb]:
                    self._water_nbrs[i, k] = nb
                    k += 1
            self._water_nbr_count[i] = k

    def water_neighbors(self, tri_idx):
        cnt = self._water_nbr_count[tri_idx]
        return self._water_nbrs[tri_idx, :cnt].tolist()

    def find_triangle(self, lat, lon):
        return int(self._delaunay.find_simplex([lat, lon]))

    def gradient(self, field, tri_idx):
        nbrs = [n for n in self.neighbors[tri_idx] if n >= 0]
        if not nbrs:
            return (0.0, 0.0)
        c0 = self.centroids[tri_idx]
        f0 = field[tri_idx]
        # cos(lat) correction: scale longitude differences to physical distance
        cos_lat = np.cos(np.radians(c0[0]))
        dlat, dlon = 0.0, 0.0
        for n in nbrs:
            cn = self.centroids[n]
            df = field[n] - f0
            dc_lat = cn[0] - c0[0]
            dc_lon = (cn[1] - c0[1]) * cos_lat
            norm = np.sqrt(dc_lat**2 + dc_lon**2)
            if norm > 0:
                dlat += df * dc_lat / norm
                dlon += df * dc_lon / norm
        mag = np.sqrt(dlat**2 + dlon**2)
        if mag > 0:
            dlat /= mag
            dlon /= mag
        return (dlat, dlon)
