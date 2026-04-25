# Multi-Resolution H3 — Feasibility Analysis

**Question:** Can the HexSimPy salmon IBM use *different* H3 resolutions
for the rivers (fine) versus the Curonian Lagoon and Baltic Sea
(coarse), so simulation cost concentrates where the geometry actually
matters?

**Short answer:** Yes — possible but non-trivial. ~1-2 weeks of focused
work plus testing. A single finer uniform resolution (res 10 instead of
res 9) is the lower-effort middle ground. Details below.

## Why this question is interesting

The Nemunas Delta channels are 50–250 m wide. At H3 resolution 9
(~200 m edge), a typical channel is 1–2 cells wide and individual
delta arms (Šyša, Skirvytė) are barely resolved.

Pumping the resolution to 10 (~75 m edge) gives 7× more cells per area
and resolves delta channels well — but also 7× more cells in the open
lagoon and the Baltic Sea, which don't *need* the extra detail because
the field is smooth there. Multi-resolution is the way to put cells
where they matter.

## H3 itself supports mixed resolutions

H3's primitives don't *require* uniform resolution:

* `h3.cell_area(cell, unit="m^2")` — already per-cell; resolution is
  inferred from the cell ID.
* `h3.cell_to_parent(cell, parent_res)` and
  `h3.cell_to_children(cell, child_res)` — navigate up/down the
  resolution hierarchy. A coarse cell at res 8 contains 7 cells at
  res 9 (one of them is a centre child, six are perimeter children).
* `h3.is_pentagon(cell)` — works at any resolution.

What H3 does *not* directly provide:

* **Cross-resolution adjacency.** `h3.grid_ring(cell, 1)` returns same-
  resolution neighbours only. There's no built-in "give me the cell on
  the other side of this edge that may be at a different resolution".
* **Mixed-resolution polygon tessellation.** `h3.polygon_to_cells()`
  takes a single resolution.

## What HexSimPy assumes today

`salmon_ibm/h3mesh.py::H3Mesh` is single-resolution by construction:

* `from_h3_cells()` reads `resolution = h3.get_resolution(h3_cells[0])`
  and treats it as the mesh-wide constant.
* `neighbors` is built from `h3.grid_ring(cell, 1)` — same-resolution
  neighbours only.
* `MAX_NBRS = 6` — pentagons get a `-1` sentinel, but no provision for
  *more* than 6 (which can happen at a coarse-fine boundary; see
  below).

Movement, bioenergetics, and the deck.gl viewer all consume `mesh.
centroids` + `mesh.neighbors` + `mesh.areas` arrays. None of those
arrays inherently require uniform resolution — the constraint is the
neighbour-table assumption.

## What a multi-resolution build would need

### 1. Per-reach resolution

`configs/config_nemunas_h3.yaml` would gain a per-reach resolution map,
e.g.

```yaml
mesh_backend: h3_multi
reaches:
  Nemunas:        { polygon: ..., resolution: 11 }   # ~28 m
  Atmata:         { polygon: ..., resolution: 11 }
  Minija:         { polygon: ..., resolution: 11 }
  CuronianLagoon: { polygon: ..., resolution: 9 }    # ~200 m
  BalticCoast:    { polygon: ..., resolution: 9 }
```

The build script tessellates each reach's polygon at its own resolution
and unions the cells.

### 2. Cross-resolution neighbour table

This is where the work concentrates. At a boundary between a res-11
reach (rivers) and a res-9 reach (lagoon), a river cell on the bank has:

* 1–6 same-resolution neighbours inside the river reach.
* 1–2 *coarse* neighbours in the lagoon — found by computing the
  centroid of the fine cell's would-be 7th neighbour (which lives in
  the lagoon zone but doesn't exist there as a fine cell), calling
  `h3.latlng_to_cell(lat, lon, 9)` to find the coarse cell that
  *contains* that point, and adding that coarse cell to the fine
  cell's neighbour list.

Symmetrically, a coarse lagoon cell on the bank has:

* 5–6 same-resolution neighbours inside the lagoon.
* 4–7 *fine* neighbours in the river — found by enumerating the
  coarse cell's children at res 11 along the shared edge, then
  intersecting with the river's fine-cell list.

Implementation sketch (~200 LOC):

```python
def _build_multi_res_neighbors(cells, resolutions, id_to_idx):
    """Find each cell's neighbours, possibly at a different resolution."""
    neighbours = []
    for i, (cell, r) in enumerate(zip(cells, resolutions)):
        nb_indices = []
        # 1. Same-resolution ring neighbours that exist in the mesh.
        for nb in h3.grid_ring(cell, 1):
            j = id_to_idx.get(int(h3.str_to_int(nb)))
            if j is not None:
                nb_indices.append(j)
        # 2. Same-resolution "missing" ring neighbours: probe the
        # centroid in case a coarser cell at lower resolution covers it.
        for nb in h3.grid_ring(cell, 1):
            if int(h3.str_to_int(nb)) in id_to_idx:
                continue
            lat, lon = h3.cell_to_latlng(nb)
            for r_lower in range(r - 1, MIN_RES - 1, -1):
                parent = h3.latlng_to_cell(lat, lon, r_lower)
                j = id_to_idx.get(int(h3.str_to_int(parent)))
                if j is not None:
                    if j not in nb_indices:
                        nb_indices.append(j)
                    break
        # 3. If THIS cell is coarse, also look for fine neighbours
        # along its boundary — children of its res-1-ring at the
        # finer-reach resolution.
        for child_r in range(r + 1, MAX_RES + 1):
            for ring_nb in h3.grid_ring(cell, 1):
                for child in h3.cell_to_children(ring_nb, child_r):
                    j = id_to_idx.get(int(h3.str_to_int(child)))
                    if j is not None and j not in nb_indices:
                        nb_indices.append(j)
        neighbours.append(nb_indices)
    return neighbours
```

This is O(N · k · resolutions_in_use) — about 200 ms for our
~50 k-cell mesh.

### 3. `MAX_NBRS` raised + variable

Today `MAX_NBRS = 6` and the neighbour table is a fixed `(N, 6)`
array. Cross-resolution boundaries can produce up to **12** neighbours
for a coarse cell (the 6 normal ring neighbours + up to 6 fine
children of the cells outside the coarse zone). Two options:

* Bump `MAX_NBRS = 12` and accept some wasted slots in single-res zones.
* Switch to a CSR-style ragged array (`row_starts: (N+1,) int32`,
  `nbr_idx: (M,) int32`) — same shape Numpy/numba can handle, but
  movement kernels need to read `[row_starts[i]:row_starts[i+1]]`
  instead of fixed slots.

The CSR change touches `_step_directed_numba`, `_advection_numba`, and
the gradient computation in `H3Mesh.gradient()`. Maybe 100 LOC of
mechanical edits + tests.

### 4. Movement — disaggregation when crossing resolutions

When an agent at a coarse cell moves to a fine neighbour, it lands in
1 of (typically) 1–6 fine children of the boundary's nearest coarse
neighbour. Pick uniformly random child weighted by water-mask.

When an agent at a fine cell moves to a coarse neighbour, no
disaggregation needed — it's just one cell.

### 5. Forcing sampling

CMEMS forcing is regridded onto the mesh in
`scripts/build_nemunas_h3_landscape.py:sample_cmems()`. Currently
samples at each centroid via `RegularGridInterpolator`. Multi-res
would resample at each cell's centroid regardless of resolution — no
change needed.

Bathymetry sampling: same — point sample at the centroid.

The only subtlety: a coarse cell's "depth" is now an average over a
much bigger area (~13 km² at res 9 vs ~92 k m² at res 10). For depth
this is fine (lagoon depth varies smoothly). For currents, the coarse
cell averages out finer features like shoreline gyres — but that's
intrinsic to choosing a coarser resolution for the lagoon, not a bug.

## Recommendation

For HexSimPy's current scope (a delta-and-lagoon salmon migration
model), I recommend the **lower-effort path**:

1. **Now:** stay at uniform res 9. With the inSTREAM-polygon water
   mask (v1.2.6), inland leak is gone and the open delta channels
   are 1–2 cells wide. Sufficient for movement and bioenergetics on
   the lagoon-scale.

2. **If finer rivers are needed:** rebuild at uniform res 10 (~75 m
   edge). Cell count grows from 106 k to ~740 k total, with ~360 k
   water cells. Tractable; the simulation step is dominated by
   per-agent operations (50 agents) not per-cell, so step time
   barely changes. The viewer's `h3_hexagon_layer` payload grows
   from 5 MB to ~35 MB JSON — borderline; might need
   subsampling for the static layer. *Cost: < 1 day of work.*

3. **Multi-resolution:** revisit if a publication or stakeholder
   requirement specifically calls for sub-50 m river resolution
   alongside km-scale lagoon coverage. *Cost: 1–2 weeks of work
   + correctness testing.* The implementation sketch above is the
   skeleton.

The bigger ROI right now is in **per-reach simulation parameters**
(food density, predation, salinity tolerance) — not the geometry. The
inSTREAM example_baltic config has 9 reaches each with its own
ecology. Replicating that taxonomy in HexSimPy (mapping H3 cells to
reach IDs, sampling reach-specific bioenergetics parameters) is the
natural next step before chasing variable resolution.
