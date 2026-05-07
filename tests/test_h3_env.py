"""Unit tests for H3Environment — Phase 2.2 of the H3 backend plan.

Most tests construct a synthetic landscape NetCDF in a tmp_path so they
run quickly without depending on the Nemunas builder script.  One
end-to-end test consumes the real ``data/nemunas_h3_landscape.nc`` and
skips when it's absent.
"""
from __future__ import annotations

from pathlib import Path

import h3
import numpy as np
import pytest
import xarray as xr

from salmon_ibm.h3_env import H3Environment
from salmon_ibm.h3mesh import H3Mesh


PROJECT = Path(__file__).resolve().parent.parent
LANDSCAPE = PROJECT / "data" / "nemunas_h3_landscape.nc"


def _build_minimal_h3_nc(
    path,
    *,
    include_dist_from_sea: bool = True,
    dist_from_sea_arr: "np.ndarray | None" = None,
) -> None:
    """Construct a 4-cell synthetic H3 multires NC for env tests.

    Cells: 0 (OpenBaltic, water), 1 (CuronianLagoon, water),
           2 (Nemunas, water), 3 (Nemunas, land).
    """
    import xarray as xr
    import numpy as np

    n = 4
    h3_id = np.arange(n, dtype=np.uint64)
    resolution = np.full(n, 9, dtype=np.int8)
    lat = np.array([55.0, 55.3, 55.5, 55.51], dtype=np.float64)
    lon = np.array([21.0, 21.3, 21.5, 21.51], dtype=np.float64)
    depth = np.array([10.0, 3.0, 2.0, 0.0], dtype=np.float32)
    water_mask = np.array([True, True, True, False], dtype=bool)
    reach_id = np.array([0, 1, 2, 2], dtype=np.int8)
    # Bidirectional chain: 0-1-2; cell 3 is land (no neighbors).
    nbr_starts = np.array([0, 1, 3, 4, 4], dtype=np.int32)
    nbr_idx = np.array([1, 0, 2, 1], dtype=np.int32)
    # Time-indexed forcing fields (1 timestep, 4 cells).
    tos = np.full((1, n), 12.0, dtype=np.float32)
    sos = np.full((1, n), 7.0, dtype=np.float32)
    uo = np.zeros((1, n), dtype=np.float32)
    vo = np.zeros((1, n), dtype=np.float32)
    time = np.array(["2026-01-01"], dtype="datetime64[D]")

    data_vars = {
        "h3_id": ("cell", h3_id),
        "resolution": ("cell", resolution),
        "lat": ("cell", lat),
        "lon": ("cell", lon),
        "depth": ("cell", depth),
        "water_mask": ("cell", water_mask),
        "reach_id": ("cell", reach_id),
        "nbr_starts": ("cell_p1", nbr_starts),
        "nbr_idx": ("edge", nbr_idx),
        "tos": (("time", "cell"), tos),
        "sos": (("time", "cell"), sos),
        "uo": (("time", "cell"), uo),
        "vo": (("time", "cell"), vo),
    }
    if include_dist_from_sea:
        if dist_from_sea_arr is None:
            # Default valid: cell 0 source (0.0), distances increasing.
            dist_from_sea_arr = np.array(
                [0.0, 100.0, 200.0, np.nan], dtype=np.float32,
            )
        data_vars["dist_from_sea"] = ("cell", dist_from_sea_arr)

    coords = {"time": time}
    attrs = {
        "reach_names": "OpenBaltic,CuronianLagoon,Nemunas",
    }
    ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
    ds.to_netcdf(path, engine="h5netcdf")


def _build_mesh_from_nc(nc_path):
    """Build an H3MultiResMesh from the test NC path."""
    import xarray as xr
    import numpy as np
    from salmon_ibm.h3_multires import H3MultiResMesh

    ds = xr.open_dataset(str(nc_path), engine="h5netcdf")
    names_attr = ds.attrs.get("reach_names", "")
    reach_names = names_attr.split(",") if names_attr else []
    mesh = H3MultiResMesh(
        h3_ids=ds["h3_id"].values.astype(np.uint64),
        resolutions=ds["resolution"].values.astype(np.int8),
        centroids=np.column_stack([ds["lat"].values, ds["lon"].values]),
        nbr_starts=ds["nbr_starts"].values.astype(np.int32),
        nbr_idx=ds["nbr_idx"].values.astype(np.int32),
        water_mask=ds["water_mask"].values.astype(bool),
        depth=ds["depth"].values.astype(np.float32),
        areas=np.zeros(len(ds["h3_id"]), dtype=np.float32),
        reach_id=ds["reach_id"].values.astype(np.int8),
        reach_names=reach_names,
    )
    ds.close()
    return mesh


def _write_synthetic_landscape(tmp_path, cells, n_time=3):
    """Build a tiny landscape NetCDF mirroring the real builder's schema."""
    h3_ids = np.array([int(h3.str_to_int(c)) for c in cells], dtype=np.uint64)
    order = np.argsort(h3_ids)
    h3_ids = h3_ids[order]
    cells_sorted = [cells[i] for i in order]
    lats = np.array([h3.cell_to_latlng(c)[0] for c in cells_sorted])
    lons = np.array([h3.cell_to_latlng(c)[1] for c in cells_sorted])
    n = len(cells_sorted)
    # Per-time-step values that vary cell-by-cell so we can detect bad
    # permutations (each cell gets a unique recoverable value).
    rng = np.random.default_rng(0)
    base = np.datetime64("2011-06-01", "ns")
    times = base + np.arange(n_time, dtype="timedelta64[D]").astype("timedelta64[ns]")
    ds = xr.Dataset(
        {
            "h3_id":      (("cell",), h3_ids),
            "lat":        (("cell",), lats),
            "lon":        (("cell",), lons),
            "depth":      (("cell",), np.full(n, 5.0, dtype=np.float32)),
            "water_mask": (("cell",), np.ones(n, dtype=np.uint8)),
            "tos": (("time", "cell"),
                    rng.uniform(10, 20, (n_time, n)).astype(np.float32)),
            "sos": (("time", "cell"),
                    rng.uniform(0.5, 7.0, (n_time, n)).astype(np.float32)),
            "uo":  (("time", "cell"),
                    rng.uniform(-0.3, 0.3, (n_time, n)).astype(np.float32)),
            "vo":  (("time", "cell"),
                    rng.uniform(-0.3, 0.3, (n_time, n)).astype(np.float32)),
        },
        coords={"time": times},
    )
    out = tmp_path / "landscape.nc"
    ds.to_netcdf(out, format="NETCDF4", engine="h5netcdf")
    return out, ds


def test_h3_env_loads_canonical_field_names(tmp_path):
    """Canonical keys: tos→temperature, sos→salinity, uo→u_current,
    vo→v_current, plus a zero-filled ``ssh`` (movement.py reads it
    for upstream/downstream behaviour, see h3_env.py docstring).
    C4: ``dist_from_sea`` is also injected by from_netcdf — Case A
    (variable absent from NC) zero-fills it.

    ``env.fields`` is the per-cell snapshot at the current timestep;
    the full time-series lives on ``env._full_fields``.
    """
    center = h3.latlng_to_cell(55.3, 21.1, 9)
    cells = [center] + list(h3.grid_ring(center, 1))
    nc, _ = _write_synthetic_landscape(tmp_path, cells)
    mesh = H3Mesh.from_h3_cells(cells)
    env = H3Environment.from_netcdf(nc, mesh)

    assert set(env.fields.keys()) == {
        "temperature", "salinity", "u_current", "v_current", "ssh",
        "dist_from_sea",
    }
    for arr in env.fields.values():
        assert arr.shape == (mesh.n_cells,)
        assert arr.dtype == np.float32
    for arr in env._full_fields.values():
        assert arr.shape == (3, mesh.n_cells)
    # ssh is synthesised (zero-filled) — confirm it is.
    np.testing.assert_array_equal(env.fields["ssh"], np.zeros(mesh.n_cells))


def test_h3_env_aligns_field_columns_with_mesh_h3_ids(tmp_path):
    """The i-th element of every snapshot must correspond to mesh.h3_ids[i].

    Build a landscape, then construct the mesh with cells in REVERSE
    order — searchsorted+order must permute correctly.
    """
    center = h3.latlng_to_cell(55.3, 21.1, 9)
    cells = [center] + list(h3.grid_ring(center, 1))
    nc, ds = _write_synthetic_landscape(tmp_path, cells)
    # Mesh in REVERSE order — its h3_ids[0] is the largest cell ID.
    mesh = H3Mesh.from_h3_cells(list(reversed(cells)))
    env = H3Environment.from_netcdf(nc, mesh)
    env.advance(0)

    # For each mesh cell, find its row in the original ds and confirm
    # env.fields["temperature"][i] matches ds["tos"][0, ds_idx].
    ds_ids = ds["h3_id"].values  # already sorted ascending
    ds_temps = ds["tos"].values  # (time=3, cell=N)
    for i in range(mesh.n_cells):
        ds_idx = int(np.searchsorted(ds_ids, mesh.h3_ids[i]))
        assert env.fields["temperature"][i] == ds_temps[0, ds_idx]


def test_h3_env_advance_mutates_fields_in_place(tmp_path):
    """advance() updates env.fields in place — callers that captured a
    reference to the dict (the Simulation step loop does) must see the
    new data without re-fetching."""
    center = h3.latlng_to_cell(55.3, 21.1, 9)
    cells = [center] + list(h3.grid_ring(center, 1))
    nc, _ = _write_synthetic_landscape(tmp_path, cells, n_time=3)
    mesh = H3Mesh.from_h3_cells(cells)
    env = H3Environment.from_netcdf(nc, mesh)

    captured = env.fields  # what Simulation does: landscape["fields"] = env.fields
    env.advance(0)
    t0 = captured["temperature"].copy()
    env.advance(2)
    t2_after_advance = captured["temperature"]
    # Same dict, same array object — but contents are now t=2 values.
    assert captured is env.fields
    different = not np.array_equal(t0, t2_after_advance)
    assert different, "advance(2) didn't update the captured fields dict"

    # advance() must clamp to range — beyond-end stays at last index.
    env.advance(99)
    np.testing.assert_array_equal(
        captured["temperature"], t2_after_advance
    )


def test_h3_env_advance_hourly_step_to_daily_data(tmp_path):
    """For daily CMEMS data on an hourly simulation, advance(step) should
    map step→day via int division, not blow up with IndexError."""
    center = h3.latlng_to_cell(55.3, 21.1, 9)
    cells = [center] + list(h3.grid_ring(center, 1))
    # 30-day daily forcing.
    nc, _ = _write_synthetic_landscape(tmp_path, cells, n_time=30)
    mesh = H3Mesh.from_h3_cells(cells)
    env = H3Environment.from_netcdf(nc, mesh)

    # 720 hourly steps over 30 days — must not IndexError.
    for step in [0, 23, 24, 25, 100, 500, 719]:
        env.advance(step)  # implicit assertion: doesn't raise
        # Each call leaves env.fields populated.
        for arr in env.fields.values():
            assert arr.shape == (mesh.n_cells,)


def test_h3_env_missing_cell_raises(tmp_path):
    """If the mesh has a cell not present in the NetCDF, fail loudly."""
    center = h3.latlng_to_cell(55.3, 21.1, 9)
    ring = list(h3.grid_ring(center, 1))
    # Landscape has only the centre + 3 of the 6 ring cells.
    nc, _ = _write_synthetic_landscape(tmp_path, [center] + ring[:3])
    # Mesh has all 7 cells.
    mesh = H3Mesh.from_h3_cells([center] + ring)
    with pytest.raises(ValueError, match="not in forcing NetCDF"):
        H3Environment.from_netcdf(nc, mesh)


def test_h3_env_sample_returns_active_timestep_array(tmp_path):
    center = h3.latlng_to_cell(55.3, 21.1, 9)
    cells = [center] + list(h3.grid_ring(center, 1))
    nc, _ = _write_synthetic_landscape(tmp_path, cells)
    mesh = H3Mesh.from_h3_cells(cells)
    env = H3Environment.from_netcdf(nc, mesh)
    env.advance(1)
    sampled = env.sample("temperature")
    assert sampled.shape == (mesh.n_cells,)
    np.testing.assert_array_equal(sampled, env.fields["temperature"])


# ---------------------------------------------------------------------------
# End-to-end on the real Nemunas landscape (skipped if absent)
# ---------------------------------------------------------------------------


def _needs_landscape():
    if not LANDSCAPE.exists():
        pytest.skip(
            f"{LANDSCAPE.name} missing — run "
            "scripts/build_nemunas_h3_landscape.py to generate it"
        )


def test_h3_env_real_nemunas_landscape_envelope():
    """Real CMEMS data sanity check: tos within Baltic surface envelope."""
    _needs_landscape()
    ds = xr.open_dataset(LANDSCAPE, engine="h5netcdf")
    cells = [h3.int_to_str(int(x)) for x in ds["h3_id"].values]
    mesh = H3Mesh.from_h3_cells(
        cells,
        depth=ds["depth"].values,
        water_mask=ds["water_mask"].values.astype(bool),
    )
    env = H3Environment.from_netcdf(LANDSCAPE, mesh)
    env.advance(0)
    temps = env.current()["temperature"]
    sal = env.current()["salinity"]
    assert temps.shape == (mesh.n_cells,)
    assert -2.0 <= float(temps.min()) <= float(temps.max()) <= 30.0, (
        f"summer temp envelope violated: "
        f"[{float(temps.min()):.2f}, {float(temps.max()):.2f}]"
    )
    assert 0.0 <= float(sal.min()) <= float(sal.max()) <= 10.0


# ---------------------------------------------------------------------------
# C4: dist_from_sea load — Case A (variable absent from NC)
# ---------------------------------------------------------------------------


def test_from_netcdf_case_a_dist_from_sea_missing(tmp_path, caplog):
    """C4 Test 4: NC missing dist_from_sea variable triggers warn +
    zero-fill + flag init."""
    import logging
    from salmon_ibm.h3_env import H3Environment, ERR_DIST_FROM_SEA_MISSING

    # Build a minimal NC matching the existing schema MINUS dist_from_sea.
    nc_path = tmp_path / "test_minimal.nc"
    _build_minimal_h3_nc(nc_path, include_dist_from_sea=False)

    # Build a minimal H3MultiResMesh via the same NC.
    mesh = _build_mesh_from_nc(nc_path)

    caplog.set_level(logging.WARNING, logger="salmon_ibm.h3_env")
    env = H3Environment.from_netcdf(str(nc_path), mesh)

    # (a) Warning emitted with the err-id.
    matching = [
        r for r in caplog.records
        if r.name == "salmon_ibm.h3_env"
        and ERR_DIST_FROM_SEA_MISSING in r.getMessage()
    ]
    assert matching, (
        f"Expected warning with err-id {ERR_DIST_FROM_SEA_MISSING}; "
        f"got {[(r.name, r.getMessage()) for r in caplog.records]!r}"
    )

    # (b) fields["dist_from_sea"] exists and is all-zeros.
    assert "dist_from_sea" in env.fields
    assert env.fields["dist_from_sea"].shape == (mesh.n_cells,)
    assert np.all(env.fields["dist_from_sea"] == 0.0)
    assert env.fields["dist_from_sea"].dtype == np.float32

    # (c) Per-env latch flag initialised.
    assert env._dormant_gradient_check_done is False

    # (d) mesh.dist_from_sea attached (same array reference).
    assert mesh.dist_from_sea is env.fields["dist_from_sea"]
