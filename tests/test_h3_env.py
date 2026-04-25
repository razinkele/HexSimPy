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
