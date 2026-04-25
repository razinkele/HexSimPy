"""Shared fetchers for authoritative water-body polygons.

Two sources, fused into one shapely (Multi)Polygon:

* **OpenStreetMap** (Overpass API) — `natural=water`, `water=*`, and
  `natural=bay`.  Catches the Curonian Lagoon, Nemunas channels,
  small lakes, ponds.  Multi-segment relations are stitched via
  `shapely.ops.linemerge` + `polygonize`.
* **Natural Earth 1:10m ocean** — single global ocean polygon
  clipped to the bbox.  Catches the Baltic Sea, which OSM tags as
  `place=sea` with a relation that bbox-clipped Overpass can't
  reliably return.

Used by:
* ``scripts/build_nemunas_h3_landscape.py`` — to AND-mask the
  bathymetry-based water test, so cells on below-sea-level land
  (Nemunas Delta polders!) don't leak in.
* ``scripts/validate_h3_landscape_vs_osm.py`` — to validate an
  already-built landscape NC.

The union is cached on disk as a GeoJSON so the builder doesn't
need a live Overpass query on every rebuild.  Refresh by deleting
the cache.
"""
from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import geopandas as gpd
import requests
from shapely.geometry import LineString, Polygon, box
from shapely.ops import linemerge, polygonize, unary_union


# Public bbox tuple shared with the validator: (south, west, north, east).
BBOX = (54.9, 20.4, 55.8, 21.9)

PROJECT = Path(__file__).resolve().parent.parent

# Natural Earth — 1:10m physical ocean polygon (~1 MB zipped).
NE_OCEAN_URL = "https://naciscdn.org/naturalearth/10m/physical/ne_10m_ocean.zip"
NE_DIR = PROJECT / "data" / "naturalearth_ocean"
NE_SHP = NE_DIR / "ne_10m_ocean.shp"

# inSTREAM example_baltic shapefile — copied from
# inSTREAM/instream-py/tests/fixtures/example_baltic/Shapefile/.
# 9 reaches: Nemunas, Atmata, Minija, Sysa, Skirvyte, Leite, Gilija,
# CuronianLagoon, BalticCoast.  EPSG:3035 in the file; we reproject
# to 4326 at load time.  Far tighter inland coverage than OSM —
# OSM polygons over-include floodplain that's actually farmland.
INSTREAM_SHP = PROJECT / "data" / "instream_baltic_polygons" / "BalticExample.shp"

# Cached fused water-polygon union (regenerable; safe to delete).
WATER_UNION_GEOJSON = PROJECT / "data" / "nemunas_water_union.geojson"

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
QUERY = f"""
[out:json][timeout:90];
(
  way["natural"="water"]({BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]});
  way["water"]({BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]});
  way["natural"="bay"]({BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]});
  relation["natural"="water"]({BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]});
  relation["water"]({BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]});
  relation["natural"="bay"]({BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]});
  relation["place"="sea"]({BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]});
);
out geom;
"""

# Overpass / its CDN reject default `python-requests/X.X.X` UAs (HTTP
# 406).  Identify ourselves with a contact handle per Overpass
# etiquette — same header satisfies the Natural Earth CDN.
_HEADERS = {
    "User-Agent": "HexSimPy-validator/1.0 (razinkele@github.com)",
    "Accept": "application/json",
}


def fetch_osm_water() -> gpd.GeoDataFrame:
    """Query Overpass and return all bbox water polygons in EPSG:4326."""
    print(f"  [Overpass] fetching water polygons over "
          f"{BBOX[0]:.2f}-{BBOX[2]:.2f}N, {BBOX[1]:.2f}-{BBOX[3]:.2f}E…")
    r = requests.post(
        OVERPASS_URL,
        data={"data": QUERY},
        headers=_HEADERS,
        timeout=120,
    )
    if not r.ok:
        print(f"    HTTP {r.status_code}: {r.text[:400]}", file=sys.stderr)
        r.raise_for_status()
    elements = r.json().get("elements", [])
    print(f"    {len(elements)} OSM elements returned")

    geoms = []
    for el in elements:
        try:
            if el["type"] == "way" and "geometry" in el:
                coords = [(p["lon"], p["lat"]) for p in el["geometry"]]
                if len(coords) < 4 or coords[0] != coords[-1]:
                    continue
                geoms.append(Polygon(coords))
            elif el["type"] == "relation" and "members" in el:
                outer_lines: list[LineString] = []
                inner_lines: list[LineString] = []
                for mem in el["members"]:
                    if mem.get("type") != "way":
                        continue
                    coords = [(p["lon"], p["lat"]) for p in mem.get("geometry", [])]
                    if len(coords) < 2:
                        continue
                    role = mem.get("role", "outer")
                    line = LineString(coords)
                    if role == "inner":
                        inner_lines.append(line)
                    else:
                        outer_lines.append(line)
                if not outer_lines:
                    continue
                # Stitch member ways into closed rings.  The Curonian
                # Lagoon and Baltic coast multipolygons need this.
                try:
                    merged = linemerge(outer_lines)
                    outer_polys = list(polygonize(merged))
                except Exception:
                    outer_polys = []
                if not outer_polys:
                    continue
                inner_polys = []
                if inner_lines:
                    try:
                        inner_polys = list(polygonize(linemerge(inner_lines)))
                    except Exception:
                        inner_polys = []
                for op in outer_polys:
                    g = op
                    for ip in inner_polys:
                        if op.contains(ip):
                            g = g.difference(ip)
                    geoms.append(g)
        except Exception as e:
            print(f"    ! skipping element {el.get('id')}: {e}")

    print(f"    {len(geoms)} valid polygon geometries")
    if not geoms:
        raise RuntimeError("no OSM water polygons in bbox — Overpass empty?")
    gdf = gpd.GeoDataFrame(geometry=geoms, crs="EPSG:4326")
    gdf["geometry"] = gdf.geometry.buffer(0)  # heal self-intersections
    return gdf


def fetch_natural_earth_ocean() -> gpd.GeoDataFrame:
    """Return the NE 1:10m ocean polygon clipped to ``BBOX``."""
    if not NE_SHP.exists():
        print(f"  [NE] downloading ocean polygon → {NE_DIR}")
        NE_DIR.mkdir(parents=True, exist_ok=True)
        r = requests.get(NE_OCEAN_URL, headers=_HEADERS, timeout=60)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            zf.extractall(NE_DIR)
        if not NE_SHP.exists():
            raise RuntimeError(f"NE shapefile missing after extract: {NE_SHP}")

    ocean = gpd.read_file(NE_SHP)
    if ocean.crs is None:
        ocean = ocean.set_crs("EPSG:4326")
    elif ocean.crs.to_epsg() != 4326:
        ocean = ocean.to_crs("EPSG:4326")

    bbox_poly = box(BBOX[1], BBOX[0], BBOX[3], BBOX[2])
    clipped = ocean.intersection(bbox_poly)
    clipped = clipped[~clipped.is_empty]
    if clipped.empty:
        raise RuntimeError("NE ocean polygon does not intersect bbox")
    return gpd.GeoDataFrame(geometry=list(clipped), crs="EPSG:4326")


def fetch_instream_polygons() -> gpd.GeoDataFrame:
    """Return the inSTREAM example_baltic 9-reach polygons in EPSG:4326.

    Loads ``data/instream_baltic_polygons/BalticExample.shp`` and
    reprojects from EPSG:3035 (LAEA Europe, metric) to EPSG:4326
    (lat/lon).  Tighter inland coverage than OSM — OSM `natural=water`
    polygons over-include floodplain that's actually farmland and
    polderland.  inSTREAM polygons came from a hand-curated combination
    of OSM rivers (clipped to channel widths) plus a published Curonian
    Lagoon outline plus a Baltic-coast strip.
    """
    if not INSTREAM_SHP.exists():
        raise FileNotFoundError(
            f"inSTREAM shapefile not found: {INSTREAM_SHP}\n"
            f"Copy it from inSTREAM/instream-py/tests/fixtures/"
            f"example_baltic/Shapefile/."
        )
    gdf = gpd.read_file(INSTREAM_SHP)
    if gdf.crs is None:
        # The .prj should make this never happen, but be defensive.
        gdf = gdf.set_crs("EPSG:3035")
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    return gdf


def get_water_union(refresh: bool = False, source: str = "instream"):
    """Return the cached water multi-polygon for the H3 build mask.

    ``source`` selects which inland polygons to combine with the
    Natural Earth ocean:

    * ``"instream"`` (default since v1.2.6) — inSTREAM example_baltic
      9-reach shapefile.  Tightest inland coverage; the H3 build's
      previous OSM-based mask leaked ~1 000 cells onto Nemunas-Delta
      farmland.
    * ``"osm"`` — OpenStreetMap Overpass query (legacy default).

    The cached GeoJSON is keyed to a single source, so switching
    sources requires ``refresh=True`` to rebuild.

    Returns the shapely geometry directly — callers usually want
    ``.contains(point)`` rather than a GeoDataFrame.
    """
    if WATER_UNION_GEOJSON.exists() and not refresh:
        gdf = gpd.read_file(WATER_UNION_GEOJSON)
        return unary_union(gdf.geometry.values)

    if source == "instream":
        instream = fetch_instream_polygons()
        ne_ocean = fetch_natural_earth_ocean()
        inland_geoms = list(instream.geometry.values)
        union = unary_union(inland_geoms + list(ne_ocean.geometry.values))
        print(f"  built water union from inSTREAM ({len(instream)} polygons) "
              f"+ NE ocean ({len(ne_ocean)} polygon)")
    elif source == "osm":
        osm = fetch_osm_water()
        ne_ocean = fetch_natural_earth_ocean()
        union = unary_union(
            list(osm.geometry.values) + list(ne_ocean.geometry.values),
        )
        print(f"  built water union from OSM ({len(osm)} polygons) "
              f"+ NE ocean ({len(ne_ocean)} polygon)")
    else:
        raise ValueError(f"unknown source: {source!r}")

    WATER_UNION_GEOJSON.parent.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame(geometry=[union], crs="EPSG:4326").to_file(
        WATER_UNION_GEOJSON, driver="GeoJSON"
    )
    print(f"  cached water union → {WATER_UNION_GEOJSON.relative_to(PROJECT)}")
    return union
