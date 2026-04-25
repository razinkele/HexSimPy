"""Earth-geometric constants shared across mesh backends.

Centralised here so ``TriMesh``, ``HexMesh``, and the forthcoming
``H3Mesh`` all use the same meter-to-degree approximation for
``metric_scale(lat)``.
"""
from __future__ import annotations

# WGS84 meridional average (meters per degree of latitude). Varies by
# <0.6% across Earth due to ellipsoid flattening; a single constant is
# accurate enough for sub-continental IBM domains.
M_PER_DEG_LAT = 110540.0

# WGS84 equatorial value (meters per degree of longitude at lat = 0).
# At latitude `lat`, the actual meters per degree is
# ``M_PER_DEG_LON_EQUATOR * cos(lat)`` — see ``Mesh.metric_scale``.
M_PER_DEG_LON_EQUATOR = 111320.0
