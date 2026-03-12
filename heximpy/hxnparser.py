"""HXN binary file parser for HexSim spatial data.

Reads PATCH_HEXMAP (.hxn), PATCH_GRID (.grid), and barrier (.hbf) files.
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class HexMap:
    """Parsed hex-map data from a .hxn file."""

    version: int
    nrows: int
    ncols: int
    flag: int  # 0 = wide, 1 = narrow
    max_val: float
    min_val: float
    hexzero: float
    values: np.ndarray

    @property
    def n_hexagons(self) -> int:
        """Compute total hexagon count from grid dimensions and flag."""
        if self.flag == 0:
            return self.ncols * self.nrows
        # flag == 1 (narrow)
        if self.nrows % 2 == 0:
            n_wide = self.nrows // 2
            n_narrow = n_wide
        else:
            n_wide = (self.nrows + 1) // 2
            n_narrow = n_wide - 1
        return self.ncols * n_wide + (self.ncols - 1) * n_narrow

    @classmethod
    def from_file(cls, path: str | Path) -> HexMap:
        """Read a .hxn file, auto-detecting PATCH_HEXMAP vs plain format.

        Raises ValueError if the file is a PATCH_GRID (.grid) file.
        """
        path = Path(path)
        with open(path, "rb") as f:
            peek = f.read(12)
            if peek == b"PATCH_HEXMAP":
                return cls._read_patch_hexmap(f)
            if peek[:10] == b"PATCH_GRID":
                raise ValueError(
                    f"File is PATCH_GRID format, not a hex-map: {path}"
                )
            # Plain format – rewind and read
            f.seek(0)
            return cls._read_plain(f)

    @classmethod
    def _read_patch_hexmap(cls, f) -> HexMap:
        """Read PATCH_HEXMAP format (37-byte header)."""
        version, nrows, ncols = struct.unpack("<III", f.read(12))
        (flag,) = struct.unpack("<B", f.read(1))
        max_val, min_val, hexzero = struct.unpack("<fff", f.read(12))
        # Data runs until b"HISTORY" marker or EOF
        rest = f.read()
        hist_idx = rest.find(b"HISTORY")
        data_bytes = rest[:hist_idx] if hist_idx >= 0 else rest
        values = np.frombuffer(data_bytes, dtype=np.float32)
        return cls(
            version=version,
            nrows=nrows,
            ncols=ncols,
            flag=flag,
            max_val=max_val,
            min_val=min_val,
            hexzero=hexzero,
            values=values,
        )

    @classmethod
    def _read_plain(cls, f) -> HexMap:
        """Read plain format (44-byte header)."""
        version, ncols, nrows = struct.unpack("<iii", f.read(12))
        (cell_size,) = struct.unpack("<d", f.read(8))
        origin_x, origin_y = struct.unpack("<dd", f.read(16))
        dtype_code, nodata = struct.unpack("<ii", f.read(8))
        n = ncols * nrows
        if dtype_code == 1:
            values = np.frombuffer(f.read(n * 4), dtype=np.float32)
        elif dtype_code == 2:
            values = np.frombuffer(f.read(n * 4), dtype=np.int32).astype(
                np.float32
            )
        else:
            raise ValueError(f"Unknown dtype_code: {dtype_code}")
        return cls(
            version=version,
            nrows=nrows,
            ncols=ncols,
            flag=0,  # plain format is always wide
            max_val=float(values.max()) if len(values) > 0 else 0.0,
            min_val=float(values.min()) if len(values) > 0 else 0.0,
            hexzero=0.0,
            values=values,
        )


@dataclass
class GridMeta:
    """Metadata from a PATCH_GRID (.grid) file."""

    version: int
    n_hexes: int
    ncols: int
    nrows: int
    flag: int
    georef: tuple[float, ...]  # (x_extent, 0.0, 0.0, y_extent, row_spacing)

    @property
    def edge(self) -> float:
        """Hexagon edge length derived from row_spacing."""
        return self.georef[4] / math.sqrt(3)

    @classmethod
    def from_file(cls, path: str | Path) -> GridMeta:
        """Read a PATCH_GRID (.grid) file (67-byte header).

        Raises ValueError if the file does not start with b"PATCH_GRID".
        """
        path = Path(path)
        with open(path, "rb") as f:
            magic = f.read(10)
            if magic != b"PATCH_GRID":
                raise ValueError(
                    f"Not a PATCH_GRID file (got {magic!r}): {path}"
                )
            version, n_hexes, ncols, nrows = struct.unpack("<IIII", f.read(16))
            (flag,) = struct.unpack("<B", f.read(1))
            georef = struct.unpack("<5d", f.read(40))
        return cls(
            version=version,
            n_hexes=n_hexes,
            ncols=ncols,
            nrows=nrows,
            flag=flag,
            georef=georef,
        )


@dataclass
class Barrier:
    """A single barrier edge entry from an .hbf file."""

    hex_id: int
    edge: int
    class_id: int
    class_name: str


def read_barriers(path: str | Path) -> list[Barrier]:
    """Parse an .hbf barrier file.

    The file contains C (classification) and E (edge) lines:
        C <id> <p1> <p2> "<name>"
        E <hex_id> <edge> <class_id>

    Returns a list of Barrier with class_name populated from classifications.
    Raises KeyError if an edge references an undefined classification.
    """
    path = Path(path)
    classifications: dict[int, str] = {}
    barriers: list[Barrier] = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if parts[0] == "C":
                class_id = int(parts[1])
                # Name is the last quoted token
                name = line.split('"')[1]
                classifications[class_id] = name
            elif parts[0] == "E":
                hex_id = int(parts[1])
                edge = int(parts[2])
                class_id = int(parts[3])
                class_name = classifications[class_id]
                barriers.append(
                    Barrier(
                        hex_id=hex_id,
                        edge=edge,
                        class_id=class_id,
                        class_name=class_name,
                    )
                )
    return barriers
