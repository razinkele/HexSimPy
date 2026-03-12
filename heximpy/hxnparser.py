"""HXN binary file parser for HexSim spatial data.

Reads PATCH_HEXMAP (.hxn), PATCH_GRID (.grid), and barrier (.hbf) files.
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

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


@dataclass
class Barrier:
    """A single barrier edge entry from an .hbf file."""

    hex_id: int
    edge: int
    class_id: int
    class_name: str


def read_barriers(path: str | Path) -> list[Barrier]:
    raise NotImplementedError
