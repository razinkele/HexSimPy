"""Origin enum for tracking wild vs hatchery agents.

Tier C1 of the hatchery-vs-wild plan (see
docs/superpowers/specs/2026-04-30-hatchery-origin-c1-design.md).
Used as int8 metadata on AgentPool agents; no behaviour change in
C1 — physiology divergence ships in C2.

Mirrors the DOState precedent in salmon_ibm/estuary.py:61-69 — IntEnum
class plus module-constant aliases plus a names tuple for YAML / CSV
serialization.
"""
from enum import IntEnum


class Origin(IntEnum):
    WILD = 0
    HATCHERY = 1


ORIGIN_WILD = Origin.WILD
ORIGIN_HATCHERY = Origin.HATCHERY
ORIGIN_NAMES = ("wild", "hatchery")  # index aligns with enum value
