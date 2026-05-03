"""Sea-age enum for tracking marine-residency duration of returning salmon.

Tier C3.2 of the hatchery-vs-wild plan (see
docs/superpowers/specs/2026-05-03-hatchery-c3.2-seaage-design.md).
Used as int8 metadata on AgentPool agents; permanent at introduction.
Sentinel UNSET (-1) covers offspring (set by ReproductionEvent's
add_agents default) and non-Baltic legacy scenarios (where the
IntroductionEvent isinstance-discriminator skips sampling).

Mirrors the salmon_ibm/origin.py pattern: IntEnum class plus
module-constant aliases plus a names tuple for YAML / CSV
serialization.
"""
from enum import IntEnum


class SeaAge(IntEnum):
    UNSET = -1
    SW1 = 1
    SW2 = 2
    SW3 = 3


SEA_AGE_UNSET = SeaAge.UNSET
SEA_AGE_1SW = SeaAge.SW1
SEA_AGE_2SW = SeaAge.SW2
SEA_AGE_3SW = SeaAge.SW3
VALID_SEA_AGES: tuple[int, ...] = (SEA_AGE_1SW, SEA_AGE_2SW, SEA_AGE_3SW)

# Indexed by (SeaAge.value + 1) — UNSET=-1 → 0, SW1=1 → 2, SW2=2 → 3,
# SW3=3 → 4. Index 1 is the SW0 slot which is structurally unused
# (no agent ever has sea_age == 0 — the smolt phase is not modelled
# as a sea_age value but as a separate prior life stage). Future 4SW
# extension would append "4SW" at index 5.
SEA_AGE_NAMES: tuple[str, ...] = ("unset", "_unused_sw0", "1SW", "2SW", "3SW")
