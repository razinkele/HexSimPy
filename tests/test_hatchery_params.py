"""Tests for the C2 hatchery vs wild parameter divergence (Tier C2).

Spec: docs/superpowers/specs/2026-05-01-hatchery-c2-bioparams-design.md
"""
import pytest


def test_activity_by_behavior_rejects_nonpositive_value():
    """BalticBioParams.__post_init__ rejects activity_by_behavior with
    non-positive values. Tests both negative AND zero (boundary case)
    so a future 'optimisation' that changes guard from `v <= 0` to
    `v < 0` would regress visibly."""
    from salmon_ibm.baltic_params import BalticBioParams
    with pytest.raises(ValueError, match="positive floats"):
        BalticBioParams(activity_by_behavior={0: 1.0, 1: -0.5, 2: 0.8, 3: 1.5, 4: 1.0})
    with pytest.raises(ValueError, match="positive floats"):
        BalticBioParams(activity_by_behavior={0: 1.0, 1: 0.0, 2: 0.8, 3: 1.5, 4: 1.0})
