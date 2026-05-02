"""Tests for the C3.1 hatchery vs wild pre-spawn skip probability.

Spec: docs/superpowers/specs/2026-05-02-hatchery-c3-spawn-design.md
"""
import pytest


def test_pre_spawn_skip_prob_rejects_out_of_range():
    """BalticBioParams.__post_init__ rejects pre_spawn_skip_prob outside
    [0, 1]. Locks the validation contract; covers both negative and
    >1.0 boundary cases. C3.1 spec mandates 0.0 <= p <= 1.0."""
    from salmon_ibm.baltic_params import BalticBioParams
    with pytest.raises(ValueError, match=r"pre_spawn_skip_prob must be"):
        BalticBioParams(pre_spawn_skip_prob=-0.1)
    with pytest.raises(ValueError, match=r"pre_spawn_skip_prob must be"):
        BalticBioParams(pre_spawn_skip_prob=1.5)
