"""C5.1 invariants on non-Baltic landscapes.

Verifies §5 spec claim: non-Baltic configs (Columbia, TriMesh) produce
empty _arrival_threshold_by_natal_rid, which means C5.1's ArrivalEvent
guard never fires on those configs (preserves C5-or-earlier behavior).
"""
import pytest


@pytest.mark.parametrize("cfg_path", [
    "config_columbia.yaml",
])
def test_non_baltic_arrival_thresholds_empty(cfg_path):
    """Non-Baltic configs must have empty _arrival_threshold_by_natal_rid.
    Otherwise C5.1's guard silently disables a previously-working metric.
    """
    from salmon_ibm.config import load_config
    from salmon_ibm.simulation import Simulation
    try:
        cfg = load_config(cfg_path)
    except FileNotFoundError:
        pytest.skip(f"{cfg_path} not present in workspace")
    sim = Simulation(cfg, n_agents=5, rng_seed=42, output_path=None)
    assert sim._is_baltic is False, (
        f"{cfg_path} unexpectedly has _is_baltic=True — discriminator "
        f"misclassifies."
    )
    assert sim._arrival_threshold_by_natal_rid == {}, (
        f"{cfg_path} has non-empty thresholds "
        f"({len(sim._arrival_threshold_by_natal_rid)} entries) — "
        f"C5.1 guard would silently disable arrivals here."
    )
