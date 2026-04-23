import numpy as np
import pytest
from salmon_ibm.estuary import salinity_cost, do_override, seiche_pause


def test_salinity_cost_below_tolerance():
    cost = salinity_cost(np.array([3.0]), S_opt=0.5, S_tol=6.0, k=0.6)
    np.testing.assert_allclose(cost, [1.0])


def test_salinity_cost_above_tolerance():
    cost = salinity_cost(np.array([10.0]), S_opt=0.5, S_tol=6.0, k=0.6)
    np.testing.assert_allclose(cost, [3.1])


def test_do_override_normal():
    override = do_override(np.array([8.0]), lethal=2.0, high=4.0)
    assert override[0] == 0


def test_do_override_high():
    override = do_override(np.array([3.0]), lethal=2.0, high=4.0)
    assert override[0] == 1


def test_do_override_lethal():
    override = do_override(np.array([1.5]), lethal=2.0, high=4.0)
    assert override[0] == 2


def test_seiche_pause_calm():
    paused = seiche_pause(np.array([0.005]), thresh=0.02)
    assert not paused[0]


def test_seiche_pause_active():
    paused = seiche_pause(np.array([0.05]), thresh=0.02)
    assert paused[0]


def test_salinity_cost_capped():
    """Salinity cost should not exceed a maximum cap."""
    from salmon_ibm.estuary import salinity_cost

    extreme_sal = np.array([50.0])  # extreme ocean salinity
    cost = salinity_cost(extreme_sal)
    assert cost[0] <= 5.0, f"Cost {cost[0]} should be capped at 5.0"
    assert cost[0] > 1.0, "Cost should still be > 1.0 for high salinity"


def test_salinity_cost_nan_treated_as_zero():
    """NaN salinity should produce cost = 1.0 (no penalty)."""
    from salmon_ibm.estuary import salinity_cost

    sal = np.array([np.nan, 5.0, np.nan])
    cost = salinity_cost(sal)
    assert cost[0] == pytest.approx(1.0), "NaN salinity should give neutral cost"
    assert not np.isnan(cost).any(), "No NaN should propagate"


def test_do_override_nan_treated_as_ok():
    """NaN dissolved oxygen should be classified as DO_OK, not DO_ESCAPE."""
    from salmon_ibm.estuary import do_override, DO_OK

    do_vals = np.array([np.nan, 5.0, np.nan])
    result = do_override(do_vals)
    assert result[0] == DO_OK, "NaN DO should be classified as DO_OK"
    assert result[2] == DO_OK, "NaN DO should be classified as DO_OK"


def test_do_override_rejects_inverted_thresholds():
    """lethal > high should raise ValueError."""
    from salmon_ibm.estuary import do_override

    with pytest.raises(ValueError, match="lethal.*must be.*<=.*high"):
        do_override(np.array([3.0]), lethal=5.0, high=2.0)


def test_estuary_params_defaults_match_liland_2024():
    from salmon_ibm.estuary import EstuaryParams
    p = EstuaryParams()
    assert p.do_lethal == 3.0
    assert p.do_high == 5.5
    assert p.s_opt == 0.5
    assert p.s_tol == 6.0
    assert p.seiche_threshold_m_per_s == 0.02


def test_validate_do_field_units_accepts_mg_per_l():
    """Typical mg/L field (0-20 range) passes."""
    import numpy as np
    from salmon_ibm.estuary import validate_do_field_units
    validate_do_field_units(np.array([0.0, 5.0, 10.0, 15.0, 20.0]))
    # NaN handling
    validate_do_field_units(np.array([np.nan, 8.0, 10.0]))


def test_validate_do_field_units_rejects_mmol_per_m3():
    """Typical mmol/m^3 field (150-400 range) must raise."""
    import pytest
    import numpy as np
    from salmon_ibm.estuary import validate_do_field_units
    with pytest.raises(ValueError, match="mmol/m"):
        validate_do_field_units(np.array([250.0, 300.0, 280.0]))


def test_do_override_uses_new_defaults():
    """do_override with default args uses Liland 2024 thresholds (3.0 / 5.5)."""
    import numpy as np
    from salmon_ibm.estuary import do_override, DO_OK, DO_ESCAPE, DO_LETHAL
    # 6.0 is above high=5.5 → OK; 4.0 < high → ESCAPE; 2.0 < lethal=3.0 → LETHAL
    out = do_override(np.array([6.0, 4.0, 2.0]))
    assert out[0] == DO_OK
    assert out[1] == DO_ESCAPE
    assert out[2] == DO_LETHAL
