import numpy as np
import pytest
from salmon_ibm.estuary import salinity_cost, do_override, seiche_pause


def test_salinity_cost_lagoon_brackish():
    """Lagoon brackish (~5 PSU) is hypo-osmotic — small cost > 1.0."""
    from salmon_ibm.estuary import EstuaryParams
    p = EstuaryParams()
    cost = salinity_cost(np.array([5.0]), p)
    # At 5 PSU with iso=10: below = (10-5)/10 = 0.5; cost = 1 + hypo_cost * 0.5
    expected = 1.0 + p.salinity_hypo_cost * 0.5
    assert cost[0] == pytest.approx(expected)
    assert cost[0] < 1.0 + p.salinity_hypo_cost  # less than full freshwater cost


def test_salinity_cost_baltic_near_iso():
    """Baltic surface (~7 PSU) is close to iso (10 PSU) — minimal cost."""
    from salmon_ibm.estuary import EstuaryParams
    p = EstuaryParams()
    cost = salinity_cost(np.array([7.0]), p)
    # Should be slightly above 1.0 but much less than the old 30% penalty
    assert 1.0 < cost[0] < 1.05


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


def test_estuary_params_defaults_match_literature():
    """Defaults: DO from Liland 2024; salinity from Wilson 2002 + Brett & Groves 1979."""
    from salmon_ibm.estuary import EstuaryParams
    p = EstuaryParams()
    # Liland 2024 — DO thresholds
    assert p.do_lethal == 3.0
    assert p.do_high == 5.5
    # Wilson 2002 — S. salar blood iso-osmotic point
    assert p.salinity_iso_osmotic == 10.0
    # Brett & Groves 1979 — hyper / hypo cost slopes
    assert p.salinity_hyper_cost == pytest.approx(0.30)
    assert p.salinity_hypo_cost == pytest.approx(0.05)
    # Seiche (default unchanged)
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


class TestEstuaryParamsValidation:
    def test_negative_iso_raises(self):
        from salmon_ibm.estuary import EstuaryParams
        with pytest.raises(ValueError, match="salinity_iso_osmotic"):
            EstuaryParams(salinity_iso_osmotic=-1.0)

    def test_iso_above_35_raises(self):
        from salmon_ibm.estuary import EstuaryParams
        with pytest.raises(ValueError, match="salinity_iso_osmotic"):
            EstuaryParams(salinity_iso_osmotic=40.0)

    def test_negative_hyper_cost_raises(self):
        from salmon_ibm.estuary import EstuaryParams
        with pytest.raises(ValueError, match="salinity_hyper_cost"):
            EstuaryParams(salinity_hyper_cost=-0.1)


def test_salinity_cost_at_iso_returns_unity():
    """Cost should be exactly 1.0 at the iso-osmotic point (default 10 PSU)."""
    from salmon_ibm.estuary import salinity_cost, EstuaryParams
    cost = salinity_cost(np.array([10.0]), EstuaryParams())
    assert cost[0] == pytest.approx(1.0)


def test_salinity_cost_marine_matches_brett_groves():
    """At full marine salinity (35 PSU), cost ≈ 1 + hyper_cost."""
    from salmon_ibm.estuary import salinity_cost, EstuaryParams
    p = EstuaryParams()
    cost = salinity_cost(np.array([35.0]), p)
    assert cost[0] == pytest.approx(1.0 + p.salinity_hyper_cost)


def test_salinity_cost_freshwater_above_one_and_below_marine():
    """At 0 PSU, cost > 1.0 (hypo-osmotic stress) but < marine cost (asymmetry)."""
    from salmon_ibm.estuary import salinity_cost, EstuaryParams
    p = EstuaryParams()
    fresh_cost = salinity_cost(np.array([0.0]), p)
    marine_cost = salinity_cost(np.array([35.0]), p)
    assert fresh_cost[0] > 1.0
    assert fresh_cost[0] < marine_cost[0]
    assert fresh_cost[0] == pytest.approx(1.0 + p.salinity_hypo_cost)


def test_salinity_cost_smooth_monotonic_outside_iso():
    """Cost increases monotonically as |salinity - iso| grows in either direction."""
    from salmon_ibm.estuary import salinity_cost, EstuaryParams
    p = EstuaryParams()
    iso = p.salinity_iso_osmotic
    # Sweep above iso
    above = np.linspace(iso, 35.0, 50)
    above_costs = salinity_cost(above, p)
    assert np.all(np.diff(above_costs) >= -1e-12), (
        f"Cost should be monotonic non-decreasing above iso; "
        f"got max negative delta {np.diff(above_costs).min()}"
    )
    # Sweep below iso (reverse direction since cost rises as salinity falls)
    below = np.linspace(0.0, iso, 50)
    below_costs = salinity_cost(below, p)
    assert np.all(np.diff(below_costs) <= 1e-12), (
        f"Cost should be monotonic non-increasing as salinity rises toward iso; "
        f"got max positive delta {np.diff(below_costs).max()}"
    )


def test_salinity_cost_handles_nan():
    """NaN salinity → cost = 1.0 (treated as iso, no penalty)."""
    from salmon_ibm.estuary import salinity_cost, EstuaryParams
    p = EstuaryParams()
    sal = np.array([np.nan, p.salinity_iso_osmotic, 35.0])
    cost = salinity_cost(sal, p)
    assert cost[0] == pytest.approx(1.0)
    assert cost[1] == pytest.approx(1.0)
    assert cost[2] == pytest.approx(1.0 + p.salinity_hyper_cost)
    assert not np.isnan(cost).any()
