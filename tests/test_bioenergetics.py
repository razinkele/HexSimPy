import numpy as np
import pytest
from salmon_ibm.bioenergetics import BioParams, hourly_respiration, update_energy


def test_bio_params_defaults():
    p = BioParams()
    assert p.RA == pytest.approx(0.00264)
    assert p.ED_MORTAL == pytest.approx(4.0)


def test_hourly_respiration_increases_with_temperature():
    p = BioParams()
    mass = np.array([3000.0])
    r_cold = hourly_respiration(mass, np.array([10.0]), np.array([1.0]), p)
    r_warm = hourly_respiration(mass, np.array([16.0]), np.array([1.0]), p)
    assert r_warm[0] > r_cold[0]


def test_hourly_respiration_increases_with_activity():
    p = BioParams()
    mass = np.array([3000.0])
    r_rest = hourly_respiration(mass, np.array([15.0]), np.array([1.0]), p)
    r_active = hourly_respiration(mass, np.array([15.0]), np.array([1.5]), p)
    assert r_active[0] > r_rest[0]


def test_update_energy_decreases_total_energy():
    """Total energy (ED * mass) must decrease; per-gram ED may rise if
    catabolized tissue is less energy-dense than the pool average."""
    p = BioParams()
    ed = np.array([6.5])
    mass = np.array([3000.0])
    temps = np.array([15.0])
    activity = np.array([1.0])
    salinity_cost = np.array([1.0])
    new_ed, dead, new_mass = update_energy(ed, mass, temps, activity, salinity_cost, p)
    e_before = ed[0] * 1000.0 * mass[0]
    e_after = new_ed[0] * 1000.0 * new_mass[0]
    assert e_after < e_before, "Total energy must decrease each step"
    assert not dead[0]


def test_mortality_at_low_energy():
    """Fish below ED_MORTAL threshold are immediately flagged dead.
    Under the proportional mass-loss model, ED stays flat rather than
    rising (old bug) or declining in a single step. Mortality is detected
    when ED drops below ED_MORTAL (4.0 kJ/g). A fish starting just above
    the threshold will not be flagged dead until the 50%-mass floor causes
    ED to fall; therefore we test with a fish already at the threshold."""
    p = BioParams()
    # Fish with ED right at the lethal threshold should be flagged dead.
    ed = np.array([3.99])
    mass = np.array([3000.0])
    new_ed, dead, new_mass = update_energy(
        ed, mass, np.array([25.0]), np.array([1.5]), np.array([1.5]), p
    )
    assert dead[0], "Fish at ED < ED_MORTAL should be flagged dead immediately"
    # Confirm ED is non-negative
    assert new_ed[0] >= 0.0


def test_respiration_monotonically_increases_with_temperature():
    """Respiration should increase with temperature across the full range.
    For non-feeding migrants, there is no consumption dome — only the
    exponential R(T) from the Wisconsin model applies."""
    p = BioParams()
    mass = np.array([3000.0])
    temps = [5.0, 10.0, 16.0, 20.0, 24.0, 26.0]
    resps = [
        float(hourly_respiration(mass, np.array([t]), np.array([1.0]), p)[0])
        for t in temps
    ]
    for i in range(len(resps) - 1):
        assert resps[i + 1] > resps[i], (
            f"R({temps[i + 1]}) = {resps[i + 1]:.4f} should exceed R({temps[i]}) = {resps[i]:.4f}"
        )


def test_mass_decreases_after_energy_update():
    p = BioParams()
    ed = np.array([6.5])
    mass = np.array([3000.0])
    new_ed, dead, new_mass = update_energy(
        ed,
        mass,
        np.array([15.0]),
        np.array([1.0]),
        np.array([1.0]),
        p,
    )
    assert new_mass[0] < 3000.0, "Mass should decrease during fasting"


def test_energy_conservation_single_step():
    """Total energy lost should equal respiration cost (no double-counting).
    energy_before - energy_after == R_hourly (within floating point tolerance)."""
    p = BioParams()
    ed = np.array([6.5])
    mass = np.array([3000.0])
    temps = np.array([15.0])
    activity = np.array([1.0])
    sal_cost = np.array([1.0])

    e_before = ed[0] * 1000.0 * mass[0]  # total energy in J
    r_hourly = float(hourly_respiration(mass, temps, activity, p)[0] * sal_cost[0])

    new_ed, dead, new_mass = update_energy(ed, mass, temps, activity, sal_cost, p)
    e_after = new_ed[0] * 1000.0 * new_mass[0]  # total energy in J

    energy_lost = e_before - e_after
    assert energy_lost == pytest.approx(r_hourly, rel=1e-6), (
        f"Energy lost ({energy_lost:.2f} J) should equal respiration ({r_hourly:.2f} J)"
    )


def test_hourly_respiration_handles_zero_mass():
    """Zero mass should produce finite respiration (clamped), not inf/nan."""
    from salmon_ibm.bioenergetics import hourly_respiration, BioParams

    params = BioParams()
    mass = np.array([0.0, -1.0, 1e-10, 3500.0])
    temp = np.full(4, 15.0)
    activity = np.ones(4)
    result = hourly_respiration(mass, temp, activity, params)
    assert np.all(np.isfinite(result)), f"Non-finite values: {result}"


def test_energy_density_decreases_monotonically():
    """ED must decrease (or stay flat) every hour for non-feeding migrants."""
    from salmon_ibm.bioenergetics import update_energy, BioParams

    params = BioParams()
    n = 100
    ed = np.full(n, 6.0)
    mass = np.full(n, 3500.0)
    temp = np.full(n, 15.0)
    activity = np.ones(n)
    sal_cost = np.ones(n)

    for _ in range(48):
        new_ed, dead, new_mass = update_energy(
            ed, mass, temp, activity, sal_cost, params
        )
        # Energy density must never increase for starving fish
        assert np.all(new_ed[~dead] <= ed[~dead] + 1e-12), (
            f"ED increased: max delta = {(new_ed[~dead] - ed[~dead]).max()}"
        )
        ed = new_ed
        mass = new_mass


def test_energy_density_declines_under_starvation():
    """Under lipid-first catabolism (Option A), ED must actively DECLINE
    toward ED_MORTAL during sustained starvation — not stay flat. The
    pre-fix proportional-mass formula keeps ED constant until the mass
    floor engages, hiding starvation behind a step function.

    See docs/superpowers/specs/2026-04-23-bioenergetics-starvation-decision.md
    """
    from salmon_ibm.bioenergetics import update_energy, BioParams

    params = BioParams()
    ed = np.array([6.0])
    mass = np.array([3500.0])
    temp = np.array([15.0])
    activity = np.ones(1)
    sal_cost = np.ones(1)

    initial_ed = float(ed[0])
    for _ in range(72):  # 3 days at 15°C with no feeding
        new_ed, dead, new_mass = update_energy(
            ed, mass, temp, activity, sal_cost, params
        )
        if dead.any():
            break
        ed = new_ed
        mass = new_mass

    # Lipid-first catabolism should produce a measurable, smooth ED decline.
    assert ed[0] < initial_ed - 0.01, (
        f"ED should decline under starvation; started at {initial_ed:.4f}, "
        f"ended at {ed[0]:.4f} after 72 h — likely the proportional-mass "
        f"formula is still in place (ED stays flat by construction)."
    )


def test_starvation_triggers_mortality_when_energy_depleted():
    """Starvation mortality should fire when ED reaches ED_MORTAL smoothly,
    not as an abrupt cliff. Under the pre-fix physics, ED stays flat then
    plummets to ~0 in a single step when the mass floor engages — the ED
    at death is not near ED_MORTAL but near zero.

    Under Option A, the trajectory is smooth: ED declines over many hours
    and crosses ED_MORTAL while the agent's body is still at a sane mass.
    """
    from salmon_ibm.bioenergetics import update_energy, BioParams

    params = BioParams()
    ed = np.array([params.ED_MORTAL + 0.6])  # start close so test stays bounded
    mass = np.array([3500.0])
    temp = np.array([15.0])
    activity = np.ones(1)
    sal_cost = np.ones(1)

    ed_at_death = None
    mass_at_death = None
    for _ in range(5000):  # cap iterations as a safety net
        new_ed, dead, new_mass = update_energy(
            ed, mass, temp, activity, sal_cost, params
        )
        if dead.any():
            ed_at_death = float(new_ed[0])
            mass_at_death = float(new_mass[0])
            break
        ed = new_ed
        mass = new_mass

    assert ed_at_death is not None, (
        f"Agent failed to starve to mortality within 5000 hours; "
        f"final ED={ed[0]:.4f}, ED_MORTAL={params.ED_MORTAL}"
    )

    # Under Option A, ED at death should be near ED_MORTAL — within
    # ~one hourly respiration step. Under the pre-fix proportional-mass
    # formula, ED at death is ~0 (abrupt cliff once mass floor engages),
    # which fails this assertion.
    assert ed_at_death > params.ED_MORTAL - 1.0, (
        f"ED at death ({ed_at_death:.4f}) should be near ED_MORTAL "
        f"({params.ED_MORTAL}); the abrupt-cliff signature suggests the "
        f"proportional-mass formula is still in place."
    )


class TestBioParamsValidation:
    def test_negative_ra_raises(self):
        with pytest.raises(ValueError, match="RA"):
            BioParams(RA=-0.001)

    def test_negative_rq_raises(self):
        with pytest.raises(ValueError, match="RQ"):
            BioParams(RQ=-0.01)

    def test_t_max_below_t_opt_raises(self):
        with pytest.raises(ValueError, match="T_MAX.*T_OPT"):
            BioParams(T_OPT=20.0, T_MAX=15.0)

    def test_mass_floor_out_of_range_raises(self):
        with pytest.raises(ValueError, match="MASS_FLOOR"):
            BioParams(MASS_FLOOR_FRACTION=1.5)

    def test_valid_params_no_error(self):
        bp = BioParams()
        assert bp.RA > 0
