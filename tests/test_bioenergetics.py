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
    p = BioParams()
    ed = np.array([4.01])
    mass = np.array([3000.0])
    new_ed, dead, new_mass = update_energy(
        ed, mass, np.array([25.0]), np.array([1.5]), np.array([1.5]), p
    )
    assert new_ed[0] < ed[0]
    for _ in range(100):
        new_ed, dead, new_mass = update_energy(
            new_ed, new_mass, np.array([25.0]), np.array([1.5]), np.array([1.5]), p
        )
        if dead[0]:
            break
    assert dead[0], "Fish should eventually die from starvation at 25C"



def test_respiration_monotonically_increases_with_temperature():
    """Respiration should increase with temperature across the full range.
    For non-feeding migrants, there is no consumption dome — only the
    exponential R(T) from the Wisconsin model applies."""
    p = BioParams()
    mass = np.array([3000.0])
    temps = [5.0, 10.0, 16.0, 20.0, 24.0, 26.0]
    resps = [float(hourly_respiration(mass, np.array([t]), np.array([1.0]), p)[0]) for t in temps]
    for i in range(len(resps) - 1):
        assert resps[i + 1] > resps[i], (
            f"R({temps[i+1]}) = {resps[i+1]:.4f} should exceed R({temps[i]}) = {resps[i]:.4f}"
        )


def test_mass_decreases_after_energy_update():
    p = BioParams()
    ed = np.array([6.5])
    mass = np.array([3000.0])
    new_ed, dead, new_mass = update_energy(
        ed, mass, np.array([15.0]), np.array([1.0]), np.array([1.0]), p,
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
