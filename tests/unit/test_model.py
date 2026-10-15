import numpy as np
import pint
import pytest

from neural_budget.model import (ACTIVITY_DEPENDENT, Q, atp_free_energy_power, cortical_budget, housekeeping_rate,
                                 resting_atp_rate)
from neural_budget.params import reference_parameters, values

BASE = values(reference_parameters())


def test_resting_equation_balances_pump_and_leaks():
    r, v, ena, ek, f = 2e8, -0.07, 0.05, -0.1, 96485.0
    p = resting_atp_rate(Q(r, "ohm"), Q(v, "V"), Q(ena, "V"), Q(ek, "V"), Q(f, "C/mol")).magnitude
    g_na = 3 * p * f / (ena - v)
    g_k = 2 * p * f / (v - ek)
    assert 1.0 / (g_na + g_k) == pytest.approx(r, rel=1e-12)
    net = g_na * (v - ena) + g_k * (v - ek) + p * f
    assert abs(net) < 1e-12 * p * f


def test_wrong_units_are_refused():
    with pytest.raises(pint.errors.DimensionalityError):
        resting_atp_rate(Q(2e8, "V"), Q(-0.07, "V"), Q(0.05, "V"), Q(-0.1, "V"), Q(96495, "C/mol"))


def test_soma_action_potential_cost_matches_hand_calculation():
    b = cortical_budget(BASE)
    d, cm, dv, e = BASE["soma_diameter"], BASE["membrane_capacitance"], BASE["ap_voltage_change"], BASE["elementary_charge"]
    expected = np.pi * d ** 2 * cm * dv / e * BASE["na_overlap_factor"] / 3 * BASE["firing_rate"]
    assert b.components["action_potential_soma"] == pytest.approx(expected, rel=1e-14)


def test_housekeeping_bases():
    assert housekeeping_rate(300.0, "signaling", 1 / 3) == pytest.approx(100.0)
    assert housekeeping_rate(300.0, "total", 0.25) == pytest.approx(100.0)
    assert housekeeping_rate(300.0, "signaling", 0.25) == pytest.approx(75.0)
    with pytest.raises(ValueError):
        housekeeping_rate(300.0, "total", 1.0)
    with pytest.raises(ValueError):
        housekeeping_rate(300.0, "budget", 0.2)


def test_glia_scaling_modes():
    two = dict(BASE, astrocytes_per_neuron=2.0)
    sheet = cortical_budget(two)
    wired = cortical_budget(two, glia_scaling="parameter")
    ref = cortical_budget(BASE)
    assert sheet.components["resting_potential_glia"] == ref.components["resting_potential_glia"]
    assert wired.components["resting_potential_glia"] == pytest.approx(2 * ref.components["resting_potential_glia"])
    with pytest.raises(ValueError):
        cortical_budget(BASE, glia_scaling="guess")


def test_vesicle_cost_multiplier_only_touches_presynaptic():
    once = cortical_budget(dict(BASE, presynaptic_vesicle_cost_multiplier=1.0))
    ref = cortical_budget(BASE)
    diff = ref.components["presynaptic"] - once.components["presynaptic"]
    expected = (BASE["atp_membrane_fusion_per_vesicle"] + BASE["atp_endocytosis_control_per_vesicle"]) * ref.vesicles_per_s
    assert diff == pytest.approx(expected)
    for c in ref.components:
        if c != "presynaptic":
            assert once.components[c] == ref.components[c]


def test_firing_rate_scales_activity_terms_linearly():
    a = cortical_budget(dict(BASE, firing_rate=2.0))
    b = cortical_budget(dict(BASE, firing_rate=6.0))
    for c in ACTIVITY_DEPENDENT:
        assert b.components[c] == pytest.approx(3 * a.components[c], rel=1e-14)


def test_free_energy_power_units():
    assert atp_free_energy_power(6.022e23, -50e3, 6.022e23) == pytest.approx(5e4)


@pytest.mark.parametrize("basis,fraction", [("signaling", -.1), ("signaling", np.nan),
                                         ("signaling", np.inf), ("total", -.1), ("total", 1)])
def test_invalid_housekeeping_inputs_are_rejected(basis, fraction):
    with pytest.raises(ValueError):
        housekeeping_rate(300.0, basis, fraction)
    with pytest.raises(ValueError):
        cortical_budget(BASE, housekeeping_basis=basis, housekeeping_fraction=fraction)


@pytest.mark.parametrize("changes", [{"release_probability": 1.5}, {"firing_rate": -1}, {"firing_rate": np.nan},
                                    {"neuron_input_resistance": 0}, {"e_k": -.05}])
def test_direct_budget_calls_cannot_bypass_physical_domains(changes):
    with pytest.raises(ValueError):
        cortical_budget(dict(BASE, **changes))


def test_equivalent_unit_quantities_preserve_the_budget():
    converted = dict(BASE, neuron_input_resistance=Q(BASE["neuron_input_resistance"] / 1e6, "megaohm"),
                     neuron_resting_potential=Q(BASE["neuron_resting_potential"] * 1e3, "mV"),
                     membrane_capacitance=Q(BASE["membrane_capacitance"] * 100, "uF/cm^2"))
    # 1 F/m^2 = 100 uF/cm^2.
    assert cortical_budget(converted).signaling == pytest.approx(cortical_budget(BASE).signaling, rel=1e-14)
    with pytest.raises(pint.errors.DimensionalityError):
        cortical_budget(dict(BASE, neuron_input_resistance=Q(200, "mV")))


def test_zero_budget_has_no_fabricated_percentages():
    from neural_budget.run import budget_rows

    budget = cortical_budget(dict(BASE, firing_rate=0, neuron_resting_potential=BASE["e_k"],
                                  astrocyte_resting_potential=BASE["e_k"]))
    assert budget.total == 0
    assert all(np.isnan(r["percent_of_signaling"]) for r in budget_rows("zero", budget, "scenario"))
