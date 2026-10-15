import pytest

from neural_budget.params import STATUSES, apply_scenario, parameter_table, reference_parameters
from neural_budget.run import bundled_scenarios

FIELDS = ("name", "value", "unit", "source", "source_location", "species_and_tissue",
          "experimental_or_model_condition", "status", "uncertainty_basis", "allowed_domain")


def test_every_parameter_carries_the_contract():
    rows = parameter_table(reference_parameters())
    # 33 spreadsheet parameters (unused Q10s excluded) + 9 formula constants
    assert len(rows) == 42
    for r in rows:
        for f in FIELDS:
            assert f in r
        assert r["status"] in STATUSES
        assert r["source"] and r["source_location"] and r["unit"]


def test_reference_is_immutable_and_scenarios_are_copies():
    ref = reference_parameters()
    sc = apply_scenario(ref, {"firing_rate": 7.8}, "test")
    assert sc["firing_rate"].value == 7.8 and sc["firing_rate"].status == "scenario"
    assert ref["firing_rate"].value == 4.0
    assert reference_parameters()["firing_rate"].value == 4.0
    with pytest.raises(KeyError):
        apply_scenario(ref, {"not_a_parameter": 1.0}, "test")
    with pytest.raises(ValueError, match="allowed domain"):
        apply_scenario(ref, {"release_probability": 1.5}, "test")


def test_bundled_scenarios_load_and_name_real_parameters():
    ref = reference_parameters()
    scs = bundled_scenarios()
    assert len(scs) == 7
    for sc in scs:
        assert sc["source"]
        for k in sc["changes"]:
            assert k in ref


def test_parameter_units_are_converted_when_magnitudes_are_extracted():
    from dataclasses import replace
    from neural_budget.params import values

    ref = reference_parameters()
    converted = dict(ref)
    converted["neuron_input_resistance"] = replace(ref["neuron_input_resistance"], value=200,
                                                  unit="megaohm", allowed_domain=(1, 1e5))
    assert values(converted)["neuron_input_resistance"] == pytest.approx(values(ref)["neuron_input_resistance"])


def test_nonfinite_parameter_values_and_domains_are_rejected():
    from dataclasses import replace

    ref = reference_parameters()["firing_rate"]
    with pytest.raises(ValueError, match="finite"):
        replace(ref, value=float("nan"))
    with pytest.raises(ValueError, match="finite"):
        replace(ref, allowed_domain=(0, float("inf")))


def test_unknown_scenario_options_are_not_silently_ignored(tmp_path):
    from neural_budget.params import load_scenario
    from neural_budget.run import _budget_options

    with pytest.raises(ValueError, match="unknown scenario options"):
        _budget_options({"options": {"housekeeping_fracton": .5}})
    p = tmp_path / "bad.toml"
    p.write_text('[scenario]\nname = "typo"\n[changes]\nfiring_rate = 4\n[options]\nhousekeeping_fracton = 0.5\n')
    with pytest.raises(ValueError, match="unknown scenario options"):
        load_scenario(p)
