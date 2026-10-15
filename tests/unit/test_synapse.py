import numpy as np
import pytest
from scipy.integrate import solve_ivp
from neural_budget.synapse import resource_state, depleted_budget
from neural_budget.model import cortical_budget
from neural_budget.params import reference_parameters, values


def test_resources_match_independent_ode_and_zero_firing_recovery():
    sol = solve_ivp(lambda t,r: (1-r)/0.8 - 0.25*12*r, (0,2), [1], rtol=1e-10, atol=1e-12)
    assert resource_state(12,.25,.8,elapsed_s=2) == pytest.approx(sol.y[0,-1], abs=1e-9)
    assert resource_state(0,.25,.8,initial=.2,elapsed_s=.8) == pytest.approx(1-.8/np.e)
    assert resource_state(12,.25,.8) == pytest.approx(1/3.4)


def test_depletion_changes_release_costs_without_changing_action_or_resting_costs():
    v = values(reference_parameters()); ref = cortical_budget(v)
    changed, result = depleted_budget(v,resource=.5)
    assert changed.components["postsynaptic_ampa"] == pytest.approx(ref.components["postsynaptic_ampa"] / 2)
    for key in ["action_potential_axon", "resting_potential_neuron", "resting_potential_glia"]:
        assert changed.components[key] == ref.components[key]
    assert result["effective_release"] == .125
    assert depleted_budget(v,resource=1)[0].total == ref.total


@pytest.mark.parametrize("args", [(-1,.25,.8), (4,1.1,.8), (4,.25,0), (4,.25,np.nan)])
def test_invalid_resource_controls_fail(args):
    with pytest.raises(ValueError): resource_state(*args)
