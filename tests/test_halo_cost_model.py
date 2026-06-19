from neurooptai.core.halo_cost_model import HALOCostModel


def test_halo_should_intervene_when_cost_is_lower():
    halo = HALOCostModel(meta_control_cost=1.0, optimization_cost=2.0)

    assert halo.total_intervention_cost() == 3.0
    assert halo.should_intervene(avoided_bad_branch_cost=10.0) is True


def test_halo_should_not_intervene_when_cost_is_higher():
    halo = HALOCostModel(meta_control_cost=5.0, optimization_cost=5.0)

    assert halo.total_intervention_cost() == 10.0
    assert halo.should_intervene(avoided_bad_branch_cost=3.0) is False
