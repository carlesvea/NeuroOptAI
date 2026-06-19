from neurooptai.core.halo_cost_model import HALOCostModel

halo = HALOCostModel(meta_control_cost=1.0, optimization_cost=2.0)

avoided_bad_branch_cost = 10.0

print("Total intervention cost:", halo.total_intervention_cost())
print("Avoided bad branch cost:", avoided_bad_branch_cost)
print("Should intervene:", halo.should_intervene(avoided_bad_branch_cost))
