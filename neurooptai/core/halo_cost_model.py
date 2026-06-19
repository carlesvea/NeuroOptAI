class HALOCostModel:
    """
    HALO principle:
    intervene only when intervention cost is lower than avoided bad-branch cost.
    """

    def __init__(self, meta_control_cost=0.0, optimization_cost=0.0):
        self.meta_control_cost = float(meta_control_cost)
        self.optimization_cost = float(optimization_cost)

    def total_intervention_cost(self):
        return self.meta_control_cost + self.optimization_cost

    def should_intervene(self, avoided_bad_branch_cost):
        return self.total_intervention_cost() < float(avoided_bad_branch_cost)
