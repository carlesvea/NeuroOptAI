class UniversalHALO:
    def should_intervene(self, intervention_cost, avoided_cost):
        return avoided_cost > intervention_cost

    def score(self, intervention_cost, avoided_cost):
        return avoided_cost / intervention_cost

    def classify(self, intervention_cost, avoided_cost):
        s = self.score(intervention_cost, avoided_cost)

        if s < 1:
            return "avoid"
        if s < 5:
            return "intervene"
        if s < 10:
            return "priority"
        return "critical"

    def expected_avoided_cost(self, probability_of_failure, avoided_cost):
        return probability_of_failure * avoided_cost

    def should_intervene_probabilistic(
        self,
        intervention_cost,
        probability_of_failure,
        avoided_cost,
    ):
        return self.expected_avoided_cost(
            probability_of_failure,
            avoided_cost,
        ) > intervention_cost
