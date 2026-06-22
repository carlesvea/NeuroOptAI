class UniversalHALO:
    def _validate_costs(self, intervention_cost, avoided_cost):
        if intervention_cost <= 0:
            raise ValueError("intervention_cost must be greater than 0.")
        if avoided_cost < 0:
            raise ValueError("avoided_cost must be non-negative.")

    def _validate_probability(self, probability_of_failure):
        if probability_of_failure < 0 or probability_of_failure > 1:
            raise ValueError("probability_of_failure must be between 0 and 1.")

    def should_intervene(self, intervention_cost, avoided_cost):
        self._validate_costs(intervention_cost, avoided_cost)
        return avoided_cost > intervention_cost

    def score(self, intervention_cost, avoided_cost):
        self._validate_costs(intervention_cost, avoided_cost)
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
        self._validate_probability(probability_of_failure)
        if avoided_cost < 0:
            raise ValueError("avoided_cost must be non-negative.")
        return probability_of_failure * avoided_cost

    def should_intervene_probabilistic(self, intervention_cost, probability_of_failure, avoided_cost):
        self._validate_costs(intervention_cost, avoided_cost)
        return self.expected_avoided_cost(probability_of_failure, avoided_cost) > intervention_cost
