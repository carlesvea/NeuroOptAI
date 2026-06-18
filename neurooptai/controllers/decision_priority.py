class DecisionPrioritySystem:
    """
    Decideix quina acció té prioritat quan hi ha diverses alertes.
    """

    def __init__(self):
        self.priorities = {
            "gradient_clipping": 3,
            "early_stopping": 2,
            "reduce_learning_rate": 1,
            "continue_training": 0,
        }

    def choose(self, actions):
        if not actions:
            return {
                "action": "continue_training",
                "recommendation": "Training looks stable.",
                "should_stop": False,
            }

        return max(actions, key=lambda action: self.priorities.get(action.get("action"), 0))
