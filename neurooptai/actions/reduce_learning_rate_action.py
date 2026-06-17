class ReduceLearningRateAction:
    """
    Acció simple: recomana reduir el learning rate quan hi ha estancament.
    """

    def __init__(self, factor=0.5):
        self.factor = float(factor)

    def execute(self, current_lr=None):
        recommendation = f"Reduce learning rate by factor={self.factor}."

        if current_lr is not None:
            new_lr = float(current_lr) * self.factor
        else:
            new_lr = None

        return {
            "action": "reduce_learning_rate",
            "recommendation": recommendation,
            "factor": self.factor,
            "new_learning_rate": new_lr,
        }
