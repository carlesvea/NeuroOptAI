class ReduceLearningRateAction:
    """
    Acció simple: recomana reduir el learning rate quan hi ha estancament.
    """

    def __init__(self, factor=0.5):
        self.factor = float(factor)

    def execute(self, current_lr=None):
        new_lr = float(current_lr) * self.factor if current_lr is not None else None

        return {
            "action": "reduce_learning_rate",
            "recommendation": f"Reduce learning rate by factor={self.factor}.",
            "factor": self.factor,
            "new_learning_rate": new_lr,
        }
