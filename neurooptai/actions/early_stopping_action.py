class EarlyStoppingAction:
    """
    Acció simple: recomana parar l'entrenament quan hi ha overfitting.
    """

    def __init__(self, message="Overfitting detected. Consider early stopping."):
        self.message = message

    def execute(self):
        return {
            "action": "early_stopping",
            "recommendation": self.message,
            "should_stop": True,
        }
