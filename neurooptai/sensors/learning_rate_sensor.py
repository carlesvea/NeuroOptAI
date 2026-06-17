class LearningRateSensor:
    """
    Sensor simple per registrar el learning rate durant l'entrenament.
    """

    def __init__(self):
        self.learning_rates = []

    def log_learning_rate(self, value):
        self.learning_rates.append(float(value))

    def latest(self):
        return self.learning_rates[-1] if self.learning_rates else None

    def is_too_low(self, threshold=1e-6):
        latest_value = self.latest()
        if latest_value is None:
            return False
        return latest_value < threshold
