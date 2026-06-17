class GradientNormSensor:
    """
    Sensor simple per registrar la mida total del gradient.
    """

    def __init__(self):
        self.gradient_norms = []

    def log_gradient_norm(self, value):
        self.gradient_norms.append(float(value))

    def latest(self):
        return self.gradient_norms[-1] if self.gradient_norms else None

    def is_too_high(self, threshold=1.0):
        latest_value = self.latest()
        if latest_value is None:
            return False
        return latest_value > threshold
