class LossSensor:
    """
    Sensor simple per registrar train loss i validation loss.
    """

    def __init__(self):
        self.train_losses = []
        self.validation_losses = []

    def log_train_loss(self, value):
        self.train_losses.append(float(value))

    def log_validation_loss(self, value):
        self.validation_losses.append(float(value))

    def latest(self):
        return {
            "train_loss": self.train_losses[-1] if self.train_losses else None,
            "validation_loss": self.validation_losses[-1] if self.validation_losses else None,
        }
