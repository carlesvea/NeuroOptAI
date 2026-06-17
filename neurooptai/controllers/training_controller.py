from neurooptai.sensors.loss_sensor import LossSensor
from neurooptai.diagnostics.overfitting_detector import OverfittingDetector
from neurooptai.actions.early_stopping_action import EarlyStoppingAction


class TrainingController:
    """
    Primer cervell de NeuroOptAI:
    registra losses, detecta overfitting i recomana accions.
    """

    def __init__(self):
        self.loss_sensor = LossSensor()
        self.overfitting_detector = OverfittingDetector()
        self.early_stopping_action = EarlyStoppingAction()

    def log_epoch(self, train_loss, validation_loss):
        self.loss_sensor.log_train_loss(train_loss)
        self.loss_sensor.log_validation_loss(validation_loss)

        is_overfitting = self.overfitting_detector.detect(
            self.loss_sensor.train_losses,
            self.loss_sensor.validation_losses
        )

        if is_overfitting:
            return self.early_stopping_action.execute()

        return {
            "action": "continue_training",
            "recommendation": "Training looks stable.",
            "should_stop": False,
        }
