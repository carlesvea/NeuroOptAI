from neurooptai.sensors.loss_sensor import LossSensor
from neurooptai.sensors.gradient_norm_sensor import GradientNormSensor
from neurooptai.diagnostics.overfitting_detector import OverfittingDetector
from neurooptai.actions.early_stopping_action import EarlyStoppingAction
from neurooptai.actions.gradient_clipping_action import GradientClippingAction


class TrainingController:
    def __init__(self, gradient_threshold=1.0):
        self.loss_sensor = LossSensor()
        self.gradient_norm_sensor = GradientNormSensor()
        self.overfitting_detector = OverfittingDetector()
        self.early_stopping_action = EarlyStoppingAction()
        self.gradient_clipping_action = GradientClippingAction(max_norm=gradient_threshold)
        self.gradient_threshold = gradient_threshold

    def log_epoch(self, train_loss, validation_loss, gradient_norm=None):
        self.loss_sensor.log_train_loss(train_loss)
        self.loss_sensor.log_validation_loss(validation_loss)

        if gradient_norm is not None:
            self.gradient_norm_sensor.log_gradient_norm(gradient_norm)

            if self.gradient_norm_sensor.is_too_high(self.gradient_threshold):
                return self.gradient_clipping_action.execute()

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
