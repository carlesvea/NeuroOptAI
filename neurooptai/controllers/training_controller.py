from neurooptai.sensors.loss_sensor import LossSensor
from neurooptai.sensors.gradient_norm_sensor import GradientNormSensor
from neurooptai.sensors.learning_rate_sensor import LearningRateSensor
from neurooptai.diagnostics.overfitting_detector import OverfittingDetector
from neurooptai.diagnostics.stagnation_detector import StagnationDetector
from neurooptai.actions.early_stopping_action import EarlyStoppingAction
from neurooptai.actions.gradient_clipping_action import GradientClippingAction
from neurooptai.actions.reduce_learning_rate_action import ReduceLearningRateAction
from neurooptai.controllers.decision_priority import DecisionPrioritySystem


class TrainingController:
    def __init__(self, gradient_threshold=1.0, stagnation_patience=3):
        self.loss_sensor = LossSensor()
        self.gradient_norm_sensor = GradientNormSensor()
        self.learning_rate_sensor = LearningRateSensor()

        self.overfitting_detector = OverfittingDetector()
        self.stagnation_detector = StagnationDetector(patience=stagnation_patience)

        self.early_stopping_action = EarlyStoppingAction()
        self.gradient_clipping_action = GradientClippingAction(max_norm=gradient_threshold)
        self.reduce_lr_action = ReduceLearningRateAction(factor=0.5)

        self.priority_system = DecisionPrioritySystem()
        self.gradient_threshold = gradient_threshold

    def log_epoch(self, train_loss, validation_loss, gradient_norm=None, learning_rate=None):
        actions = []

        self.loss_sensor.log_train_loss(train_loss)
        self.loss_sensor.log_validation_loss(validation_loss)

        if learning_rate is not None:
            self.learning_rate_sensor.log_learning_rate(learning_rate)

        if gradient_norm is not None:
            self.gradient_norm_sensor.log_gradient_norm(gradient_norm)

            if self.gradient_norm_sensor.is_too_high(self.gradient_threshold):
                actions.append(self.gradient_clipping_action.execute())

        if self.overfitting_detector.detect(
            self.loss_sensor.train_losses,
            self.loss_sensor.validation_losses
        ):
            actions.append(self.early_stopping_action.execute())

        if self.stagnation_detector.detect(self.loss_sensor.validation_losses):
            actions.append(self.reduce_lr_action.execute(current_lr=learning_rate))

        if actions:
            return self.priority_system.choose(actions)

        return {
            "action": "continue_training",
            "recommendation": "Training looks stable.",
            "should_stop": False,
        }

    def status(self):
        latest_losses = self.loss_sensor.latest()

        return {
            "epochs_logged": len(self.loss_sensor.train_losses),
            "latest_train_loss": latest_losses["train_loss"],
            "latest_validation_loss": latest_losses["validation_loss"],
            "latest_gradient_norm": self.gradient_norm_sensor.latest(),
            "latest_learning_rate": self.learning_rate_sensor.latest(),
        }
