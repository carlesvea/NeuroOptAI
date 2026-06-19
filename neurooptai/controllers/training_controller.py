from neurooptai.core.training_state import TrainingState
from neurooptai.core.halo_cost_model import HALOCostModel
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
    def __init__(
        self,
        gradient_threshold=1.0,
        stagnation_patience=3,
        halo_enabled=False,
        meta_control_cost=0.0,
        optimization_cost=0.0,
    ):
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

        self.halo_enabled = halo_enabled
        self.halo_cost_model = HALOCostModel(
            meta_control_cost=meta_control_cost,
            optimization_cost=optimization_cost,
        )

    def log_state(self, state: TrainingState):
        return self.log_epoch(
            train_loss=state.train_loss,
            validation_loss=state.validation_loss,
            gradient_norm=state.gradient_norm,
            learning_rate=state.learning_rate,
        )

    def log_epoch(
        self,
        train_loss,
        validation_loss,
        gradient_norm=None,
        learning_rate=None,
        avoided_bad_branch_cost=None,
    ):
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
            chosen_action = self.priority_system.choose(actions)

            if self.halo_enabled:
                if avoided_bad_branch_cost is None:
                    avoided_bad_branch_cost = 0.0

                if not self.halo_cost_model.should_intervene(avoided_bad_branch_cost):
                    return {
                        "action": "continue_training",
                        "recommendation": "HALO blocked intervention because expected benefit does not exceed intervention cost.",
                        "should_stop": False,
                        "halo": {
                            "enabled": True,
                            "intervention_cost": self.halo_cost_model.total_intervention_cost(),
                            "avoided_bad_branch_cost": float(avoided_bad_branch_cost),
                            "blocked_action": chosen_action,
                        },
                    }

            return chosen_action

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
