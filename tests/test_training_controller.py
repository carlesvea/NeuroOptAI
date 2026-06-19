from neurooptai.controllers.training_controller import TrainingController


def test_gradient_clipping_priority():
    controller = TrainingController(gradient_threshold=1.0, stagnation_patience=2)

    controller.log_epoch(1.0, 1.0, 0.5, 0.001)
    controller.log_epoch(0.8, 1.05, 0.5, 0.001)
    result = controller.log_epoch(0.6, 1.10, 2.5, 0.001)

    assert result["action"] == "gradient_clipping"


def test_reduce_learning_rate_on_stagnation():
    controller = TrainingController(gradient_threshold=1.0, stagnation_patience=2)

    controller.log_epoch(1.0, 0.90, 0.5, 0.001)
    controller.log_epoch(1.0, 0.91, 0.5, 0.001)
    result = controller.log_epoch(1.0, 0.91, 0.5, 0.001)

    assert result["action"] == "reduce_learning_rate"
    assert result["new_learning_rate"] == 0.0005


def test_controller_status():
    controller = TrainingController(gradient_threshold=1.0, stagnation_patience=2)

    controller.log_epoch(1.0, 0.9, 0.5, 0.001)

    status = controller.status()

    assert status["epochs_logged"] == 1
    assert status["latest_train_loss"] == 1.0
    assert status["latest_validation_loss"] == 0.9
    assert status["latest_gradient_norm"] == 0.5
    assert status["latest_learning_rate"] == 0.001


def test_log_state_uses_training_state():
    from neurooptai.core.training_state import TrainingState

    controller = TrainingController(gradient_threshold=1.0, stagnation_patience=2)

    state = TrainingState(
        epoch=1,
        train_loss=1.0,
        validation_loss=0.9,
        gradient_norm=0.5,
        learning_rate=0.001,
    )

    result = controller.log_state(state)
    status = controller.status()

    assert result["action"] == "continue_training"
    assert status["epochs_logged"] == 1
    assert status["latest_train_loss"] == 1.0
    assert status["latest_validation_loss"] == 0.9
    assert status["latest_gradient_norm"] == 0.5
    assert status["latest_learning_rate"] == 0.001
