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
