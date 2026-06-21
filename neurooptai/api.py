from neurooptai.controllers.training_controller import TrainingController


REQUIRED_HISTORY_KEYS = ("train_loss", "validation_loss")


def _validate_history_state(state, index):
    if not isinstance(state, dict):
        raise ValueError(f"History item {index} must be a dictionary.")

    missing = [key for key in REQUIRED_HISTORY_KEYS if key not in state]
    if missing:
        raise ValueError(f"History item {index} is missing required keys: {missing}")


def analyze_training_state(
    train_loss,
    validation_loss,
    gradient_norm=None,
    learning_rate=None,
    gradient_threshold=1.0,
    stagnation_patience=3,
    halo_enabled=False,
    meta_control_cost=0.0,
    optimization_cost=0.0,
    avoided_bad_branch_cost=None,
):
    controller = TrainingController(
        gradient_threshold=gradient_threshold,
        stagnation_patience=stagnation_patience,
        halo_enabled=halo_enabled,
        meta_control_cost=meta_control_cost,
        optimization_cost=optimization_cost,
    )

    return controller.log_epoch(
        train_loss=train_loss,
        validation_loss=validation_loss,
        gradient_norm=gradient_norm,
        learning_rate=learning_rate,
        avoided_bad_branch_cost=avoided_bad_branch_cost,
    )


def analyze_training_history(
    history,
    gradient_threshold=1.0,
    stagnation_patience=3,
    halo_enabled=False,
    meta_control_cost=0.0,
    optimization_cost=0.0,
):
    if not isinstance(history, list):
        raise ValueError("Training history must be a list of dictionaries.")

    controller = TrainingController(
        gradient_threshold=gradient_threshold,
        stagnation_patience=stagnation_patience,
        halo_enabled=halo_enabled,
        meta_control_cost=meta_control_cost,
        optimization_cost=optimization_cost,
    )

    results = []

    for index, state in enumerate(history):
        _validate_history_state(state, index)

        result = controller.log_epoch(
            train_loss=state["train_loss"],
            validation_loss=state["validation_loss"],
            gradient_norm=state.get("gradient_norm"),
            learning_rate=state.get("learning_rate"),
            avoided_bad_branch_cost=state.get("avoided_bad_branch_cost"),
        )
        results.append(result)

    return results
