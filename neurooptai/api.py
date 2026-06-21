from neurooptai.controllers.training_controller import TrainingController


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
    controller = TrainingController(
        gradient_threshold=gradient_threshold,
        stagnation_patience=stagnation_patience,
        halo_enabled=halo_enabled,
        meta_control_cost=meta_control_cost,
        optimization_cost=optimization_cost,
    )

    results = []

    for state in history:
        result = controller.log_epoch(
            train_loss=state["train_loss"],
            validation_loss=state["validation_loss"],
            gradient_norm=state.get("gradient_norm"),
            learning_rate=state.get("learning_rate"),
            avoided_bad_branch_cost=state.get("avoided_bad_branch_cost"),
        )
        results.append(result)

    return results
