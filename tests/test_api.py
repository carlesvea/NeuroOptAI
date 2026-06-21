from neurooptai import analyze_training_state


def test_analyze_training_state_api():
    result = analyze_training_state(
        train_loss=1.0,
        validation_loss=1.1,
        gradient_norm=2.5,
        learning_rate=0.001,
    )

    assert result["action"] == "gradient_clipping"


def test_analyze_training_state_api_with_halo_blocks_intervention():
    result = analyze_training_state(
        train_loss=1.0,
        validation_loss=1.1,
        gradient_norm=2.5,
        learning_rate=0.001,
        halo_enabled=True,
        meta_control_cost=5.0,
        optimization_cost=5.0,
        avoided_bad_branch_cost=3.0,
    )

    assert result["action"] == "continue_training"
    assert "halo" in result


def test_analyze_training_history_api():
    from neurooptai import analyze_training_history

    history = [
        {"train_loss": 1.0, "validation_loss": 1.0, "gradient_norm": 0.5, "learning_rate": 0.001},
        {"train_loss": 0.8, "validation_loss": 1.1, "gradient_norm": 2.5, "learning_rate": 0.001},
    ]

    results = analyze_training_history(history)

    assert len(results) == 2
    assert results[-1]["action"] == "gradient_clipping"
