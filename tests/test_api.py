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
