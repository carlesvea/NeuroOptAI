from neurooptai import analyze_training_state


def test_analyze_training_state_api():
    result = analyze_training_state(
        train_loss=1.0,
        validation_loss=1.1,
        gradient_norm=2.5,
        learning_rate=0.001,
    )

    assert result["action"] == "gradient_clipping"
