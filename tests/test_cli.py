import subprocess
import sys


def test_cli_version():
    result = subprocess.run(
        [sys.executable, "-m", "neurooptai.cli", "version"],
        capture_output=True,
        text=True,
        check=True,
    )

    assert "NeuroOptAI 0.1.0" in result.stdout


def test_cli_demo():
    result = subprocess.run(
        [sys.executable, "-m", "neurooptai.cli", "demo"],
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Epoch 1" in result.stdout
    assert "gradient_clipping" in result.stdout


def test_cli_halo():
    result = subprocess.run(
        [sys.executable, "-m", "neurooptai.cli", "halo"],
        capture_output=True,
        text=True,
        check=True,
    )

    assert "HALO blocked case" in result.stdout
    assert "HALO allowed case" in result.stdout
    assert "gradient_clipping" in result.stdout


def test_cli_analyze():
    result = subprocess.run(
        [
            sys.executable, "-m", "neurooptai.cli", "analyze",
            "--train-loss", "1.0",
            "--validation-loss", "1.1",
            "--gradient-norm", "2.5",
            "--learning-rate", "0.001",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert "gradient_clipping" in result.stdout


def test_cli_analyze_json_output():
    import json

    result = subprocess.run(
        [
            sys.executable, "-m", "neurooptai.cli", "analyze",
            "--train-loss", "1.0",
            "--validation-loss", "1.1",
            "--gradient-norm", "2.5",
            "--learning-rate", "0.001",
            "--json-output",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    parsed = json.loads(result.stdout)
    assert parsed["action"] == "gradient_clipping"


def test_cli_analyze_log_file(tmp_path):
    import json

    log_file = tmp_path / "analyze_log.jsonl"

    result = subprocess.run(
        [
            sys.executable, "-m", "neurooptai.cli", "analyze",
            "--train-loss", "1.0",
            "--validation-loss", "1.1",
            "--gradient-norm", "2.5",
            "--learning-rate", "0.001",
            "--log-file", str(log_file),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert "gradient_clipping" in result.stdout
    assert log_file.exists()

    parsed = json.loads(log_file.read_text().strip())
    assert parsed["command"] == "analyze"
    assert parsed["decision"]["action"] == "gradient_clipping"


def test_cli_analyze_history(tmp_path):
    import json

    input_file = tmp_path / "history.json"
    input_file.write_text(json.dumps([
        {"train_loss": 1.0, "validation_loss": 1.0, "gradient_norm": 0.5, "learning_rate": 0.001},
        {"train_loss": 0.8, "validation_loss": 1.1, "gradient_norm": 2.5, "learning_rate": 0.001},
    ]))

    result = subprocess.run(
        [
            sys.executable, "-m", "neurooptai.cli", "analyze-history",
            "--input-file", str(input_file),
            "--json-output",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    parsed = json.loads(result.stdout)
    assert parsed[-1]["action"] == "gradient_clipping"
