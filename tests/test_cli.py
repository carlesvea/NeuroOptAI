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
