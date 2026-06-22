[![NeuroOptAI Tests](https://github.com/carlesvea/NeuroOptAI/actions/workflows/tests.yml/badge.svg)](https://github.com/carlesvea/NeuroOptAI/actions/workflows/tests.yml)

# NeuroOptAI

Intelligent autopilot for monitoring and optimizing neural network training.

NeuroOptAI observes training signals, detects common problems, prioritizes alerts, recommends optimization actions, and logs decisions.

## Core loop

sensors -> diagnostics -> actions -> decision priority -> controller -> logging

## Current MVP

- Loss monitoring
- Gradient norm monitoring
- Learning rate monitoring
- Overfitting detection
- Stagnation detection
- Early stopping recommendation
- Gradient clipping recommendation
- Learning rate reduction recommendation
- Decision priority system
- JSON logging

## Components

### Sensors

- LossSensor
- GradientNormSensor
- LearningRateSensor

### Diagnostics

- OverfittingDetector
- StagnationDetector

### Actions

- EarlyStoppingAction
- GradientClippingAction
- ReduceLearningRateAction

### Controllers

- TrainingController
- DecisionPrioritySystem

### Utils

- JSONLogger

## Examples

PYTHONPATH=. python experiments/demo_controller.py

PYTHONPATH=. python experiments/demo_stagnation.py

PYTHONPATH=. python experiments/demo_priority.py

PYTHONPATH=. python experiments/demo_json_logging.py

PYTHONPATH=. python experiments/demo_status.py

## Roadmap

- Add CLI interface
- Add PyTorch training loop integration
- Add MNIST experiment
- Add CIFAR-10 experiment
- Package as installable Python library

## Author

Carles X. Vea

## Installation

Install NeuroOptAI locally in editable mode:

pip install -e .

## CLI usage

Run the demo from the command line:

neurooptai demo

## CLI commands

Show the installed version:

neurooptai version

Run the demo:

neurooptai demo

## Tests

Run the test suite locally:

pytest

## Development install

Install NeuroOptAI with development dependencies:

pip install -e ".[dev]"

Run tests:

pytest

## HALO Principle

HALO means:

meta-control cost + optimization cost < cost of avoided bad training branches

NeuroOptAI should only intervene when the expected benefit is greater than the cost of intervention.

Run the HALO demo:

PYTHONPATH=. python experiments/demo_halo.py

## HALO controller integration

The TrainingController can optionally use HALO as a cost-aware gate.

When HALO is enabled, NeuroOptAI only applies an intervention if:

intervention cost < avoided bad branch cost

Run the HALO controller demo:

PYTHONPATH=. python experiments/demo_halo_controller.py

## HALO CLI

Run the HALO cost-aware controller demo:

neurooptai halo

This shows two cases:

- HALO blocks an intervention when the intervention cost is too high.
- HALO allows an intervention when the avoided bad-branch cost is higher.

## Analyze command

Analyze one training state directly from the CLI:

neurooptai analyze --train-loss 1.0 --validation-loss 1.1 --gradient-norm 2.5 --learning-rate 0.001

With HALO enabled:

neurooptai analyze --train-loss 1.0 --validation-loss 1.1 --gradient-norm 2.5 --learning-rate 0.001 --halo-enabled --meta-control-cost 1.0 --optimization-cost 1.0 --avoided-bad-branch-cost 10.0

## JSON output

Return analyze results as JSON:

neurooptai analyze --train-loss 1.0 --validation-loss 1.1 --gradient-norm 2.5 --learning-rate 0.001 --json-output

## Analyze logging

Save analyze decisions to a JSON Lines log file:

neurooptai analyze --train-loss 1.0 --validation-loss 1.1 --gradient-norm 2.5 --learning-rate 0.001 --log-file neurooptai_analyze_log.jsonl

## Python API

Use NeuroOptAI directly from Python:

from neurooptai import analyze_training_state

result = analyze_training_state(
    train_loss=1.0,
    validation_loss=1.1,
    gradient_norm=2.5,
    learning_rate=0.001,
)

print(result)

## Python API with HALO

Use HALO cost-aware gating from Python:

from neurooptai import analyze_training_state

result = analyze_training_state(
    train_loss=1.0,
    validation_loss=1.1,
    gradient_norm=2.5,
    learning_rate=0.001,
    halo_enabled=True,
    meta_control_cost=1.0,
    optimization_cost=1.0,
    avoided_bad_branch_cost=10.0,
)

print(result)

## Python API for training history

Analyze multiple training states:

from neurooptai import analyze_training_history

history = [
    {"train_loss": 1.0, "validation_loss": 1.0, "gradient_norm": 0.5, "learning_rate": 0.001},
    {"train_loss": 0.8, "validation_loss": 1.1, "gradient_norm": 2.5, "learning_rate": 0.001},
]

results = analyze_training_history(history)
print(results)

## Analyze training history from CLI

Analyze a full training history from a JSON file:

neurooptai analyze-history --input-file experiments/history_sample.json --json-output

You can also save the decisions:

neurooptai analyze-history --input-file experiments/history_sample.json --json-output --log-file neurooptai_history_log.jsonl

## Training history validation

`analyze-history` validates the input JSON.

The input file must contain a list of objects, and each object must include:

- train_loss
- validation_loss

If the file is missing, invalid JSON, or missing required keys, NeuroOptAI returns a clear CLI error.

## Universal HALO

Universal HALO generalizes cost-aware intervention beyond neural training.

Core rule:

intervention_cost < avoided_cost

Probabilistic rule:

intervention_cost < probability_of_failure * avoided_cost

Example:

PYTHONPATH=. python experiments/demo_universal_halo.py
