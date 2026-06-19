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
