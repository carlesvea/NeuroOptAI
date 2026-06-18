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
