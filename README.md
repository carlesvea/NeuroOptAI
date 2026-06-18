# NeuroOptAI

Intelligent autopilot for monitoring and optimizing neural network training.

NeuroOptAI is a Python library designed to observe neural network training, detect common problems, prioritize alerts, and recommend optimization actions.

## Core loop

sensors -> diagnostics -> actions -> decision priority -> controller

## Current MVP

NeuroOptAI currently supports:

- Loss monitoring
- Gradient norm monitoring
- Learning rate monitoring
- Overfitting detection
- Stagnation detection
- Early stopping recommendation
- Gradient clipping recommendation
- Learning rate reduction recommendation
- Decision priority system

## Available components

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

## Examples

Run the main demo:

PYTHONPATH=. python experiments/demo_controller.py

Run the stagnation demo:

PYTHONPATH=. python experiments/demo_stagnation.py

Run the priority demo:

PYTHONPATH=. python experiments/demo_priority.py

## Project structure

neurooptai/
  sensors/
  diagnostics/
  actions/
  controllers/
  utils/

experiments/
  demo_controller.py
  demo_stagnation.py
  demo_priority.py

## Roadmap

- Add JSON logging
- Add CLI interface
- Add PyTorch training loop integration
- Add MNIST experiment
- Add CIFAR-10 experiment
- Package as installable Python library

## Author

Carles X. Vea
