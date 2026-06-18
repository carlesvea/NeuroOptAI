# NeuroOptAI

Intelligent autopilot for monitoring and optimizing neural network training.

NeuroOptAI is a Python library designed to observe neural network training, detect common problems, and recommend optimization actions.

## Core loop

sensors -> diagnostics -> actions -> controller

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

### Controller

- TrainingController

## Examples

Run the main demo:

PYTHONPATH=. python experiments/demo_controller.py

Run the stagnation demo:

PYTHONPATH=. python experiments/demo_stagnation.py

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

## Roadmap

- Add PyTorch training loop integration
- Add MNIST experiment
- Add CIFAR-10 experiment
- Add automatic decision priority system
- Add JSON logging
- Add CLI interface
- Package as installable Python library

## Author

Carles X. Vea
