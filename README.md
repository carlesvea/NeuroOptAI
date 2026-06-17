# NeuroOptAI

Intelligent autopilot for monitoring and optimizing neural network training.

NeuroOptAI is a Python library designed to observe neural network training, detect common problems, and recommend or trigger optimization actions.

## Core idea

NeuroOptAI follows a simple loop:

sensors -> diagnostics -> actions -> controller

It monitors signals such as:

- Train loss
- Validation loss
- Gradient behavior
- Learning rate
- Training stability

## Current MVP

The current MVP includes:

- LossSensor
- OverfittingDetector
- EarlyStoppingAction
- TrainingController
- Demo experiment

## Example

Run the demo:

PYTHONPATH=. python experiments/demo_controller.py

Expected behavior:

Epoch 1: continue training
Epoch 2: continue training
Epoch 3: early stopping recommendation

## Project structure

neurooptai/
  sensors/
  diagnostics/
  actions/
  controllers/
  utils/

experiments/
  demo_controller.py

## Roadmap

- Add gradient norm sensor
- Add learning rate monitor
- Add stagnation detector
- Add adaptive learning rate action
- Add PyTorch training loop integration
- Test on MNIST
- Test on CIFAR-10

## Author

Carles X. Vea
