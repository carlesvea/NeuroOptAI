from dataclasses import dataclass


@dataclass
class TrainingState:
    epoch: int
    train_loss: float
    validation_loss: float
    gradient_norm: float | None = None
    learning_rate: float | None = None
