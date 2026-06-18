from neurooptai.controllers.training_controller import TrainingController

controller = TrainingController(gradient_threshold=1.0, stagnation_patience=2)

history = [
    (1.0, 1.00, 0.5, 0.001),
    (0.8, 1.05, 0.5, 0.001),
    (0.6, 1.10, 2.5, 0.001),
]

for epoch, (train_loss, val_loss, grad_norm, lr) in enumerate(history, start=1):
    result = controller.log_epoch(train_loss, val_loss, grad_norm, lr)
    print(f"Epoch {epoch}: {result}")
