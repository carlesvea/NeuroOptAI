from neurooptai.controllers.training_controller import TrainingController

controller = TrainingController(gradient_threshold=1.0, stagnation_patience=2)

history = [
    (1.0, 0.90, 0.5, 0.001),
    (1.0, 0.91, 0.5, 0.001),
    (1.0, 0.91, 0.5, 0.001),
]

for epoch, (train_loss, val_loss, grad_norm, lr) in enumerate(history, start=1):
    result = controller.log_epoch(train_loss, val_loss, grad_norm, lr)
    print(f"Epoch {epoch}: {result}")
