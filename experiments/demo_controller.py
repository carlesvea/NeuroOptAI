from neurooptai.controllers.training_controller import TrainingController

controller = TrainingController(gradient_threshold=1.0)

history = [
    (1.0, 1.0, 0.5),
    (0.8, 0.9, 0.7),
    (0.7, 0.95, 2.5),
    (0.6, 1.1, 0.8),
]

for epoch, (train_loss, val_loss, grad_norm) in enumerate(history, start=1):
    result = controller.log_epoch(train_loss, val_loss, grad_norm)
    print(f"Epoch {epoch}: {result}")
