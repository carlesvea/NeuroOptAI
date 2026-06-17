from neurooptai.controllers.training_controller import TrainingController

controller = TrainingController()

history = [
    (1.0, 1.0),
    (0.8, 0.9),
    (0.6, 1.1),
]

for epoch, (train_loss, val_loss) in enumerate(history, start=1):
    result = controller.log_epoch(train_loss, val_loss)
    print(f"Epoch {epoch}: {result}")
