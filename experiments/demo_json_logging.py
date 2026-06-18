from neurooptai.controllers.training_controller import TrainingController
from neurooptai.utils.json_logger import JSONLogger

controller = TrainingController(gradient_threshold=1.0, stagnation_patience=2)
logger = JSONLogger("demo_neurooptai_log.jsonl")

history = [
    (1.0, 1.00, 0.5, 0.001),
    (0.8, 1.05, 0.5, 0.001),
    (0.6, 1.10, 2.5, 0.001),
]

for epoch, (train_loss, val_loss, grad_norm, lr) in enumerate(history, start=1):
    result = controller.log_epoch(train_loss, val_loss, grad_norm, lr)

    event = {
        "epoch": epoch,
        "train_loss": train_loss,
        "validation_loss": val_loss,
        "gradient_norm": grad_norm,
        "learning_rate": lr,
        "decision": result,
    }

    logger.log(event)
    print(f"Epoch {epoch}: {result}")
