from neurooptai.controllers.training_controller import TrainingController

controller = TrainingController()

controller.log_epoch(train_loss=1.0, validation_loss=0.9, gradient_norm=0.5, learning_rate=0.001)

print(controller.status())
