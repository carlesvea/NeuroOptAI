from neurooptai.controllers.training_controller import TrainingController

controller = TrainingController(
    gradient_threshold=1.0,
    halo_enabled=True,
    meta_control_cost=2.0,
    optimization_cost=2.0,
)

print("Case 1: intervention blocked")
result = controller.log_epoch(
    train_loss=1.0,
    validation_loss=1.0,
    gradient_norm=2.5,
    learning_rate=0.001,
    avoided_bad_branch_cost=3.0,
)
print(result)

print("Case 2: intervention allowed")
result = controller.log_epoch(
    train_loss=0.9,
    validation_loss=1.1,
    gradient_norm=2.5,
    learning_rate=0.001,
    avoided_bad_branch_cost=10.0,
)
print(result)
