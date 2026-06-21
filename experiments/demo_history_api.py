from neurooptai import analyze_training_history

history = [
    {"train_loss": 1.0, "validation_loss": 1.0, "gradient_norm": 0.5, "learning_rate": 0.001},
    {"train_loss": 0.8, "validation_loss": 1.1, "gradient_norm": 2.5, "learning_rate": 0.001},
]

results = analyze_training_history(history)

for epoch, result in enumerate(results, start=1):
    print(f"Epoch {epoch}: {result}")
