import argparse
from neurooptai.controllers.training_controller import TrainingController

VERSION = "0.1.0"


def run_demo():
    controller = TrainingController(gradient_threshold=1.0, stagnation_patience=2)

    history = [
        (1.0, 1.00, 0.5, 0.001),
        (0.8, 1.05, 0.5, 0.001),
        (0.6, 1.10, 2.5, 0.001),
    ]

    for epoch, (train_loss, val_loss, grad_norm, lr) in enumerate(history, start=1):
        result = controller.log_epoch(train_loss, val_loss, grad_norm, lr)
        print(f"Epoch {epoch}: {result}")


def main():
    parser = argparse.ArgumentParser(description="NeuroOptAI command line interface")
    parser.add_argument("command", choices=["demo", "version"], help="Command to execute")
    args = parser.parse_args()

    if args.command == "demo":
        run_demo()
    elif args.command == "version":
        print(f"NeuroOptAI {VERSION}")


if __name__ == "__main__":
    main()
