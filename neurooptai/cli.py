import argparse
from neurooptai import __version__
from neurooptai.controllers.training_controller import TrainingController


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


def run_halo_demo():
    controller = TrainingController(
        gradient_threshold=1.0,
        halo_enabled=True,
        meta_control_cost=2.0,
        optimization_cost=2.0,
    )

    blocked = controller.log_epoch(
        train_loss=1.0,
        validation_loss=1.0,
        gradient_norm=2.5,
        learning_rate=0.001,
        avoided_bad_branch_cost=3.0,
    )

    allowed = controller.log_epoch(
        train_loss=0.9,
        validation_loss=1.1,
        gradient_norm=2.5,
        learning_rate=0.001,
        avoided_bad_branch_cost=10.0,
    )

    print("HALO blocked case:", blocked)
    print("HALO allowed case:", allowed)


def main():
    parser = argparse.ArgumentParser(description="NeuroOptAI command line interface")
    parser.add_argument("command", choices=["demo", "halo", "version"], help="Command to execute")
    args = parser.parse_args()

    if args.command == "demo":
        run_demo()
    elif args.command == "halo":
        run_halo_demo()
    elif args.command == "version":
        print(f"NeuroOptAI {__version__}")


if __name__ == "__main__":
    main()
