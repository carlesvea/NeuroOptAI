import argparse
import json
import sys
from neurooptai import __version__, UniversalHALO
from neurooptai.api import analyze_training_history
from neurooptai.controllers.training_controller import TrainingController
from neurooptai.utils.json_logger import JSONLogger


def run_demo():
    controller = TrainingController(gradient_threshold=1.0, stagnation_patience=2)
    history = [(1.0, 1.00, 0.5, 0.001), (0.8, 1.05, 0.5, 0.001), (0.6, 1.10, 2.5, 0.001)]

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

    blocked = controller.log_epoch(1.0, 1.0, 2.5, 0.001, avoided_bad_branch_cost=3.0)
    allowed = controller.log_epoch(0.9, 1.1, 2.5, 0.001, avoided_bad_branch_cost=10.0)

    print("HALO blocked case:", blocked)
    print("HALO allowed case:", allowed)


def run_analyze(args):
    controller = TrainingController(
        gradient_threshold=args.gradient_threshold,
        stagnation_patience=args.stagnation_patience,
        halo_enabled=args.halo_enabled,
        meta_control_cost=args.meta_control_cost,
        optimization_cost=args.optimization_cost,
    )

    result = controller.log_epoch(
        train_loss=args.train_loss,
        validation_loss=args.validation_loss,
        gradient_norm=args.gradient_norm,
        learning_rate=args.learning_rate,
        avoided_bad_branch_cost=args.avoided_bad_branch_cost,
    )

    if args.log_file:
        logger = JSONLogger(args.log_file)
        logger.log({
            "command": "analyze",
            "train_loss": args.train_loss,
            "validation_loss": args.validation_loss,
            "gradient_norm": args.gradient_norm,
            "learning_rate": args.learning_rate,
            "decision": result,
        })

    print(json.dumps(result, indent=2) if args.json_output else result)


def run_analyze_history(args):
    try:
        with open(args.input_file, "r", encoding="utf-8") as file:
            history = json.load(file)

        results = analyze_training_history(
            history,
            gradient_threshold=args.gradient_threshold,
            stagnation_patience=args.stagnation_patience,
            halo_enabled=args.halo_enabled,
            meta_control_cost=args.meta_control_cost,
            optimization_cost=args.optimization_cost,
        )

        if args.log_file:
            logger = JSONLogger(args.log_file)
            logger.log({
                "command": "analyze-history",
                "input_file": args.input_file,
                "decisions": results,
            })

        print(json.dumps(results, indent=2) if args.json_output else results)

    except FileNotFoundError:
        print(f"Error: input file not found: {args.input_file}", file=sys.stderr)
        raise SystemExit(1)
    except json.JSONDecodeError as error:
        print(f"Error: invalid JSON input: {error}", file=sys.stderr)
        raise SystemExit(1)
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)


def run_universal_halo(args):
    halo = UniversalHALO()

    try:
        result = halo.decision(
            intervention_cost=args.intervention_cost,
            avoided_cost=args.avoided_cost,
            probability_of_failure=args.probability_of_failure,
        )

        print(json.dumps(result, indent=2) if args.json_output else result)
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)

def main():
    parser = argparse.ArgumentParser(description="NeuroOptAI command line interface")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("demo")
    subparsers.add_parser("halo")
    subparsers.add_parser("version")

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--train-loss", type=float, required=True)
    analyze.add_argument("--validation-loss", type=float, required=True)
    analyze.add_argument("--gradient-norm", type=float, default=None)
    analyze.add_argument("--learning-rate", type=float, default=None)
    analyze.add_argument("--gradient-threshold", type=float, default=1.0)
    analyze.add_argument("--stagnation-patience", type=int, default=3)
    analyze.add_argument("--halo-enabled", action="store_true")
    analyze.add_argument("--meta-control-cost", type=float, default=0.0)
    analyze.add_argument("--optimization-cost", type=float, default=0.0)
    analyze.add_argument("--avoided-bad-branch-cost", type=float, default=None)
    analyze.add_argument("--json-output", action="store_true")
    analyze.add_argument("--log-file", type=str, default=None)

    history = subparsers.add_parser("analyze-history")
    history.add_argument("--input-file", type=str, required=True)
    history.add_argument("--gradient-threshold", type=float, default=1.0)
    history.add_argument("--stagnation-patience", type=int, default=3)
    history.add_argument("--halo-enabled", action="store_true")
    history.add_argument("--meta-control-cost", type=float, default=0.0)
    history.add_argument("--optimization-cost", type=float, default=0.0)
    history.add_argument("--json-output", action="store_true")
    history.add_argument("--log-file", type=str, default=None)

    universal = subparsers.add_parser("universal-halo")
    universal.add_argument("--intervention-cost", type=float, required=True)
    universal.add_argument("--avoided-cost", type=float, required=True)
    universal.add_argument("--probability-of-failure", type=float, default=None)
    universal.add_argument("--json-output", action="store_true")

    args = parser.parse_args()

    if args.command == "demo":
        run_demo()
    elif args.command == "halo":
        run_halo_demo()
    elif args.command == "version":
        print(f"NeuroOptAI {__version__}")
    elif args.command == "analyze":
        run_analyze(args)
    elif args.command == "analyze-history":
        run_analyze_history(args)
    elif args.command == "universal-halo":
        run_universal_halo(args)


if __name__ == "__main__":
    main()
