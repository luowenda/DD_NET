from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import DEFAULT_EXPERIMENT, list_models, load_config


def _model_arguments(parser):
    parser.add_argument("--model", default="ddnet", choices=list_models())
    parser.add_argument("--seed", type=int)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DD-Net and reviewed experiment reproduction")
    parser.add_argument("--experiment", default=str(DEFAULT_EXPERIMENT))
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list-models")
    listing.add_argument("--group", choices=("main", "fine", "coarse"))
    _model_arguments(commands.add_parser("validate-config"))
    audit = commands.add_parser("audit-data")
    audit.add_argument("--output", default="artifacts/data_manifest.json")
    audit.add_argument("--verify-reference", action="store_true")
    training = commands.add_parser("train")
    _model_arguments(training)
    training.add_argument("--run-name", required=True)
    training.add_argument("--resume")
    training.add_argument("--stop-after-epoch", type=int)
    prediction = commands.add_parser("predict")
    _model_arguments(prediction)
    prediction.add_argument("--run-name", required=True)
    prediction.add_argument("--checkpoint")
    evaluation = commands.add_parser("evaluate")
    _model_arguments(evaluation)
    evaluation.add_argument("--run-name", required=True)
    evaluation.add_argument("--predictions", required=True)
    smoke = commands.add_parser("smoke-test")
    _model_arguments(smoke)
    smoke.add_argument("--batch-size", type=int, default=1)
    smoke.add_argument("--backward", action="store_true")
    profile = commands.add_parser("profile")
    _model_arguments(profile)
    profile.add_argument("--checkpoint")
    profile.add_argument("--architecture-only", action="store_true")
    profile.add_argument("--batch-size", type=int, default=1)
    profile.add_argument("--warmup", type=int, default=10)
    profile.add_argument("--iterations", type=int, default=100)
    profile.add_argument("--skip-flops", action="store_true")
    profile.add_argument("--output", required=True)
    return parser


def _audit_data(config, output, verify_reference):
    from .data import inspect_npy
    from .provenance import write_json

    references = {
        "train": "ece842a05fcca781fa2136ad0f23de3c9367dba568b0bf13d7f1c69e34ce7121",
        "test": "648ed414dbb80609c2613de2cfc3b9fd0bfefcba9e994010f191f512728bea81",
    }
    manifest = {}
    for split in ("train", "test"):
        record = inspect_npy(config["data"][f"{split}_path"])
        expected = [40, config["data"][f"{split}_samples"], 64, 64]
        if record["shape"] != expected:
            raise ValueError(f"{split}: expected T,N,H,W shape {expected}, got {record['shape']}")
        if record["nan_count"] or record["inf_count"] or record["min"] < 0 or record["max"] > 70:
            raise ValueError(f"{split}: data must be finite physical reflectivity in 0-70 dBZ")
        if verify_reference and record["sha256"] != references[split]:
            raise ValueError(f"{split}: reference dataset checksum mismatch")
        manifest[split] = record
    write_json(output, manifest)
    print(Path(output).resolve())


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "list-models":
        print("\n".join(list_models(args.group)))
        return
    config = load_config(getattr(args, "model", "ddnet"), experiment_path=args.experiment)
    if getattr(args, "seed", None) is not None:
        config["experiment"]["seed"] = args.seed
    if args.command == "validate-config":
        print(json.dumps(config, indent=2))
        return
    if args.command == "audit-data":
        _audit_data(config, args.output, args.verify_reference)
        return

    if args.command == "profile":
        if config["model"]["family"] == "trainable" and not args.checkpoint and not args.architecture_only:
            raise ValueError("Supply a trained checkpoint, or explicitly select --architecture-only")
        from .profiling import profile_model
        from .provenance import write_json

        result = profile_model(config, args.batch_size, args.warmup, args.iterations,
                               args.checkpoint, include_flops=not args.skip_flops)
        write_json(args.output, result)
        print(Path(args.output).resolve())
        return

    from .engine import evaluate, predict_pysteps, predict_trainable, smoke_test, train

    if args.command == "train":
        print(train(config, args.run_name, args.resume,
                    approve_proposed_config=True, stop_after_epoch=args.stop_after_epoch))
    elif args.command == "predict":
        if config["model"]["family"] == "non_trainable":
            if args.checkpoint:
                raise ValueError("PySTEPS does not use checkpoints")
            print(predict_pysteps(config, args.run_name))
        else:
            if not args.checkpoint:
                raise ValueError("Neural prediction requires --checkpoint")
            print(predict_trainable(config, args.checkpoint, args.run_name))
    elif args.command == "evaluate":
        print(evaluate(config, args.predictions, args.run_name))
    elif args.command == "smoke-test":
        print(json.dumps(smoke_test(config, args.batch_size, args.backward), indent=2))


if __name__ == "__main__":
    main()
