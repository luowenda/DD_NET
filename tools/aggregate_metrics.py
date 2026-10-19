"""Aggregate each independently evaluated run; never pool counts across seeds."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ddnet_revision.config import list_models, load_config, validate_config


def flatten(value, prefix=""):
    output = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            output.update(flatten(item, name))
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            output[name] = float(item)
    return output


def collect(suite, seeds, artifact_root):
    result = {}
    shared_test_hash = None
    shared_train_hash = None
    for alias in list_models(suite):
        config = load_config(alias)
        deterministic = config["model"]["family"] == "non_trainable"
        samples = []
        used_seeds = [0] if deterministic else seeds
        for seed in used_seeds:
            run = "deterministic" if deterministic else f"seed{seed}"
            directory = artifact_root / config["model"]["name"] / run
            manifest = json.loads((directory / "prediction_manifest.json").read_text())
            validate_config(manifest["config"])
            if manifest["config"]["model"]["name"] != config["model"]["name"]:
                raise ValueError(f"Prediction/model mismatch: {directory}")
            if not deterministic and manifest["config"]["experiment"]["seed"] != seed:
                raise ValueError(f"Prediction/seed mismatch: {directory}")
            test_hash = manifest["datasets"]["test"]["sha256"]
            if shared_test_hash is None:
                shared_test_hash = test_hash
            elif shared_test_hash != test_hash:
                raise ValueError("Cannot combine different test datasets")
            if not deterministic:
                training = json.loads((directory / "training_manifest.json").read_text())
                validate_config(training["config"])
                if training["config"]["experiment"]["seed"] != seed:
                    raise ValueError("Training seed does not match the evaluated run")
                train_hash = training["datasets"]["train"]["sha256"]
                if shared_train_hash is None:
                    shared_train_hash = train_hash
                elif shared_train_hash != train_hash:
                    raise ValueError("Cannot combine different training datasets")
                records = [json.loads(line) for line in (directory / "train.jsonl").read_text().splitlines() if line.strip()]
                if len(records) != 100 or records[-1]["epoch"] != 100 or records[-1]["global_step"] != 13700:
                    raise ValueError("Cannot summarize an incomplete E100 training run")
            if deterministic:
                status = json.loads((directory / "pysteps_status.json").read_text())
                if status["fallback_count"]:
                    raise ValueError("Do not summarize PySTEPS as pure extrapolation after persistence fallbacks")
            metrics = json.loads((directory / "metrics.json").read_text())
            if metrics.get("samples") != 88 or metrics.get("comparison") != ">=":
                raise ValueError("Metric sample count or threshold convention changed")
            samples.append(flatten(metrics["overall"]))
        keys = set(samples[0])
        if any(set(item) != keys for item in samples):
            raise ValueError(f"Metric schema changed across seeds: {alias}")
        result[alias] = {
            "seeds": [] if deterministic else used_seeds,
            "deterministic": deterministic,
            "metrics": {key: {"mean": statistics.mean(item[key] for item in samples),
                              "std": statistics.stdev(item[key] for item in samples)
                              if len(samples) > 1 else None}
                        for key in sorted(keys)},
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, choices=("main", "fine", "coarse"))
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--artifact-root", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seeds = args.seeds if args.seeds is not None else ([0, 1, 2] if args.suite == "main" else [0])
    if not seeds or len(set(seeds)) != len(seeds):
        parser.error("Supply distinct seeds")
    result = collect(args.suite, seeds, args.artifact_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
