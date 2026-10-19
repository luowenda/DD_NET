"""Print a reproducible sequential plan; run it only with --execute."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ddnet_revision.config import DEFAULT_EXPERIMENT, list_models, load_config


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reuse_completed_step(command):
    from ddnet_revision.config import validate_config

    if "train" in command:
        operation = "train"
    elif "predict" in command:
        operation = "predict"
    else:
        return False
    alias = command[command.index("--model") + 1]
    config = load_config(alias, command[command.index("--experiment") + 1])
    seed = int(command[command.index("--seed") + 1])
    config["experiment"]["seed"] = seed
    run = command[command.index("--run-name") + 1]
    root = Path(config["artifacts"]["root"])
    root = root if root.is_absolute() else ROOT / root
    directory = root / config["model"]["name"] / run
    if operation == "train":
        checkpoint = directory / "last.pt"
        if not checkpoint.exists():
            return False
        manifest = json.loads((directory / "training_manifest.json").read_text())
        if manifest["datasets"]["train"]["sha256"] != _file_sha256(config["data"]["train_path"]):
            raise ValueError("Existing training used different input data")
        import torch
        from ddnet_revision.engine import _validate_resume_config

        payload = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
        _validate_resume_config(config, payload["config"])
        expected_steps = config["training"]["epochs"] * (config["data"]["train_samples"] // config["training"]["batch_size"])
        if payload["epoch"] + 1 != config["training"]["epochs"] or payload["global_step"] != expected_steps:
            raise RuntimeError(f"Incomplete run at {directory}; resume it explicitly before continuing the suite")
        records = [json.loads(line) for line in (directory / "train.jsonl").read_text().splitlines() if line.strip()]
        if len(records) != config["training"]["epochs"] or records[-1]["global_step"] != expected_steps:
            raise ValueError("Completed checkpoint does not match a full training log")
        return True
    predictions = directory / "predictions.npy"
    if not predictions.exists():
        return False
    import numpy as np

    manifest = json.loads((directory / "prediction_manifest.json").read_text())
    validate_config(manifest["config"])
    if manifest["datasets"]["test"]["sha256"] != _file_sha256(config["data"]["test_path"]):
        raise ValueError("Existing prediction used different test input data")
    if manifest["config"]["model"]["name"] != config["model"]["name"] or manifest["config"]["experiment"]["seed"] != seed:
        raise ValueError("Existing prediction model/seed does not match this suite")
    if "--checkpoint" in command:
        checkpoint = Path(command[command.index("--checkpoint") + 1])
        if manifest["checkpoint"]["sha256"] != _file_sha256(checkpoint):
            raise ValueError("Existing prediction came from a different checkpoint")
    value = np.load(predictions, mmap_mode="r", allow_pickle=False)
    if value.shape != (88, 20, 64, 64) or not np.isfinite(value).all():
        raise ValueError("Existing prediction array is incomplete or non-finite")
    return True


def plan(suite: str, seeds: list[int], experiment: str = str(DEFAULT_EXPERIMENT)):
    commands = []
    for alias in list_models(suite):
        config = load_config(alias, experiment)
        selected_seeds = [0] if config["model"]["family"] == "non_trainable" else seeds
        for seed in selected_seeds:
            run = "deterministic" if config["model"]["family"] == "non_trainable" else f"seed{seed}"
            root = Path(config["artifacts"]["root"])
            root = root if root.is_absolute() else ROOT / root
            directory = root / config["model"]["name"] / run
            prefix = [sys.executable, "-m", "ddnet_revision.cli", "--experiment", experiment]
            shared = ["--model", alias, "--run-name", run, "--seed", str(seed)]
            if config["model"]["family"] == "trainable":
                commands.append(prefix + ["train", *shared])
                checkpoint = ["--checkpoint", str(directory / "last.pt")]
            else:
                checkpoint = []
            commands.append(prefix + ["predict", *shared, *checkpoint])
            commands.append(prefix + ["evaluate", *shared, "--predictions", str(directory / "predictions.npy")])
    return commands


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, choices=("main", "fine", "coarse"))
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--experiment", default=str(DEFAULT_EXPERIMENT))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    seeds = args.seeds if args.seeds is not None else ([0, 1, 2] if args.suite == "main" else [0])
    if len(set(seeds)) != len(seeds):
        parser.error("Seeds must be distinct")
    commands = plan(args.suite, seeds, args.experiment)
    if args.execute:
        import os

        environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
        for command in commands:
            print(shlex.join(command), flush=True)
            if reuse_completed_step(command):
                print("# Reused matching completed artifact; no retraining or overwrite.", flush=True)
                continue
            subprocess.run(command, cwd=ROOT, env=environment, check=True)
    else:
        print("# Plan only. No model is run. Add --execute to run sequentially.")
        for command in commands:
            print(shlex.join(command))


if __name__ == "__main__":
    main()
