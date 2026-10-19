from __future__ import annotations

import copy
import json
import tomllib
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENT = PROJECT_ROOT / "configs" / "experiment.toml"
MODEL_CONFIG_DIR = PROJECT_ROOT / "configs" / "models"
FINAL_MODEL = "ddnet_incidence_attention_large_e100"


def load_toml(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().open("rb") as handle:
        return tomllib.load(handle)


def registry() -> dict[str, Any]:
    return json.loads((PROJECT_ROOT / "MODEL_REGISTRY.json").read_text(encoding="utf-8"))


def model_entry(name: str) -> dict[str, Any]:
    entries = registry()["entries"]
    if name in entries:
        return entries[name]
    for entry in entries.values():
        if entry["model_id"] == name:
            return entry
    raise ValueError(f"Unknown released model: {name}")


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def load_config(
    model_name: str = "ddnet",
    experiment_path: str | Path = DEFAULT_EXPERIMENT,
    model_path: str | Path | None = None,
) -> dict[str, Any]:
    entry = model_entry(model_name)
    selected = Path(model_path) if model_path else PROJECT_ROOT / entry["config"]
    config = deep_merge(load_toml(experiment_path), load_toml(selected))
    for split in ("train", "test"):
        key = f"{split}_path"
        config["data"][key] = str(resolve_project_path(config["data"][key]).resolve())
    config["_meta"] = {
        "project_root": str(PROJECT_ROOT),
        "experiment_config": str(Path(experiment_path).resolve()),
        "model_config": str(selected.resolve()),
    }
    validate_config(config)
    return config


def comparable_model(model: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(model)
    if "reference_config" in result:
        # This field describes the recipe's origin; it is not a runtime input.
        result["reference_config"] = Path(result["reference_config"]).name
    return result


def validate_config(config: dict[str, Any]) -> None:
    model = config["model"]
    entry = model_entry(model["name"])
    if comparable_model(model) != comparable_model(entry["model"]):
        raise ValueError("Architecture must match the selected released model/ablation exactly")
    if config.get("training", {}) != entry["training"]:
        raise ValueError("Training recipe must match the selected recorded experiment exactly")
    exp, data, selection = config["experiment"], config["data"], config["selection"]
    if exp["input_frames"] != 20 or exp["output_frames"] != 20:
        raise ValueError("The released experiments use a 20-to-20 forecast")
    if exp["frame_interval_minutes"] != 6 or exp["normalization_scale"] != 70.0:
        raise ValueError("Use six-minute frames and physical dBZ normalized by 70")
    if (exp["evaluation_min"], exp["evaluation_max"]) != (0.0, 70.0):
        raise ValueError("The evaluation domain must be 0-70 dBZ")
    if exp["thresholds_dbz"] != [10.0, 20.0, 30.0, 40.0]:
        raise ValueError("Use the released 10/20/30/40 dBZ evaluation thresholds")
    if data["canonical_layout"] != "T,N,H,W" or data["model_layout"] != "N,T,C,H,W":
        raise ValueError("Use T,N,H,W arrays and N,T,C,H,W model tensors")
    if (data["height"], data["width"]) != (64, 64):
        raise ValueError("The released adapters use 64x64 radar frames")
    if any(int(data[key]) <= 0 for key in ("train_samples", "test_samples")):
        raise ValueError("Dataset sample counts must be positive")
    if selection["validation_enabled"] or selection["test_during_training"]:
        raise ValueError("The released runs optimize on training data only")
    if selection["strategy"] != "last_epoch":
        raise ValueError("Checkpoint selection must use the final epoch")
    if data["train_samples"] != 551 or data["test_samples"] != 88:
        raise ValueError("The released E100 recipes are locked to 551 train / 88 test sequences")
    if data["train_years"] != "2011-2017,2019-2021" or data["test_year"] != 2018:
        raise ValueError("Retain the recorded year-wise split metadata")
    if Path(data["train_path"]).resolve() == Path(data["test_path"]).resolve():
        raise ValueError("Training and test paths must be different")
    required_evaluation = {
        "comparison": ">=", "categorical_aggregation": "pooled",
        "continuous_aggregation": "sample_frame_macro", "zero_denominator": 0.0,
        "save_predictions": True, "clip_predictions": True,
    }
    if config["evaluation"] != required_evaluation:
        raise ValueError("Retain the shared released evaluation protocol")
    if int(config["runtime"]["checkpoint_every"]) <= 0:
        raise ValueError("checkpoint_every must be positive")
    if int(config["runtime"]["prediction_batch_size"]) != 4:
        raise ValueError("The recorded accuracy protocol uses prediction batch size 4")
    if config["runtime"]["deterministic_cudnn"] is not True:
        raise ValueError("Retain the recorded deterministic cuDNN setting")


def list_models(group: str | None = None) -> list[str]:
    selected = registry()
    return list(selected["groups"][group]) if group else list(selected["entries"])
