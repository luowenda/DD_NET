from __future__ import annotations

import fcntl
import json
import os
import random
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .config import comparable_model, resolve_project_path
from .data import (
    IndexedRadarSequenceDataset,
    RadarSequenceDataset,
    load_targets_dbz,
    load_year_split_indices,
    sha256_file,
)
from .metrics import RevisionEvaluator
from .models import PySTEPSOpticalFlow, build_trainable_adapter
from .provenance import build_run_manifest, write_json


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int, deterministic_cudnn: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic_cudnn
    torch.backends.cudnn.benchmark = not deterministic_cudnn


def _run_dir(config: dict[str, Any], run_name: str) -> Path:
    if not run_name or Path(run_name).name != run_name or run_name in {".", ".."}:
        raise ValueError("run_name must be a single directory name")
    root = resolve_project_path(config["artifacts"]["root"])
    return root / config["model"]["name"] / run_name


@contextmanager
def _training_run_lock(run_dir: Path, resume: str | Path | None):
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / ".training.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Training is already running in {run_dir}") from error
        try:
            if resume is None:
                if any(path.name != ".training.lock" for path in run_dir.iterdir()):
                    raise FileExistsError(
                        f"Refusing to overwrite {run_dir}; use --resume with its last.pt"
                    )
            elif Path(resume).resolve() != (run_dir / "last.pt").resolve():
                raise ValueError("Resume must use this run directory's last.pt")
            elif not Path(resume).is_file():
                raise FileNotFoundError(resume)
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _prediction_artifact_names(split: str) -> tuple[str, str]:
    if split == "test":
        return "predictions.npy", "prediction_manifest.json"
    if split == "validation":
        return "validation_predictions.npy", "validation_prediction_manifest.json"
    raise ValueError(f"Unsupported prediction split: {split}")


@contextmanager
def _prediction_run_lock(run_dir: Path, split: str = "test"):
    run_dir.mkdir(parents=True, exist_ok=True)
    output_name, manifest_name = _prediction_artifact_names(split)
    with (run_dir / f".{split}_prediction.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Prediction is already running in {run_dir}") from error
        try:
            protected = (
                run_dir / output_name,
                run_dir / f"{output_name}.tmp",
                run_dir / manifest_name,
            )
            if any(path.exists() for path in protected):
                raise FileExistsError(
                    f"Refusing to overwrite prediction artifacts in {run_dir}; use a new run name"
                )
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _validate_resume_config(current: dict[str, Any], saved: dict[str, Any]) -> None:
    for section in ("model", "training", "selection", "experiment", "data", "runtime"):
        ignored = {"device", "num_workers"} if section == "experiment" else set()
        if section == "data":
            ignored = {"train_path", "test_path"}
        left = {key: value for key, value in current[section].items() if key not in ignored}
        right = {key: value for key, value in saved[section].items() if key not in ignored}
        if section == "model":
            left, right = comparable_model(left), comparable_model(right)
        if left != right:
            raise ValueError(f"Resume configuration mismatch in {section}")


def _validate_checkpoint_config(current: dict[str, Any], saved: dict[str, Any]) -> None:
    for section in ("model", "training", "experiment", "data"):
        ignored = {"device", "num_workers", "seed"} if section == "experiment" else set()
        if section == "data":
            ignored = {"train_path", "test_path"}
        left = {key: value for key, value in current[section].items() if key not in ignored}
        right = {key: value for key, value in saved[section].items() if key not in ignored}
        if section == "model":
            left, right = comparable_model(left), comparable_model(right)
        if left != right:
            raise ValueError(f"Checkpoint configuration mismatch in {section}")


def _progress(event: str, **fields: Any) -> None:
    print(
        json.dumps(
            {"event": event, "utc": datetime.now(UTC).isoformat(), **fields}, allow_nan=False
        ),
        flush=True,
    )


def _dataset(config: dict[str, Any], split: str) -> Dataset:
    exp = config["experiment"]
    data = config["data"]
    source_split = "train" if split == "validation" else split
    dataset = RadarSequenceDataset(
        data[f"{source_split}_path"],
        input_frames=exp["input_frames"],
        output_frames=exp["output_frames"],
        expected_samples=data[f"{source_split}_samples"],
        expected_height=data["height"],
        expected_width=data["width"],
    )
    if config["selection"]["validation_enabled"] and split in {"train", "validation"}:
        indices = load_year_split_indices(
            data["validation_split_manifest"], int(data["train_samples"]), split
        )
        expected = int(
            data["tuning_train_samples"] if split == "train" else data["validation_samples"]
        )
        if len(indices) != expected:
            raise ValueError(f"Expected {expected} {split} samples, got {len(indices)}")
        return IndexedRadarSequenceDataset(dataset, indices)
    if split == "validation":
        raise ValueError("Validation split requested while validation is disabled")
    return dataset


def _loader(config: dict[str, Any], split: str, batch_size: int, shuffle: bool) -> DataLoader:
    drop_last = bool(config["training"].get("drop_last", False)) if split == "train" else False
    return DataLoader(
        _dataset(config, split),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(config["experiment"]["num_workers"]),
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
        persistent_workers=int(config["experiment"]["num_workers"]) > 0,
    )


def _optimizer(config: dict[str, Any], parameters: Any) -> torch.optim.Optimizer:
    training = config["training"]
    kwargs = {
        "lr": float(training["learning_rate"]),
        "weight_decay": float(training.get("weight_decay", 0.0)),
    }
    if "optimizer_betas" in training:
        kwargs["betas"] = tuple(float(value) for value in training["optimizer_betas"])
    if "optimizer_eps" in training:
        kwargs["eps"] = float(training["optimizer_eps"])
    name = training["optimizer"].lower()
    if name == "adam":
        return torch.optim.Adam(parameters, **kwargs)
    if name == "adamw":
        return torch.optim.AdamW(parameters, **kwargs)
    raise ValueError(f"Unsupported optimizer: {name}")


def _teacher_forcing_ratio(training: dict[str, Any], global_step: int) -> float:
    if not training.get("scheduled_sampling", False):
        return 0.0
    if global_step >= int(training["sampling_stop_iter"]):
        return 0.0
    sampling_step = global_step + int(bool(training.get("sampling_update_before_draw", False)))
    return max(
        0.0,
        float(training["sampling_start_value"])
        - sampling_step * float(training["sampling_changing_rate"]),
    )


def _drop_path_rate_for_epoch(config: dict[str, Any], epoch: int) -> float | None:
    model = config["model"]
    schedule = model.get("drop_path_schedule")
    if schedule is None:
        return None
    if schedule != "openstl_late":
        raise ValueError(f"Unsupported DropPath schedule: {schedule}")
    cutoff_epoch = int(model["drop_path_cutoff_epoch"])
    return 0.0 if epoch < cutoff_epoch else float(model["drop_path_rate"])


def _apply_drop_path_schedule(
    config: dict[str, Any], adapter: torch.nn.Module, epoch: int
) -> float | None:
    rate = _drop_path_rate_for_epoch(config, epoch)
    if rate is None:
        return None
    model = getattr(adapter, "model", adapter)
    update = getattr(model, "update_drop_path", None)
    if update is None:
        raise TypeError(
            f"Model {config['model']['name']} declares a DropPath schedule "
            "but does not implement update_drop_path"
        )
    update(rate)
    return rate


def _scheduler(
    config: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    steps_per_epoch: int,
) -> tuple[Any | None, bool]:
    training = config["training"]
    name = str(training.get("scheduler", "none")).lower()
    if name == "none":
        return None, False
    if name == "warmup_cosine":
        warmup_steps = int(training["warmup_epochs"]) * steps_per_epoch
        total_steps = int(training["epochs"]) * steps_per_epoch
        if warmup_steps <= 0 or warmup_steps >= total_steps:
            raise ValueError("warmup_epochs must be between 0 and the total epoch count")
        start_factor = float(training.get("warmup_start_factor", 0.01))
        if not 0.0 < start_factor <= 1.0:
            raise ValueError("warmup_start_factor must be in (0, 1]")
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=start_factor,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        decay = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=float(training.get("minimum_learning_rate", 0.0)),
        )
        return (
            torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup, decay],
                milestones=[warmup_steps],
            ),
            True,
        )
    if name == "cosine":
        return (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=int(training["epochs"]),
                eta_min=float(training.get("minimum_learning_rate", 0.0)),
            ),
            False,
        )
    if name == "onecycle":
        return (
            torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=float(training["learning_rate"]),
                epochs=int(training["epochs"]),
                steps_per_epoch=steps_per_epoch,
            ),
            True,
        )
    if name == "multistep":
        return (
            torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[int(value) for value in training["scheduler_milestones"]],
                gamma=float(training.get("scheduler_gamma", 0.1)),
            ),
            False,
        )
    raise ValueError(f"Unsupported scheduler: {name}")


def _save_checkpoint(
    path: Path,
    adapter: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    epoch: int,
    global_step: int,
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema_version": 2,
            "model": adapter.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "global_step": global_step,
            "config": config,
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        },
        temporary,
    )
    os.replace(temporary, path)


def load_model_checkpoint(
    adapter: torch.nn.Module,
    path: str | Path,
    device: torch.device,
    config: dict[str, Any] | None = None,
) -> None:
    payload = torch.load(path, map_location=device, weights_only=False)
    if config is not None:
        if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict):
            raise ValueError(
                "Checkpoint has no embedded pipeline configuration; refusing an "
                "architecture-ambiguous load"
            )
        _validate_checkpoint_config(config, payload["config"])
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    if config is not None and isinstance(state, dict):
        current = adapter.state_dict()
        fixed = "hffl.dwt.filters"
        missing = current.keys() - state.keys()
        unexpected = state.keys() - current.keys()
        if missing == {fixed} and not unexpected and config["training"]["loss"] != "mse_hffl":
            # Only the constant, prediction-unused loss buffer may be restored.
            # Learned state and HFFL-dependent checkpoints always load strictly.
            from .losses import HaarDWT2D

            torch.testing.assert_close(current[fixed].cpu(), HaarDWT2D().filters, atol=0, rtol=0)
            state = dict(state)
            state[fixed] = current[fixed]
    try:
        adapter.load_state_dict(state, strict=True)
    except RuntimeError:
        if not hasattr(adapter, "model"):
            raise
        adapter.model.load_state_dict(state, strict=True)


def train(
    config: dict[str, Any],
    run_name: str,
    resume: str | Path | None = None,
    approve_proposed_config: bool = False,
    stop_after_epoch: int | None = None,
) -> Path:
    if config["model"]["family"] != "trainable":
        raise ValueError(f"{config['model']['name']} is not trainable")
    approved_statuses = {
        "LOCKED_BY_USER",
        "LOCKED_BY_USER_HISTORICAL_REPRODUCTION",
        "REVIEWER_Q7_SINGLE_FACTOR_ABLATION",
        "REVIEWER_Q7_STRICT_FIXED_ARCHITECTURE_ABLATION",
        "SOURCE_AUDITED_RADAR_ADAPTED",
    }
    if (
        config["training"].get("protocol_status") not in approved_statuses
        and not approve_proposed_config
    ):
        raise RuntimeError(
            "Training configuration is still proposed. Review it, then pass "
            "--approve-proposed-config or use an audited protocol status."
        )
    run_dir = _run_dir(config, run_name)
    with _training_run_lock(run_dir, resume):
        return _train_run(config, run_dir, resume, stop_after_epoch)


def _train_run(
    config: dict[str, Any],
    run_dir: Path,
    resume: str | Path | None,
    stop_after_epoch: int | None = None,
) -> Path:
    payload = None
    manifest_splits = (
        ("train", "validation") if config["selection"]["validation_enabled"] else ("train",)
    )
    manifest = build_run_manifest(config, dataset_splits=manifest_splits)
    if resume:
        payload = torch.load(resume, map_location="cpu", weights_only=False)
        _validate_resume_config(config, payload["config"])
        original = json.loads((run_dir / "training_manifest.json").read_text(encoding="utf-8"))
        if original["datasets"] != manifest["datasets"]:
            raise ValueError("Training or validation data manifest changed; refusing resume")
        if int(payload["epoch"]) + 1 >= int(config["training"]["epochs"]):
            _progress("training_already_complete", checkpoint=str(resume))
            return Path(resume)
        manifest["resume_epoch"] = int(payload["epoch"]) + 1
        write_json(run_dir / "resume_manifests" / f"{time.time_ns()}.json", manifest)
    else:
        write_json(run_dir / "training_manifest.json", manifest)

    seed = int(config["experiment"]["seed"])
    seed_everything(seed, config["runtime"]["deterministic_cudnn"])
    device = select_device(config["experiment"]["device"])
    adapter = build_trainable_adapter(config, device).to(device)
    optimizer = _optimizer(config, adapter.parameters())
    loader = _loader(config, "train", int(config["training"]["batch_size"]), True)
    scheduler, scheduler_per_batch = _scheduler(config, optimizer, len(loader))
    start_epoch = 0
    global_step = 0
    if payload is not None:
        adapter.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        if scheduler is not None and payload.get("scheduler") is not None:
            scheduler.load_state_dict(payload["scheduler"])
        start_epoch = int(payload["epoch"]) + 1
        global_step = int(payload["global_step"])
        if "rng" in payload:
            random.setstate(payload["rng"]["python"])
            np.random.set_state(payload["rng"]["numpy"])
            torch.set_rng_state(payload["rng"]["torch"])
            if device.type == "cuda" and payload["rng"]["cuda"] is not None:
                torch.cuda.set_rng_state_all(payload["rng"]["cuda"])
        del payload

    log_path = run_dir / "train.jsonl"
    epochs = int(config["training"]["epochs"])
    final_epoch = epochs if stop_after_epoch is None else int(stop_after_epoch)
    if final_epoch <= 0 or final_epoch > epochs:
        raise ValueError("stop_after_epoch must be between 1 and the configured epoch count")
    if final_epoch <= start_epoch:
        raise ValueError("stop_after_epoch must be greater than the checkpoint epoch")
    checkpoint_every = int(config["runtime"]["checkpoint_every"])
    for epoch in range(start_epoch, final_epoch):
        epoch_start = time.perf_counter()
        drop_path_rate = _apply_drop_path_schedule(config, adapter, epoch)
        _progress(
            "epoch_started",
            epoch=epoch + 1,
            epochs=epochs,
            global_step=global_step,
            drop_path_rate=drop_path_rate,
        )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        adapter.train()
        loss_sum = 0.0
        sample_count = 0
        for batch_index, batch in enumerate(loader):
            input_dbz = batch["input_dbz"].to(device, non_blocking=True)
            target_dbz = batch["target_dbz"].to(device, non_blocking=True)
            ratio = _teacher_forcing_ratio(config["training"], global_step)
            optimizer.zero_grad(set_to_none=True)
            output = adapter.training_step(input_dbz, target_dbz, ratio, global_step)
            if not torch.isfinite(output.loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch + 1}, batch {batch_index + 1}")
            output.loss.backward()
            gradient_clip = float(config["training"].get("gradient_clip_norm", 0.0))
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), gradient_clip)
            optimizer.step()
            if scheduler is not None and scheduler_per_batch:
                scheduler.step()
            batch_size = input_dbz.shape[0]
            batch_loss = float(output.loss.detach().cpu())
            loss_sum += batch_loss * batch_size
            sample_count += batch_size
            global_step += 1
            if batch_index == 0 or (batch_index + 1) % 25 == 0:
                _progress(
                    "optimizer_step_completed",
                    epoch=epoch + 1,
                    batch=batch_index + 1,
                    batches=len(loader),
                    global_step=global_step,
                    loss=batch_loss,
                    peak_allocated_mib=(
                        torch.cuda.max_memory_allocated(device) / 1024**2
                        if device.type == "cuda"
                        else None
                    ),
                )
            del output, input_dbz, target_dbz
        if scheduler is not None and not scheduler_per_batch:
            scheduler.step()
        record = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "train_loss": loss_sum / sample_count,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": time.perf_counter() - epoch_start,
        }
        if drop_path_rate is not None:
            record["drop_path_rate"] = drop_path_rate
        _save_checkpoint(
            run_dir / "last.pt", adapter, optimizer, scheduler, epoch, global_step, config
        )
        if (epoch + 1) % checkpoint_every == 0:
            _save_checkpoint(
                run_dir / "checkpoints" / f"epoch_{epoch + 1:03d}.pt",
                adapter,
                optimizer,
                scheduler,
                epoch,
                global_step,
                config,
            )
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        _progress("checkpoint_saved", checkpoint=str(run_dir / "last.pt"), **record)
    if final_epoch == epochs:
        _progress("training_complete", epochs=epochs, global_step=global_step)
    else:
        _progress(
            "training_stopped_at_planned_epoch",
            epoch=final_epoch,
            configured_epochs=epochs,
            global_step=global_step,
        )
    return run_dir / "last.pt"


def predict_trainable(
    config: dict[str, Any], checkpoint: str | Path, run_name: str, split: str = "test"
) -> Path:
    if split == "validation" and not config["selection"]["validation_enabled"]:
        raise ValueError("Validation prediction requires validation_enabled=true")
    run_dir = _run_dir(config, run_name)
    output_name, manifest_name = _prediction_artifact_names(split)
    with _prediction_run_lock(run_dir, split):
        seed_everything(int(config["experiment"]["seed"]), config["runtime"]["deterministic_cudnn"])
        device = select_device(config["experiment"]["device"])
        adapter = build_trainable_adapter(config, device).to(device)
        load_model_checkpoint(adapter, checkpoint, device, config)
        adapter.eval()
        loader = _loader(config, split, int(config["runtime"]["prediction_batch_size"]), False)
        prediction_manifest = build_run_manifest(config, dataset_splits=(split,))
        prediction_manifest["checkpoint"] = {
            "path": str(Path(checkpoint).resolve()),
            "sha256": sha256_file(checkpoint),
        }
        output_path = run_dir / output_name
        temporary_output = run_dir / f"{output_name}.tmp"
        dataset = loader.dataset
        predictions = np.lib.format.open_memmap(
            temporary_output,
            mode="w+",
            dtype=np.float32,
            shape=(
                len(dataset),
                config["experiment"]["output_frames"],
                config["data"]["height"],
                config["data"]["width"],
            ),
        )
        lower = float(config["experiment"]["evaluation_min"])
        upper = float(config["experiment"]["evaluation_max"])
        with torch.inference_mode():
            for batch in loader:
                indices = batch["index"].numpy()
                output = adapter.predict(batch["input_dbz"].to(device, non_blocking=True))
                value = output.squeeze(2).float().cpu().numpy()
                if config["evaluation"]["clip_predictions"]:
                    value = np.clip(value, lower, upper)
                predictions[indices] = value
        predictions.flush()
        del predictions
        os.replace(temporary_output, output_path)
        write_json(run_dir / manifest_name, prediction_manifest)
        return output_path


def predict_pysteps(config: dict[str, Any], run_name: str) -> Path:
    run_dir = _run_dir(config, run_name)
    with _prediction_run_lock(run_dir):
        dataset = _dataset(config, "test")
        baseline = PySTEPSOpticalFlow(config)
        manifest = build_run_manifest(config, dataset_splits=("test",))
        output_path = run_dir / "predictions.npy"
        temporary_output = run_dir / "predictions.npy.tmp"
        predictions = np.lib.format.open_memmap(
            temporary_output,
            mode="w+",
            dtype=np.float32,
            shape=(
                len(dataset),
                config["experiment"]["output_frames"],
                config["data"]["height"],
                config["data"]["width"],
            ),
        )
        for index in range(len(dataset)):
            input_dbz = dataset[index]["input_dbz"].squeeze(1).numpy()
            predictions[index] = baseline.predict_sample(input_dbz)
        predictions.flush()
        del predictions
        os.replace(temporary_output, output_path)
        write_json(run_dir / "prediction_manifest.json", manifest)
        write_json(run_dir / "pysteps_status.json", {"fallback_count": baseline.fallback_count})
        return output_path


def evaluate(
    config: dict[str, Any], predictions_path: str | Path, run_name: str, split: str = "test"
) -> Path:
    if split == "validation" and not config["selection"]["validation_enabled"]:
        raise ValueError("Validation evaluation requires validation_enabled=true")
    predictions = np.load(predictions_path, mmap_mode="r", allow_pickle=False)
    source_split = "train" if split == "validation" else split
    indices = None
    if split == "validation":
        indices = load_year_split_indices(
            config["data"]["validation_split_manifest"],
            int(config["data"]["train_samples"]),
            "validation",
        )
    targets = load_targets_dbz(
        config["data"][f"{source_split}_path"],
        config["experiment"]["input_frames"],
        config["experiment"]["output_frames"],
        indices,
    )
    if predictions.shape != targets.shape:
        raise ValueError(f"Prediction shape {predictions.shape} != target shape {targets.shape}")
    evaluator = RevisionEvaluator(
        output_frames=config["experiment"]["output_frames"],
        thresholds_dbz=tuple(config["experiment"]["thresholds_dbz"]),
        data_range=config["experiment"]["normalization_scale"],
        frame_interval_minutes=config["experiment"]["frame_interval_minutes"],
    )
    batch_size = int(config["runtime"]["prediction_batch_size"])
    for start in range(0, predictions.shape[0], batch_size):
        stop = min(start + batch_size, predictions.shape[0])
        evaluator.update(predictions[start:stop], targets[start:stop])
    output_name = "validation_metrics.json" if split == "validation" else "metrics.json"
    output_path = _run_dir(config, run_name) / output_name
    write_json(output_path, evaluator.result())
    return output_path


def smoke_test(
    config: dict[str, Any], batch_size: int = 1, backward: bool = False
) -> dict[str, Any]:
    if config["model"]["family"] != "trainable":
        raise ValueError("Use predict-pysteps for the non-trainable optical-flow baseline")
    seed_everything(int(config["experiment"]["seed"]), config["runtime"]["deterministic_cudnn"])
    device = select_device(config["experiment"]["device"])
    adapter = build_trainable_adapter(config, device).to(device)
    batch = next(iter(_loader(config, "train", batch_size, False)))
    input_dbz = batch["input_dbz"].to(device)
    target_dbz = batch["target_dbz"].to(device)
    adapter.train()
    output = adapter.training_step(input_dbz, target_dbz, teacher_forcing_ratio=0.0, global_step=0)
    if tuple(output.prediction_dbz.shape) != tuple(target_dbz.shape):
        raise RuntimeError(
            f"Prediction shape {tuple(output.prediction_dbz.shape)} does not match "
            f"target {tuple(target_dbz.shape)}"
        )
    if not torch.isfinite(output.loss):
        raise RuntimeError("Smoke-test loss is non-finite")
    if backward:
        output.loss.backward()
    return {
        "model": config["model"]["name"],
        "device": str(device),
        "batch_size": batch_size,
        "prediction_shape": list(output.prediction_dbz.shape),
        "loss": float(output.loss.detach().cpu()),
        "backward": backward,
        "parameters": sum(parameter.numel() for parameter in adapter.parameters()),
    }
