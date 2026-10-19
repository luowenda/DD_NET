from __future__ import annotations

import csv
import json
import platform
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .data import RadarSequenceDataset
from .engine import (
    _dataset, _loader, _validate_checkpoint_config, load_model_checkpoint,
    seed_everything, select_device,
)
from .models import PySTEPSOpticalFlow, build_trainable_adapter


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _timed_iterations(
    operation: Callable[[], Any],
    device: torch.device,
    warmup: int,
    iterations: int,
) -> tuple[list[float], Any]:
    result: Any = None
    with torch.inference_mode():
        for _ in range(warmup):
            result = operation()
        _synchronize(device)

        durations_ms = []
        for _ in range(iterations):
            start = time.perf_counter()
            result = operation()
            _synchronize(device)
            durations_ms.append((time.perf_counter() - start) * 1000.0)
    return durations_ms, result


def _timing_summary(durations_ms: list[float], batch_size: int) -> dict[str, float]:
    values = np.asarray(durations_ms, dtype=np.float64)
    mean_ms = float(values.mean())
    return {
        "mean_ms_per_batch": mean_ms,
        "median_ms_per_batch": float(np.median(values)),
        "std_ms_per_batch": float(values.std(ddof=0)),
        "p05_ms_per_batch": float(np.percentile(values, 5)),
        "p95_ms_per_batch": float(np.percentile(values, 95)),
        "mean_ms_per_sample": mean_ms / batch_size,
        "samples_per_second": 1000.0 * batch_size / mean_ms,
    }


def _runtime_metadata(device: torch.device) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "device": str(device),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cpu_threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        metadata["gpu"] = {
            "name": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "total_memory_mb": properties.total_memory / (1024**2),
        }
    return metadata


def _protocol_metadata(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "input_frames": int(config["experiment"]["input_frames"]),
        "output_frames": int(config["experiment"]["output_frames"]),
        "height": int(config["data"]["height"]),
        "width": int(config["data"]["width"]),
        "input_units": "dBZ",
        "profiled_at_utc": datetime.now(UTC).isoformat(),
        "experiment_config": config["_meta"]["experiment_config"],
        "model_config": config["_meta"]["model_config"],
    }


def _estimate_flops(
    operation: Callable[[], torch.Tensor], device: torch.device
) -> dict[str, Any]:
    try:
        from torch.utils.flop_counter import FlopCounterMode

        with torch.inference_mode(), FlopCounterMode(display=False) as counter:
            operation()
            _synchronize(device)
        captured_flops = int(counter.get_total_flops())
        return {
            "captured_flops_per_batch": captured_flops or None,
            "method": "torch.utils.flop_counter.FlopCounterMode",
            "scope": "one full input-to-forecast inference call",
            "limitation": (
                "PyTorch FlopCounterMode covers its registered operators only; custom and "
                "unsupported operators may be omitted, so this is not asserted as an exact "
                "theoretical total."
            ),
        }
    except Exception as error:  # noqa: BLE001 - counter/operator support varies by build.
        return {
            "captured_flops_per_batch": None,
            "method": "torch.utils.flop_counter.FlopCounterMode",
            "scope": "one full input-to-forecast inference call",
            "error": f"{type(error).__name__}: {error}",
        }


def _load_profile_checkpoint(adapter, checkpoint, device, config):
    payload = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    if isinstance(payload, dict) and isinstance(payload.get("model"), dict):
        state = payload["model"]
        current = adapter.state_dict()
        fixed = "hffl.dwt.filters"
        missing = current.keys() - state.keys()
        unexpected = state.keys() - current.keys()
        if missing == {fixed} and not unexpected and config["training"]["loss"] != "mse_hffl":
            _validate_checkpoint_config(config, payload["config"])
            # Old native-loss adapters omitted this fixed, prediction-unused
            # Haar buffer. Never relax loading for learned/model state.
            from .losses import HaarDWT2D

            torch.testing.assert_close(current[fixed].cpu(), HaarDWT2D().filters, atol=0, rtol=0)
            restored = dict(state)
            restored[fixed] = current[fixed]
            adapter.load_state_dict(restored, strict=True)
            return {"restored_fixed_loss_buffer": fixed,
                    "scope": "Fixed Haar loss buffer only; unused by predict; no checkpoint file changed."}
    del payload
    load_model_checkpoint(adapter, checkpoint, device, config)
    return None


def _profile_trainable(
    config: dict[str, Any],
    batch_size: int,
    warmup: int,
    iterations: int,
    checkpoint: str | Path | None,
    include_flops: bool,
) -> dict[str, Any]:
    device = select_device(config["experiment"]["device"])
    if device.type == "cuda":
        torch.cuda.empty_cache()
    adapter = build_trainable_adapter(config, device).to(device).eval()
    checkpoint_record: dict[str, Any]
    if checkpoint is not None:
        compatibility = _load_profile_checkpoint(adapter, checkpoint, device, config)
        checkpoint_record = {"mode": "loaded", "path": str(Path(checkpoint).resolve())}
        if compatibility:
            checkpoint_record["compatibility"] = compatibility
    else:
        checkpoint_record = {
            "mode": "random_initialization",
            "note": "Weights do not change architecture, operator shapes, or parameter count.",
        }

    batch = next(iter(_loader(config, "test", batch_size, False)))
    input_dbz = batch["input_dbz"].to(device, non_blocking=True)
    operation = lambda: adapter.predict(input_dbz)

    durations_ms, output = _timed_iterations(operation, device, warmup, iterations)
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"Expected tensor prediction, got {type(output).__name__}")
    if not torch.isfinite(output).all():
        raise RuntimeError("Profiling prediction contains non-finite values")
    output_shape = list(output.shape)
    del output

    memory: dict[str, Any]
    if device.type == "cuda":
        _synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        baseline_allocated = torch.cuda.memory_allocated(device)
        with torch.inference_mode():
            memory_output = operation()
            _synchronize(device)
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
        memory = {
            "baseline_allocated_mb": baseline_allocated / (1024**2),
            "peak_allocated_mb": peak_allocated / (1024**2),
            "peak_reserved_mb": peak_reserved / (1024**2),
            "inference_incremental_peak_mb": (peak_allocated - baseline_allocated) / (1024**2),
            "scope": "model, input batch, output, and inference intermediates in this process",
        }
        del memory_output
    else:
        memory = {
            "peak_allocated_mb": None,
            "reason": "CUDA allocator metrics are unavailable on the selected device.",
        }

    parameters = [parameter for parameter in adapter.parameters()]
    hypergraph = None
    encoder = getattr(getattr(adapter, "model", None), "conv_in", None)
    if encoder is not None and hasattr(encoder, "scales"):
        incidences, edges = {}, {}
        for scale in encoder.scales:
            index = getattr(encoder, f"hyperedge_{scale}")
            incidences[scale] = int(index.shape[1])
            edges[scale] = int(index[1].max().item()) + 1
        hypergraph = {
            "nodes_per_sample": int(np.prod((encoder.temporal_size, *encoder.spatial_size))),
            "retained_incidence_counts": incidences,
            "edges_per_scale": edges,
            "active_scales": [scale for scale, gate in zip(encoder.scales, encoder.scale_gates.tolist()) if gate],
            "retained_incidence_buffer_bytes": sum(
                getattr(encoder, f"hyperedge_{scale}").numel() *
                getattr(encoder, f"hyperedge_{scale}").element_size() for scale in encoder.scales
            ),
        }
    result = {
        "schema_version": 1,
        "model": config["model"]["name"],
        "family": "trainable",
        "input_shape": list(input_dbz.shape),
        "output_shape": output_shape,
        "batch_size": batch_size,
        "warmup_iterations": warmup,
        "measured_iterations": iterations,
        "parameters": {
            "total": sum(parameter.numel() for parameter in parameters),
            "trainable": sum(parameter.numel() for parameter in parameters if parameter.requires_grad),
        },
        "timing": _timing_summary(durations_ms, batch_size),
        "durations_ms": durations_ms,
        "memory": memory,
        "flops": _estimate_flops(operation, device) if include_flops else {
            "captured_flops_per_batch": None,
            "reason": "FLOPs collection was disabled by command-line option.",
        },
        "checkpoint": checkpoint_record,
        "hypergraph": hypergraph,
        "protocol": _protocol_metadata(config),
        "runtime": _runtime_metadata(device),
    }
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _profile_pysteps(
    config: dict[str, Any], batch_size: int, warmup: int, iterations: int
) -> dict[str, Any]:
    dataset: RadarSequenceDataset = _dataset(config, "test")
    if batch_size > len(dataset):
        raise ValueError(f"batch_size {batch_size} exceeds test-set size {len(dataset)}")
    samples = np.stack(
        [dataset[index]["input_dbz"].squeeze(1).numpy() for index in range(batch_size)], axis=0
    )
    baseline = PySTEPSOpticalFlow(config)
    operation = lambda: baseline.predict_batch(samples)
    durations_ms, output = _timed_iterations(
        operation, torch.device("cpu"), warmup, iterations
    )
    if not np.isfinite(output).all():
        raise RuntimeError("PySTEPS profiling prediction contains non-finite values")
    return {
        "schema_version": 1,
        "model": config["model"]["name"],
        "family": "non_trainable",
        "input_shape": list(samples.shape),
        "output_shape": list(output.shape),
        "batch_size": batch_size,
        "warmup_iterations": warmup,
        "measured_iterations": iterations,
        "parameters": {"total": 0, "trainable": 0},
        "timing": _timing_summary(durations_ms, batch_size),
        "durations_ms": durations_ms,
        "memory": {
            "peak_allocated_mb": None,
            "reason": "PySTEPS runs on CPU; CUDA allocator memory is not applicable.",
        },
        "flops": {
            "captured_flops_per_batch": None,
            "reason": "PySTEPS is a NumPy/SciPy/OpenCV algorithm outside PyTorch profiling.",
        },
        "fallback_count": baseline.fallback_count,
        "sample_calls": batch_size * (warmup + iterations),
        "protocol": _protocol_metadata(config),
        "runtime": _runtime_metadata(torch.device("cpu")),
    }


def profile_model(
    config: dict[str, Any],
    batch_size: int = 1,
    warmup: int = 5,
    iterations: int = 20,
    checkpoint: str | Path | None = None,
    include_flops: bool = True,
) -> dict[str, Any]:
    """Profile the production inference path on real test inputs."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    seed_everything(
        int(config["experiment"]["seed"]), config["runtime"]["deterministic_cudnn"]
    )
    if config["model"]["family"] == "non_trainable":
        if checkpoint is not None:
            raise ValueError("A checkpoint cannot be supplied for PySTEPS")
        return _profile_pysteps(config, batch_size, warmup, iterations)
    return _profile_trainable(
        config, batch_size, warmup, iterations, checkpoint, include_flops
    )


def aggregate_profile_results(
    profile_dir: str | Path, output_prefix: str | Path
) -> tuple[Path, Path]:
    """Create machine-readable and reviewer-readable efficiency tables."""
    source_dir = Path(profile_dir)
    paths = sorted(source_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No profile JSON files found in {source_dir}")

    rows = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
        flops = result["flops"].get("captured_flops_per_batch")
        rows.append(
            {
                "model": result["model"],
                "device": result["runtime"].get("gpu", {}).get(
                    "name", result["runtime"]["device"]
                ),
                "batch_size": result["batch_size"],
                "parameters_m": result["parameters"]["total"] / 1_000_000,
                "captured_gflops_per_batch": None if flops is None else flops / 1_000_000_000,
                "mean_latency_ms_per_sample": result["timing"]["mean_ms_per_sample"],
                "std_latency_ms_per_batch": result["timing"]["std_ms_per_batch"],
                "samples_per_second": result["timing"]["samples_per_second"],
                "peak_allocated_mb": result["memory"].get("peak_allocated_mb"),
                "incremental_peak_mb": result["memory"].get(
                    "inference_incremental_peak_mb"
                ),
            }
        )

    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = prefix.with_suffix(".csv")
    markdown_path = prefix.with_suffix(".md")
    fields = list(rows[0])
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    def display(value: Any, decimals: int = 3) -> str:
        if value is None:
            return "N/A"
        if isinstance(value, float):
            return f"{value:.{decimals}f}"
        return str(value)

    lines = [
        "# Inference Efficiency Profile",
        "",
        (
            "| Model | Device | Batch | Params (M) | Captured GFLOPs/batch | "
            "Latency (ms/sample) | Throughput (samples/s) | Peak allocated (MB) |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['device']} | {row['batch_size']} | "
            f"{display(row['parameters_m'])} | {display(row['captured_gflops_per_batch'])} | "
            f"{display(row['mean_latency_ms_per_sample'])} | "
            f"{display(row['samples_per_second'])} | {display(row['peak_allocated_mb'])} |"
        )
    lines.extend(
        [
            "",
            (
                "Parameter counts are exact. Captured GFLOPs are reported by PyTorch "
                "`FlopCounterMode` for registered operators and may omit custom or "
                "unsupported operations. Latency uses synchronized inference after warm-up. CUDA "
                "peak allocated memory includes model parameters, the input/output batch, and "
                "inference intermediates. PySTEPS runs on CPU, so CUDA memory and PyTorch FLOPs are "
                "not applicable."
            ),
            "",
        ]
    )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, markdown_path
