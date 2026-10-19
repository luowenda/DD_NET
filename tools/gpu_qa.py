"""Explicit synthetic CUDA integration QA, not a formal training experiment."""

from __future__ import annotations

import argparse
import gc
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def check_one(alias, output_directory):
    import numpy as np
    import torch
    from ddnet_revision.config import load_config
    from ddnet_revision.engine import (
        _optimizer, _save_checkpoint, _scheduler, load_model_checkpoint, seed_everything,
    )
    from ddnet_revision.metrics import RevisionEvaluator
    from ddnet_revision.models import PySTEPSOpticalFlow, build_trainable_adapter

    config = load_config(alias)
    seed_everything(0)
    if config["model"]["family"] == "non_trainable":
        y, x = np.mgrid[:64, :64]
        observed = np.stack([40 * np.exp(-((x - 16 - t / 2) ** 2 + (y - 32) ** 2) / 60)
                             for t in range(20)]).astype(np.float32)
        baseline = PySTEPSOpticalFlow(config)
        prediction = baseline.predict_sample(observed)
        if not np.isfinite(prediction).all() or baseline.fallback_count:
            raise RuntimeError("Synthetic PySTEPS integration failed or used persistence fallback")
        return {"status": "passed", "synthetic_only": True, "model": alias,
                "prediction_shape": list(prediction.shape), "fallback_count": 0}
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required; this tool will not run neural models on the local CPU")
    device = torch.device("cuda")
    adapter = build_trainable_adapter(config, device).to(device)
    optimizer = _optimizer(config, adapter.parameters())
    scheduler, per_batch = _scheduler(config, optimizer, 551 // 4)
    observed = torch.rand(1, 20, 1, 64, 64, device=device) * 70
    target = torch.rand_like(observed) * 70
    adapter.train()
    optimizer.zero_grad(set_to_none=True)
    result = adapter.training_step(observed, target, 0.0, 0)
    if not torch.isfinite(result.loss):
        raise RuntimeError("Synthetic loss is non-finite")
    result.loss.backward()
    gradients = [p.grad for p in adapter.parameters() if p.grad is not None]
    if not gradients or any(not torch.isfinite(gradient).all() for gradient in gradients):
        raise RuntimeError("Synthetic gradients are missing or non-finite")
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    if any(not torch.isfinite(parameter).all() for parameter in adapter.parameters()):
        raise RuntimeError("Non-finite parameters after one optimizer update")
    adapter.eval()
    seed_everything(0)
    with torch.inference_mode():
        before = adapter.predict(observed).detach().clone()
    if before.shape != target.shape or not torch.isfinite(before).all():
        raise RuntimeError("Synthetic inference shape/values are invalid")
    estimated_bytes = sum(t.numel() * t.element_size() for t in adapter.state_dict().values())
    estimated_bytes += sum(t.numel() * t.element_size() for state in optimizer.state.values()
                           for t in state.values() if isinstance(t, torch.Tensor))
    if shutil.disk_usage(output_directory).free < estimated_bytes * 1.2:
        raise RuntimeError("Insufficient disk for the temporary QA checkpoint; no experiment files will be deleted")
    with tempfile.TemporaryDirectory(prefix=f"qa-{alias}-", dir=output_directory) as temporary:
        checkpoint = Path(temporary) / "last.pt"
        _save_checkpoint(checkpoint, adapter, optimizer, scheduler, 0, 1, config)
        parameter = next(parameter for parameter in adapter.parameters() if parameter.numel())
        expected = parameter.detach().flatten()[0].clone()
        with torch.no_grad():
            parameter.view(-1)[0].add_(1.0)
        load_model_checkpoint(adapter, checkpoint, torch.device("cpu"), config)
        if not torch.equal(parameter.detach().flatten()[0], expected):
            raise RuntimeError("Checkpoint did not restore the parameter")
        seed_everything(0)
        with torch.inference_mode():
            after = adapter.predict(observed)
        torch.testing.assert_close(after, before, rtol=1e-4, atol=1e-4)
    evaluator = RevisionEvaluator()
    evaluator.update(after.squeeze(2).cpu().numpy(), target.squeeze(2).cpu().numpy())
    measured = evaluator.result()
    if measured["overall"]["MSE"] < 0:
        raise RuntimeError("Invalid synthetic evaluation")
    loss = float(result.loss.detach().cpu())
    del result, before, after, gradients, optimizer, scheduler, adapter, observed, target
    gc.collect()
    torch.cuda.empty_cache()
    return {"status": "passed", "synthetic_only": True, "not_paper_metrics": True,
            "model": alias, "optimizer_updates": 1, "test_target_used_for_prediction": False,
            "checkpoint_reload_verified": True, "loss": loss,
            "epoch_scheduler": not per_batch}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["ddnet"])
    parser.add_argument("--output-directory", type=Path, default=ROOT / "artifacts/gpu-qa")
    args = parser.parse_args()
    from ddnet_revision.config import model_entry
    from ddnet_revision.provenance import write_json

    for alias in args.models:
        model_entry(alias)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    for alias in args.models:
        try:
            result = check_one(alias, args.output_directory)
            write_json(args.output_directory / f"{alias}.json", result)
            print(f"{alias}: synthetic integration QA passed", flush=True)
        except Exception as error:
            write_json(args.output_directory / f"{alias}.json", {
                "status": "failed", "synthetic_only": True, "error": str(error),
            })
            raise


if __name__ == "__main__":
    main()
