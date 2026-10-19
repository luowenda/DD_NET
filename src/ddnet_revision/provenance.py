from __future__ import annotations

import importlib.metadata
import hashlib
import json
import math
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT
from .data import load_year_split_indices, sha256_file

PACKAGES = (
    "numpy",
    "pysteps",
    "scikit-image",
    "timm",
    "torch",
    "torch-geometric",
    "torchvision",
)


def package_versions() -> dict[str, str]:
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "NOT_INSTALLED"
    return versions


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNVERSIONED_WORKTREE"


def git_dirty() -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=PROJECT_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return bool(result.stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def source_tree_sha256() -> str:
    source_root = PROJECT_ROOT / "src" / "ddnet_revision"
    digest = hashlib.sha256()
    for path in sorted(source_root.rglob("*.py")):
        relative = path.relative_to(PROJECT_ROOT).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def build_run_manifest(
    config: dict[str, Any],
    include_checksums: bool = True,
    dataset_splits: tuple[str, ...] = ("train", "test"),
) -> dict[str, Any]:
    data = config["data"]
    datasets = {}
    checksums: dict[Path, str] = {}
    for split in dataset_splits:
        source_split = "train" if split == "validation" else split
        path = Path(data[f"{source_split}_path"])
        if include_checksums and path not in checksums:
            checksums[path] = sha256_file(path)
        datasets[split] = {
            "path": str(path.resolve()),
            "sha256": checksums[path] if include_checksums else "NOT_COMPUTED",
        }
        if config["selection"]["validation_enabled"] and split in {"train", "validation"}:
            manifest_path = Path(data["validation_split_manifest"])
            split_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if include_checksums and split_manifest["source_sha256"] != checksums[path]:
                raise ValueError("Validation manifest checksum does not match the training data")
            indices = load_year_split_indices(manifest_path, int(data["train_samples"]), split)
            datasets[split].update(
                {
                    "source_split": "train",
                    "year": data["validation_year"]
                    if split == "validation"
                    else data["train_years"],
                    "samples": len(indices),
                    "source_indices": list(indices),
                    "split_manifest": str(manifest_path.resolve()),
                    "split_manifest_sha256": sha256_file(manifest_path),
                }
            )
    return {
        "created_utc": datetime.now(UTC).isoformat(),
        "pipeline_git_revision": git_revision(),
        "pipeline_git_dirty": git_dirty(),
        "pipeline_source_sha256": source_tree_sha256(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": package_versions(),
        "datasets": datasets,
        "config": config,
    }


def write_json(path: str | Path, value: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value
