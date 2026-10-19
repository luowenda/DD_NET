from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


def sha256_file(path: str | Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RadarSequenceDataset(Dataset):
    """Read audited radar sequences and expose canonical physical dBZ tensors."""

    def __init__(
        self,
        path: str | Path,
        input_frames: int = 20,
        output_frames: int = 20,
        expected_samples: int | None = None,
        expected_height: int | None = None,
        expected_width: int | None = None,
    ) -> None:
        self.path = Path(path)
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.array = np.load(self.path, mmap_mode="r", allow_pickle=False)
        expected_frames = input_frames + output_frames
        if self.array.ndim != 4:
            raise ValueError(f"Expected T,N,H,W data, got shape {self.array.shape}")
        if self.array.shape[0] != expected_frames:
            raise ValueError(
                f"Expected {expected_frames} frames, got {self.array.shape[0]} in {self.path}"
            )
        if expected_samples is not None and self.array.shape[1] != expected_samples:
            raise ValueError(
                f"Expected {expected_samples} samples, got {self.array.shape[1]} in {self.path}"
            )
        if expected_height is not None and self.array.shape[2] != expected_height:
            raise ValueError(f"Expected height {expected_height}, got {self.array.shape[2]}")
        if expected_width is not None and self.array.shape[3] != expected_width:
            raise ValueError(f"Expected width {expected_width}, got {self.array.shape[3]}")

    def __len__(self) -> int:
        return int(self.array.shape[1])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = np.array(self.array[:, index], dtype=np.float32, copy=True)
        sequence = torch.from_numpy(sample).unsqueeze(1)
        return {
            "input_dbz": sequence[: self.input_frames],
            "target_dbz": sequence[self.input_frames :],
            "index": torch.tensor(index, dtype=torch.int64),
        }


class IndexedRadarSequenceDataset(Dataset):
    """Expose an audited subset while keeping output indices contiguous."""

    def __init__(self, dataset: RadarSequenceDataset, source_indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.source_indices = tuple(int(index) for index in source_indices)
        if len(set(self.source_indices)) != len(self.source_indices):
            raise ValueError("Split indices must be unique")
        if any(index < 0 or index >= len(dataset) for index in self.source_indices):
            raise ValueError("Split index is outside the source dataset")

    def __len__(self) -> int:
        return len(self.source_indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        source_index = self.source_indices[index]
        sample = self.dataset[source_index]
        sample["index"] = torch.tensor(index, dtype=torch.int64)
        sample["source_index"] = torch.tensor(source_index, dtype=torch.int64)
        return sample


def load_year_split_indices(
    manifest_path: str | Path, source_samples: int, split: str
) -> tuple[int, ...]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if int(manifest["source_samples"]) != source_samples:
        raise ValueError("Validation manifest source sample count does not match the dataset")
    validation = tuple(int(index) for index in manifest["validation_indices"])
    if len(set(validation)) != len(validation):
        raise ValueError("Validation indices must be unique")
    if any(index < 0 or index >= source_samples for index in validation):
        raise ValueError("Validation index is outside the source dataset")
    if split == "validation":
        return validation
    if split == "train":
        held_out = set(validation)
        return tuple(index for index in range(source_samples) if index not in held_out)
    raise ValueError(f"Year split is not defined for {split!r}")


def load_targets_dbz(
    path: str | Path,
    input_frames: int = 20,
    output_frames: int = 20,
    indices: Sequence[int] | None = None,
) -> np.ndarray:
    source = np.load(path, mmap_mode="r", allow_pickle=False)
    targets = source[input_frames : input_frames + output_frames]
    if indices is not None:
        targets = targets[:, list(indices)]
    return np.moveaxis(targets, 1, 0)


def inspect_npy(path: str | Path, scan_values: bool = True) -> dict[str, Any]:
    source_path = Path(path)
    array = np.load(source_path, mmap_mode="r", allow_pickle=False)
    result: dict[str, Any] = {
        "path": str(source_path.resolve()),
        "sha256": sha256_file(source_path),
        "size_bytes": source_path.stat().st_size,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }
    if scan_values:
        result.update(
            {
                "min": float(np.nanmin(array)),
                "max": float(np.nanmax(array)),
                "nan_count": int(np.isnan(array).sum()),
                "inf_count": int(np.isinf(array).sum()),
            }
        )
    return result
