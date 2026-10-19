from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from skimage.metrics import structural_similarity


def _as_bt_hw(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array, dtype=np.float64)
    if value.ndim == 5 and value.shape[2] == 1:
        value = value[:, :, 0]
    if value.ndim != 4:
        raise ValueError(f"Expected B,T,H,W or B,T,1,H,W, got {value.shape}")
    return value


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


@dataclass
class ContinuousSums:
    mse: np.ndarray
    mae: np.ndarray
    psnr: np.ndarray
    ssim: np.ndarray
    frames: np.ndarray


class RevisionEvaluator:
    """Single evaluator for all models in the canonical 0-70 dBZ domain."""

    def __init__(
        self,
        output_frames: int = 20,
        thresholds_dbz: tuple[float, ...] = (10.0, 20.0, 30.0, 40.0),
        data_range: float = 70.0,
        frame_interval_minutes: int = 6,
    ) -> None:
        self.output_frames = output_frames
        self.thresholds = np.asarray(thresholds_dbz, dtype=np.float64)
        self.data_range = float(data_range)
        self.frame_interval_minutes = int(frame_interval_minutes)
        self.counts = np.zeros((len(self.thresholds), 3), dtype=np.int64)
        self.counts_by_lead = np.zeros((len(self.thresholds), output_frames, 3), dtype=np.int64)
        zeros = np.zeros(output_frames, dtype=np.float64)
        self.continuous = ContinuousSums(
            mse=zeros.copy(),
            mae=zeros.copy(),
            psnr=zeros.copy(),
            ssim=zeros.copy(),
            frames=np.zeros(output_frames, dtype=np.int64),
        )
        self.samples = 0

    def update(self, prediction_dbz: np.ndarray, target_dbz: np.ndarray) -> None:
        prediction = _as_bt_hw(prediction_dbz)
        target = _as_bt_hw(target_dbz)
        if prediction.shape != target.shape:
            raise ValueError(f"Prediction {prediction.shape} != target {target.shape}")
        if prediction.shape[1] != self.output_frames:
            raise ValueError(f"Expected {self.output_frames} lead times, got {prediction.shape[1]}")
        if not np.isfinite(prediction).all() or not np.isfinite(target).all():
            raise ValueError("Predictions and targets must contain only finite dBZ values")

        self.samples += prediction.shape[0]
        for threshold_index, threshold in enumerate(self.thresholds):
            pred_event = prediction >= threshold
            true_event = target >= threshold
            tp_by_lead = np.count_nonzero(pred_event & true_event, axis=(0, 2, 3))
            fp_by_lead = np.count_nonzero(pred_event & ~true_event, axis=(0, 2, 3))
            fn_by_lead = np.count_nonzero(~pred_event & true_event, axis=(0, 2, 3))
            lead_counts = np.stack([tp_by_lead, fp_by_lead, fn_by_lead], axis=-1)
            self.counts_by_lead[threshold_index] += lead_counts
            self.counts[threshold_index] += lead_counts.sum(axis=0)

        for batch_index in range(prediction.shape[0]):
            for lead in range(self.output_frames):
                pred_frame = prediction[batch_index, lead]
                true_frame = target[batch_index, lead]
                error = pred_frame - true_frame
                mse = float(np.mean(error * error))
                self.continuous.mse[lead] += mse
                self.continuous.mae[lead] += float(np.mean(np.abs(error)))
                self.continuous.psnr[lead] += (
                    float("inf") if mse == 0.0 else 20.0 * np.log10(self.data_range / np.sqrt(mse))
                )
                self.continuous.ssim[lead] += float(
                    structural_similarity(pred_frame, true_frame, data_range=self.data_range)
                )
                self.continuous.frames[lead] += 1

    @staticmethod
    def _categorical(counts: np.ndarray) -> dict[str, Any]:
        tp, fp, fn = (int(value) for value in counts)
        return {
            "CSI": _ratio(tp, tp + fp + fn),
            "FAR": _ratio(fp, tp + fp),
            "POD": _ratio(tp, tp + fn),
            "TP": tp,
            "FP": fp,
            "FN": fn,
        }

    def result(self) -> dict[str, Any]:
        if self.samples == 0:
            raise RuntimeError("No samples were evaluated")
        frame_counts = self.continuous.frames.astype(np.float64)
        if np.any(frame_counts == 0):
            raise RuntimeError("At least one lead time has no evaluated frames")
        per_lead_mse = self.continuous.mse / frame_counts
        per_lead_mae = self.continuous.mae / frame_counts
        per_lead_psnr = self.continuous.psnr / frame_counts
        per_lead_ssim = self.continuous.ssim / frame_counts

        thresholds = {
            f"{threshold:g}": self._categorical(self.counts[index])
            for index, threshold in enumerate(self.thresholds)
        }
        leads = []
        for lead in range(self.output_frames):
            leads.append(
                {
                    "frame": lead + 1,
                    "minutes": (lead + 1) * self.frame_interval_minutes,
                    "MSE": float(per_lead_mse[lead]),
                    "MAE": float(per_lead_mae[lead]),
                    "PSNR": float(per_lead_psnr[lead]),
                    "SSIM": float(per_lead_ssim[lead]),
                    "thresholds": {
                        f"{threshold:g}": self._categorical(self.counts_by_lead[index, lead])
                        for index, threshold in enumerate(self.thresholds)
                    },
                }
            )
        return {
            "schema_version": 1,
            "evaluation_domain": "dBZ",
            "comparison": ">=",
            "categorical_aggregation": "pooled_over_sample_time_space",
            "continuous_aggregation": "equal_weight_per_sample_frame",
            "data_range": self.data_range,
            "samples": self.samples,
            "overall": {
                "MSE": float(np.mean(per_lead_mse)),
                "MAE": float(np.mean(per_lead_mae)),
                "PSNR": float(np.mean(per_lead_psnr)),
                "SSIM": float(np.mean(per_lead_ssim)),
                "thresholds": thresholds,
            },
            "by_lead_time": leads,
        }


def evaluate_arrays(
    prediction_dbz: np.ndarray,
    target_dbz: np.ndarray,
    thresholds_dbz: tuple[float, ...] = (10.0, 20.0, 30.0, 40.0),
    data_range: float = 70.0,
    frame_interval_minutes: int = 6,
) -> dict[str, Any]:
    prediction = _as_bt_hw(prediction_dbz)
    evaluator = RevisionEvaluator(
        output_frames=prediction.shape[1],
        thresholds_dbz=thresholds_dbz,
        data_range=data_range,
        frame_interval_minutes=frame_interval_minutes,
    )
    evaluator.update(prediction, target_dbz)
    return evaluator.result()
