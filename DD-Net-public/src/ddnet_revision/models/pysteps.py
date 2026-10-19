from __future__ import annotations

from typing import Any

import numpy as np


class PySTEPSOpticalFlow:
    """Lucas-Kanade motion plus deterministic semi-Lagrangian extrapolation."""

    def __init__(self, config: dict[str, Any]) -> None:
        model = config["model"]
        experiment = config["experiment"]
        self.output_frames = int(experiment["output_frames"])
        self.motion_input_frames = int(model["motion_input_frames"])
        self.motion_method = str(model["motion_method"])
        self.nowcast_method = str(model["nowcast_method"])
        self.extrapolation_method = str(model["extrapolation_method"])
        self.outval = model["outval"]
        self.fallback_count = 0

    def predict_sample(self, input_dbz: np.ndarray) -> np.ndarray:
        from pysteps import motion, nowcasts

        sequence = np.asarray(input_dbz, dtype=np.float64)
        if sequence.ndim == 4 and sequence.shape[1] == 1:
            sequence = sequence[:, 0]
        if sequence.ndim != 3:
            raise ValueError(f"Expected T,H,W input, got {sequence.shape}")
        motion_input = sequence[-self.motion_input_frames :]
        try:
            estimate_motion = motion.get_method(self.motion_method)
            velocity = estimate_motion(motion_input)
            extrapolate = nowcasts.get_method(self.nowcast_method)
            forecast = extrapolate(
                sequence[-1],
                velocity,
                self.output_frames,
                extrap_method=self.extrapolation_method,
                extrap_kwargs={"outval": self.outval},
            )
            forecast = np.asarray(forecast, dtype=np.float32)
            if forecast.shape != (self.output_frames, *sequence.shape[-2:]):
                raise RuntimeError(f"Unexpected PySTEPS output shape: {forecast.shape}")
            if not np.isfinite(forecast).all():
                raise RuntimeError("PySTEPS returned non-finite values")
            return forecast
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            self.fallback_count += 1
            return np.repeat(sequence[-1:, :, :], self.output_frames, axis=0).astype(np.float32)

    def predict_batch(self, input_dbz: np.ndarray) -> np.ndarray:
        return np.stack([self.predict_sample(sample) for sample in input_dbz], axis=0)
