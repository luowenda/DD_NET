from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class StepOutput:
    loss: torch.Tensor
    prediction_dbz: torch.Tensor


class TrainableNowcastAdapter(nn.Module, ABC):
    def __init__(self, normalization_scale: float, output_frames: int) -> None:
        super().__init__()
        self.normalization_scale = normalization_scale
        self.output_frames = output_frames

    def normalize(self, value_dbz: torch.Tensor) -> torch.Tensor:
        return value_dbz / self.normalization_scale

    def denormalize(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.normalization_scale

    @abstractmethod
    def training_step(
        self,
        input_dbz: torch.Tensor,
        target_dbz: torch.Tensor,
        teacher_forcing_ratio: float,
        global_step: int = 0,
    ) -> StepOutput:
        raise NotImplementedError

    @abstractmethod
    def predict(self, input_dbz: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
