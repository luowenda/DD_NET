from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class HaarDWT2D(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        scale = 2.0**-0.5
        low = torch.tensor([scale, scale], dtype=torch.float32)
        high = torch.tensor([-scale, scale], dtype=torch.float32)
        filters = torch.stack(
            [
                torch.outer(low, low),
                torch.outer(high, low),
                torch.outer(low, high),
                torch.outer(high, high),
            ]
        ).unsqueeze(1)
        self.register_buffer("filters", filters)

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, ...]:
        channels = value.shape[1]
        filters = self.filters.to(dtype=value.dtype).repeat(channels, 1, 1, 1)
        output = F.conv2d(value, filters, stride=2, groups=channels)
        return torch.chunk(output, 4, dim=1)


class HighFocalFrequencyLoss(nn.Module):
    """Device-safe implementation of the audited historical HFFL semantics."""

    def __init__(
        self,
        loss_weight: float = 0.001,
        level: int = 1,
        tau: float = 0.1,
        alpha: float = 1.0,
        log_matrix: bool = True,
    ) -> None:
        super().__init__()
        self.loss_weight = loss_weight
        self.level = level
        self.tau = tau
        self.alpha = alpha
        self.log_matrix = log_matrix
        self.dwt = HaarDWT2D()

    def _high_frequency_mask(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        y, x = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing="ij",
        )
        radius = self.tau * max(height, width)
        mask = torch.ones((height, width), dtype=torch.float32, device=device)
        for center_y, center_x in (
            (0, 0),
            (0, width - 1),
            (height - 1, 0),
            (height - 1, width - 1),
        ):
            distance = torch.sqrt((x - center_x).float().square() + (y - center_y).float().square())
            mask[distance <= radius] = 0.0
        return mask

    def _frequency_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_frequency = torch.view_as_real(torch.fft.fft2(prediction, norm="ortho"))
        true_frequency = torch.view_as_real(torch.fft.fft2(target, norm="ortho"))
        difference = pred_frequency - true_frequency
        magnitude = torch.sqrt(difference[..., 0].square() + difference[..., 1].square())
        weights = magnitude.pow(self.alpha)
        if self.log_matrix:
            weights = torch.log1p(weights)
        normalizer = weights.amax(dim=(-2, -1), keepdim=True)
        weights = torch.where(normalizer > 0, weights / normalizer, torch.zeros_like(weights))
        weights = weights.detach().clamp_(0.0, 1.0)
        frequency_distance = difference[..., 0].square() + difference[..., 1].square()
        mask = self._high_frequency_mask(
            frequency_distance.shape[-2], frequency_distance.shape[-1], prediction.device
        )
        return torch.mean(weights * frequency_distance * mask)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if prediction.ndim == 5:
            prediction = prediction.flatten(0, 1)
            target = target.flatten(0, 1)
        loss = prediction.new_zeros(())
        for _ in range(self.level):
            prediction = self.dwt(prediction)[0]
            target = self.dwt(target)[0]
            loss = loss + self._frequency_loss(prediction, target)
        return loss * self.loss_weight


def mse_mae_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(prediction, target) + F.l1_loss(prediction, target)
