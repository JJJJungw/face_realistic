"""Losses without external pretrained checkpoints for the first baseline."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def gradient_loss(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    prediction_dx = prediction[:, :, :, 1:] - prediction[:, :, :, :-1]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    prediction_dy = prediction[:, :, 1:, :] - prediction[:, :, :-1, :]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(prediction_dx * mask[:, :, :, 1:], target_dx * mask[:, :, :, 1:]) + F.l1_loss(
        prediction_dy * mask[:, :, 1:, :], target_dy * mask[:, :, 1:, :]
    )


@dataclass(frozen=True, slots=True)
class LossWeights:
    composite: float = 1.0
    face: float = 2.0
    alpha: float = 0.5
    gradient: float = 0.25


class ReconstructionObjective(nn.Module):
    def __init__(self, weights: LossWeights | None = None):
        super().__init__()
        self.weights = weights or LossWeights()

    def forward(
        self, outputs: dict[str, Tensor], target: Tensor, face_mask: Tensor
    ) -> dict[str, Tensor]:
        safe_mask = face_mask.clamp(0.0, 1.0)
        safe_alpha = outputs["alpha"].clamp(1e-6, 1.0 - 1e-6)
        composite = F.l1_loss(outputs["composite"], target)
        face = F.l1_loss(outputs["generated"] * safe_mask, target * safe_mask)
        alpha = F.binary_cross_entropy(safe_alpha, safe_mask)
        edges = gradient_loss(outputs["generated"], target, safe_mask)
        total = (
            self.weights.composite * composite
            + self.weights.face * face
            + self.weights.alpha * alpha
            + self.weights.gradient * edges
        )
        return {
            "total": total,
            "composite": composite.detach(),
            "face": face.detach(),
            "alpha": alpha.detach(),
            "gradient": edges.detach(),
        }
