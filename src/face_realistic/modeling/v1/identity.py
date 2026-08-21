"""Multi-reference identity encoding without external checkpoints."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .blocks import ConvNormAct, ResidualDownBlock
from .contracts import reference_weights


class MultiReferenceIdentityEncoder(nn.Module):
    def __init__(self, channels: tuple[int, ...], identity_dim: int) -> None:
        super().__init__()
        # Reference pyramids start at H/2 and end at H/16.
        self.stem = ConvNormAct(3, channels[0], kernel_size=7, stride=2)
        self.down = nn.ModuleList(
            ResidualDownBlock(source, target)
            for source, target in zip(channels[:3], channels[1:4], strict=True)
        )
        self.global_projection = nn.Linear(channels[3], identity_dim)

    def forward(
        self,
        faces: Tensor,
        quality: Tensor | None = None,
        valid: Tensor | None = None,
    ) -> tuple[Tensor, list[Tensor], Tensor]:
        batch, references, channels, height, width = faces.shape
        flattened = faces.reshape(batch * references, channels, height, width)
        pyramid = [self.stem(flattened)]
        for block in self.down:
            pyramid.append(block(pyramid[-1]))
        per_reference = F.normalize(
            self.global_projection(pyramid[-1].mean(dim=(2, 3))), dim=1
        ).reshape(batch, references, -1)
        weights = reference_weights(faces, quality, valid)
        global_identity = F.normalize(
            (per_reference * weights[..., None]).sum(dim=1), dim=1
        )
        aggregated: list[Tensor] = []
        for features in pyramid:
            shaped = features.reshape(batch, references, *features.shape[1:])
            aggregated.append((shaped * weights[:, :, None, None, None]).sum(dim=1))
        return global_identity, aggregated, weights


class MotionEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        hidden = max(128, output_dim)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, blendshapes: Tensor, head_pose: Tensor) -> Tensor:
        return self.network(torch.cat((blendshapes, head_pose), dim=1))
