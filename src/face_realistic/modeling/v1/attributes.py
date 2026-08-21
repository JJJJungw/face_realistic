"""Identity-suppressed target attribute encoders."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .blocks import PyramidEncoder


class TargetAttributeEncoder(nn.Module):
    def __init__(self, geometry_channels: int, channels: tuple[int, ...]) -> None:
        super().__init__()
        self.geometry_encoder = PyramidEncoder(geometry_channels, channels)
        self.appearance_encoder = PyramidEncoder(7, channels)
        self.fusion = nn.ModuleList(
            nn.Conv2d(channel * 2, channel, 1) for channel in channels
        )

    def forward(
        self,
        geometry_maps: Tensor,
        lowfreq_illumination: Tensor,
        context_ring: Tensor,
        occlusion_mask: Tensor,
    ) -> tuple[list[Tensor], Tensor, Tensor]:
        size = context_ring.shape[-2:]
        geometry = F.interpolate(geometry_maps, size=size, mode="bilinear", align_corners=False)
        illumination = F.interpolate(
            lowfreq_illumination, size=size, mode="bilinear", align_corners=False
        )
        face_support = geometry[:, -1:].clamp(0.0, 1.0)
        safe_occlusion = occlusion_mask.clamp(0.0, 1.0)
        sanitized_context = context_ring * (1.0 - face_support)
        geometry_pyramid = self.geometry_encoder(geometry)
        appearance_pyramid = self.appearance_encoder(
            torch.cat((illumination, sanitized_context, safe_occlusion), dim=1)
        )
        fused = [
            projection(torch.cat((geometry_feature, appearance_feature), dim=1))
            for projection, geometry_feature, appearance_feature in zip(
                self.fusion, geometry_pyramid, appearance_pyramid, strict=True
            )
        ]
        return fused, sanitized_context, face_support
