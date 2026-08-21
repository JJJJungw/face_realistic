"""Feed-forward canonical face generator with explicit output heads."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .blocks import ConvNormAct
from .modulation import CanonicalIdentityAttention, PartialIdentityResidualBlock


class CanonicalSwapGenerator(nn.Module):
    def __init__(
        self,
        channels: tuple[int, ...],
        style_dim: int,
        attention_heads: int,
    ) -> None:
        super().__init__()
        deepest = channels[-1]
        self.canonical_attention = CanonicalIdentityAttention(deepest, attention_heads)
        self.bottleneck = PartialIdentityResidualBlock(
            deepest, deepest, channels[3], style_dim
        )
        decoder_channels = tuple(reversed(channels[:-1]))
        local_channels = (channels[2], channels[1], channels[0], channels[0])
        self.projections = nn.ModuleList()
        self.blocks = nn.ModuleList()
        current = deepest
        for output_channels, identity_channels in zip(
            decoder_channels, local_channels, strict=True
        ):
            self.projections.append(ConvNormAct(current, output_channels))
            self.blocks.append(
                PartialIdentityResidualBlock(
                    output_channels,
                    output_channels,
                    identity_channels,
                    style_dim,
                )
            )
            current = output_channels
        self.rgb_head = nn.Sequential(nn.Conv2d(current, 3, 3, padding=1), nn.Tanh())
        self.alpha_head = nn.Sequential(nn.Conv2d(current, 1, 3, padding=1), nn.Sigmoid())
        self.visibility_head = nn.Sequential(nn.Conv2d(current, 1, 3, padding=1), nn.Sigmoid())
        self.confidence_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(deepest, 1), nn.Sigmoid()
        )

    def forward(
        self,
        attributes: list[Tensor],
        local_identity: list[Tensor],
        style: Tensor,
        identity_region: Tensor,
    ) -> dict[str, Tensor]:
        output = self.canonical_attention(attributes[-1], local_identity[-1])
        confidence = self.confidence_head(output)
        output, bottleneck_gate = self.bottleneck(
            output, attributes[-1], local_identity[-1], style, identity_region
        )
        gates = [bottleneck_gate]
        decoder_attributes = list(reversed(attributes[:-1]))
        decoder_identity = [local_identity[2], local_identity[1], local_identity[0], local_identity[0]]
        for projection, block, attribute, identity in zip(
            self.projections,
            self.blocks,
            decoder_attributes,
            decoder_identity,
            strict=True,
        ):
            output = F.interpolate(output, size=attribute.shape[-2:], mode="bilinear", align_corners=False)
            output = projection(output)
            output, gate = block(output, attribute, identity, style, identity_region)
            gates.append(gate)
        full_gate = torch.stack(
            [F.interpolate(gate, size=output.shape[-2:], mode="bilinear", align_corners=False) for gate in gates],
            dim=0,
        ).mean(dim=0)
        return {
            "generated": self.rgb_head(output),
            "raw_alpha": self.alpha_head(output),
            "raw_visibility": self.visibility_head(output),
            "confidence": confidence,
            "identity_gate": full_gate,
        }
