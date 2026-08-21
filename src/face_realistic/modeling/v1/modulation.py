"""Canonical-space identity injection blocks."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .blocks import group_count


class CanonicalIdentityAttention(nn.Module):
    def __init__(self, channels: int, heads: int) -> None:
        super().__init__()
        self.target_norm = nn.LayerNorm(channels)
        self.reference_norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.output_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, target: Tensor, reference: Tensor) -> Tensor:
        batch, channels, height, width = target.shape
        query = target.flatten(2).transpose(1, 2)
        key_value = reference.flatten(2).transpose(1, 2)
        attended, _ = self.attention(
            self.target_norm(query),
            self.reference_norm(key_value),
            self.reference_norm(key_value),
            need_weights=False,
        )
        attended = attended.transpose(1, 2).reshape(batch, channels, height, width)
        return target + self.output_scale.tanh() * attended


class PartialIdentityModulation(nn.Module):
    def __init__(
        self,
        channels: int,
        attribute_channels: int,
        local_identity_channels: int,
        style_dim: int,
    ) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(group_count(channels), channels, affine=False)
        self.attribute = nn.Conv2d(attribute_channels, channels * 2, 3, padding=1)
        self.local_identity = nn.Conv2d(local_identity_channels, channels, 1)
        self.style = nn.Linear(style_dim, channels * 2)
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2 + 1, channels, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, 1, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        inputs: Tensor,
        attributes: Tensor,
        local_identity: Tensor,
        style: Tensor,
        identity_region: Tensor,
    ) -> tuple[Tensor, Tensor]:
        normalized = self.norm(inputs)
        attributes = F.interpolate(attributes, size=inputs.shape[-2:], mode="bilinear", align_corners=False)
        local = F.interpolate(local_identity, size=inputs.shape[-2:], mode="bilinear", align_corners=False)
        local = self.local_identity(local)
        region = F.interpolate(identity_region, size=inputs.shape[-2:], mode="bilinear", align_corners=False)
        attribute_gamma, attribute_beta = self.attribute(attributes).chunk(2, dim=1)
        identity_gamma, identity_beta = self.style(style).chunk(2, dim=1)
        attribute_value = (1.0 + attribute_gamma) * normalized + attribute_beta
        identity_value = (1.0 + identity_gamma[:, :, None, None]) * normalized
        identity_value = identity_value + identity_beta[:, :, None, None] + local
        gate = self.gate(torch.cat((normalized, local, region), dim=1)) * region
        return gate * identity_value + (1.0 - gate) * attribute_value, gate


class PartialIdentityResidualBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        attribute_channels: int,
        local_identity_channels: int,
        style_dim: int,
    ) -> None:
        super().__init__()
        self.first = PartialIdentityModulation(
            channels, attribute_channels, local_identity_channels, style_dim
        )
        self.second = PartialIdentityModulation(
            channels, attribute_channels, local_identity_channels, style_dim
        )
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.activation = nn.SiLU(inplace=True)

    def forward(
        self,
        inputs: Tensor,
        attributes: Tensor,
        local_identity: Tensor,
        style: Tensor,
        identity_region: Tensor,
    ) -> tuple[Tensor, Tensor]:
        output, first_gate = self.first(
            inputs, attributes, local_identity, style, identity_region
        )
        output = self.conv1(self.activation(output))
        output, second_gate = self.second(
            output, attributes, local_identity, style, identity_region
        )
        output = self.conv2(self.activation(output))
        return inputs + output, (first_gate + second_gate) * 0.5
