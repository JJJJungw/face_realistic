"""A compact 256px identity-conditioned face generator.

Design goals for the first baseline:

* a source image contributes identity;
* an identity-suppressed target crop contributes coarse pose and lighting;
* an explicit motion vector contributes expression and head pose;
* AAD-style blocks decide spatially between identity and target attributes;
* the model predicts RGB and an alpha mask for paste-back.

This is a clean-room implementation.  It does not load or distill InSwapper or
GHOST checkpoints.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )


class ResidualDownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.body = nn.Sequential(
            ConvNormAct(in_channels, out_channels, stride=2),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(_group_count(out_channels), out_channels),
        )
        self.skip = nn.Conv2d(in_channels, out_channels, 1, stride=2)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.activation(self.body(inputs) + self.skip(inputs))


class IdentityEncoder(nn.Module):
    """Trainable source encoder; no external face-recognition weights."""

    def __init__(self, base_channels: int, max_channels: int, identity_dim: int):
        super().__init__()
        channels = [base_channels, base_channels * 2, base_channels * 4, max_channels]
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], 7, stride=2, padding=3),
            nn.GroupNorm(_group_count(channels[0]), channels[0]),
            nn.SiLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            ResidualDownBlock(channels[0], channels[1]),
            ResidualDownBlock(channels[1], channels[2]),
            ResidualDownBlock(channels[2], channels[3]),
            ResidualDownBlock(channels[3], channels[3]),
        )
        self.projection = nn.Linear(channels[3], identity_dim)

    def forward(self, source: Tensor) -> Tensor:
        features = self.blocks(self.stem(source)).mean(dim=(2, 3))
        return F.normalize(self.projection(features), dim=1)


class AttributeEncoder(nn.Module):
    """Return a feature pyramid from a low-frequency target crop."""

    def __init__(self, base_channels: int, max_channels: int):
        super().__init__()
        channels = [
            base_channels,
            base_channels * 2,
            base_channels * 4,
            max_channels,
            max_channels,
        ]
        self.stem = ConvNormAct(3, channels[0])
        self.down = nn.ModuleList(
            ResidualDownBlock(in_channels, out_channels)
            for in_channels, out_channels in zip(channels[:-1], channels[1:], strict=True)
        )
        self.channels = channels

    def forward(self, target: Tensor) -> list[Tensor]:
        pyramid = [self.stem(target)]
        for block in self.down:
            pyramid.append(block(pyramid[-1]))
        return pyramid


class MotionEncoder(nn.Module):
    def __init__(self, motion_dim: int, output_dim: int):
        super().__init__()
        hidden = max(128, output_dim)
        self.network = nn.Sequential(
            nn.Linear(motion_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, motion: Tensor) -> Tensor:
        return self.network(motion)


class AdaptiveAttributeDenorm(nn.Module):
    """Spatially mix target attributes and global identity/motion style."""

    def __init__(self, channels: int, attribute_channels: int, style_dim: int):
        super().__init__()
        self.norm = nn.InstanceNorm2d(channels, affine=False)
        self.attribute = nn.Conv2d(attribute_channels, channels * 2, 3, padding=1)
        self.style = nn.Linear(style_dim, channels * 2)
        self.attention = nn.Sequential(
            nn.Conv2d(channels + attribute_channels, channels, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, inputs: Tensor, attributes: Tensor, style: Tensor) -> Tensor:
        normalized = self.norm(inputs)
        attribute_gamma, attribute_beta = self.attribute(attributes).chunk(2, dim=1)
        style_gamma, style_beta = self.style(style).chunk(2, dim=1)
        style_gamma = style_gamma[:, :, None, None]
        style_beta = style_beta[:, :, None, None]
        attribute_value = (1.0 + attribute_gamma) * normalized + attribute_beta
        identity_value = (1.0 + style_gamma) * normalized + style_beta
        gate = self.attention(torch.cat((normalized, attributes), dim=1))
        return gate * identity_value + (1.0 - gate) * attribute_value


class AADResidualBlock(nn.Module):
    def __init__(self, channels: int, attribute_channels: int, style_dim: int):
        super().__init__()
        self.aad1 = AdaptiveAttributeDenorm(channels, attribute_channels, style_dim)
        self.aad2 = AdaptiveAttributeDenorm(channels, attribute_channels, style_dim)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, inputs: Tensor, attributes: Tensor, style: Tensor) -> Tensor:
        output = self.conv1(self.activation(self.aad1(inputs, attributes, style)))
        output = self.conv2(self.activation(self.aad2(output, attributes, style)))
        return inputs + output


class AADGenerator(nn.Module):
    def __init__(self, attribute_channels: list[int], style_dim: int):
        super().__init__()
        deepest = attribute_channels[-1]
        self.input_projection = nn.Conv2d(deepest, deepest, 3, padding=1)
        self.deep_block = AADResidualBlock(deepest, deepest, style_dim)
        self.upsample_projections = nn.ModuleList()
        self.blocks = nn.ModuleList()
        current = deepest
        for attribute_channel in reversed(attribute_channels[:-1]):
            self.upsample_projections.append(ConvNormAct(current, attribute_channel))
            self.blocks.append(
                AADResidualBlock(attribute_channel, attribute_channel, style_dim)
            )
            current = attribute_channel
        self.rgb_head = nn.Sequential(nn.Conv2d(current, 3, 3, padding=1), nn.Tanh())
        self.alpha_head = nn.Sequential(nn.Conv2d(current, 1, 3, padding=1), nn.Sigmoid())

    def forward(self, pyramid: list[Tensor], style: Tensor) -> tuple[Tensor, Tensor]:
        output = self.input_projection(pyramid[-1])
        output = self.deep_block(output, pyramid[-1], style)
        for projection, block, attributes in zip(
            self.upsample_projections,
            self.blocks,
            reversed(pyramid[:-1]),
            strict=True,
        ):
            output = F.interpolate(output, size=attributes.shape[-2:], mode="bilinear", align_corners=False)
            output = projection(output)
            output = block(output, attributes, style)
        return self.rgb_head(output), self.alpha_head(output)


@dataclass(frozen=True, slots=True)
class ModelConfig:
    motion_dim: int
    image_size: int = 256
    base_channels: int = 32
    max_channels: int = 256
    identity_dim: int = 256
    motion_embedding_dim: int = 128
    target_bottleneck: int = 16

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


class CleanRoomFaceSwapModel(nn.Module):
    """One-shot face generator with explicit identity suppression on target RGB."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.image_size < 32 or config.image_size % 16:
            raise ValueError("image_size must be >= 32 and divisible by 16")
        if not 2 <= config.target_bottleneck <= config.image_size:
            raise ValueError("target_bottleneck must be between 2 and image_size")
        self.config = config
        self.identity_encoder = IdentityEncoder(
            config.base_channels, config.max_channels, config.identity_dim
        )
        self.attribute_encoder = AttributeEncoder(config.base_channels, config.max_channels)
        self.motion_encoder = MotionEncoder(config.motion_dim, config.motion_embedding_dim)
        style_dim = config.identity_dim + config.motion_embedding_dim
        self.generator = AADGenerator(self.attribute_encoder.channels, style_dim)

    def suppress_target_identity(self, target: Tensor) -> Tensor:
        coarse = F.adaptive_avg_pool2d(
            target, (self.config.target_bottleneck, self.config.target_bottleneck)
        )
        return F.interpolate(coarse, size=target.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, source: Tensor, target: Tensor, motion: Tensor) -> dict[str, Tensor]:
        if source.shape != target.shape or source.ndim != 4 or source.shape[1] != 3:
            raise ValueError("source and target must have matching BCHW RGB shapes")
        identity = self.identity_encoder(source)
        suppressed_target = self.suppress_target_identity(target)
        attributes = self.attribute_encoder(suppressed_target)
        motion_embedding = self.motion_encoder(motion)
        style = torch.cat((identity, motion_embedding), dim=1)
        generated, alpha = self.generator(attributes, style)
        composite = generated * alpha + target * (1.0 - alpha)
        return {
            "generated": generated,
            "alpha": alpha,
            "composite": composite,
            "identity": identity,
            "suppressed_target": suppressed_target,
        }

