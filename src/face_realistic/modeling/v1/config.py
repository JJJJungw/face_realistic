"""Configuration for the FRS-v1 architecture draft."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class V1ModelConfig:
    image_size: int = 256
    geometry_channels: int = 16
    blendshape_dim: int = 52
    pose_dim: int = 6
    base_channels: int = 32
    max_channels: int = 256
    identity_dim: int = 256
    motion_embedding_dim: int = 128
    attention_heads: int = 4
    max_references: int = 5

    def __post_init__(self) -> None:
        if self.image_size < 32 or self.image_size % 16:
            raise ValueError("image_size must be >= 32 and divisible by 16")
        if self.geometry_channels < 2:
            raise ValueError("geometry_channels must include feature maps and face support")
        if self.blendshape_dim < 0 or self.pose_dim <= 0:
            raise ValueError("blendshape_dim must be non-negative and pose_dim must be positive")
        if self.base_channels < 8 or self.max_channels < self.base_channels * 2:
            raise ValueError("channel configuration is too small")
        if self.max_channels % self.attention_heads:
            raise ValueError("max_channels must be divisible by attention_heads")
        if self.max_references < 1:
            raise ValueError("max_references must be positive")

    @property
    def channels(self) -> tuple[int, int, int, int, int]:
        return (
            self.base_channels,
            min(self.base_channels * 2, self.max_channels),
            min(self.base_channels * 4, self.max_channels),
            self.max_channels,
            self.max_channels,
        )

    @property
    def style_dim(self) -> int:
        return self.identity_dim + self.motion_embedding_dim

    def to_dict(self) -> dict[str, int]:
        return asdict(self)
