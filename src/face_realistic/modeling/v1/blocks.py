"""Small reusable convolution blocks."""

from __future__ import annotations

from torch import Tensor, nn


def group_count(channels: int) -> int:
    for groups in (16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
    ) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
            nn.GroupNorm(group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )


class ResidualDownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            ConvNormAct(in_channels, out_channels, stride=2),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(group_count(out_channels), out_channels),
        )
        self.skip = nn.Conv2d(in_channels, out_channels, 1, stride=2)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.activation(self.body(inputs) + self.skip(inputs))


class PyramidEncoder(nn.Module):
    """Return H, H/2, H/4, H/8, and H/16 feature maps."""

    def __init__(self, input_channels: int, channels: tuple[int, ...]) -> None:
        super().__init__()
        self.stem = ConvNormAct(input_channels, channels[0])
        self.down = nn.ModuleList(
            ResidualDownBlock(source, target)
            for source, target in zip(channels[:-1], channels[1:], strict=True)
        )

    def forward(self, inputs: Tensor) -> list[Tensor]:
        pyramid = [self.stem(inputs)]
        for block in self.down:
            pyramid.append(block(pyramid[-1]))
        return pyramid
