"""Rasterize MediaPipe landmarks into identity-neutral spatial conditions."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


LANDMARK_GROUPS: tuple[tuple[int, ...], ...] = (
    (33, 133, 159, 145, 160, 144, 158, 153),
    (362, 263, 386, 374, 385, 380, 387, 373),
    (70, 63, 105, 66, 107),
    (336, 296, 334, 293, 300),
    (1, 2, 4, 5, 6, 168, 197, 195),
    (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 0),
    (78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 13),
    (10, 338, 297, 284, 389, 454, 361, 397, 379, 400, 152, 176, 150, 172, 132, 234, 162, 54, 67, 109),
    (50, 101, 205, 206, 187),
    (280, 330, 425, 426, 411),
    (468, 469, 470, 471, 472),
    (473, 474, 475, 476, 477),
)


class MediaPipeGeometryRasterizer(nn.Module):
    """Create 16 geometry maps from normalized `[x, y, z, visibility]` landmarks.

    Channels 0..11 are semantic group heatmaps, 12 is all-point density, 13 is
    normalized depth, 14 is visibility, and 15 is an analytic face-support map.
    """

    output_channels = 16

    def __init__(
        self,
        output_size: int = 64,
        sigma: float = 1.6,
        groups: Sequence[Sequence[int]] = LANDMARK_GROUPS,
    ) -> None:
        super().__init__()
        if output_size < 8:
            raise ValueError("output_size must be at least 8")
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        if len(groups) != 12:
            raise ValueError("exactly 12 semantic landmark groups are required")
        self.output_size = output_size
        self.sigma = sigma
        self.groups = tuple(tuple(group) for group in groups)

    def _heatmap(self, points: Tensor, weights: Tensor | None = None) -> Tensor:
        batch = points.shape[0]
        size = self.output_size
        coordinates = torch.linspace(0.0, 1.0, size, device=points.device, dtype=points.dtype)
        grid_y, grid_x = torch.meshgrid(coordinates, coordinates, indexing="ij")
        grid = torch.stack((grid_x, grid_y), dim=-1)[None, None]
        distance = ((grid - points[:, :, None, None, :]) ** 2).sum(dim=-1)
        sigma = self.sigma / max(1, size - 1)
        heat = torch.exp(-distance / (2.0 * sigma * sigma))
        if weights is not None:
            heat = heat * weights[:, :, None, None]
        return heat.amax(dim=1) if heat.shape[1] else points.new_zeros((batch, size, size))

    def forward(self, landmarks: Tensor) -> Tensor:
        if landmarks.ndim != 3 or landmarks.shape[-1] != 4:
            raise ValueError("landmarks must have shape [B, K, 4]")
        if landmarks.shape[1] < 1:
            raise ValueError("at least one landmark is required")
        xy = landmarks[..., :2].clamp(0.0, 1.0)
        depth = landmarks[..., 2]
        visibility = landmarks[..., 3].clamp(0.0, 1.0)
        maps: list[Tensor] = []
        for group in self.groups:
            indices = [index for index in group if index < landmarks.shape[1]]
            if indices:
                maps.append(self._heatmap(xy[:, indices], visibility[:, indices]))
            else:
                maps.append(xy.new_zeros((xy.shape[0], self.output_size, self.output_size)))

        all_heat = self._heatmap(xy, visibility)
        size = self.output_size
        coordinates = torch.linspace(0.0, 1.0, size, device=xy.device, dtype=xy.dtype)
        grid_y, grid_x = torch.meshgrid(coordinates, coordinates, indexing="ij")
        grid = torch.stack((grid_x, grid_y), dim=-1)[None, None]
        sigma = self.sigma / max(1, size - 1)
        kernels = torch.exp(-((grid - xy[:, :, None, None]) ** 2).sum(-1) / (2 * sigma * sigma))
        weighted = kernels * visibility[:, :, None, None]
        normalizer = weighted.sum(dim=1).clamp_min(1e-6)
        depth_centered = depth - depth.mean(dim=1, keepdim=True)
        depth_scale = depth_centered.abs().amax(dim=1, keepdim=True).clamp_min(1e-6)
        normalized_depth = depth_centered / depth_scale
        depth_map = (weighted * normalized_depth[:, :, None, None]).sum(dim=1) / normalizer
        visibility_map = weighted.sum(dim=1).clamp(0.0, 1.0)

        center = xy.mean(dim=1)
        # A degenerate or partially observed landmark set must still produce a
        # conservative inner-face mask instead of exposing target RGB.
        spread = (xy - center[:, None]).abs().quantile(0.9, dim=1).clamp_min(0.18)
        normalized_x = (grid_x[None] - center[:, 0, None, None]) / (spread[:, 0, None, None] * 1.15)
        normalized_y = (grid_y[None] - center[:, 1, None, None]) / (spread[:, 1, None, None] * 1.10)
        radius = normalized_x.square() + normalized_y.square()
        face_support = torch.sigmoid((1.0 - radius) * 10.0)

        maps.extend((all_heat, depth_map, visibility_map, face_support))
        return torch.stack(maps, dim=1)
