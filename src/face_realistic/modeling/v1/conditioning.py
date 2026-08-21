"""Build identity-suppressed target conditions from an aligned crop."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from .geometry import MediaPipeGeometryRasterizer


def build_target_conditions(
    target_crop: Tensor,
    landmarks: Tensor,
    *,
    rasterizer: MediaPipeGeometryRasterizer,
    occlusion_mask: Tensor | None = None,
    illumination_size: int = 16,
) -> dict[str, Tensor]:
    """Create geometry, low-frequency illumination, context, and occlusion tensors.

    `target_crop` uses normalized BCHW RGB values.  The returned context has
    the analytic inner-face support removed so no high-frequency target face
    pixels can reach the generator through this helper.
    """
    if target_crop.ndim != 4 or target_crop.shape[1] != 3:
        raise ValueError("target_crop must have shape [B, 3, H, W]")
    if target_crop.shape[-2] != target_crop.shape[-1]:
        raise ValueError("target_crop must be square")
    if landmarks.shape[0] != target_crop.shape[0]:
        raise ValueError("target_crop and landmarks batch sizes must match")
    if not 2 <= illumination_size <= target_crop.shape[-1]:
        raise ValueError("illumination_size must be between 2 and the crop size")
    geometry_maps = rasterizer(landmarks)
    face_support = F.interpolate(
        geometry_maps[:, -1:],
        size=target_crop.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).clamp(0.0, 1.0)
    lowfreq = F.adaptive_avg_pool2d(target_crop, (illumination_size, illumination_size))
    if occlusion_mask is None:
        occlusion = torch.zeros_like(face_support)
    else:
        if occlusion_mask.shape != face_support.shape:
            raise ValueError(f"occlusion_mask must have shape {face_support.shape}")
        occlusion = occlusion_mask.to(
            device=target_crop.device, dtype=target_crop.dtype
        ).clamp(0.0, 1.0)
    context_ring = target_crop * (1.0 - face_support)
    return {
        "geometry_maps": geometry_maps,
        "lowfreq_illumination": lowfreq,
        "context_ring": context_ring,
        "occlusion_mask": occlusion,
        "face_support": face_support,
    }
