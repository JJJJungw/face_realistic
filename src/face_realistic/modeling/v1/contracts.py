"""Tensor contracts shared by training and inference."""

from __future__ import annotations

import torch
from torch import Tensor

from .config import V1ModelConfig


def _expect(name: str, tensor: Tensor, shape: tuple[int | None, ...]) -> None:
    if tensor.ndim != len(shape):
        raise ValueError(f"{name} must have {len(shape)} dimensions, got {tensor.shape}")
    for actual, expected in zip(tensor.shape, shape, strict=True):
        if expected is not None and actual != expected:
            raise ValueError(f"{name} has invalid shape {tensor.shape}; expected {shape}")


def validate_v1_inputs(
    *,
    config: V1ModelConfig,
    reference_faces: Tensor,
    geometry_maps: Tensor,
    blendshapes: Tensor,
    head_pose: Tensor,
    lowfreq_illumination: Tensor,
    context_ring: Tensor,
    occlusion_mask: Tensor,
    reference_quality: Tensor | None,
    reference_valid: Tensor | None,
) -> tuple[int, int]:
    batch, references = reference_faces.shape[:2]
    size = config.image_size
    _expect("reference_faces", reference_faces, (batch, references, 3, size, size))
    if not 1 <= references <= config.max_references:
        raise ValueError(f"reference count must be in [1, {config.max_references}]")
    _expect("geometry_maps", geometry_maps, (batch, config.geometry_channels, None, None))
    _expect("blendshapes", blendshapes, (batch, config.blendshape_dim))
    _expect("head_pose", head_pose, (batch, config.pose_dim))
    _expect("lowfreq_illumination", lowfreq_illumination, (batch, 3, None, None))
    _expect("context_ring", context_ring, (batch, 3, size, size))
    _expect("occlusion_mask", occlusion_mask, (batch, 1, size, size))
    if geometry_maps.shape[-2] != geometry_maps.shape[-1]:
        raise ValueError("geometry_maps must be square")
    if lowfreq_illumination.shape[-2] != lowfreq_illumination.shape[-1]:
        raise ValueError("lowfreq_illumination must be square")
    if reference_quality is not None:
        _expect("reference_quality", reference_quality, (batch, references))
    if reference_valid is not None:
        _expect("reference_valid", reference_valid, (batch, references))
    tensors = (
        geometry_maps,
        blendshapes,
        head_pose,
        lowfreq_illumination,
        context_ring,
        occlusion_mask,
    )
    if any(tensor.device != reference_faces.device for tensor in tensors):
        raise ValueError("all FRS-v1 inputs must be on the same device")
    return batch, references


def reference_weights(
    reference_faces: Tensor,
    quality: Tensor | None,
    valid: Tensor | None,
) -> Tensor:
    batch, references = reference_faces.shape[:2]
    device, dtype = reference_faces.device, reference_faces.dtype
    weights = torch.ones((batch, references), device=device, dtype=dtype)
    if quality is not None:
        weights = weights * quality.to(device=device, dtype=dtype).clamp_min(0.0)
    if valid is not None:
        weights = weights * valid.to(device=device, dtype=dtype).clamp(0.0, 1.0)
    denominator = weights.sum(dim=1, keepdim=True)
    if torch.any(denominator <= 0):
        raise ValueError("each sample must contain at least one valid reference")
    return weights / denominator
