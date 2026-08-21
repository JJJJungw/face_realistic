"""Paste a generated aligned face crop into an untouched source frame."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .color import masked_color_match
from .masks import build_visible_alpha


@dataclass(frozen=True, slots=True)
class CompositorConfig:
    feather_pixels: float = 2.0
    color_match_strength: float = 0.65
    maximum_color_gain: float = 1.8

    def __post_init__(self) -> None:
        if self.feather_pixels < 0:
            raise ValueError("feather_pixels must be non-negative")
        if not 0.0 <= self.color_match_strength <= 1.0:
            raise ValueError("color_match_strength must be in [0, 1]")
        if self.maximum_color_gain < 1.0:
            raise ValueError("maximum_color_gain must be at least 1")


def _validate_images(original: np.ndarray, generated: np.ndarray) -> None:
    if original.shape != generated.shape or original.ndim != 3 or original.shape[2] != 3:
        raise ValueError("original and generated must be matching HWC three-channel images")
    if original.dtype != np.uint8 or generated.dtype != np.uint8:
        raise ValueError("original and generated must use uint8 pixels")


def compose_aligned_crop(
    original: np.ndarray,
    generated: np.ndarray,
    alpha: np.ndarray,
    *,
    visibility: np.ndarray | None = None,
    occlusion: np.ndarray | None = None,
    face_support: np.ndarray | None = None,
    config: CompositorConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Harmonize and blend an already aligned generated crop."""
    _validate_images(original, generated)
    settings = config or CompositorConfig()
    visible_alpha = build_visible_alpha(
        alpha,
        visibility=visibility,
        occlusion=occlusion,
        face_support=face_support,
        feather_pixels=settings.feather_pixels,
    )
    corrected = masked_color_match(
        generated,
        original,
        visible_alpha,
        strength=settings.color_match_strength,
        maximum_gain=settings.maximum_color_gain,
    )
    weight = visible_alpha[..., None]
    output = corrected.astype(np.float32) * weight + original.astype(np.float32) * (1.0 - weight)
    output = np.clip(output, 0.0, 255.0).round().astype(np.uint8)
    # This assignment is intentional: pixels outside the final alpha are
    # byte-identical to the input, regardless of color harmonization.
    output[visible_alpha <= 0.0] = original[visible_alpha <= 0.0]
    return output, visible_alpha


def pasteback_to_frame(
    original_frame: np.ndarray,
    generated_crop: np.ndarray,
    alpha: np.ndarray,
    frame_to_crop: np.ndarray,
    *,
    visibility: np.ndarray | None = None,
    occlusion: np.ndarray | None = None,
    face_support: np.ndarray | None = None,
    config: CompositorConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Color-match in crop space, inverse-warp, and blend into the frame."""
    if original_frame.ndim != 3 or original_frame.shape[2] != 3 or original_frame.dtype != np.uint8:
        raise ValueError("original_frame must be a uint8 HWC image")
    if generated_crop.ndim != 3 or generated_crop.shape[2] != 3 or generated_crop.dtype != np.uint8:
        raise ValueError("generated_crop must be a uint8 HWC image")
    transform = np.asarray(frame_to_crop, dtype=np.float32)
    if transform.shape != (2, 3):
        raise ValueError("frame_to_crop must have shape (2, 3)")
    crop_height, crop_width = generated_crop.shape[:2]
    target_crop = cv2.warpAffine(
        original_frame,
        transform,
        (crop_width, crop_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    settings = config or CompositorConfig()
    crop_alpha = build_visible_alpha(
        alpha,
        visibility=visibility,
        occlusion=occlusion,
        face_support=face_support,
        feather_pixels=settings.feather_pixels,
    )
    corrected = masked_color_match(
        generated_crop,
        target_crop,
        crop_alpha,
        strength=settings.color_match_strength,
        maximum_gain=settings.maximum_color_gain,
    )
    crop_to_frame = cv2.invertAffineTransform(transform)
    frame_height, frame_width = original_frame.shape[:2]
    warped_face = cv2.warpAffine(
        corrected,
        crop_to_frame,
        (frame_width, frame_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
    )
    warped_alpha = cv2.warpAffine(
        crop_alpha,
        crop_to_frame,
        (frame_width, frame_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    warped_alpha = np.clip(warped_alpha, 0.0, 1.0)
    weight = warped_alpha[..., None]
    output = warped_face.astype(np.float32) * weight + original_frame.astype(np.float32) * (1.0 - weight)
    output = np.clip(output, 0.0, 255.0).round().astype(np.uint8)
    output[warped_alpha <= 0.0] = original_frame[warped_alpha <= 0.0]
    return output, warped_alpha
