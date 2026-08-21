"""Mask operations for face paste-back."""

from __future__ import annotations

import cv2
import numpy as np


def _mask(name: str, value: np.ndarray | None, shape: tuple[int, int], default: float) -> np.ndarray:
    if value is None:
        return np.full(shape, default, dtype=np.float32)
    array = np.asarray(value, dtype=np.float32)
    if array.shape == (*shape, 1):
        array = array[..., 0]
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    return np.clip(array, 0.0, 1.0)


def build_visible_alpha(
    alpha: np.ndarray,
    *,
    visibility: np.ndarray | None = None,
    occlusion: np.ndarray | None = None,
    face_support: np.ndarray | None = None,
    feather_pixels: float = 0.0,
) -> np.ndarray:
    """Combine generator masks while guaranteeing that occluders remain original."""
    base = np.asarray(alpha, dtype=np.float32)
    if base.ndim == 3 and base.shape[-1] == 1:
        base = base[..., 0]
    if base.ndim != 2:
        raise ValueError("alpha must be a 2D mask")
    shape = base.shape
    visible = _mask("visibility", visibility, shape, 1.0)
    blocker = _mask("occlusion", occlusion, shape, 0.0)
    support = _mask("face_support", face_support, shape, 1.0)
    combined = np.clip(base, 0.0, 1.0) * visible * support * (1.0 - blocker)
    if feather_pixels > 0:
        combined = cv2.GaussianBlur(combined, (0, 0), feather_pixels)
        # Gaussian feathering must not bleed generated pixels over a hand,
        # glasses, or any other target occluder.
        combined *= 1.0 - blocker
    return np.clip(combined, 0.0, 1.0)
