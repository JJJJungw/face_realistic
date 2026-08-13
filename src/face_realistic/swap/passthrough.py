"""Preserve performance-critical pixels from the original target frame."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

# MediaPipe Face Mesh contours. The mask deliberately excludes brows, nose and
# the outer face so that most identity-bearing geometry still comes from swap.
LEFT_EYE = (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246)
RIGHT_EYE = (362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398)
OUTER_LIPS = (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185)


def _expanded_hull(
    landmarks: Sequence[object],
    indices: Sequence[int],
    width: int,
    height: int,
    expansion: float,
) -> np.ndarray:
    points = np.asarray(
        [
            (float(landmarks[index].x) * width, float(landmarks[index].y) * height)
            for index in indices
        ],
        dtype=np.float32,
    )
    center = points.mean(axis=0, keepdims=True)
    points = center + (points - center) * expansion
    points[:, 0] = np.clip(points[:, 0], 0, width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, height - 1)
    return cv2.convexHull(np.rint(points).astype(np.int32))


def make_performance_mask(
    landmarks: Sequence[object],
    frame_size: tuple[int, int],
    *,
    eye_expansion: float = 1.15,
    mouth_expansion: float = 1.35,
    feather_pixels: float = 6.0,
) -> np.ndarray:
    """Return a float alpha mask for original eyes, eyelids, lips and mouth."""
    width, height = frame_size
    if len(landmarks) < 468:
        raise ValueError(f"Expected at least 468 landmarks, got {len(landmarks)}")
    if eye_expansion < 1.0 or mouth_expansion < 1.0:
        raise ValueError("Region expansion must be at least 1.0")
    if feather_pixels < 0:
        raise ValueError("feather_pixels must not be negative")

    mask = np.zeros((height, width), dtype=np.uint8)
    for indices, expansion in (
        (LEFT_EYE, eye_expansion),
        (RIGHT_EYE, eye_expansion),
        (OUTER_LIPS, mouth_expansion),
    ):
        cv2.fillConvexPoly(
            mask,
            _expanded_hull(landmarks, indices, width, height, expansion),
            255,
            lineType=cv2.LINE_AA,
        )
    alpha = mask.astype(np.float32) / 255.0
    if feather_pixels > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), feather_pixels)
    return np.clip(alpha, 0.0, 1.0)


def restore_performance_regions(
    original: np.ndarray,
    swapped: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    """Composite original performance pixels over a swapped BGR frame."""
    if original.shape != swapped.shape:
        raise ValueError("original and swapped frames must have identical shapes")
    if alpha.shape != original.shape[:2]:
        raise ValueError("alpha mask must match the frame height and width")
    alpha_3d = alpha[..., None]
    output = original.astype(np.float32) * alpha_3d + swapped.astype(np.float32) * (1.0 - alpha_3d)
    return np.clip(np.rint(output), 0, 255).astype(np.uint8)
