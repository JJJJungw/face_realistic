"""Lightweight masked color harmonization."""

from __future__ import annotations

import numpy as np


def masked_color_match(
    generated: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    *,
    strength: float = 0.65,
    maximum_gain: float = 1.8,
) -> np.ndarray:
    """Match per-channel mean/std inside a soft mask.

    This deliberately avoids a learned relighting checkpoint.  It is a stable
    baseline that can later be replaced without changing the compositor API.
    """
    if generated.shape != target.shape or generated.ndim != 3 or generated.shape[2] != 3:
        raise ValueError("generated and target must have matching HWC three-channel shapes")
    if mask.shape != generated.shape[:2]:
        raise ValueError("mask must match the image height and width")
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be in [0, 1]")
    weights = np.asarray(mask, dtype=np.float32).reshape(-1, 1)
    total = float(weights.sum())
    if total < 1.0 or strength == 0.0:
        return generated.copy()
    source = generated.astype(np.float32).reshape(-1, 3)
    destination = target.astype(np.float32).reshape(-1, 3)
    source_mean = (source * weights).sum(axis=0) / total
    target_mean = (destination * weights).sum(axis=0) / total
    source_var = ((source - source_mean) ** 2 * weights).sum(axis=0) / total
    target_var = ((destination - target_mean) ** 2 * weights).sum(axis=0) / total
    gain = np.sqrt(target_var + 1e-4) / np.sqrt(source_var + 1e-4)
    gain = np.clip(gain, 1.0 / maximum_gain, maximum_gain)
    matched = (source - source_mean) * gain + target_mean
    mixed = source * (1.0 - strength) + matched * strength
    return np.clip(mixed.reshape(generated.shape), 0.0, 255.0).astype(np.uint8)
