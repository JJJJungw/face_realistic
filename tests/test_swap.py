import numpy as np
import pytest

from face_realistic.swap.baseline import _largest_face
from face_realistic.swap.passthrough import make_performance_mask, restore_performance_regions


class FakeFace:
    def __init__(self, bbox: list[float]) -> None:
        self.bbox = bbox


def test_largest_face() -> None:
    small = FakeFace([0, 0, 10, 10])
    large = FakeFace([0, 0, 20, 30])
    assert _largest_face([small, large]) is large


def test_largest_face_rejects_empty_list() -> None:
    try:
        _largest_face([])
    except RuntimeError:
        pass
    else:
        raise AssertionError("empty face list must fail")


class Landmark:
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y


def test_performance_mask_contains_eyes_and_mouth() -> None:
    landmarks = [Landmark(0.5, 0.5) for _ in range(478)]
    for index in (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246):
        landmarks[index] = Landmark(0.35 + (index % 3) * 0.02, 0.4 + (index % 2) * 0.01)
    for index in (362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398):
        landmarks[index] = Landmark(0.6 + (index % 3) * 0.02, 0.4 + (index % 2) * 0.01)
    for index in (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185):
        landmarks[index] = Landmark(0.45 + (index % 4) * 0.025, 0.62 + (index % 2) * 0.025)

    mask = make_performance_mask(landmarks, (200, 100), feather_pixels=0)

    assert mask.shape == (100, 200)
    assert mask.max() == 1.0
    assert mask.mean() > 0


def test_restore_performance_regions_uses_original_where_masked() -> None:
    original = np.full((4, 4, 3), 200, dtype=np.uint8)
    swapped = np.full((4, 4, 3), 20, dtype=np.uint8)
    alpha = np.zeros((4, 4), dtype=np.float32)
    alpha[1:3, 1:3] = 1.0

    output = restore_performance_regions(original, swapped, alpha)

    assert np.all(output[1:3, 1:3] == 200)
    assert np.all(output[0, 0] == 20)


def test_performance_mask_rejects_too_few_landmarks() -> None:
    with pytest.raises(ValueError, match="at least 468"):
        make_performance_mask([], (100, 100))
