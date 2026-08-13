import numpy as np

from face_realistic.performance.reenactment_compositor import (
    composite_face,
    estimate_face_transform,
    smooth_transform,
)


class Landmark:
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y


def landmarks(offset_x: float = 0.0, offset_y: float = 0.0):
    points = [Landmark(0.5, 0.5) for _ in range(478)]
    coordinates = {
        33: (0.3, 0.4),
        263: (0.7, 0.4),
        168: (0.5, 0.45),
        6: (0.5, 0.5),
        197: (0.5, 0.55),
        195: (0.5, 0.6),
    }
    for index, (x, y) in coordinates.items():
        points[index] = Landmark(x + offset_x, y + offset_y)
    return points


def test_estimate_face_transform_recovers_translation():
    transform = estimate_face_transform(
        landmarks(), landmarks(0.1, 0.05), (100, 100), (100, 100)
    )

    assert np.allclose(transform[:, :2], np.eye(2), atol=1e-4)
    assert np.allclose(transform[:, 2], [10, 5], atol=1e-3)


def test_smooth_transform_uses_ema():
    previous = np.float32([[1, 0, 0], [0, 1, 0]])
    current = np.float32([[1, 0, 10], [0, 1, 20]])

    smoothed = smooth_transform(previous, current, 0.25)

    assert np.allclose(smoothed[:, 2], [2.5, 5.0])


def test_composite_face_respects_alpha():
    original = np.zeros((2, 2, 3), dtype=np.uint8)
    generated = np.full((2, 2, 3), 200, dtype=np.uint8)
    alpha = np.float32([[0.0, 0.5], [1.0, 0.0]])

    output = composite_face(original, generated, alpha)

    assert output[0, 0, 0] == 0
    assert output[0, 1, 0] == 100
    assert output[1, 0, 0] == 200

