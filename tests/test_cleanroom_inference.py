from types import SimpleNamespace

import numpy as np

from face_realistic.modeling.infer import (
    LEFT_EYE,
    MOUTH,
    RIGHT_EYE,
    alignment_transform,
    motion_vector,
    paste_generated_crop,
)


def _landmarks() -> list[SimpleNamespace]:
    points = [SimpleNamespace(x=0.5, y=0.5) for _ in range(478)]
    for index in LEFT_EYE:
        points[index] = SimpleNamespace(x=0.31, y=0.38)
    for index in RIGHT_EYE:
        points[index] = SimpleNamespace(x=0.69, y=0.38)
    for index in MOUTH:
        points[index] = SimpleNamespace(x=0.50, y=0.68)
    return points


def test_alignment_transform_maps_landmarks_to_canonical_points():
    transform = alignment_transform(_landmarks(), (100, 100), 100)
    source = np.float32([[31, 38, 1], [69, 38, 1], [50, 68, 1]])
    mapped = source @ transform.T

    np.testing.assert_allclose(mapped, [[31, 38], [69, 38], [50, 68]], atol=2.0)


def test_motion_vector_uses_checkpoint_order_and_normalizes_pose():
    blendshapes = [SimpleNamespace(category_name="jawOpen", score=0.75)]
    matrix = np.eye(4, dtype=np.float32)

    values = motion_vector(("jawOpen", "missing", "head_pitch"), blendshapes, matrix)

    np.testing.assert_allclose(values, [0.75, 0.0, 0.0])


def test_paste_generated_crop_respects_alpha():
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    generated = np.full((16, 16, 3), 200, dtype=np.uint8)
    alpha = np.ones((16, 16), dtype=np.float32)
    transform = np.float32([[1, 0, 0], [0, 1, 0]])

    composite, ratio = paste_generated_crop(frame, generated, alpha, transform)

    assert np.array_equal(composite, generated)
    assert ratio == 1.0
