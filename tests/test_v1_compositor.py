import cv2
import numpy as np

from face_realistic.compositing import (
    CompositorConfig,
    build_visible_alpha,
    compose_aligned_crop,
    pasteback_to_frame,
)


def test_visible_alpha_restores_target_occlusion():
    alpha = np.ones((8, 8), dtype=np.float32)
    occlusion = np.zeros((8, 8), dtype=np.float32)
    occlusion[:, :3] = 1.0

    visible = build_visible_alpha(alpha, occlusion=occlusion, feather_pixels=1.0)

    assert np.all(visible[:, :3] == 0.0)
    assert np.all(visible[:, 5:] > 0.0)


def test_aligned_compositor_keeps_zero_alpha_pixels_byte_exact():
    original = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    generated = np.full_like(original, 220)
    alpha = np.zeros((8, 8), dtype=np.float32)
    alpha[2:6, 2:6] = 1.0

    output, visible = compose_aligned_crop(
        original,
        generated,
        alpha,
        config=CompositorConfig(feather_pixels=0, color_match_strength=0),
    )

    assert np.array_equal(output[visible == 0], original[visible == 0])
    assert np.all(output[3:5, 3:5] == 220)


def test_pasteback_identity_transform_changes_only_masked_region():
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    generated = np.full((8, 8, 3), 180, dtype=np.uint8)
    alpha = np.zeros((8, 8), dtype=np.float32)
    cv2.circle(alpha, (4, 4), 2, 1.0, -1)
    frame_to_crop = np.float32([[1, 0, 0], [0, 1, 0]])

    output, warped_alpha = pasteback_to_frame(
        frame,
        generated,
        alpha,
        frame_to_crop,
        config=CompositorConfig(feather_pixels=0, color_match_strength=0),
    )

    assert np.array_equal(output[warped_alpha == 0], frame[warped_alpha == 0])
    assert output[4, 4, 0] == 180
