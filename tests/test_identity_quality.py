import numpy as np

from face_realistic.identity.register import score_image_quality


def test_quality_scores_are_bounded() -> None:
    image = np.full((512, 512, 3), 127, dtype=np.uint8)
    result = score_image_quality(
        image,
        [0.2, 0.15, 0.8, 0.9],
        {"euler_xyz_deg": [0.0, 0.0, 0.0], "translation": [0.0, 0.0, 0.0]},
    )
    for value in (result.total, result.sharpness, result.exposure, result.frontality, result.face_coverage):
        assert 0.0 <= value <= 1.0


def test_frontal_image_scores_above_profile() -> None:
    image = np.full((512, 512, 3), 127, dtype=np.uint8)
    bbox = [0.2, 0.15, 0.8, 0.9]
    frontal = score_image_quality(
        image, bbox, {"euler_xyz_deg": [0.0, 0.0, 0.0], "translation": [0.0, 0.0, 0.0]}
    )
    profile = score_image_quality(
        image, bbox, {"euler_xyz_deg": [0.0, 80.0, 0.0], "translation": [0.0, 0.0, 0.0]}
    )
    assert frontal.total > profile.total
