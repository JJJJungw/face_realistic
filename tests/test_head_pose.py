import numpy as np

from face_realistic.performance.head_pose import matrix_to_head_pose


def test_identity_matrix_has_zero_pose() -> None:
    result = matrix_to_head_pose(np.eye(4))
    assert result["euler_xyz_deg"] == [0.0, -0.0, 0.0]
    assert result["translation"] == [0.0, 0.0, 0.0]


def test_translation_is_preserved() -> None:
    matrix = np.eye(4)
    matrix[:3, 3] = [1.25, -2.5, 3.75]
    result = matrix_to_head_pose(matrix)
    assert result["translation"] == [1.25, -2.5, 3.75]
