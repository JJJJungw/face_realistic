"""얼굴 변환행렬에서 사람이 읽기 쉬운 자세 값을 계산한다."""

from __future__ import annotations

import math

import numpy as np


def matrix_to_head_pose(matrix: np.ndarray) -> dict[str, list[float]]:
    """4x4 얼굴 변환행렬을 XYZ Euler 각도와 이동 벡터로 변환한다."""
    transform = np.asarray(matrix, dtype=float)
    if transform.shape != (4, 4):
        raise ValueError(f"4x4 변환행렬이 필요합니다. 현재 shape={transform.shape}")

    rotation = transform[:3, :3]
    scale = np.linalg.norm(rotation, axis=0)
    scale[scale == 0.0] = 1.0
    rotation = rotation / scale

    singularity = math.hypot(rotation[0, 0], rotation[1, 0])
    if singularity > 1e-6:
        roll_x = math.atan2(rotation[2, 1], rotation[2, 2])
        pitch_y = math.atan2(-rotation[2, 0], singularity)
        yaw_z = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        roll_x = math.atan2(-rotation[1, 2], rotation[1, 1])
        pitch_y = math.atan2(-rotation[2, 0], singularity)
        yaw_z = 0.0

    return {
        "euler_xyz_deg": [
            round(math.degrees(roll_x), 4),
            round(math.degrees(pitch_y), 4),
            round(math.degrees(yaw_z), 4),
        ],
        "translation": [round(float(value), 6) for value in transform[:3, 3]],
    }
