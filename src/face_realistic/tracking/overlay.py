"""추적 결과 확인용 OpenCV 오버레이."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np


FEATURE_LOOPS: tuple[tuple[int, ...], ...] = (
    (10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109),
    (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246),
    (362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398),
    (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185),
    (70, 63, 105, 66, 107),
    (336, 296, 334, 293, 300),
    (468, 469, 470, 471, 472),
    (473, 474, 475, 476, 477),
)


def _pixel(landmark: Any, width: int, height: int) -> tuple[int, int]:
    return int(landmark.x * width), int(landmark.y * height)


def draw_tracking_overlay(
    frame: Any,
    faces: Sequence[Sequence[Any]],
    blendshapes: Sequence[Sequence[Any]],
    fps: float,
) -> Any:
    """랜드마크 특징선과 상위 표정 값을 프레임에 그린다."""
    height, width = frame.shape[:2]
    for face_index, landmarks in enumerate(faces):
        for landmark in landmarks[::8]:
            cv2.circle(frame, _pixel(landmark, width, height), 1, (90, 220, 90), -1)
        for loop in FEATURE_LOOPS:
            points = [_pixel(landmarks[index], width, height) for index in loop if index < len(landmarks)]
            if len(points) > 1:
                cv2.polylines(
                    frame,
                    [np.asarray(points, dtype=np.int32)],
                    True,
                    (70, 255, 180),
                    1,
                    cv2.LINE_AA,
                )

        if face_index < len(blendshapes):
            active = sorted(
                (item for item in blendshapes[face_index] if item.category_name != "_neutral"),
                key=lambda item: item.score,
                reverse=True,
            )[:4]
            for row, item in enumerate(active):
                label = f"{item.category_name}: {item.score:.2f}"
                cv2.putText(frame, label, (16, 54 + row * 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)

    status = f"faces: {len(faces)} | processing: {fps:.1f} fps"
    cv2.rectangle(frame, (0, 0), (width, 32), (18, 18, 18), -1)
    cv2.putText(frame, status, (14, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.57, (70, 255, 180), 1, cv2.LINE_AA)
    return frame
