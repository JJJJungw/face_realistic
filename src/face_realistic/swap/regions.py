"""세분화된 퍼포먼스 영역 정의와 조합 가능한 마스크 빌더.

passthrough.py 는 눈 전체 + 입술 전체라는 굵은 두 덩어리만 복원한다.
그 설정 하나만으로는 "표정을 얼마나 얻고 신원을 얼마나 잃는지"의
중간 지점을 알 수 없어서, 여기서 부위를 더 잘게 쪼갠다.

신원 단서의 세기는 부위마다 다르다는 것이 전제다.
치아·혀·홍채는 사람을 특정하는 힘이 약하고,
눈꺼풀 모양·눈썹·눈 주위 윤곽은 강하다.
그 가정이 맞는지를 재기 위한 재료다.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

# MediaPipe Face Mesh 인덱스. 468~477 은 refine_landmarks 로 추가되는 홍채다.
LEFT_EYE = (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246)
RIGHT_EYE = (362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398)
LEFT_IRIS = (468, 469, 470, 471, 472)
RIGHT_IRIS = (473, 474, 475, 476, 477)
INNER_LIPS = (78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191)
OUTER_LIPS = (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185)
LEFT_BROW = (70, 63, 105, 66, 107, 55, 65, 52, 53, 46)
RIGHT_BROW = (300, 293, 334, 296, 336, 285, 295, 282, 283, 276)


@dataclass(frozen=True, slots=True)
class RegionSpec:
    """볼록껍질 하나. expansion 은 중심 기준 확대 배율."""

    name: str
    indices: tuple[int, ...]
    expansion: float


# 부위 그룹. 값이 작을수록 신원 누출이 적을 것으로 기대하는 순서로 적었다.
REGION_GROUPS: dict[str, tuple[RegionSpec, ...]] = {
    "mouth_inner": (RegionSpec("mouth_inner", INNER_LIPS, 1.05),),
    "iris": (
        RegionSpec("left_iris", LEFT_IRIS, 1.20),
        RegionSpec("right_iris", RIGHT_IRIS, 1.20),
    ),
    "eye_open": (
        RegionSpec("left_eye_open", LEFT_EYE, 1.00),
        RegionSpec("right_eye_open", RIGHT_EYE, 1.00),
    ),
    "mouth_outer": (RegionSpec("mouth_outer", OUTER_LIPS, 1.10),),
    "eyelid": (
        RegionSpec("left_eyelid", LEFT_EYE, 1.45),
        RegionSpec("right_eyelid", RIGHT_EYE, 1.45),
    ),
    "brow": (
        RegionSpec("left_brow", LEFT_BROW, 1.15),
        RegionSpec("right_brow", RIGHT_BROW, 1.15),
    ),
    # 현재 passthrough.py 의 기본값을 그대로 재현한 대조군
    "legacy_eye": (
        RegionSpec("legacy_left_eye", LEFT_EYE, 1.15),
        RegionSpec("legacy_right_eye", RIGHT_EYE, 1.15),
    ),
    "legacy_mouth": (RegionSpec("legacy_mouth", OUTER_LIPS, 1.35),),
}

IRIS_GROUPS = frozenset({"iris"})
MIN_LANDMARKS = 468
IRIS_LANDMARKS = 478


def resolve_groups(names: Sequence[str]) -> tuple[RegionSpec, ...]:
    specs: list[RegionSpec] = []
    for name in names:
        if name not in REGION_GROUPS:
            known = ", ".join(sorted(REGION_GROUPS))
            raise KeyError(f"알 수 없는 부위 그룹입니다: {name} (사용 가능: {known})")
        specs.extend(REGION_GROUPS[name])
    return tuple(specs)


def _hull(
    points_xy: np.ndarray,
    indices: Sequence[int],
    expansion: float,
    width: int,
    height: int,
) -> np.ndarray:
    points = points_xy[list(indices)].astype(np.float32)
    center = points.mean(axis=0, keepdims=True)
    points = center + (points - center) * expansion
    points[:, 0] = np.clip(points[:, 0], 0, width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, height - 1)
    return cv2.convexHull(np.rint(points).astype(np.int32))


def build_region_mask(
    points_xy: np.ndarray,
    frame_size: tuple[int, int],
    group_names: Sequence[str],
    *,
    feather_pixels: float = 6.0,
) -> np.ndarray:
    """선택한 부위 그룹의 합집합에 대한 float alpha 마스크를 만든다.

    points_xy 는 (N, 2) 픽셀 좌표. 정규화 좌표가 아니다.
    """
    width, height = frame_size
    points_xy = np.asarray(points_xy, dtype=np.float32)
    if points_xy.ndim != 2 or points_xy.shape[1] != 2:
        raise ValueError(f"points_xy 는 (N, 2) 여야 합니다: {points_xy.shape}")
    if points_xy.shape[0] < MIN_LANDMARKS:
        raise ValueError(f"랜드마크가 {MIN_LANDMARKS}개 미만입니다: {points_xy.shape[0]}")
    if any(name in IRIS_GROUPS for name in group_names) and points_xy.shape[0] < IRIS_LANDMARKS:
        raise ValueError(
            "홍채 그룹은 478개 랜드마크가 필요합니다. FaceLandmarker 에서 홍채가 꺼져 있습니다."
        )
    if feather_pixels < 0:
        raise ValueError("feather_pixels 는 0 이상이어야 합니다.")

    mask = np.zeros((height, width), dtype=np.uint8)
    for spec in resolve_groups(group_names):
        cv2.fillConvexPoly(
            mask,
            _hull(points_xy, spec.indices, spec.expansion, width, height),
            255,
            lineType=cv2.LINE_AA,
        )
    alpha = mask.astype(np.float32) / 255.0
    if feather_pixels > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), feather_pixels)
    return np.clip(alpha, 0.0, 1.0)


def composite(original: np.ndarray, swapped: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """alpha 가 1인 곳은 원본, 0인 곳은 스왑 결과."""
    if original.shape != swapped.shape:
        raise ValueError("원본과 스왑 프레임의 shape 이 다릅니다.")
    if alpha.shape != original.shape[:2]:
        raise ValueError("alpha 마스크 크기가 프레임과 다릅니다.")
    a = alpha[..., None]
    out = original.astype(np.float32) * a + swapped.astype(np.float32) * (1.0 - a)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)
