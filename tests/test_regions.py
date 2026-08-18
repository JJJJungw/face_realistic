"""부위별 마스크 빌더 검증. 모델 없이 도는 순수 기하 테스트."""

from __future__ import annotations

import numpy as np
import pytest

from face_realistic.swap.regions import (
    REGION_GROUPS,
    build_region_mask,
    composite,
    resolve_groups,
)

FRAME = (256, 256)


def _synthetic_points(count: int = 478) -> np.ndarray:
    rng = np.random.default_rng(0)
    points = rng.uniform(60, 196, size=(count, 2)).astype(np.float32)
    return points


def test_groups_resolve() -> None:
    specs = resolve_groups(["mouth_inner", "iris"])
    assert len(specs) == 3
    assert {spec.name for spec in specs} == {"mouth_inner", "left_iris", "right_iris"}


def test_unknown_group_rejected() -> None:
    with pytest.raises(KeyError):
        resolve_groups(["nose"])


def test_mask_grows_monotonically_with_more_regions() -> None:
    points = _synthetic_points()
    ladder = [
        ["mouth_inner"],
        ["mouth_inner", "iris"],
        ["mouth_inner", "iris", "eye_open"],
        ["mouth_inner", "iris", "eye_open", "mouth_outer"],
    ]
    areas = [
        float(build_region_mask(points, FRAME, groups, feather_pixels=0.0).mean())
        for groups in ladder
    ]
    assert all(later >= earlier for earlier, later in zip(areas, areas[1:]))
    assert areas[0] > 0.0


def test_eyelid_covers_eye_open() -> None:
    points = _synthetic_points()
    open_mask = build_region_mask(points, FRAME, ["eye_open"], feather_pixels=0.0)
    lid_mask = build_region_mask(points, FRAME, ["eyelid"], feather_pixels=0.0)
    assert lid_mask.mean() > open_mask.mean()


def test_iris_requires_478_landmarks() -> None:
    with pytest.raises(ValueError, match="478"):
        build_region_mask(_synthetic_points(468), FRAME, ["iris"], feather_pixels=0.0)


def test_too_few_landmarks_rejected() -> None:
    with pytest.raises(ValueError, match="468"):
        build_region_mask(_synthetic_points(200), FRAME, ["mouth_inner"], feather_pixels=0.0)


def test_all_groups_produce_nonempty_mask() -> None:
    points = _synthetic_points()
    for name in REGION_GROUPS:
        mask = build_region_mask(points, FRAME, [name], feather_pixels=0.0)
        assert mask.mean() > 0.0, name


def test_composite_endpoints() -> None:
    original = np.full((8, 8, 3), 10, dtype=np.uint8)
    swapped = np.full((8, 8, 3), 200, dtype=np.uint8)
    assert composite(original, swapped, np.ones((8, 8), np.float32)).mean() == pytest.approx(10)
    assert composite(original, swapped, np.zeros((8, 8), np.float32)).mean() == pytest.approx(200)


def test_composite_shape_guard() -> None:
    original = np.zeros((8, 8, 3), np.uint8)
    with pytest.raises(ValueError):
        composite(original, np.zeros((4, 4, 3), np.uint8), np.zeros((8, 8), np.float32))
