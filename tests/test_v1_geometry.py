import pytest

torch = pytest.importorskip("torch")

from face_realistic.modeling.v1 import MediaPipeGeometryRasterizer, build_target_conditions


def test_geometry_rasterizer_builds_identity_neutral_maps():
    landmarks = torch.zeros(2, 478, 4)
    landmarks[..., 0] = 0.5
    landmarks[..., 1] = 0.5
    landmarks[..., 3] = 1.0
    landmarks[1, :, 0] = 0.6
    rasterizer = MediaPipeGeometryRasterizer(output_size=16)

    maps = rasterizer(landmarks)

    assert maps.shape == (2, 16, 16, 16)
    assert torch.isfinite(maps).all()
    assert maps[:, 15].min() >= 0
    assert maps[:, 15].max() <= 1
    assert not torch.equal(maps[0], maps[1])


def test_geometry_rasterizer_rejects_invalid_shape():
    rasterizer = MediaPipeGeometryRasterizer(output_size=8)

    with pytest.raises(ValueError, match="shape"):
        rasterizer(torch.zeros(1, 478, 3))


def test_condition_builder_removes_inner_face_rgb():
    target = torch.ones(1, 3, 32, 32)
    landmarks = torch.zeros(1, 478, 4)
    landmarks[..., :2] = 0.5
    landmarks[..., 3] = 1.0

    conditions = build_target_conditions(
        target,
        landmarks,
        rasterizer=MediaPipeGeometryRasterizer(output_size=8),
        illumination_size=4,
    )

    assert conditions["geometry_maps"].shape == (1, 16, 8, 8)
    assert conditions["lowfreq_illumination"].shape == (1, 3, 4, 4)
    assert conditions["context_ring"][0, :, 16, 16].abs().max() < 0.01
