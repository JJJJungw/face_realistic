import pytest

torch = pytest.importorskip("torch")

from face_realistic.modeling.losses import ReconstructionObjective
from face_realistic.modeling.network import CleanRoomFaceSwapModel, ModelConfig


def test_cleanroom_model_forward_backward():
    config = ModelConfig(
        motion_dim=12,
        image_size=32,
        base_channels=8,
        max_channels=32,
        identity_dim=16,
        motion_embedding_dim=8,
        target_bottleneck=4,
    )
    model = CleanRoomFaceSwapModel(config)
    source = torch.randn(2, 3, 32, 32)
    target = torch.randn(2, 3, 32, 32).clamp(-1, 1)
    motion = torch.randn(2, 12)
    face_mask = torch.ones(2, 1, 32, 32)

    outputs = model(source, target, motion)
    losses = ReconstructionObjective()(outputs, target, face_mask)
    losses["total"].backward()

    assert outputs["generated"].shape == (2, 3, 32, 32)
    assert outputs["alpha"].shape == (2, 1, 32, 32)
    assert outputs["identity"].shape == (2, 16)
    assert torch.isfinite(losses["total"])
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_target_identity_suppression_reduces_spatial_resolution():
    model = CleanRoomFaceSwapModel(
        ModelConfig(motion_dim=4, image_size=32, base_channels=8, max_channels=32, target_bottleneck=2)
    )
    target = torch.randn(1, 3, 32, 32)

    suppressed = model.suppress_target_identity(target)

    assert suppressed.shape == target.shape
    assert not torch.allclose(suppressed, target)

