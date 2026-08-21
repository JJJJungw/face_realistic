import pytest

torch = pytest.importorskip("torch")

from face_realistic.modeling.v1 import FaceRealisticSwapV1, V1ModelConfig


def make_inputs(batch: int = 2, references: int = 2):
    size = 32
    geometry = torch.rand(batch, 16, 8, 8)
    geometry[:, -1] = 1.0
    return {
        "reference_faces": torch.randn(batch, references, 3, size, size).clamp(-1, 1),
        "geometry_maps": geometry,
        "blendshapes": torch.randn(batch, 8),
        "head_pose": torch.randn(batch, 6),
        "lowfreq_illumination": torch.randn(batch, 3, 8, 8).clamp(-1, 1),
        "context_ring": torch.randn(batch, 3, size, size).clamp(-1, 1),
        "occlusion_mask": torch.zeros(batch, 1, size, size),
        "reference_quality": torch.tensor([[1.0, 0.5]]).expand(batch, -1).clone(),
        "reference_valid": torch.ones(batch, references),
    }


def config():
    return V1ModelConfig(
        image_size=32,
        blendshape_dim=8,
        base_channels=8,
        max_channels=32,
        identity_dim=16,
        motion_embedding_dim=8,
        attention_heads=4,
    )


def test_v1_model_forward_backward_contract():
    model = FaceRealisticSwapV1(config())
    inputs = make_inputs()

    outputs = model(**inputs)
    loss = outputs["generated"].abs().mean() + outputs["alpha"].mean() + outputs["confidence"].mean()
    loss.backward()

    assert outputs["generated"].shape == (2, 3, 32, 32)
    assert outputs["alpha"].shape == (2, 1, 32, 32)
    assert outputs["visibility"].shape == (2, 1, 32, 32)
    assert outputs["composite_alpha"].shape == (2, 1, 32, 32)
    assert outputs["confidence"].shape == (2, 1)
    assert outputs["identity"].shape == (2, 16)
    assert outputs["reference_weights"].shape == (2, 2)
    assert torch.all((outputs["alpha"] >= 0) & (outputs["alpha"] <= 1))
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_invalid_reference_is_excluded_from_identity_average():
    model = FaceRealisticSwapV1(config()).eval()
    inputs = make_inputs(batch=1)
    inputs["reference_valid"] = torch.tensor([[1.0, 0.0]])

    first = model(**inputs)
    inputs["reference_faces"][:, 1] = torch.randn_like(inputs["reference_faces"][:, 1])
    second = model(**inputs)

    assert torch.equal(first["identity"], second["identity"])
    assert torch.equal(first["generated"], second["generated"])
    assert torch.equal(first["reference_weights"], torch.tensor([[1.0, 0.0]]))


def test_context_pixels_are_removed_inside_face_support():
    model = FaceRealisticSwapV1(config()).eval()
    inputs = make_inputs(batch=1)
    first = model(**inputs)
    inputs["context_ring"] = -inputs["context_ring"]
    second = model(**inputs)

    assert torch.equal(first["sanitized_context"], torch.zeros_like(first["sanitized_context"]))
    assert torch.equal(first["generated"], second["generated"])
