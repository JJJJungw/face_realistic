import pytest

torch = pytest.importorskip("torch")

from face_realistic.modeling.v1.config import V1ModelConfig
from face_realistic.modeling.v1.smoke import run_smoke


def test_v1_smoke_report_on_cpu():
    report = run_smoke(
        device=torch.device("cpu"),
        config=V1ModelConfig(
            image_size=32,
            base_channels=8,
            max_channels=32,
            identity_dim=16,
            motion_embedding_dim=8,
        ),
        references=1,
        warmup=0,
        iterations=1,
    )

    assert report["device"] == "cpu"
    assert report["parameters"] > 0
    assert report["generator_fps"] > 0
    assert report["output_shapes"]["generated"] == [1, 3, 32, 32]
    assert report["trained_weights"] is False
