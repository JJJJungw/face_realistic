from face_realistic import __version__
from face_realistic.config import TrackingConfig


def test_package_version() -> None:
    assert __version__ == "0.1.0"


def test_config_rejects_missing_input(tmp_path) -> None:
    config = TrackingConfig(
        input_path=tmp_path / "missing.mp4",
        model_path=tmp_path / "model.task",
        jsonl_path=tmp_path / "tracking.jsonl",
        preview_path=None,
    )
    try:
        config.validate()
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("missing input must fail validation")
