from pathlib import Path

import pytest

from face_realistic.performance.liveportrait import (
    build_liveportrait_command,
    find_generated_video,
)


def test_build_liveportrait_command_uses_expression_defaults():
    command = build_liveportrait_command(
        python=Path("/lp/.venv/bin/python"),
        liveportrait_dir=Path("/lp"),
        source_image=Path("/data/source.png"),
        driving_video=Path("/data/driving.mp4"),
        output_dir=Path("/data/out"),
    )

    assert command[:2] == ["/lp/.venv/bin/python", "/lp/inference.py"]
    assert "--flag_crop_driving_video" in command
    assert command[command.index("--animation_region") + 1] == "all"
    assert command[command.index("--driving_option") + 1] == "expression-friendly"


def test_find_generated_video_excludes_concat(tmp_path: Path):
    expected = tmp_path / "source--driving.mp4"
    expected.touch()
    (tmp_path / "source--driving_concat.mp4").touch()

    assert find_generated_video(tmp_path) == expected


def test_find_generated_video_rejects_ambiguous_output(tmp_path: Path):
    (tmp_path / "one.mp4").touch()
    (tmp_path / "two.mp4").touch()

    with pytest.raises(RuntimeError, match="Expected one"):
        find_generated_video(tmp_path)
