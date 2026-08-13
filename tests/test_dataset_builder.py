import json
from pathlib import Path

from face_realistic.dataset.builder import (
    VideoProbe,
    build_pair_records,
    discover_files,
    make_job_id,
)


def test_discover_files_is_recursive_sorted_and_ignores_hidden(tmp_path: Path):
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "b.MOV").touch()
    (tmp_path / "a.mp4").touch()
    (tmp_path / ".hidden.mp4").touch()
    (tmp_path / "notes.txt").touch()

    files = discover_files(tmp_path, {".mp4", ".mov"})

    assert [path.name for path in files] == ["a.mp4", "b.MOV"]


def test_job_id_is_stable_and_disambiguates_equal_stems(tmp_path: Path):
    first = tmp_path / "a" / "clip.mp4"
    second = tmp_path / "b" / "clip.mp4"
    first.parent.mkdir()
    second.parent.mkdir()
    first.touch()
    second.touch()

    assert make_job_id(first, tmp_path) == make_job_id(first, tmp_path)
    assert make_job_id(first, tmp_path) != make_job_id(second, tmp_path)


def test_pair_records_use_shorter_video_and_generated_fps(tmp_path: Path):
    driving = VideoProbe(3.0, 30.0, 1920, 1080, 90)
    generated = VideoProbe(2.9, 25.0, 512, 512, 72)

    records = list(
        build_pair_records(
            identity_id="person_01",
            job_id="clip_1234",
            source_image=tmp_path / "face.png",
            driving_video=tmp_path / "driving.mp4",
            generated_video=tmp_path / "generated.mp4",
            driving=driving,
            generated=generated,
        )
    )

    assert len(records) == 72
    assert records[1]["timestamp_ms"] == 40.0
    assert records[-1]["frame_index"] == 71
    json.dumps(records[0])

