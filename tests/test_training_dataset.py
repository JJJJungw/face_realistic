import json
from pathlib import Path

import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from face_realistic.modeling.dataset import IdentityStillDataset, make_oval_mask, split_records


def _write_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((32, 32, 3), value, dtype=np.uint8))


def test_identity_still_dataset_builds_motion_vector(tmp_path: Path):
    images = tmp_path / "aligned"
    records = []
    for index, name in enumerate(("front_neutral", "smile", "left", "blink")):
        image_path = images / f"{name}.jpg"
        _write_image(image_path, 50 + index * 20)
        records.append(
            {
                "source_path": f"assets/{name}.png",
                "source_sha256": str(index),
                "aligned_path": str(image_path),
                "detected": True,
                "head_pose": {"euler_xyz_deg": [10.0, -20.0, 5.0]},
                "blendshapes": {"jawOpen": 0.25, "eyeBlinkLeft": 0.5},
            }
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"identity_id": "person_01", "images": records}))

    dataset = IdentityStillDataset(
        manifest, project_root=tmp_path, image_size=32, split="all"
    )
    item = dataset[1]

    assert len(dataset) == 4
    assert dataset.motion_names == (
        "eyeBlinkLeft",
        "jawOpen",
        "head_pitch",
        "head_yaw",
        "head_roll",
    )
    assert item["source"].shape == (3, 32, 32)
    assert item["motion"].shape == (5,)
    assert item["face_mask"].shape == (1, 32, 32)


def test_split_records_is_deterministic():
    records = [{"source_path": f"{index}.jpg", "source_sha256": str(index)} for index in range(20)]

    first = split_records(records, 0.2)
    second = split_records(records, 0.2)

    assert first == second
    assert first[0]
    assert first[1]


def test_oval_mask_stays_in_probability_range():
    mask = make_oval_mask(256)

    assert torch.isfinite(mask).all()
    assert mask.min().item() >= 0.0
    assert mask.max().item() <= 1.0
