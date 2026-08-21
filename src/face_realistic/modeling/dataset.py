"""Still-image pairs for the first one-identity overfit experiment."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


POSE_NAMES = ("head_pitch", "head_yaw", "head_roll")


def _resolve(path_value: str, project_root: Path) -> Path:
    path = Path(path_value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _stable_validation_member(record: dict[str, Any], validation_fraction: float) -> bool:
    key = record.get("source_sha256") or record["source_path"]
    bucket = int(hashlib.sha256(str(key).encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return bucket < validation_fraction


def split_records(
    records: list[dict[str, Any]], validation_fraction: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be in (0, 0.5)")
    validation = [record for record in records if _stable_validation_member(record, validation_fraction)]
    training = [record for record in records if record not in validation]
    if not validation and len(records) > 1:
        validation = [records[-1]]
        training = records[:-1]
    if not training and validation:
        training = [validation.pop()]
    return training, validation


def make_oval_mask(size: int) -> Tensor:
    mask = np.zeros((size, size), dtype=np.float32)
    cv2.ellipse(
        mask,
        (size // 2, round(size * 0.52)),
        (round(size * 0.36), round(size * 0.45)),
        0,
        0,
        360,
        1.0,
        thickness=-1,
        lineType=cv2.LINE_AA,
    )
    mask = cv2.GaussianBlur(mask, (0, 0), max(1.0, size * 0.015))
    # OpenCV's float blur can overshoot 1.0 by a few ULPs. CUDA BCE treats
    # even that tiny overshoot as an invalid target and raises a device-side
    # assertion, so enforce the probability range before creating the tensor.
    mask = np.clip(mask, 0.0, 1.0)
    return torch.from_numpy(mask[None, ...])


@dataclass(frozen=True, slots=True)
class DatasetInfo:
    identity_id: str
    image_count: int
    source_image: str
    motion_names: tuple[str, ...]


class IdentityStillDataset(Dataset[dict[str, Tensor]]):
    """Use a fixed neutral source and each registered still as the target."""

    def __init__(
        self,
        manifest_path: Path,
        *,
        project_root: Path,
        image_size: int = 256,
        split: str = "train",
        validation_fraction: float = 0.2,
        source_stem: str = "front_neutral",
    ):
        self.manifest_path = manifest_path.expanduser().resolve()
        self.project_root = project_root.expanduser().resolve()
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        valid = [
            record
            for record in manifest.get("images", [])
            if record.get("detected") and record.get("aligned_path") and record.get("head_pose")
        ]
        valid = [record for record in valid if _resolve(record["aligned_path"], self.project_root).is_file()]
        if len(valid) < 2:
            raise ValueError("At least two registered and locally available identity images are required")
        source_record = next(
            (record for record in valid if Path(record["source_path"]).stem == source_stem),
            None,
        )
        if source_record is None:
            raise ValueError(f"Source reference '{source_stem}' is missing from the manifest")

        train_records, validation_records = split_records(valid, validation_fraction)
        if split == "train":
            records = train_records
        elif split in {"validation", "val"}:
            records = validation_records
        elif split == "all":
            records = valid
        else:
            raise ValueError("split must be train, validation, or all")
        if not records:
            raise ValueError(f"The {split} split is empty")

        blendshape_names = sorted(
            {name for record in valid for name in record.get("blendshapes", {}) if name != "_neutral"}
        )
        self.records = records
        self.source_path = _resolve(source_record["aligned_path"], self.project_root)
        self.image_size = image_size
        self.motion_names = tuple(blendshape_names) + POSE_NAMES
        self.mask = make_oval_mask(image_size)
        self.info = DatasetInfo(
            identity_id=str(manifest.get("identity_id", self.manifest_path.parent.name)),
            image_count=len(records),
            source_image=str(self.source_path),
            motion_names=self.motion_names,
        )
        self._source = self._load_image(self.source_path)

    @property
    def motion_dim(self) -> int:
        return len(self.motion_names)

    def _load_image(self, path: Path) -> Tensor:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not decode training image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        array = np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32) / 127.5 - 1.0
        return torch.from_numpy(array)

    def _motion(self, record: dict[str, Any]) -> Tensor:
        blendshapes = record.get("blendshapes", {})
        values = [float(blendshapes.get(name, 0.0)) for name in self.motion_names[:-3]]
        pitch, yaw, roll = record["head_pose"]["euler_xyz_deg"]
        values.extend(
            (
                float(np.clip(pitch / 45.0, -1.0, 1.0)),
                float(np.clip(yaw / 60.0, -1.0, 1.0)),
                float(np.clip(roll / 45.0, -1.0, 1.0)),
            )
        )
        return torch.tensor(values, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        record = self.records[index]
        target_path = _resolve(record["aligned_path"], self.project_root)
        return {
            "source": self._source.clone(),
            "target": self._load_image(target_path),
            "motion": self._motion(record),
            "face_mask": self.mask.clone(),
        }
