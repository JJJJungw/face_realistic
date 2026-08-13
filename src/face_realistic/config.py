"""추적 실행 설정."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class TrackingConfig:
    input_path: Path
    model_path: Path
    jsonl_path: Path
    preview_path: Path | None
    max_seconds: float | None = 3.0
    num_faces: int = 1
    min_face_detection_confidence: float = 0.5
    min_face_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

    def validate(self) -> None:
        if not self.input_path.is_file():
            raise FileNotFoundError(f"입력 영상을 찾을 수 없습니다: {self.input_path}")
        if self.max_seconds is not None and self.max_seconds <= 0:
            raise ValueError("max_seconds는 0보다 커야 합니다.")
        if self.num_faces < 1:
            raise ValueError("num_faces는 1 이상이어야 합니다.")
        for name, value in (
            ("min_face_detection_confidence", self.min_face_detection_confidence),
            ("min_face_presence_confidence", self.min_face_presence_confidence),
            ("min_tracking_confidence", self.min_tracking_confidence),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name}는 0과 1 사이여야 합니다.")
