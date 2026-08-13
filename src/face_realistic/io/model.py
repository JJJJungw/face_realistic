"""MediaPipe 모델 파일 준비."""

from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path


FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
YUNET_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)
SFACE_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/"
    "face_recognition_sface_2021dec.onnx"
)


def ensure_face_landmarker_model(
    path: Path,
    url: str = FACE_LANDMARKER_MODEL_URL,
) -> Path:
    """모델이 없으면 임시 파일로 내려받은 뒤 원자적으로 배치한다."""
    if path.is_file() and path.stat().st_size > 0:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".part")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "face-realistic-mvp/0.1"})
        with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
            with temporary_path.open("wb") as output:
                shutil.copyfileobj(response, output)
        if temporary_path.stat().st_size == 0:
            raise RuntimeError("다운로드한 모델 파일이 비어 있습니다.")
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path


def ensure_opencv_face_models(
    yunet_path: Path,
    sface_path: Path,
) -> tuple[Path, Path]:
    """OpenCV Zoo의 YuNet 검출기와 SFace 인식 모델을 준비한다."""
    return (
        _ensure_model(yunet_path, YUNET_MODEL_URL, minimum_bytes=100_000),
        _ensure_model(sface_path, SFACE_MODEL_URL, minimum_bytes=10_000_000),
    )


def _ensure_model(path: Path, url: str, minimum_bytes: int) -> Path:
    if path.is_file() and path.stat().st_size >= minimum_bytes:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".part")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "face-realistic-mvp/0.1"})
        with urllib.request.urlopen(request, timeout=180) as response:  # noqa: S310
            with temporary_path.open("wb") as output:
                shutil.copyfileobj(response, output)
        if temporary_path.stat().st_size < minimum_bytes:
            raise RuntimeError(f"다운로드한 모델이 예상보다 작습니다: {temporary_path.stat().st_size} bytes")
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path
