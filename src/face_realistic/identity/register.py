"""대체 얼굴 ID 이미지 등록과 품질 평가."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from face_realistic.performance.head_pose import matrix_to_head_pose


ALIGNED_SIZE = 512
LEFT_EYE = (33, 133, 159, 145)
RIGHT_EYE = (362, 263, 386, 374)
MOUTH = (13, 14, 61, 291)
REFERENCE_PRIORITY = (
    "front_neutral",
    "front_slight_smile",
    "pose_left_15",
    "pose_right_15",
    "pose_left_30",
    "pose_right_30",
    "slight_smile",
    "focused",
)


@dataclass(frozen=True, slots=True)
class ImageQuality:
    total: float
    sharpness: float
    exposure: float
    frontality: float
    face_coverage: float
    sharpness_raw: float
    brightness_raw: float


def _load_mediapipe() -> Any:
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError("MediaPipe가 없습니다. `uv sync --extra dev`를 실행하세요.") from exc
    return mp


def _mean_point(landmarks: list[Any], indices: tuple[int, ...], width: int, height: int) -> np.ndarray:
    return np.mean(
        [[landmarks[index].x * width, landmarks[index].y * height] for index in indices],
        axis=0,
        dtype=np.float32,
    )


def align_face(image: np.ndarray, landmarks: list[Any], size: int = ALIGNED_SIZE) -> np.ndarray:
    """눈과 입 중심을 사용해 얼굴을 고정 크기의 정사각형으로 정렬한다."""
    height, width = image.shape[:2]
    eyes = [
        _mean_point(landmarks, LEFT_EYE, width, height),
        _mean_point(landmarks, RIGHT_EYE, width, height),
    ]
    image_left_eye, image_right_eye = sorted(eyes, key=lambda point: float(point[0]))
    mouth = _mean_point(landmarks, MOUTH, width, height)
    source = np.float32([image_left_eye, image_right_eye, mouth])
    target = np.float32(
        [
            [0.31 * size, 0.38 * size],
            [0.69 * size, 0.38 * size],
            [0.50 * size, 0.68 * size],
        ]
    )
    transform, _ = cv2.estimateAffinePartial2D(source, target, method=cv2.LMEDS)
    if transform is None:
        raise RuntimeError("얼굴 정렬 변환을 계산하지 못했습니다.")
    return cv2.warpAffine(
        image,
        transform,
        (size, size),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def normalized_bbox(landmarks: list[Any]) -> list[float]:
    xs = [float(point.x) for point in landmarks]
    ys = [float(point.y) for point in landmarks]
    return [
        round(max(0.0, min(xs)), 6),
        round(max(0.0, min(ys)), 6),
        round(min(1.0, max(xs)), 6),
        round(min(1.0, max(ys)), 6),
    ]


def score_image_quality(
    aligned: np.ndarray,
    bbox: list[float],
    head_pose: dict[str, list[float]] | None,
) -> ImageQuality:
    """0~1 범위의 등록 품질 점수와 구성 요소를 계산한다."""
    gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    sharpness_raw = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness_raw = float(gray.mean())
    sharpness = min(1.0, math.log1p(sharpness_raw) / math.log1p(1000.0))
    exposure = max(0.0, 1.0 - abs(brightness_raw - 127.5) / 127.5)

    if head_pose is None:
        frontality = 0.0
    else:
        pitch, yaw, roll = head_pose["euler_xyz_deg"]
        frontality = math.exp(-((abs(pitch) / 35.0) ** 2 + (abs(yaw) / 45.0) ** 2 + (abs(roll) / 45.0) ** 2))

    width = max(0.0, bbox[2] - bbox[0])
    height = max(0.0, bbox[3] - bbox[1])
    coverage_raw = width * height
    face_coverage = min(1.0, coverage_raw / 0.24)
    total = 0.30 * sharpness + 0.15 * exposure + 0.35 * frontality + 0.20 * face_coverage
    return ImageQuality(
        total=round(total, 6),
        sharpness=round(sharpness, 6),
        exposure=round(exposure, 6),
        frontality=round(frontality, 6),
        face_coverage=round(face_coverage, 6),
        sharpness_raw=round(sharpness_raw, 3),
        brightness_raw=round(brightness_raw, 3),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _create_contact_sheet(records: list[dict[str, Any]], output_path: Path) -> None:
    valid = [record for record in records if record["detected"]]
    if not valid:
        return
    columns, tile, label_height = 5, 190, 42
    rows = math.ceil(len(valid) / columns)
    sheet = np.full((rows * (tile + label_height), columns * tile, 3), 245, dtype=np.uint8)
    for index, record in enumerate(valid):
        image = cv2.imread(record["aligned_path"])
        if image is None:
            continue
        image = cv2.resize(image, (tile, tile), interpolation=cv2.INTER_AREA)
        x = (index % columns) * tile
        y = (index // columns) * (tile + label_height)
        sheet[y : y + tile, x : x + tile] = image
        label = f'{Path(record["source_path"]).stem}  {record["quality"]["total"]:.2f}'
        cv2.putText(sheet, label, (x + 4, y + tile + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (25, 25, 25), 1, cv2.LINE_AA)
    cv2.imwrite(str(output_path), sheet)


def register_identity(identity_dir: Path, model_path: Path, output_root: Path) -> dict[str, Any]:
    """한 ID 폴더의 모든 사진을 등록하고 manifest를 반환한다."""
    if not identity_dir.is_dir():
        raise FileNotFoundError(f"ID 폴더를 찾을 수 없습니다: {identity_dir}")
    image_paths = sorted(
        path for path in identity_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    if not image_paths:
        raise ValueError(f"등록할 이미지가 없습니다: {identity_dir}")

    mp = _load_mediapipe()
    identity_id = identity_dir.name
    output_dir = output_root / identity_id
    aligned_dir = output_dir / "aligned"
    aligned_dir.mkdir(parents=True, exist_ok=True)
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path.resolve())),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=2,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )

    records: list[dict[str, Any]] = []
    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
        for source_path in image_paths:
            record: dict[str, Any] = {
                "source_path": str(source_path),
                "source_sha256": _sha256(source_path),
                "detected": False,
                "error": None,
            }
            try:
                image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise RuntimeError("이미지를 디코딩하지 못했습니다.")
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                media_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
                result = landmarker.detect(media_image)
                record["faces_detected"] = len(result.face_landmarks)
                if not result.face_landmarks:
                    record["error"] = "face_not_detected"
                    records.append(record)
                    continue

                landmarks = result.face_landmarks[0]
                matrix = np.asarray(result.facial_transformation_matrixes[0], dtype=float)
                head_pose = matrix_to_head_pose(matrix)
                bbox = normalized_bbox(landmarks)
                aligned = align_face(image, landmarks)
                aligned_path = aligned_dir / f"{source_path.stem}.jpg"
                if not cv2.imwrite(str(aligned_path), aligned, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise RuntimeError("정렬 이미지를 저장하지 못했습니다.")
                quality = score_image_quality(aligned, bbox, head_pose)
                blendshapes = {
                    item.category_name: round(float(item.score), 6)
                    for item in result.face_blendshapes[0]
                }
                record.update(
                    {
                        "detected": True,
                        "aligned_path": str(aligned_path),
                        "landmark_count": len(landmarks),
                        "bbox_normalized": bbox,
                        "head_pose": head_pose,
                        "quality": asdict(quality),
                        "blendshapes": blendshapes,
                    }
                )
            except Exception as exc:  # 한 장의 실패가 전체 등록을 중단하지 않게 한다.
                record["error"] = f"{type(exc).__name__}: {exc}"
            records.append(record)

    valid = [record for record in records if record["detected"]]
    ranked = sorted(valid, key=lambda item: item["quality"]["total"], reverse=True)
    neutral = next((record for record in valid if Path(record["source_path"]).stem == "front_neutral"), None)
    representative = neutral or (ranked[0] if ranked else None)
    records_by_stem = {Path(record["source_path"]).stem: record for record in valid}
    references = [records_by_stem[stem] for stem in REFERENCE_PRIORITY if stem in records_by_stem]
    if len(references) < 8:
        references.extend(record for record in ranked if record not in references)
    manifest = {
        "schema_version": 1,
        "identity_id": identity_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_directory": str(identity_dir),
        "total_images": len(records),
        "registered_images": len(valid),
        "failed_images": len(records) - len(valid),
        "mean_quality": round(statistics.mean(item["quality"]["total"] for item in valid), 6) if valid else 0.0,
        "representative": representative["aligned_path"] if representative else None,
        "recommended_references": [item["aligned_path"] for item in references[:8]],
        "images": records,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _create_contact_sheet(records, output_dir / "contact_sheet.jpg")
    return manifest
