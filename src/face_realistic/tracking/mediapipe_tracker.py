"""MediaPipe Face Landmarker 영상 추적 파이프라인."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from face_realistic.config import TrackingConfig
from face_realistic.performance.head_pose import matrix_to_head_pose
from face_realistic.tracking.overlay import draw_tracking_overlay


@dataclass(frozen=True, slots=True)
class TrackingSummary:
    input_path: str
    frames_processed: int
    frames_with_face: int
    source_fps: float
    source_size: tuple[int, int]
    elapsed_seconds: float
    processing_fps: float
    realtime_factor: float
    jsonl_path: str
    preview_path: str | None


def _load_mediapipe() -> Any:
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError(
            "MediaPipe가 설치되어 있지 않습니다. `uv sync --extra dev`를 먼저 실행하세요."
        ) from exc
    return mp


def _serialize_face(
    landmarks: list[Any],
    blendshapes: list[Any],
    matrix: np.ndarray | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "landmarks": [
            [round(float(point.x), 7), round(float(point.y), 7), round(float(point.z), 7)]
            for point in landmarks
        ],
        "blendshapes": {
            item.category_name: round(float(item.score), 7) for item in blendshapes
        },
        "transformation_matrix": None,
        "head_pose": None,
    }
    if matrix is not None:
        matrix_array = np.asarray(matrix, dtype=float)
        payload["transformation_matrix"] = np.round(matrix_array, 7).tolist()
        payload["head_pose"] = matrix_to_head_pose(matrix_array)
    return payload


def _create_landmarker(mp: Any, config: TrackingConfig) -> Any:
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(config.model_path.resolve())),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=config.num_faces,
        min_face_detection_confidence=config.min_face_detection_confidence,
        min_face_presence_confidence=config.min_face_presence_confidence,
        min_tracking_confidence=config.min_tracking_confidence,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )
    return mp.tasks.vision.FaceLandmarker.create_from_options(options)


def _open_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"미리보기 영상을 생성할 수 없습니다: {path}")
    return writer


def track_video(config: TrackingConfig) -> TrackingSummary:
    """영상 프레임을 추적하고 JSONL 및 선택적 미리보기를 생성한다."""
    config.validate()
    mp = _load_mediapipe()
    capture = cv2.VideoCapture(str(config.input_path))
    if not capture.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {config.input_path}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if source_fps <= 0:
        source_fps = 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_limit = None if config.max_seconds is None else max(1, round(config.max_seconds * source_fps))

    config.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    writer = _open_writer(config.preview_path, source_fps, (width, height)) if config.preview_path else None
    processed = 0
    detected = 0
    previous_timestamp = -1
    started = time.perf_counter()

    try:
        with config.jsonl_path.open("w", encoding="utf-8") as output, _create_landmarker(mp, config) as landmarker:
            while frame_limit is None or processed < frame_limit:
                ok, frame = capture.read()
                if not ok:
                    break

                timestamp_ms = max(previous_timestamp + 1, round(processed * 1000.0 / source_fps))
                previous_timestamp = timestamp_ms
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                media_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))

                inference_started = time.perf_counter()
                result = landmarker.detect_for_video(media_image, timestamp_ms)
                inference_ms = (time.perf_counter() - inference_started) * 1000.0
                if result.face_landmarks:
                    detected += 1

                faces: list[dict[str, Any]] = []
                for index, landmarks in enumerate(result.face_landmarks):
                    shapes = result.face_blendshapes[index] if index < len(result.face_blendshapes) else []
                    matrix = (
                        result.facial_transformation_matrixes[index]
                        if index < len(result.facial_transformation_matrixes)
                        else None
                    )
                    faces.append(_serialize_face(landmarks, shapes, matrix))

                record = {
                    "schema_version": 1,
                    "frame_index": processed,
                    "timestamp_ms": timestamp_ms,
                    "detected": bool(faces),
                    "inference_ms": round(inference_ms, 3),
                    "faces": faces,
                }
                output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

                if writer is not None:
                    elapsed = max(time.perf_counter() - started, 1e-9)
                    draw_tracking_overlay(frame, result.face_landmarks, result.face_blendshapes, (processed + 1) / elapsed)
                    writer.write(frame)
                processed += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    elapsed_seconds = time.perf_counter() - started
    processing_fps = processed / elapsed_seconds if elapsed_seconds > 0 else 0.0
    processed_duration = processed / source_fps
    summary = TrackingSummary(
        input_path=str(config.input_path),
        frames_processed=processed,
        frames_with_face=detected,
        source_fps=round(source_fps, 3),
        source_size=(width, height),
        elapsed_seconds=round(elapsed_seconds, 3),
        processing_fps=round(processing_fps, 3),
        realtime_factor=round(elapsed_seconds / processed_duration, 3) if processed_duration else 0.0,
        jsonl_path=str(config.jsonl_path),
        preview_path=str(config.preview_path) if config.preview_path else None,
    )
    summary_path = config.jsonl_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(asdict(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary
