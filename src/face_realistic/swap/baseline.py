"""InsightFace InSwapper를 이용한 연구용 영상 face-swap 기준선."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2

from face_realistic.io.model import ensure_face_landmarker_model
from face_realistic.swap.passthrough import make_performance_mask, restore_performance_regions


@dataclass(frozen=True, slots=True)
class SwapSummary:
    input_path: str
    source_identity_path: str
    output_path: str
    frames_processed: int
    frames_swapped: int
    frames_failed: int
    source_fps: float
    elapsed_seconds: float
    processing_fps: float
    realtime_factor: float
    audio_muxed: bool
    performance_passthrough: bool
    passthrough_frames: int
    landmark_failures: int
    eye_expansion: float
    mouth_expansion: float
    feather_ratio: float
    model_license: str


def _load_insightface() -> Any:
    try:
        import onnxruntime as ort

        if "CUDAExecutionProvider" in ort.get_available_providers() and hasattr(ort, "preload_dlls"):
            ort.preload_dlls(directory="")
        import insightface
        from insightface.app import FaceAnalysis
    except ImportError as exc:
        raise RuntimeError(
            "Face swap 의존성이 없습니다. `uv sync --extra dev --extra swap`을 실행하세요."
        ) from exc
    return insightface, FaceAnalysis


def _create_performance_landmarker(model_path: Path) -> tuple[Any, Any]:
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError("MediaPipe가 없어 performance pass-through를 사용할 수 없습니다.") from exc
    ensure_face_landmarker_model(model_path)
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path.resolve())),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return mp, mp.tasks.vision.FaceLandmarker.create_from_options(options)


def resolve_providers(mode: str) -> list[str]:
    """요청한 실행 모드와 설치된 ONNX Runtime provider를 맞춘다."""
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("ONNX Runtime이 없습니다. swap extra를 설치하세요.") from exc
    available = set(ort.get_available_providers())
    if mode == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                "CUDAExecutionProvider를 사용할 수 없습니다. CPU용 onnxruntime을 제거하고 "
                "EC2 CUDA 버전에 맞는 onnxruntime-gpu를 설치하세요."
            )
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if mode == "cpu":
        return ["CPUExecutionProvider"]
    if mode != "auto":
        raise ValueError(f"지원하지 않는 provider 모드입니다: {mode}")
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _largest_face(faces: list[Any]) -> Any:
    if not faces:
        raise RuntimeError("얼굴을 검출하지 못했습니다.")
    return max(faces, key=lambda face: float((face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])))


def _mux_audio(silent_path: Path, source_path: Path, output_path: Path, duration: float) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        silent_path.replace(output_path)
        return False
    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(silent_path),
        "-i",
        str(source_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-t",
        f"{duration:.6f}",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    subprocess.run(command, check=True)
    silent_path.unlink(missing_ok=True)
    return True


def run_face_swap(
    input_path: Path,
    source_identity_path: Path,
    output_path: Path,
    model_root: Path,
    swapper_model_path: Path,
    max_seconds: float | None = 3.0,
    provider: str = "auto",
    preserve_performance: bool = False,
    landmarker_model_path: Path = Path("models/face_landmarker.task"),
    eye_expansion: float = 1.15,
    mouth_expansion: float = 1.35,
    feather_ratio: float = 0.012,
) -> SwapSummary:
    """원본 영상의 가장 큰 얼굴을 지정한 대체 ID로 교체한다."""
    insightface, FaceAnalysis = _load_insightface()
    if not input_path.is_file():
        raise FileNotFoundError(f"입력 영상이 없습니다: {input_path}")
    if not source_identity_path.is_file():
        raise FileNotFoundError(f"대체 ID 이미지가 없습니다: {source_identity_path}")
    if not swapper_model_path.is_file():
        raise FileNotFoundError(
            f"InSwapper 모델이 없습니다: {swapper_model_path}\n"
            "InsightFace에서 허가된 연구/상용 모델을 내려받아 해당 경로에 배치하세요."
        )
    if eye_expansion < 1.0 or mouth_expansion < 1.0:
        raise ValueError("eye_expansion과 mouth_expansion은 1.0 이상이어야 합니다.")
    if feather_ratio < 0:
        raise ValueError("feather_ratio는 0 이상이어야 합니다.")

    providers = resolve_providers(provider)
    analysis = FaceAnalysis(
        name="buffalo_l",
        root=str(model_root),
        allowed_modules=["detection", "recognition"],
        providers=providers,
    )
    analysis.prepare(ctx_id=0 if providers[0] == "CUDAExecutionProvider" else -1, det_thresh=0.5, det_size=(640, 640))
    swapper = insightface.model_zoo.get_model(str(swapper_model_path.resolve()), providers=providers)

    source_image = cv2.imread(str(source_identity_path), cv2.IMREAD_COLOR)
    if source_image is None:
        raise RuntimeError(f"대체 ID 이미지를 읽지 못했습니다: {source_identity_path}")
    source_face = _largest_face(analysis.get(source_image, max_num=1))
    mp = None
    landmarker = None
    if preserve_performance:
        mp, landmarker = _create_performance_landmarker(landmarker_model_path)

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"입력 영상을 열지 못했습니다: {input_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_limit = None if max_seconds is None else max(1, round(max_seconds * fps))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    silent_path = output_path.with_name(output_path.stem + ".silent.mp4")
    writer = cv2.VideoWriter(str(silent_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"임시 출력 영상을 생성하지 못했습니다: {silent_path}")

    records: list[dict[str, Any]] = []
    processed = 0
    swapped = 0
    passthrough_frames = 0
    landmark_failures = 0
    previous_timestamp = -1
    started = time.perf_counter()
    try:
        while frame_limit is None or processed < frame_limit:
            ok, frame = capture.read()
            if not ok:
                break
            frame_started = time.perf_counter()
            record: dict[str, Any] = {
                "frame_index": processed,
                "timestamp_ms": round(processed * 1000.0 / fps),
                "swapped": False,
            }
            original_frame = frame.copy()
            performance_landmarks = None
            if landmarker is not None and mp is not None:
                timestamp_ms = max(previous_timestamp + 1, round(processed * 1000.0 / fps))
                previous_timestamp = timestamp_ms
                rgb = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
                media_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb.copy(),
                )
                landmark_result = landmarker.detect_for_video(media_image, timestamp_ms)
                if landmark_result.face_landmarks:
                    performance_landmarks = landmark_result.face_landmarks[0]
                else:
                    landmark_failures += 1

            faces = analysis.get(original_frame, max_num=1)
            if faces:
                target_face = _largest_face(faces)
                frame = swapper.get(original_frame, target_face, source_face, paste_back=True)
                swapped += 1
                record["swapped"] = True
                record["target_bbox"] = [round(float(value), 2) for value in target_face.bbox]
                record["detection_score"] = round(float(target_face.det_score), 6)
                if performance_landmarks is not None:
                    face_width = max(float(target_face.bbox[2] - target_face.bbox[0]), 1.0)
                    alpha = make_performance_mask(
                        performance_landmarks,
                        (width, height),
                        eye_expansion=eye_expansion,
                        mouth_expansion=mouth_expansion,
                        feather_pixels=max(1.0, face_width * feather_ratio),
                    )
                    frame = restore_performance_regions(original_frame, frame, alpha)
                    passthrough_frames += 1
                    record["performance_passthrough"] = True
                    record["passthrough_area_ratio"] = round(float(alpha.mean()), 7)
            else:
                record["error"] = "face_not_detected"
            record["processing_ms"] = round((time.perf_counter() - frame_started) * 1000.0, 3)
            records.append(record)
            writer.write(frame)
            processed += 1
    finally:
        capture.release()
        writer.release()
        if landmarker is not None:
            landmarker.close()

    elapsed = time.perf_counter() - started
    duration = processed / fps if fps else 0.0
    audio_muxed = _mux_audio(silent_path, input_path, output_path, duration)
    summary = SwapSummary(
        input_path=str(input_path),
        source_identity_path=str(source_identity_path),
        output_path=str(output_path),
        frames_processed=processed,
        frames_swapped=swapped,
        frames_failed=processed - swapped,
        source_fps=round(fps, 3),
        elapsed_seconds=round(elapsed, 3),
        processing_fps=round(processed / elapsed, 3) if elapsed else 0.0,
        realtime_factor=round(elapsed / duration, 3) if duration else 0.0,
        audio_muxed=audio_muxed,
        performance_passthrough=preserve_performance,
        passthrough_frames=passthrough_frames,
        landmark_failures=landmark_failures,
        eye_expansion=eye_expansion,
        mouth_expansion=mouth_expansion,
        feather_ratio=feather_ratio,
        model_license=(
            "InsightFace pretrained models: non-commercial research unless separately licensed; "
            "verify the MediaPipe task-model license before commercial use"
            if preserve_performance
            else "InsightFace pretrained models: non-commercial research unless separately licensed"
        ),
    )
    report_path = output_path.with_suffix(".json")
    report_path.write_text(
        json.dumps({"summary": asdict(summary), "frames": records}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
