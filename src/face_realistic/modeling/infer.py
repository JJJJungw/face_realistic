"""Run the clean-room face generator on a target video."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from face_realistic.io.model import ensure_face_landmarker_model
from face_realistic.modeling.network import CleanRoomFaceSwapModel, ModelConfig
from face_realistic.modeling.train import resolve_device
from face_realistic.performance.head_pose import matrix_to_head_pose


LEFT_EYE = (33, 133, 159, 145)
RIGHT_EYE = (362, 263, 386, 374)
MOUTH = (13, 14, 61, 291)


@dataclass(frozen=True, slots=True)
class InferenceSummary:
    input_path: str
    checkpoint_path: str
    source_image: str
    output_path: str
    device: str
    frames_processed: int
    frames_generated: int
    frames_failed: int
    source_fps: float
    elapsed_seconds: float
    processing_fps: float
    realtime_factor: float
    audio_muxed: bool
    motion_smoothing: float


def _load_mediapipe() -> Any:
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError("MediaPipe가 없습니다. 학습 환경을 다시 설치하세요.") from exc
    return mp


def _mean_point(
    landmarks: list[Any], indices: tuple[int, ...], width: int, height: int
) -> np.ndarray:
    return np.mean(
        [[landmarks[index].x * width, landmarks[index].y * height] for index in indices],
        axis=0,
        dtype=np.float32,
    )


def alignment_transform(
    landmarks: list[Any], frame_size: tuple[int, int], output_size: int
) -> np.ndarray:
    """Return the same frame-to-canonical transform used by identity registration."""
    width, height = frame_size
    eyes = [
        _mean_point(landmarks, LEFT_EYE, width, height),
        _mean_point(landmarks, RIGHT_EYE, width, height),
    ]
    image_left_eye, image_right_eye = sorted(eyes, key=lambda point: float(point[0]))
    mouth = _mean_point(landmarks, MOUTH, width, height)
    source = np.float32([image_left_eye, image_right_eye, mouth])
    target = np.float32(
        [
            [0.31 * output_size, 0.38 * output_size],
            [0.69 * output_size, 0.38 * output_size],
            [0.50 * output_size, 0.68 * output_size],
        ]
    )
    transform, _ = cv2.estimateAffinePartial2D(source, target, method=cv2.LMEDS)
    if transform is None:
        raise RuntimeError("얼굴 정렬 변환을 계산하지 못했습니다.")
    return transform.astype(np.float32)


def motion_vector(
    motion_names: list[str] | tuple[str, ...],
    blendshapes: list[Any],
    transformation_matrix: np.ndarray,
) -> np.ndarray:
    """Build the exact blendshape/head-pose vector stored in the checkpoint."""
    scores = {item.category_name: float(item.score) for item in blendshapes}
    pose = matrix_to_head_pose(np.asarray(transformation_matrix, dtype=float))
    first, second, third = pose["euler_xyz_deg"]
    pose_values = {
        "head_pitch": float(np.clip(first / 45.0, -1.0, 1.0)),
        "head_yaw": float(np.clip(second / 60.0, -1.0, 1.0)),
        "head_roll": float(np.clip(third / 45.0, -1.0, 1.0)),
    }
    values = [pose_values[name] if name in pose_values else scores.get(name, 0.0) for name in motion_names]
    return np.asarray(values, dtype=np.float32)


def paste_generated_crop(
    frame: np.ndarray,
    generated_bgr: np.ndarray,
    alpha: np.ndarray,
    frame_to_crop: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Inverse-warp generated RGB and alpha, then blend over the original frame."""
    height, width = frame.shape[:2]
    crop_to_frame = cv2.invertAffineTransform(frame_to_crop)
    warped_face = cv2.warpAffine(
        generated_bgr,
        crop_to_frame,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
    )
    warped_alpha = cv2.warpAffine(
        np.clip(alpha, 0.0, 1.0).astype(np.float32),
        crop_to_frame,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    warped_alpha = np.clip(warped_alpha, 0.0, 1.0)[..., None]
    composite = warped_face.astype(np.float32) * warped_alpha + frame.astype(np.float32) * (
        1.0 - warped_alpha
    )
    return np.clip(composite, 0, 255).round().astype(np.uint8), float(warped_alpha.mean())


def _image_tensor(image_bgr: np.ndarray, size: int, device: torch.device) -> torch.Tensor:
    resized = cv2.resize(image_bgr, (size, size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    array = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).unsqueeze(0).to(device)


def _generated_images(outputs: dict[str, torch.Tensor]) -> tuple[np.ndarray, np.ndarray]:
    generated = outputs["generated"][0].detach().float().cpu().clamp(-1, 1).numpy()
    generated = ((generated.transpose(1, 2, 0) + 1.0) * 127.5).round().astype(np.uint8)
    generated_bgr = cv2.cvtColor(generated, cv2.COLOR_RGB2BGR)
    alpha = outputs["alpha"][0, 0].detach().float().cpu().clamp(0, 1).numpy()
    return generated_bgr, alpha


def _mux_audio(silent_path: Path, source_path: Path, output_path: Path, duration: float) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        silent_path.replace(output_path)
        return False
    subprocess.run(
        [
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
        ],
        check=True,
    )
    silent_path.unlink(missing_ok=True)
    return True


def run_video_inference(args: argparse.Namespace) -> InferenceSummary:
    input_path = args.input.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"입력 영상이 없습니다: {input_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"체크포인트가 없습니다: {checkpoint_path}")
    if not 0.0 <= args.motion_smoothing < 1.0:
        raise ValueError("motion_smoothing은 0 이상 1 미만이어야 합니다.")

    device = resolve_device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = ModelConfig(**checkpoint["model_config"])
    motion_names = tuple(checkpoint["motion_names"])
    if len(motion_names) != config.motion_dim:
        raise ValueError("체크포인트의 motion_names와 motion_dim이 일치하지 않습니다.")
    model = CleanRoomFaceSwapModel(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()

    source_value = args.source_image or checkpoint.get("source_image")
    if not source_value:
        raise ValueError("체크포인트에 source_image가 없습니다. --source-image를 지정하세요.")
    source_path = Path(source_value).expanduser()
    if not source_path.is_absolute():
        source_path = (args.project_root.expanduser().resolve() / source_path).resolve()
    source_image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if source_image is None:
        raise FileNotFoundError(f"Source 이미지를 읽지 못했습니다: {source_path}")
    source_tensor = _image_tensor(source_image, config.image_size, device)

    model_path = ensure_face_landmarker_model(args.landmarker_model)
    mp = _load_mediapipe()
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path.resolve())),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"입력 영상을 열지 못했습니다: {input_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_limit = None if args.max_seconds is None else max(1, round(args.max_seconds * fps))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    silent_path = output_path.with_name(output_path.stem + ".silent.mp4")
    writer = cv2.VideoWriter(
        str(silent_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"임시 출력 영상을 생성하지 못했습니다: {silent_path}")

    records: list[dict[str, Any]] = []
    processed = generated_count = 0
    previous_timestamp = -1
    smoothed_motion: np.ndarray | None = None
    started = time.perf_counter()
    try:
        with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker, torch.inference_mode():
            while frame_limit is None or processed < frame_limit:
                ok, frame = capture.read()
                if not ok:
                    break
                frame_started = time.perf_counter()
                timestamp_ms = max(previous_timestamp + 1, round(processed * 1000.0 / fps))
                previous_timestamp = timestamp_ms
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                media_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)
                )
                result = landmarker.detect_for_video(media_image, timestamp_ms)
                record: dict[str, Any] = {
                    "frame_index": processed,
                    "timestamp_ms": timestamp_ms,
                    "generated": False,
                }
                if (
                    result.face_landmarks
                    and result.face_blendshapes
                    and result.facial_transformation_matrixes
                ):
                    landmarks = result.face_landmarks[0]
                    transform = alignment_transform(landmarks, (width, height), config.image_size)
                    target_crop = cv2.warpAffine(
                        frame,
                        transform,
                        (config.image_size, config.image_size),
                        flags=cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_REFLECT_101,
                    )
                    current_motion = motion_vector(
                        motion_names,
                        result.face_blendshapes[0],
                        np.asarray(result.facial_transformation_matrixes[0], dtype=float),
                    )
                    if smoothed_motion is None:
                        smoothed_motion = current_motion
                    else:
                        keep = args.motion_smoothing
                        smoothed_motion = keep * smoothed_motion + (1.0 - keep) * current_motion
                    target_tensor = _image_tensor(target_crop, config.image_size, device)
                    motion_tensor = torch.from_numpy(smoothed_motion.copy()).unsqueeze(0).to(device)
                    outputs = model(source_tensor, target_tensor, motion_tensor)
                    generated_crop, alpha = _generated_images(outputs)
                    frame, alpha_ratio = paste_generated_crop(frame, generated_crop, alpha, transform)
                    generated_count += 1
                    record.update(
                        {
                            "generated": True,
                            "alpha_area_ratio": round(alpha_ratio, 7),
                            "bbox_normalized": [
                                round(max(0.0, min(float(point.x) for point in landmarks)), 6),
                                round(max(0.0, min(float(point.y) for point in landmarks)), 6),
                                round(min(1.0, max(float(point.x) for point in landmarks)), 6),
                                round(min(1.0, max(float(point.y) for point in landmarks)), 6),
                            ],
                        }
                    )
                else:
                    smoothed_motion = None
                    record["error"] = "face_not_detected"
                record["processing_ms"] = round((time.perf_counter() - frame_started) * 1000.0, 3)
                records.append(record)
                writer.write(frame)
                processed += 1
    finally:
        capture.release()
        writer.release()

    elapsed = time.perf_counter() - started
    duration = processed / fps if fps else 0.0
    audio_muxed = _mux_audio(silent_path, input_path, output_path, duration)
    summary = InferenceSummary(
        input_path=str(input_path),
        checkpoint_path=str(checkpoint_path),
        source_image=str(source_path),
        output_path=str(output_path),
        device=str(device),
        frames_processed=processed,
        frames_generated=generated_count,
        frames_failed=processed - generated_count,
        source_fps=round(fps, 3),
        elapsed_seconds=round(elapsed, 3),
        processing_fps=round(processed / elapsed, 3) if elapsed else 0.0,
        realtime_factor=round(elapsed / duration, 3) if duration else 0.0,
        audio_muxed=audio_muxed,
        motion_smoothing=args.motion_smoothing,
    )
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pipeline": "clean-room one-identity video inference",
        "summary": asdict(summary),
        "frames": records,
        "limitations": [
            "one-identity training does not prove source identity conditioning",
            "the target low-frequency crop can leak target identity",
            "the current alpha supervision is an approximate oval rather than face parsing",
        ],
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument("--input", type=Path, default=root / "assets/source/swap2.mp4")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=root / "outputs/training/person_01_2k/checkpoint.pt",
    )
    parser.add_argument("--source-image", type=Path)
    parser.add_argument(
        "--output", type=Path, default=root / "outputs/cleanroom_swap_3s.mp4"
    )
    parser.add_argument(
        "--landmarker-model", type=Path, default=root / "models/face_landmarker.task"
    )
    parser.add_argument("--max-seconds", type=float, default=3.0)
    parser.add_argument("--motion-smoothing", type=float, default=0.65)
    parser.add_argument("--device", default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_seconds is not None and args.max_seconds <= 0:
        args.max_seconds = None
    run_video_inference(args)


if __name__ == "__main__":
    main()
