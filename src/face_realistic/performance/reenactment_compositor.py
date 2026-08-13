"""Paste a one-shot reenacted master face back into the original scene."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from face_realistic.io.model import ensure_face_landmarker_model


# MediaPipe face oval. The polygon is contracted before blending so hair, ears,
# neck, and most of the source portrait background never enter the target scene.
FACE_OVAL = (
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
)

# Relatively stable central points for similarity alignment. Using only three
# expression-sensitive points made the pasted face pulse while speaking.
ALIGNMENT_LANDMARKS = (33, 263, 168, 6, 197, 195)


def landmark_points(landmarks: Sequence[Any], indices: Sequence[int], size: tuple[int, int]) -> np.ndarray:
    """Convert normalized MediaPipe landmarks to pixel coordinates."""
    width, height = size
    return np.float32(
        [[float(landmarks[index].x) * width, float(landmarks[index].y) * height] for index in indices]
    )


def estimate_face_transform(
    generated_landmarks: Sequence[Any],
    target_landmarks: Sequence[Any],
    generated_size: tuple[int, int],
    target_size: tuple[int, int],
) -> np.ndarray:
    """Estimate a rotation/scale/translation transform from generated to target."""
    source = landmark_points(generated_landmarks, ALIGNMENT_LANDMARKS, generated_size)
    target = landmark_points(target_landmarks, ALIGNMENT_LANDMARKS, target_size)
    transform, _ = cv2.estimateAffinePartial2D(source, target, method=cv2.LMEDS)
    if transform is None:
        raise RuntimeError("Could not align the reenacted face to the target")
    return transform.astype(np.float32)


def smooth_transform(previous: np.ndarray | None, current: np.ndarray, alpha: float) -> np.ndarray:
    """EMA the similarity parameters without blending arbitrary affine matrices."""
    if previous is None:
        return current
    return ((1.0 - alpha) * previous + alpha * current).astype(np.float32)


def make_face_mask(
    generated_landmarks: Sequence[Any],
    generated_size: tuple[int, int],
    target_size: tuple[int, int],
    transform: np.ndarray,
    contract: float,
    feather_pixels: float,
) -> np.ndarray:
    """Make a soft inner-face mask in target coordinates."""
    polygon = landmark_points(generated_landmarks, FACE_OVAL, generated_size)
    center = polygon.mean(axis=0, keepdims=True)
    polygon = center + (polygon - center) * contract
    polygon = cv2.transform(polygon[None, :, :], transform)[0]
    width, height = target_size
    polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
    mask = np.zeros((height, width), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.round(polygon).astype(np.int32), 1.0, lineType=cv2.LINE_AA)
    if feather_pixels > 0:
        kernel = max(3, round(feather_pixels * 4) | 1)
        mask = cv2.GaussianBlur(mask, (kernel, kernel), feather_pixels)
    return np.clip(mask, 0.0, 1.0)


def composite_face(original: np.ndarray, warped_face: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    alpha_3d = alpha[..., None]
    output = original.astype(np.float32) * (1.0 - alpha_3d) + warped_face.astype(np.float32) * alpha_3d
    return np.clip(output, 0, 255).astype(np.uint8)


def _create_landmarker(model_path: Path) -> tuple[Any, Any]:
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError("MediaPipe is required for reenactment paste-back") from exc
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


def _detect(mp: Any, landmarker: Any, frame: np.ndarray, timestamp_ms: int) -> Sequence[Any] | None:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    result = landmarker.detect_for_video(image, timestamp_ms)
    return result.face_landmarks[0] if result.face_landmarks else None


def _mux_audio(silent_path: Path, source_path: Path, output_path: Path, duration: float) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        silent_path.replace(output_path)
        return False
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
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
            "fast",
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


def run_compositor(args: argparse.Namespace) -> dict[str, object]:
    original_path = args.original.expanduser().resolve()
    reenacted_path = args.reenacted.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    report_path = args.report.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    for path, label in ((original_path, "Original video"), (reenacted_path, "Reenacted video")):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")
    if not 0.5 <= args.mask_contract <= 1.0:
        raise ValueError("mask_contract must be between 0.5 and 1.0")
    if not 0.0 < args.transform_alpha <= 1.0:
        raise ValueError("transform_alpha must be in (0, 1]")

    original_capture = cv2.VideoCapture(str(original_path))
    reenacted_capture = cv2.VideoCapture(str(reenacted_path))
    if not original_capture.isOpened() or not reenacted_capture.isOpened():
        original_capture.release()
        reenacted_capture.release()
        raise RuntimeError("Could not open input videos")
    fps = float(original_capture.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(original_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(original_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    generated_width = int(reenacted_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    generated_height = int(reenacted_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_limit = None if args.max_seconds == 0 else max(1, round(args.max_seconds * fps))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    silent_path = output_path.with_name(output_path.stem + ".silent.mp4")
    writer = cv2.VideoWriter(str(silent_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        original_capture.release()
        reenacted_capture.release()
        raise RuntimeError(f"Could not create output: {silent_path}")

    mp, original_landmarker = _create_landmarker(model_path)
    _, generated_landmarker = _create_landmarker(model_path)
    processed = 0
    composited = 0
    failures = 0
    previous_transform: np.ndarray | None = None
    records: list[dict[str, object]] = []
    try:
        while frame_limit is None or processed < frame_limit:
            original_ok, original = original_capture.read()
            generated_ok, generated = reenacted_capture.read()
            if not original_ok or not generated_ok:
                break
            timestamp_ms = round(processed * 1000.0 / fps)
            original_landmarks = _detect(mp, original_landmarker, original, timestamp_ms)
            generated_landmarks = _detect(mp, generated_landmarker, generated, timestamp_ms)
            record: dict[str, object] = {"frame_index": processed, "timestamp_ms": timestamp_ms}
            if original_landmarks is not None and generated_landmarks is not None:
                try:
                    transform = estimate_face_transform(
                        generated_landmarks,
                        original_landmarks,
                        (generated_width, generated_height),
                        (width, height),
                    )
                    transform = smooth_transform(previous_transform, transform, args.transform_alpha)
                    previous_transform = transform
                    warped = cv2.warpAffine(
                        generated,
                        transform,
                        (width, height),
                        flags=cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_REFLECT_101,
                    )
                    mask = make_face_mask(
                        generated_landmarks,
                        (generated_width, generated_height),
                        (width, height),
                        transform,
                        contract=args.mask_contract,
                        feather_pixels=args.feather_pixels,
                    )
                    original = composite_face(original, warped, mask)
                    composited += 1
                    record.update(
                        {
                            "composited": True,
                            "mask_area_ratio": round(float(mask.mean()), 7),
                            "transform": np.round(transform, 5).tolist(),
                        }
                    )
                except RuntimeError as exc:
                    failures += 1
                    record.update({"composited": False, "error": str(exc)})
            else:
                failures += 1
                record.update({"composited": False, "error": "face_not_detected"})
            records.append(record)
            writer.write(original)
            processed += 1
    finally:
        original_capture.release()
        reenacted_capture.release()
        writer.release()
        original_landmarker.close()
        generated_landmarker.close()

    duration = processed / fps
    audio_muxed = _mux_audio(silent_path, original_path, output_path, duration)
    report = {
        "pipeline": "one-shot reenactment face-only paste-back",
        "original_path": str(original_path),
        "reenacted_path": str(reenacted_path),
        "output_path": str(output_path),
        "frames_processed": processed,
        "frames_composited": composited,
        "frames_failed": failures,
        "source_fps": round(fps, 3),
        "mask_contract": args.mask_contract,
        "feather_pixels": args.feather_pixels,
        "transform_alpha": args.transform_alpha,
        "audio_muxed": audio_muxed,
        "occlusion_restoration": False,
        "records": records,
        "license_note": (
            "LivePortrait code is MIT; its bundled InsightFace detection weights are "
            "non-commercial research unless replaced or separately licensed."
        ),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    printable = {key: value for key, value in report.items() if key != "records"}
    print(json.dumps(printable, ensure_ascii=False, indent=2))
    return report


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, default=root / "assets/source/swap2.mp4")
    parser.add_argument(
        "--reenacted", type=Path, default=root / "outputs/liveportrait_all_baseline.mp4"
    )
    parser.add_argument(
        "--output", type=Path, default=root / "outputs/master_reenactment_swap.mp4"
    )
    parser.add_argument(
        "--report", type=Path, default=root / "outputs/master_reenactment_swap.json"
    )
    parser.add_argument("--model", type=Path, default=root / "models/face_landmarker.task")
    parser.add_argument("--max-seconds", type=float, default=3.0)
    parser.add_argument("--mask-contract", type=float, default=0.88)
    parser.add_argument("--feather-pixels", type=float, default=9.0)
    parser.add_argument("--transform-alpha", type=float, default=0.35)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_seconds < 0:
        raise SystemExit("--max-seconds must be zero or greater")
    if args.feather_pixels < 0:
        raise SystemExit("--feather-pixels must be zero or greater")
    run_compositor(args)


if __name__ == "__main__":
    main()
