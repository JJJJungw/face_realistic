"""passthrough 부위별 ablation — 표정 보존과 신원 누출의 교환곡선을 만든다.

■ 왜 필요한가
  현재 알려진 것은 두 극단뿐이다.
    passthrough 없음 → 신원 0.3532, 입 벌림 상관 0.118 (표정이 죽음)
    눈+입 전체 복원 → 신원 0.4142, 픽셀오차 76% 감소 (신원이 샘)
  중간을 모르면 어디를 깎아야 할지 알 수 없다. 부위를 하나씩 켜면서
  (표정 보존, 신원 누출) 두 숫자를 찍어 파레토 곡선을 만든다.

■ 설계
  스왑 결과는 마스크와 무관하다. 그래서 스왑은 딱 한 번만 돌리고
  프레임을 캐시한 뒤, 설정별로 합성과 측정만 반복한다.
  설정 8개를 위해 스왑을 8번 돌리는 것은 낭비다.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from face_realistic.identity.embedding import SFACE_LFW_COSINE_THRESHOLD, SFaceEmbedder, cosine_similarity
from face_realistic.io.model import ensure_face_landmarker_model, ensure_opencv_face_models
from face_realistic.swap.baseline import _largest_face, _load_insightface, resolve_providers
from face_realistic.swap.regions import build_region_mask, composite

# 사다리. 위에서 아래로 갈수록 원본을 더 많이 복원한다.
# 신원 단서가 약할 것으로 기대되는 부위부터 켠다.
LADDER: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("none", ()),
    ("A_mouth_inner", ("mouth_inner",)),
    ("B_plus_iris", ("mouth_inner", "iris")),
    ("C_plus_eye_open", ("mouth_inner", "iris", "eye_open")),
    ("D_plus_mouth_outer", ("mouth_inner", "iris", "eye_open", "mouth_outer")),
    ("E_plus_eyelid", ("mouth_inner", "iris", "eye_open", "mouth_outer", "eyelid")),
    ("F_plus_brow", ("mouth_inner", "iris", "eye_open", "mouth_outer", "eyelid", "brow")),
    ("legacy_current", ("legacy_eye", "legacy_mouth")),
)

UPPER_INNER_LIP = 13
LOWER_INNER_LIP = 14
FACE_LEFT = 234
FACE_RIGHT = 454
EAR_LEFT = ((159, 145), (33, 133))
EAR_RIGHT = ((386, 374), (362, 263))


@dataclass(frozen=True, slots=True)
class FrameCache:
    original: Path
    swapped: Path
    points: np.ndarray


def _landmarker(model_path: Path, video_mode: bool):
    import mediapipe as mp

    ensure_face_landmarker_model(model_path)
    mode = mp.tasks.vision.RunningMode.VIDEO if video_mode else mp.tasks.vision.RunningMode.IMAGE
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path.resolve())),
        running_mode=mode,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp, mp.tasks.vision.FaceLandmarker.create_from_options(options)


def _points(landmarks, width: int, height: int) -> np.ndarray:
    return np.asarray(
        [(float(p.x) * width, float(p.y) * height) for p in landmarks], dtype=np.float32
    )


def _detect_points(mp_module, landmarker, frame: np.ndarray) -> np.ndarray | None:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp_module.Image(image_format=mp_module.ImageFormat.SRGB, data=rgb.copy())
    result = landmarker.detect(image)
    if not result.face_landmarks:
        return None
    return _points(result.face_landmarks[0], frame.shape[1], frame.shape[0])


def _mouth_open(points: np.ndarray) -> float:
    face_width = float(np.linalg.norm(points[FACE_RIGHT] - points[FACE_LEFT]))
    if face_width <= 1e-6:
        return 0.0
    gap = float(np.linalg.norm(points[UPPER_INNER_LIP] - points[LOWER_INNER_LIP]))
    return gap / face_width


def _ear(points: np.ndarray, spec) -> float:
    (top, bottom), (left, right) = spec
    horizontal = float(np.linalg.norm(points[right] - points[left]))
    if horizontal <= 1e-6:
        return 0.0
    return float(np.linalg.norm(points[top] - points[bottom])) / horizontal


def _eye_open(points: np.ndarray) -> float:
    return 0.5 * (_ear(points, EAR_LEFT) + _ear(points, EAR_RIGHT))


def _pearson(a: list[float], b: list[float]) -> float:
    if len(a) < 3:
        return float("nan")
    mean_a, mean_b = statistics.mean(a), statistics.mean(b)
    da = [value - mean_a for value in a]
    db = [value - mean_b for value in b]
    denom = math.sqrt(sum(x * x for x in da)) * math.sqrt(sum(x * x for x in db))
    if denom <= 1e-12:
        return float("nan")
    return sum(x * y for x, y in zip(da, db, strict=True)) / denom


def build_cache(
    input_path: Path,
    identity_path: Path,
    cache_dir: Path,
    model_root: Path,
    swapper_model_path: Path,
    landmarker_model_path: Path,
    max_seconds: float | None,
    provider: str,
) -> dict[str, Any]:
    """스왑을 한 번만 돌려 원본·스왑 프레임과 랜드마크를 저장한다."""
    insightface, FaceAnalysis = _load_insightface()
    providers = resolve_providers(provider)
    analysis = FaceAnalysis(
        name="buffalo_l",
        root=str(model_root),
        allowed_modules=["detection", "recognition"],
        providers=providers,
    )
    analysis.prepare(
        ctx_id=0 if providers[0] == "CUDAExecutionProvider" else -1,
        det_thresh=0.5,
        det_size=(640, 640),
    )
    swapper = insightface.model_zoo.get_model(str(swapper_model_path.resolve()), providers=providers)

    identity_image = cv2.imread(str(identity_path), cv2.IMREAD_COLOR)
    if identity_image is None:
        raise RuntimeError(f"대체 ID 이미지를 읽지 못했습니다: {identity_path}")
    source_face = _largest_face(analysis.get(identity_image, max_num=1))

    mp_module, landmarker = _landmarker(landmarker_model_path, video_mode=True)
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"입력 영상을 열지 못했습니다: {input_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    limit = None if max_seconds in (None, 0) else max(1, round(max_seconds * fps))

    original_dir = cache_dir / "original"
    swapped_dir = cache_dir / "swapped"
    original_dir.mkdir(parents=True, exist_ok=True)
    swapped_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    index = 0
    previous_timestamp = -1
    started = time.perf_counter()
    try:
        while limit is None or index < limit:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp_ms = max(previous_timestamp + 1, round(index * 1000.0 / fps))
            previous_timestamp = timestamp_ms
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp_module.Image(image_format=mp_module.ImageFormat.SRGB, data=rgb.copy())
            landmark_result = landmarker.detect_for_video(image, timestamp_ms)
            faces = analysis.get(frame, max_num=1)
            usable = bool(faces) and bool(landmark_result.face_landmarks)
            if usable:
                target = _largest_face(faces)
                swapped = swapper.get(frame, target, source_face, paste_back=True)
                name = f"{index:06d}.png"
                cv2.imwrite(str(original_dir / name), frame)
                cv2.imwrite(str(swapped_dir / name), swapped)
                points = _points(landmark_result.face_landmarks[0], frame.shape[1], frame.shape[0])
                records.append(
                    {
                        "frame_index": index,
                        "name": name,
                        "face_width": round(
                            float(target.bbox[2] - target.bbox[0]), 3
                        ),
                        "points": [[round(float(x), 3), round(float(y), 3)] for x, y in points],
                    }
                )
            index += 1
    finally:
        capture.release()
        landmarker.close()

    manifest = {
        "input_path": str(input_path),
        "identity_path": str(identity_path),
        "fps": round(fps, 3),
        "frames_read": index,
        "frames_cached": len(records),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "landmark_count": len(records[0]["points"]) if records else 0,
        "frames": records,
    }
    (cache_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def _load_cache(cache_dir: Path) -> tuple[dict[str, Any], list[FrameCache]]:
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"캐시가 없습니다. --build-cache 로 먼저 만드세요: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames = [
        FrameCache(
            original=cache_dir / "original" / record["name"],
            swapped=cache_dir / "swapped" / record["name"],
            points=np.asarray(record["points"], dtype=np.float32),
        )
        for record in manifest["frames"]
    ]
    return manifest, frames


def measure(
    cache_dir: Path,
    yunet_path: Path,
    sface_path: Path,
    landmarker_model_path: Path,
    feather_ratio: float,
    write_video: bool,
    fps: float,
) -> dict[str, Any]:
    manifest, frames = _load_cache(cache_dir)
    if not frames:
        raise RuntimeError("캐시에 프레임이 없습니다.")
    yunet_path, sface_path = ensure_opencv_face_models(yunet_path, sface_path)
    embedder = SFaceEmbedder(yunet_path, sface_path)
    mp_module, landmarker = _landmarker(landmarker_model_path, video_mode=False)

    identity_image = cv2.imread(manifest["identity_path"], cv2.IMREAD_COLOR)
    master_embedding, _ = embedder.embed(identity_image)

    originals = [cv2.imread(str(item.original), cv2.IMREAD_COLOR) for item in frames]
    swapped = [cv2.imread(str(item.swapped), cv2.IMREAD_COLOR) for item in frames]

    # 기준선: 원본 프레임 자체의 표정·신원
    reference_mouth = [_mouth_open(item.points) for item in frames]
    reference_eye = [_eye_open(item.points) for item in frames]
    source_embeddings = []
    for frame in originals:
        try:
            embedding, _ = embedder.embed(frame)
        except RuntimeError:
            embedding = None
        source_embeddings.append(embedding)

    results: list[dict[str, Any]] = []
    try:
        for label, groups in LADDER:
            mouth, eye, to_source, to_master, areas = [], [], [], [], []
            paired_reference_mouth, paired_reference_eye = [], []
            writer = None
            if write_video:
                height, width = originals[0].shape[:2]
                writer = cv2.VideoWriter(
                    str(cache_dir / f"ablation_{label}.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height),
                )
            for position, item in enumerate(frames):
                original, swap_frame = originals[position], swapped[position]
                if groups:
                    height, width = original.shape[:2]
                    face_width = max(
                        float(np.linalg.norm(item.points[FACE_RIGHT] - item.points[FACE_LEFT])), 1.0
                    )
                    alpha = build_region_mask(
                        item.points,
                        (width, height),
                        groups,
                        feather_pixels=max(1.0, face_width * feather_ratio),
                    )
                    output = composite(original, swap_frame, alpha)
                    areas.append(float(alpha.mean()))
                else:
                    output = swap_frame
                    areas.append(0.0)
                if writer is not None:
                    writer.write(output)

                points = _detect_points(mp_module, landmarker, output)
                if points is not None:
                    mouth.append(_mouth_open(points))
                    eye.append(_eye_open(points))
                    paired_reference_mouth.append(reference_mouth[position])
                    paired_reference_eye.append(reference_eye[position])
                try:
                    embedding, _ = embedder.embed(output)
                except RuntimeError:
                    embedding = None
                if embedding is not None:
                    to_master.append(cosine_similarity(embedding, master_embedding))
                    if source_embeddings[position] is not None:
                        to_source.append(cosine_similarity(embedding, source_embeddings[position]))
            if writer is not None:
                writer.release()

            results.append(
                {
                    "label": label,
                    "regions": list(groups),
                    "measured_frames": len(mouth),
                    "identity_frames": len(to_source),
                    "mask_area_ratio": round(statistics.mean(areas), 6) if areas else 0.0,
                    "mouth_open_corr": round(_pearson(mouth, paired_reference_mouth), 4),
                    "mouth_open_mae": round(
                        statistics.mean(
                            [abs(a - b) for a, b in zip(mouth, paired_reference_mouth, strict=True)]
                        ),
                        6,
                    )
                    if mouth
                    else float("nan"),
                    "eye_open_corr": round(_pearson(eye, paired_reference_eye), 4),
                    "identity_to_source": round(statistics.mean(to_source), 4) if to_source else float("nan"),
                    "identity_to_master": round(statistics.mean(to_master), 4) if to_master else float("nan"),
                }
            )
            row = results[-1]
            print(
                f"{label:<20} 마스크 {row['mask_area_ratio']:.4f}  "
                f"입상관 {row['mouth_open_corr']:+.3f}  눈상관 {row['eye_open_corr']:+.3f}  "
                f"원본신원 {row['identity_to_source']:.4f}  master {row['identity_to_master']:.4f}",
                flush=True,
            )
    finally:
        landmarker.close()

    report = {
        "cache_dir": str(cache_dir),
        "input_path": manifest["input_path"],
        "identity_path": manifest["identity_path"],
        "frames": len(frames),
        "feather_ratio": feather_ratio,
        "sface_lfw_threshold": SFACE_LFW_COSINE_THRESHOLD,
        "reference_mouth_open_mean": round(statistics.mean(reference_mouth), 6),
        "reference_eye_open_mean": round(statistics.mean(reference_eye), 6),
        "configs": results,
        "notes": [
            "identity_to_source 가 낮을수록 비식별이 잘 된 것이다.",
            "0.363 은 SFace 의 LFW 보고 임계값이며 이 데이터의 운영 임계값이 아니다.",
            "표정 상관은 합성 결과에 랜드마커를 다시 돌려 측정한다.",
        ],
    }
    report_path = cache_dir / "ablation_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _draw_pareto(results, cache_dir / "ablation_pareto.png")
    return report


def _draw_pareto(results: list[dict[str, Any]], path: Path, size: int = 720) -> None:
    """가로 = 원본 신원 잔존, 세로 = 입 벌림 상관. 왼쪽 위가 좋다."""
    points = [
        (row["identity_to_source"], row["mouth_open_corr"], row["label"])
        for row in results
        if not math.isnan(row["identity_to_source"]) and not math.isnan(row["mouth_open_corr"])
    ]
    if not points:
        return
    margin = 90
    canvas = np.full((size, size, 3), 255, dtype=np.uint8)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_lo, x_hi = min(xs + [0.0]), max(xs + [0.6])
    y_lo, y_hi = min(ys + [0.0]), max(ys + [1.0])
    span_x = max(x_hi - x_lo, 1e-6)
    span_y = max(y_hi - y_lo, 1e-6)

    def to_px(x: float, y: float) -> tuple[int, int]:
        px = margin + int((x - x_lo) / span_x * (size - 2 * margin))
        py = size - margin - int((y - y_lo) / span_y * (size - 2 * margin))
        return px, py

    cv2.rectangle(canvas, (margin, margin), (size - margin, size - margin), (210, 210, 210), 1)
    if x_lo <= SFACE_LFW_COSINE_THRESHOLD <= x_hi:
        tx, _ = to_px(SFACE_LFW_COSINE_THRESHOLD, y_lo)
        cv2.line(canvas, (tx, margin), (tx, size - margin), (160, 160, 235), 1, cv2.LINE_AA)
        cv2.putText(canvas, "0.363", (tx + 4, margin + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 220), 1, cv2.LINE_AA)
    for x, y, label in points:
        px, py = to_px(x, y)
        cv2.circle(canvas, (px, py), 6, (60, 60, 60), -1, cv2.LINE_AA)
        cv2.putText(canvas, label, (px + 9, py - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.putText(canvas, "identity_to_source (low = de-identified)", (margin, size - margin + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.putText(canvas, "mouth_open_corr", (margin - 60, margin - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)


def main() -> None:
    parser = argparse.ArgumentParser(description="passthrough 부위별 ablation")
    parser.add_argument("--input", type=Path, default=Path("assets/source/swap2.mp4"))
    parser.add_argument("--identity", type=Path, default=Path("assets/identities/person_01/front_neutral.png"))
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/ablation_cache"))
    parser.add_argument("--model-root", type=Path, default=Path("models"))
    parser.add_argument("--swapper-model", type=Path, default=Path("models/inswapper_128.onnx"))
    parser.add_argument("--landmarker-model", type=Path, default=Path("models/face_landmarker.task"))
    parser.add_argument("--yunet", type=Path, default=Path("models/face_detection_yunet_2023mar.onnx"))
    parser.add_argument("--sface", type=Path, default=Path("models/face_recognition_sface_2021dec.onnx"))
    parser.add_argument("--max-seconds", type=float, default=3.0, help="0 이면 전체")
    parser.add_argument("--provider", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--feather-ratio", type=float, default=0.012)
    parser.add_argument("--reuse-cache", action="store_true", help="스왑을 다시 돌리지 않고 기존 캐시 사용")
    parser.add_argument("--write-video", action="store_true", help="설정마다 확인용 영상 저장")
    args = parser.parse_args()

    if not args.reuse_cache:
        manifest = build_cache(
            args.input,
            args.identity,
            args.cache_dir,
            args.model_root,
            args.swapper_model,
            args.landmarker_model,
            None if args.max_seconds == 0 else args.max_seconds,
            args.provider,
        )
        print(
            f"[cache] 프레임 {manifest['frames_cached']}/{manifest['frames_read']} "
            f"랜드마크 {manifest['landmark_count']}개 "
            f"{manifest['elapsed_seconds']:.1f}초",
            flush=True,
        )
    else:
        manifest = json.loads((args.cache_dir / "manifest.json").read_text(encoding="utf-8"))
        print(f"[cache] 재사용 {manifest['frames_cached']}프레임", flush=True)

    report = measure(
        args.cache_dir,
        args.yunet,
        args.sface,
        args.landmarker_model,
        args.feather_ratio,
        args.write_video,
        float(manifest["fps"]),
    )
    print(f"\n원본 입 벌림 평균 {report['reference_mouth_open_mean']:.4f}")
    print(f"보고서 → {args.cache_dir / 'ablation_report.json'}")
    print(f"그래프 → {args.cache_dir / 'ablation_pareto.png'}")


if __name__ == "__main__":
    main()
