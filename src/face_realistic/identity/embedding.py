"""SFace 기반 얼굴 임베딩, centroid 및 원본-대체 ID 비교."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

import cv2
import numpy as np


SFACE_LFW_COSINE_THRESHOLD = 0.363


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(values))
    if norm <= 1e-12:
        raise ValueError("0 벡터는 정규화할 수 없습니다.")
    return values / norm


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(l2_normalize(left), l2_normalize(right)))


class SFaceEmbedder:
    def __init__(self, yunet_path: Path, sface_path: Path) -> None:
        self.detector = cv2.FaceDetectorYN_create(
            str(yunet_path.resolve()), "", (320, 320), 0.75, 0.3, 5000
        )
        self.recognizer = cv2.FaceRecognizerSF_create(str(sface_path.resolve()), "")

    def embed(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """가장 큰 얼굴을 정렬하고 L2 정규화 임베딩을 반환한다."""
        height, width = image.shape[:2]
        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(image)
        if faces is None or len(faces) == 0:
            raise RuntimeError("YuNet이 얼굴을 검출하지 못했습니다.")
        face = max(faces, key=lambda item: float(item[2] * item[3]))
        aligned = self.recognizer.alignCrop(image, face)
        feature = self.recognizer.feature(aligned)
        return l2_normalize(feature), aligned


def _weighted_centroid(features: list[np.ndarray], weights: list[float]) -> np.ndarray:
    if not features:
        raise ValueError("centroid를 만들 임베딩이 없습니다.")
    matrix = np.stack(features)
    weight_array = np.asarray(weights, dtype=np.float32)
    weight_array = weight_array / weight_array.sum()
    return l2_normalize(np.sum(matrix * weight_array[:, None], axis=0))


def _pairwise_scores(features: list[np.ndarray]) -> list[float]:
    return [
        cosine_similarity(features[left], features[right])
        for left in range(len(features))
        for right in range(left + 1, len(features))
    ]


def _source_video_embeddings(
    embedder: SFaceEmbedder,
    video_path: Path,
    sample_count: int,
    max_seconds: float,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"원본 영상을 열 수 없습니다: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    end_frame = min(total_frames, max(1, round(max_seconds * fps)))
    indices = sorted(set(int(value) for value in np.linspace(0, end_frame - 1, sample_count)))
    features: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    try:
        for frame_index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            record: dict[str, Any] = {
                "frame_index": frame_index,
                "timestamp_ms": round(frame_index * 1000.0 / fps),
                "detected": False,
            }
            if not ok:
                record["error"] = "frame_decode_failed"
                records.append(record)
                continue
            try:
                feature, _ = embedder.embed(frame)
                features.append(feature)
                record["detected"] = True
                record["embedding"] = [round(float(value), 8) for value in feature]
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
            records.append(record)
    finally:
        capture.release()
    return features, records


def build_identity_profile(
    manifest_path: Path,
    yunet_path: Path,
    sface_path: Path,
    source_video: Path | None = None,
    source_samples: int = 12,
    source_seconds: float = 3.0,
) -> dict[str, Any]:
    """등록 이미지 임베딩과 품질 가중 identity centroid를 생성한다."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_dir = manifest_path.parent
    embedder = SFaceEmbedder(yunet_path, sface_path)
    references = {Path(path).stem for path in manifest["recommended_references"]}

    image_records: list[dict[str, Any]] = []
    reference_features: list[np.ndarray] = []
    reference_weights: list[float] = []
    all_features: list[np.ndarray] = []
    for image_record in manifest["images"]:
        if not image_record["detected"]:
            continue
        aligned_path = Path(image_record["aligned_path"])
        source_path = Path(image_record["source_path"])
        record: dict[str, Any] = {
            "source_path": str(source_path),
            "aligned_path": str(aligned_path),
            "is_reference": aligned_path.stem in references,
            "embedded": False,
        }
        image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if image is None:
            record["error"] = "image_decode_failed"
            image_records.append(record)
            continue
        try:
            try:
                feature, _ = embedder.embed(image)
                record["embedding_input"] = "source"
            except RuntimeError:
                aligned_image = cv2.imread(str(aligned_path), cv2.IMREAD_COLOR)
                if aligned_image is None:
                    raise
                feature, _ = embedder.embed(aligned_image)
                record["embedding_input"] = "mediapipe_aligned_fallback"
            all_features.append(feature)
            record["embedded"] = True
            record["embedding"] = [round(float(value), 8) for value in feature]
            if record["is_reference"]:
                reference_features.append(feature)
                reference_weights.append(max(0.01, float(image_record["quality"]["total"])))
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        image_records.append(record)

    centroid = _weighted_centroid(reference_features, reference_weights)
    for record in image_records:
        if record["embedded"]:
            record["similarity_to_centroid"] = round(
                cosine_similarity(np.asarray(record["embedding"], dtype=np.float32), centroid), 6
            )

    reference_pairwise = _pairwise_scores(reference_features)
    all_similarities = [record["similarity_to_centroid"] for record in image_records if record["embedded"]]
    np.save(output_dir / "identity_centroid.npy", centroid)

    source_profile: dict[str, Any] | None = None
    if source_video is not None:
        source_features, source_records = _source_video_embeddings(
            embedder, source_video, source_samples, source_seconds
        )
        source_profile = {
            "video_path": str(source_video),
            "requested_samples": source_samples,
            "embedded_samples": len(source_features),
            "samples": source_records,
        }
        if source_features:
            source_centroid = _weighted_centroid(source_features, [1.0] * len(source_features))
            np.save(output_dir / "source_centroid.npy", source_centroid)
            similarity = cosine_similarity(centroid, source_centroid)
            source_to_target = [cosine_similarity(feature, centroid) for feature in source_features]
            source_internal = [cosine_similarity(feature, source_centroid) for feature in source_features]
            embedded_source_records = [record for record in source_records if record["detected"]]
            for record, target_score, internal_score in zip(
                embedded_source_records, source_to_target, source_internal, strict=True
            ):
                record["similarity_to_target"] = round(target_score, 6)
                record["similarity_to_source_centroid"] = round(internal_score, 6)
            source_profile.update(
                {
                    "cosine_similarity_to_target": round(similarity, 6),
                    "sample_similarity_to_target_min": round(min(source_to_target), 6),
                    "sample_similarity_to_target_mean": round(statistics.mean(source_to_target), 6),
                    "sample_similarity_to_target_max": round(max(source_to_target), 6),
                    "source_centroid_similarity_min": round(min(source_internal), 6),
                    "source_centroid_similarity_mean": round(statistics.mean(source_internal), 6),
                    "source_centroid_similarity_max": round(max(source_internal), 6),
                    "diagnostic_different_identity": similarity < SFACE_LFW_COSINE_THRESHOLD,
                }
            )

    profile = {
        "schema_version": 1,
        "identity_id": manifest["identity_id"],
        "embedding_model": "OpenCV SFace 2021-12",
        "embedding_dimension": int(centroid.size),
        "cosine_threshold_diagnostic": SFACE_LFW_COSINE_THRESHOLD,
        "embedded_images": len(all_features),
        "reference_images": len(reference_features),
        "centroid_path": str(output_dir / "identity_centroid.npy"),
        "within_identity": {
            "centroid_similarity_min": round(min(all_similarities), 6),
            "centroid_similarity_mean": round(statistics.mean(all_similarities), 6),
            "centroid_similarity_max": round(max(all_similarities), 6),
            "reference_pairwise_min": round(min(reference_pairwise), 6),
            "reference_pairwise_mean": round(statistics.mean(reference_pairwise), 6),
            "reference_pairwise_max": round(max(reference_pairwise), 6),
            "images_below_diagnostic_threshold": sum(
                score < SFACE_LFW_COSINE_THRESHOLD for score in all_similarities
            ),
        },
        "source_comparison": source_profile,
        "images": image_records,
        "notes": [
            "0.363은 SFace의 LFW 보고 임계값이며 이 데이터셋의 운영 임계값이 아닙니다.",
            "익명성 판정에는 별도 얼굴 인식 모델과 실제 데이터 기반 ROC 교차 검증이 필요합니다.",
        ],
    }
    profile_path = output_dir / "identity_profile.json"
    profile_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return profile
