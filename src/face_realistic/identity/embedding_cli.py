"""얼굴 임베딩 프로파일 생성 CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from face_realistic.identity.embedding import build_identity_profile
from face_realistic.io.model import ensure_opencv_face_models


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SFace identity centroid 및 원본 비교 생성")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/identity_registry/person_01/manifest.json"),
    )
    parser.add_argument("--source-video", type=Path, default=Path("assets/source/swap2.mp4"))
    parser.add_argument("--source-samples", type=int, default=12)
    parser.add_argument("--source-seconds", type=float, default=3.0)
    parser.add_argument("--yunet-model", type=Path, default=Path("models/face_detection_yunet_2023mar.onnx"))
    parser.add_argument("--sface-model", type=Path, default=Path("models/face_recognition_sface_2021dec.onnx"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    yunet_path, sface_path = ensure_opencv_face_models(args.yunet_model, args.sface_model)
    profile = build_identity_profile(
        args.manifest,
        yunet_path,
        sface_path,
        source_video=args.source_video,
        source_samples=args.source_samples,
        source_seconds=args.source_seconds,
    )
    print(
        json.dumps(
            {
                "identity_id": profile["identity_id"],
                "embedded_images": profile["embedded_images"],
                "reference_images": profile["reference_images"],
                "within_identity": profile["within_identity"],
                "source_comparison": {
                    key: value
                    for key, value in (profile["source_comparison"] or {}).items()
                    if key != "samples"
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
