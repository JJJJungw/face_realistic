"""대체 ID 등록 명령행 인터페이스."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from face_realistic.identity.register import register_identity
from face_realistic.io.model import ensure_face_landmarker_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="대체 얼굴 ID 이미지 등록")
    parser.add_argument("--identity-dir", type=Path, default=Path("assets/identities/person_01"))
    parser.add_argument("--model", type=Path, default=Path("models/face_landmarker.task"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/identity_registry"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model_path = ensure_face_landmarker_model(args.model)
    manifest = register_identity(args.identity_dir, model_path, args.output_root)
    print(
        json.dumps(
            {
                "identity_id": manifest["identity_id"],
                "total_images": manifest["total_images"],
                "registered_images": manifest["registered_images"],
                "failed_images": manifest["failed_images"],
                "mean_quality": manifest["mean_quality"],
                "representative": manifest["representative"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
