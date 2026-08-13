"""연구용 InSwapper 영상 기준선 CLI."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from face_realistic.swap.baseline import run_face_swap


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="InSwapper 128 영상 face-swap 기준선")
    parser.add_argument("--input", type=Path, default=Path("assets/source/swap2.mp4"))
    parser.add_argument(
        "--source-identity",
        type=Path,
        default=Path("assets/identities/person_01/front_neutral.png"),
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/swap_baseline.mp4"))
    parser.add_argument("--max-seconds", type=float, default=3.0, help="전체 영상은 0")
    parser.add_argument("--model-root", type=Path, default=Path("models/insightface"))
    parser.add_argument("--swapper-model", type=Path, default=Path("models/inswapper_128.onnx"))
    parser.add_argument("--provider", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--preserve-performance",
        action="store_true",
        help="원본 눈·입 픽셀을 soft mask로 복원하는 연구 실험",
    )
    parser.add_argument(
        "--landmarker-model", type=Path, default=Path("models/face_landmarker.task")
    )
    parser.add_argument("--eye-expansion", type=float, default=1.15)
    parser.add_argument("--mouth-expansion", type=float, default=1.35)
    parser.add_argument("--feather-ratio", type=float, default=0.012)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_face_swap(
        args.input,
        args.source_identity,
        args.output,
        args.model_root,
        args.swapper_model,
        max_seconds=None if args.max_seconds == 0 else args.max_seconds,
        provider=args.provider,
        preserve_performance=args.preserve_performance,
        landmarker_model_path=args.landmarker_model,
        eye_expansion=args.eye_expansion,
        mouth_expansion=args.mouth_expansion,
        feather_ratio=args.feather_ratio,
    )
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
