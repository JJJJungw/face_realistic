"""명령행 인터페이스."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from face_realistic.config import TrackingConfig
from face_realistic.io.model import ensure_face_landmarker_model
from face_realistic.tracking.mediapipe_tracker import track_video


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MediaPipe 얼굴 퍼포먼스 추적 MVP")
    parser.add_argument("--input", type=Path, default=Path("assets/source/swap2.mp4"))
    parser.add_argument("--model", type=Path, default=Path("models/face_landmarker.task"))
    parser.add_argument("--output-jsonl", type=Path, default=Path("outputs/tracking.jsonl"))
    parser.add_argument("--output-preview", type=Path, default=Path("outputs/tracking_preview.mp4"))
    parser.add_argument("--max-seconds", type=float, default=3.0, help="처리 길이. 전체 영상은 0")
    parser.add_argument("--num-faces", type=int, default=1)
    parser.add_argument("--no-preview", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    max_seconds = None if args.max_seconds == 0 else args.max_seconds
    model_path = ensure_face_landmarker_model(args.model)
    config = TrackingConfig(
        input_path=args.input,
        model_path=model_path,
        jsonl_path=args.output_jsonl,
        preview_path=None if args.no_preview else args.output_preview,
        max_seconds=max_seconds,
        num_faces=args.num_faces,
    )
    summary = track_video(config)
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))
