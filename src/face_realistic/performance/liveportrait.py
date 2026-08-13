"""Run the official LivePortrait repository as an isolated expression baseline."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class VideoInfo:
    duration_seconds: float
    fps: float
    width: int
    height: int
    has_audio: bool


def build_liveportrait_command(
    python: Path,
    liveportrait_dir: Path,
    source_image: Path,
    driving_video: Path,
    output_dir: Path,
    device_id: int = 0,
    driving_multiplier: float = 1.0,
    animation_region: str = "exp",
) -> list[str]:
    """Build an upstream CLI command using its documented human-video defaults."""
    return [
        str(python),
        str(liveportrait_dir / "inference.py"),
        "-s",
        str(source_image),
        "-d",
        str(driving_video),
        "-o",
        str(output_dir),
        "--flag_crop_driving_video",
        "--animation_region",
        animation_region,
        "--driving_option",
        "expression-friendly",
        "--driving_multiplier",
        str(driving_multiplier),
        "--device_id",
        str(device_id),
    ]


def find_generated_video(output_dir: Path) -> Path:
    """Return the animation, excluding LivePortrait's three-panel comparison."""
    candidates = sorted(
        path
        for path in output_dir.glob("*.mp4")
        if not path.name.endswith("_concat.mp4")
    )
    if len(candidates) != 1:
        names = ", ".join(path.name for path in candidates) or "none"
        raise RuntimeError(f"Expected one LivePortrait video, found: {names}")
    return candidates[0]


def _run(command: Sequence[str], *, cwd: Path | None = None) -> None:
    subprocess.run(list(command), cwd=cwd, check=True)


def _probe_video(path: Path) -> VideoInfo:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,width,height,avg_frame_rate:format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    data = json.loads(result.stdout)
    video = next(stream for stream in data["streams"] if stream["codec_type"] == "video")
    numerator, denominator = video["avg_frame_rate"].split("/")
    fps = float(numerator) / float(denominator) if float(denominator) else 0.0
    return VideoInfo(
        duration_seconds=float(data["format"]["duration"]),
        fps=fps,
        width=int(video["width"]),
        height=int(video["height"]),
        has_audio=any(stream["codec_type"] == "audio" for stream in data["streams"]),
    )


def _require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def _require_executable(path: Path, label: str) -> Path:
    """Validate an executable without resolving a virtualenv Python symlink.

    Python uses the invoked ``.venv/bin/python`` path to discover its virtual
    environment. Resolving that symlink first would bypass the venv and run the
    underlying uv-managed interpreter directly.
    """
    candidate = path.expanduser().absolute()
    if not candidate.is_file():
        raise FileNotFoundError(f"{label} not found: {candidate}")
    return candidate


def run(args: argparse.Namespace) -> dict[str, object]:
    project_root = args.project_root.expanduser().resolve()
    liveportrait_dir = args.liveportrait_dir.expanduser().resolve()
    source_image = _require_file(args.source_image, "Source identity image")
    input_video = _require_file(args.input_video, "Driving video")
    python = _require_executable(
        liveportrait_dir / ".venv/bin/python", "LivePortrait Python"
    )
    _require_file(liveportrait_dir / "inference.py", "LivePortrait inference entrypoint")

    output_path = args.output.expanduser().resolve()
    report_path = args.report.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    work_dir = project_root / "outputs/liveportrait_work" / run_id
    raw_dir = work_dir / "animations"
    raw_dir.mkdir(parents=True)
    driving_clip = work_dir / "driving_3s.mp4"

    clip_command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_video),
    ]
    if args.max_seconds > 0:
        clip_command.extend(["-t", str(args.max_seconds)])
    clip_command.extend(
        [
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(driving_clip),
        ]
    )
    _run(clip_command)

    command = build_liveportrait_command(
        python=python,
        liveportrait_dir=liveportrait_dir,
        source_image=source_image,
        driving_video=driving_clip,
        output_dir=raw_dir,
        device_id=args.device_id,
        driving_multiplier=args.driving_multiplier,
        animation_region=args.animation_region,
    )
    started = time.perf_counter()
    _run(command, cwd=liveportrait_dir)
    elapsed = time.perf_counter() - started

    generated = find_generated_video(raw_dir)
    shutil.copy2(generated, output_path)
    video_info = _probe_video(output_path)
    report = {
        "pipeline": "LivePortrait official expression baseline",
        "input_path": str(input_video),
        "source_identity_path": str(source_image),
        "output_path": str(output_path),
        "raw_output_path": str(generated),
        "max_seconds": args.max_seconds,
        "device_id": args.device_id,
        "animation_region": args.animation_region,
        "relative_motion": True,
        "driving_option": "expression-friendly",
        "driving_multiplier": args.driving_multiplier,
        "elapsed_seconds": round(elapsed, 3),
        "realtime_factor": round(elapsed / video_info.duration_seconds, 3),
        "video": asdict(video_info),
        "license_note": (
            "LivePortrait code is MIT; bundled InsightFace detection weights are "
            "non-commercial research unless replaced or separately licensed."
        ),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument(
        "--liveportrait-dir", type=Path, default=root / "third_party/LivePortrait"
    )
    parser.add_argument(
        "--source-image",
        type=Path,
        default=root / "assets/identities/person_01/front_neutral.png",
    )
    parser.add_argument(
        "--input-video", type=Path, default=root / "assets/source/swap2.mp4"
    )
    parser.add_argument(
        "--output", type=Path, default=root / "outputs/liveportrait_baseline.mp4"
    )
    parser.add_argument(
        "--report", type=Path, default=root / "outputs/liveportrait_baseline.json"
    )
    parser.add_argument("--max-seconds", type=float, default=3.0)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--driving-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--animation-region",
        choices=("exp", "pose", "lip", "eyes", "all"),
        default="exp",
        help="LivePortrait motion region; exp isolates expression from pose and scale.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_seconds < 0:
        raise SystemExit("--max-seconds must be zero (full video) or greater")
    if args.driving_multiplier <= 0:
        raise SystemExit("--driving-multiplier must be greater than zero")
    run(args)


if __name__ == "__main__":
    main()
