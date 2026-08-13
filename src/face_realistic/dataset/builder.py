"""Build a reproducible one-identity motion-transfer dataset index."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from face_realistic.performance.liveportrait import (
    build_liveportrait_command,
    find_generated_video,
)


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass(frozen=True, slots=True)
class VideoProbe:
    duration_seconds: float
    fps: float
    width: int
    height: int
    estimated_frames: int


def discover_files(root: Path, extensions: set[str]) -> list[Path]:
    """Discover supported files deterministically and ignore hidden files."""
    if not root.is_dir():
        return []
    return sorted(
        path.resolve()
        for path in root.rglob("*")
        if path.is_file()
        and not path.name.startswith(".")
        and path.suffix.lower() in extensions
    )


def make_job_id(path: Path, relative_to: Path) -> str:
    """Create a readable, stable identifier without filename collisions."""
    relative = path.resolve().relative_to(relative_to.resolve()).as_posix()
    safe_stem = "".join(
        character if character.isascii() and character.isalnum() else "_"
        for character in path.stem
    ).strip("_") or "video"
    digest = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:8]
    return f"{safe_stem}_{digest}"


def probe_video(path: Path) -> VideoProbe:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate:format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    data = json.loads(result.stdout)
    stream = data["streams"][0]
    numerator, denominator = stream["avg_frame_rate"].split("/")
    fps = float(numerator) / float(denominator) if float(denominator) else 0.0
    duration = float(data["format"]["duration"])
    return VideoProbe(
        duration_seconds=duration,
        fps=fps,
        width=int(stream["width"]),
        height=int(stream["height"]),
        estimated_frames=max(0, round(duration * fps)),
    )


def build_pair_records(
    *,
    identity_id: str,
    job_id: str,
    source_image: Path,
    driving_video: Path,
    generated_video: Path,
    driving: VideoProbe,
    generated: VideoProbe,
) -> Iterable[dict[str, object]]:
    """Yield timestamp-aligned lazy frame pairs without duplicating video data."""
    frame_count = min(driving.estimated_frames, generated.estimated_frames)
    fps = generated.fps or driving.fps
    for frame_index in range(frame_count):
        yield {
            "schema_version": 1,
            "identity_id": identity_id,
            "job_id": job_id,
            "frame_index": frame_index,
            "timestamp_ms": round(frame_index * 1000.0 / fps, 3),
            "source_image": str(source_image),
            "driving_video": str(driving_video),
            "generated_video": str(generated_video),
        }


def _run(command: Sequence[str], *, cwd: Path | None = None) -> None:
    subprocess.run(list(command), cwd=cwd, check=True)


def _make_driving_clip(source: Path, output: Path, max_seconds: float) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
    ]
    if max_seconds > 0:
        command.extend(["-t", str(max_seconds)])
    command.extend(
        [
            "-map",
            "0:v:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-movflags",
            "+faststart",
            str(output),
        ]
    )
    _run(command)


def _render_job(
    *,
    liveportrait_dir: Path,
    source_image: Path,
    driving_clip: Path,
    generated_video: Path,
    work_dir: Path,
    device_id: int,
    animation_region: str,
    driving_multiplier: float,
) -> None:
    python = liveportrait_dir / ".venv/bin/python"
    inference = liveportrait_dir / "inference.py"
    if not python.is_file() or not inference.is_file():
        raise FileNotFoundError(
            "LivePortrait 환경이 없습니다. 먼저 "
            "bash scripts/setup_liveportrait_ec2.sh 를 실행하세요."
        )
    raw_dir = work_dir / "animations"
    raw_dir.mkdir(parents=True, exist_ok=True)
    command = build_liveportrait_command(
        python=python,
        liveportrait_dir=liveportrait_dir,
        source_image=source_image,
        driving_video=driving_clip,
        output_dir=raw_dir,
        device_id=device_id,
        driving_multiplier=driving_multiplier,
        animation_region=animation_region,
    )
    _run(command, cwd=liveportrait_dir)
    generated_video.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(find_generated_video(raw_dir), generated_video)


def build_dataset(args: argparse.Namespace) -> dict[str, object]:
    identity_dir = args.identity_dir.expanduser().resolve()
    driving_dir = args.driving_dir.expanduser().resolve()
    source_image = args.source_image.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    liveportrait_dir = args.liveportrait_dir.expanduser().resolve()

    identity_images = discover_files(identity_dir, IMAGE_EXTENSIONS)
    driving_videos = discover_files(driving_dir, VIDEO_EXTENSIONS)
    if not identity_images:
        raise FileNotFoundError(f"Identity images not found: {identity_dir}")
    if source_image not in identity_images:
        raise FileNotFoundError(f"Source identity image not found: {source_image}")
    if not driving_videos:
        raise FileNotFoundError(f"Driving videos not found: {driving_dir}")

    dataset_dir = output_root / args.identity_id
    clips_dir = dataset_dir / "driving"
    generated_dir = dataset_dir / "generated"
    work_root = dataset_dir / "work"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[dict[str, object]] = []
    pair_records: list[dict[str, object]] = []
    projected_frames = 0
    rendered_frames = 0

    for driving_video in driving_videos:
        job_id = make_job_id(driving_video, driving_dir)
        input_probe = probe_video(driving_video)
        selected_duration = (
            min(input_probe.duration_seconds, args.max_seconds)
            if args.max_seconds > 0
            else input_probe.duration_seconds
        )
        projected = round(selected_duration * input_probe.fps)
        projected_frames += projected
        clip_path = clips_dir / f"{job_id}.mp4"
        generated_path = generated_dir / f"{job_id}.mp4"

        status = "planned"
        if args.render:
            if args.overwrite or not generated_path.is_file():
                _make_driving_clip(driving_video, clip_path, args.max_seconds)
                _render_job(
                    liveportrait_dir=liveportrait_dir,
                    source_image=source_image,
                    driving_clip=clip_path,
                    generated_video=generated_path,
                    work_dir=work_root / job_id,
                    device_id=args.device_id,
                    animation_region=args.animation_region,
                    driving_multiplier=args.driving_multiplier,
                )
            elif not clip_path.is_file():
                _make_driving_clip(driving_video, clip_path, args.max_seconds)

        if clip_path.is_file() and generated_path.is_file():
            driving_probe = probe_video(clip_path)
            generated_probe = probe_video(generated_path)
            records = list(
                build_pair_records(
                    identity_id=args.identity_id,
                    job_id=job_id,
                    source_image=source_image,
                    driving_video=clip_path,
                    generated_video=generated_path,
                    driving=driving_probe,
                    generated=generated_probe,
                )
            )
            pair_records.extend(records)
            rendered_frames += len(records)
            status = "ready"

        jobs.append(
            {
                "job_id": job_id,
                "status": status,
                "input_video": str(driving_video),
                "input_probe": asdict(input_probe),
                "selected_duration_seconds": round(selected_duration, 3),
                "projected_frames": projected,
                "driving_clip": str(clip_path),
                "generated_video": str(generated_path),
            }
        )

    pairs_path = dataset_dir / "pairs.jsonl"
    pairs_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in pair_records)
    )
    readiness = (
        "ready_for_overfit_poc"
        if rendered_frames >= args.minimum_frames
        else "needs_more_rendered_frames"
        if args.render
        else "ready_to_render"
    )
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "identity_id": args.identity_id,
        "identity_dir": str(identity_dir),
        "identity_image_count": len(identity_images),
        "source_image": str(source_image),
        "driving_video_count": len(driving_videos),
        "projected_frames": projected_frames,
        "rendered_pair_frames": rendered_frames,
        "minimum_frames_for_overfit_poc": args.minimum_frames,
        "readiness": readiness,
        "pairs_path": str(pairs_path),
        "render_enabled": args.render,
        "generator": {
            "name": "LivePortrait",
            "animation_region": args.animation_region,
            "driving_multiplier": args.driving_multiplier,
            "bootstrap_only": True,
        },
        "jobs": jobs,
        "warning": (
            "This is synthetic bootstrap data for a one-identity overfit test. "
            "It does not demonstrate cross-identity generalization and may contain "
            "LivePortrait-specific artifacts."
        ),
    }
    manifest_path = dataset_dir / "manifest.json"
    manifest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-id", default="person_01")
    parser.add_argument(
        "--identity-dir", type=Path, default=root / "assets/identities/person_01"
    )
    parser.add_argument(
        "--source-image",
        type=Path,
        default=root / "assets/identities/person_01/front_neutral.png",
    )
    parser.add_argument("--driving-dir", type=Path, default=root / "assets/source")
    parser.add_argument("--output-root", type=Path, default=root / "outputs/datasets")
    parser.add_argument(
        "--liveportrait-dir", type=Path, default=root / "third_party/LivePortrait"
    )
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=0.0,
        help="Maximum seconds per driver; zero uses the complete video.",
    )
    parser.add_argument("--minimum-frames", type=int, default=750)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--driving-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--animation-region",
        choices=("exp", "pose", "lip", "eyes", "all"),
        default="all",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_seconds < 0:
        raise SystemExit("--max-seconds must be zero (full video) or greater")
    if args.minimum_frames < 1:
        raise SystemExit("--minimum-frames must be at least one")
    if args.driving_multiplier <= 0:
        raise SystemExit("--driving-multiplier must be greater than zero")
    build_dataset(args)


if __name__ == "__main__":
    main()
