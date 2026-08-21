"""Benchmark the untrained FRS-v1 forward path on CPU or CUDA."""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import torch

from .config import V1ModelConfig
from .model import FaceRealisticSwapV1


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(value)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_smoke(
    *,
    device: torch.device,
    config: V1ModelConfig,
    references: int = 1,
    warmup: int = 10,
    iterations: int = 100,
) -> dict[str, Any]:
    if not 1 <= references <= config.max_references:
        raise ValueError(f"references must be in [1, {config.max_references}]")
    if warmup < 0 or iterations < 1:
        raise ValueError("warmup must be non-negative and iterations must be positive")

    model = FaceRealisticSwapV1(config).to(device).eval()
    size = config.image_size
    geometry_size = size // 4
    inputs = {
        "reference_faces": torch.zeros(1, references, 3, size, size, device=device),
        "geometry_maps": torch.zeros(
            1, config.geometry_channels, geometry_size, geometry_size, device=device
        ),
        "blendshapes": torch.zeros(1, config.blendshape_dim, device=device),
        "head_pose": torch.zeros(1, config.pose_dim, device=device),
        "lowfreq_illumination": torch.zeros(1, 3, 16, 16, device=device),
        "context_ring": torch.zeros(1, 3, size, size, device=device),
        "occlusion_mask": torch.zeros(1, 1, size, size, device=device),
    }
    inputs["geometry_maps"][:, -1] = 1.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        for _ in range(warmup):
            outputs = model(**inputs)
        _synchronize(device)
        started = time.perf_counter()
        for _ in range(iterations):
            outputs = model(**inputs)
        _synchronize(device)
    elapsed = time.perf_counter() - started

    report: dict[str, Any] = {
        "pipeline": "FRS-v1 untrained architecture smoke benchmark",
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
        ),
        "torch_version": torch.__version__,
        "image_size": size,
        "references": references,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "warmup": warmup,
        "iterations": iterations,
        "elapsed_seconds": round(elapsed, 4),
        "latency_ms_per_frame": round(elapsed * 1000.0 / iterations, 3),
        "generator_fps": round(iterations / elapsed, 3),
        "output_shapes": {
            name: list(outputs[name].shape)
            for name in ("generated", "alpha", "visibility", "confidence")
        },
        "trained_weights": False,
    }
    if device.type == "cuda":
        report["peak_cuda_memory_mb"] = round(
            torch.cuda.max_memory_allocated(device) / (1024 * 1024), 2
        )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--references", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_smoke(
        device=resolve_device(args.device),
        config=V1ModelConfig(image_size=args.image_size),
        references=args.references,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
