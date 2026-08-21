"""Train or smoke-test the clean-room 256px one-identity baseline."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from face_realistic.modeling.dataset import IdentityStillDataset
from face_realistic.modeling.losses import ReconstructionObjective
from face_realistic.modeling.network import CleanRoomFaceSwapModel, ModelConfig


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def _to_image(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().float().cpu().clamp(-1, 1).numpy()
    array = ((array.transpose(1, 2, 0) + 1.0) * 127.5).round().astype(np.uint8)
    return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)


def _save_preview(path: Path, batch: dict[str, torch.Tensor], outputs: dict[str, torch.Tensor]) -> None:
    alpha = outputs["alpha"][0].detach().float().cpu().numpy().transpose(1, 2, 0)
    alpha_image = np.repeat((alpha * 255).round().astype(np.uint8), 3, axis=2)
    panels = [
        _to_image(batch["source"][0]),
        _to_image(batch["target"][0]),
        _to_image(outputs["suppressed_target"][0]),
        _to_image(outputs["generated"][0]),
        _to_image(outputs["composite"][0]),
        alpha_image,
    ]
    labels = ("SOURCE ID", "TRAIN TARGET", "TARGET INPUT", "GENERATED", "COMPOSITE", "ALPHA")
    for panel, label in zip(panels, labels, strict=True):
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 28), (20, 20, 20), thickness=-1)
        cv2.putText(
            panel,
            label,
            (8, 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.hstack(panels))


def run_training(args: argparse.Namespace) -> dict[str, object]:
    project_root = args.project_root.expanduser().resolve()
    dataset = IdentityStillDataset(
        args.manifest,
        project_root=project_root,
        image_size=args.image_size,
        split="train",
        validation_fraction=args.validation_fraction,
    )
    device = resolve_device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    config = ModelConfig(
        motion_dim=dataset.motion_dim,
        image_size=args.image_size,
        base_channels=args.base_channels,
        max_channels=args.max_channels,
        identity_dim=args.identity_dim,
        motion_embedding_dim=args.motion_embedding_dim,
        target_bottleneck=args.target_bottleneck,
        condition_mode=args.condition_mode,
    )
    model = CleanRoomFaceSwapModel(config).to(device)
    objective = ReconstructionObjective().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    loader = DataLoader(
        dataset,
        batch_size=min(args.batch_size, len(dataset)),
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )

    iterator = iter(loader)
    started = time.perf_counter()
    last_losses: dict[str, float] = {}
    last_batch: dict[str, torch.Tensor] | None = None
    last_outputs: dict[str, torch.Tensor] | None = None
    model.train()
    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = {name: tensor.to(device) for name, tensor in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch["source"], batch["target"], batch["motion"])
        losses = objective(outputs, batch["target"], batch["face_mask"])
        losses["total"].backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        optimizer.step()
        last_losses = {name: round(float(value.detach().cpu()), 6) for name, value in losses.items()}
        last_losses["gradient_norm"] = round(float(gradient_norm.detach().cpu()), 6)
        last_batch, last_outputs = batch, outputs
        if step == 1 or step == args.steps or step % args.log_every == 0:
            print(json.dumps({"step": step, **last_losses}, ensure_ascii=False))

    elapsed = time.perf_counter() - started
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.pt"
    torch.save(
        {
            "schema_version": 1,
            "model_state_dict": model.state_dict(),
            "model_config": config.to_dict(),
            "motion_names": dataset.motion_names,
            "identity_id": dataset.info.identity_id,
            "source_image": dataset.info.source_image,
            "steps": args.steps,
        },
        checkpoint_path,
    )
    if last_batch is not None and last_outputs is not None:
        _save_preview(output_dir / "preview.jpg", last_batch, last_outputs)

    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pipeline": f"clean-room {args.condition_mode} one-identity reconstruction baseline",
        "device": str(device),
        "torch_version": torch.__version__,
        "dataset": asdict(dataset.info),
        "model_config": config.to_dict(),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "steps": args.steps,
        "elapsed_seconds": round(elapsed, 3),
        "last_losses": last_losses,
        "checkpoint_path": str(checkpoint_path),
        "preview_path": str(output_dir / "preview.jpg"),
        "limitations": [
            "one identity cannot validate identity disentanglement",
            "the first objective has no pretrained identity or perceptual loss",
            "still images cannot validate temporal consistency",
            *(
                ["motion-only conditioning has no spatial landmark or geometry map"]
                if args.condition_mode == "motion_only"
                else ["the low-frequency target can leak target identity"]
            ),
        ],
        "license_boundary": (
            "No InSwapper or GHOST checkpoint is loaded. Model weights are initialized "
            "from scratch and must only be trained on rights-cleared data."
        ),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "outputs/identity_registry/person_01/manifest.json",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=root / "outputs/training/person_01_overfit"
    )
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--target-bottleneck", type=int, default=16)
    parser.add_argument(
        "--condition-mode",
        choices=("target_lowpass", "motion_only"),
        default="motion_only",
    )
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--max-channels", type=int, default=256)
    parser.add_argument("--identity-dim", type=int, default=256)
    parser.add_argument("--motion-embedding-dim", type=int, default=128)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--gradient-clip", type=float, default=10.0)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.steps < 1 or args.batch_size < 1:
        raise SystemExit("--steps and --batch-size must be at least one")
    if args.log_every < 1:
        raise SystemExit("--log-every must be at least one")
    run_training(args)


if __name__ == "__main__":
    main()
